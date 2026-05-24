#!/usr/bin/env python3
"""
Monte Carlo simulation: structural limitations of VaR model selection
under a known GARCH(1,1)-t data generating process.

Shows that even with perfect knowledge of the DGP, flexible models (XGBoost
quantile) fail to improve VaR calibration when backtesting samples are small,
and that Kupiec/Christoffersen tests have low power under those conditions.

Self-contained: does NOT import from the fibras pipeline.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, norm, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
import time
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})

# ============================================================
# CONFIGURATION
# ============================================================

GARCH_OMEGA = 0.05
GARCH_ALPHA = 0.10
GARCH_BETA  = 0.85
GARCH_NU    = 6.0

N_TOTAL       = 1500
N_SIM         = 200
T_TRAIN       = 125
T_TEST_VALUES = [58, 250]
ALPHAS        = [0.05, 0.01]

XGB_PARAMS = {
    "n_estimators": 50,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": 1,
}

OUTPUT_DIR  = "output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RNG_SEED    = 42

# ============================================================
# DATA GENERATING PROCESS
# ============================================================

def simulate_garch_t(n, omega, alpha, beta, nu, rng):
    """
    Simulate GARCH(1,1)-t returns.

    r_t  = sigma_t * epsilon_t
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
    epsilon_t ~ t_nu  (standardised to unit variance)

    Returns (returns, sigma2, eps).
    """
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    r = np.zeros(n)

    sigma2[0] = omega / (1.0 - alpha - beta)
    scale = np.sqrt((nu - 2.0) / nu) if nu > 2 else 1.0
    eps[0] = rng.standard_t(nu) * scale
    r[0] = np.sqrt(sigma2[0]) * eps[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = rng.standard_t(nu) * scale
        r[t] = np.sqrt(sigma2[t]) * eps[t]

    return r, sigma2, eps


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def fit_garch(returns, dist="t"):
    """Fit GARCH(1,1) with specified distribution. Returns fitted result or None."""
    try:
        from arch import arch_model
        import pandas as pd
        y = pd.Series(returns, dtype=float)
        model = arch_model(y, vol="Garch", p=1, q=1, dist=dist)
        res = model.fit(disp="off")
        return res
    except Exception:
        return None


def garch_forecast_1step(res, returns_train, returns_test, alpha):
    """
    One-step-ahead GARCH VaR forecasts using fitted result.

    The model is fitted on returns_train; conditional variance is
    updated day-by-day through the test period with observed returns,
    then converted to VaR using the appropriate quantile.
    """
    params = res.params
    omega_val  = params["omega"]
    a_val      = params.get("alpha[1]", params.get("alpha.1", 0.0))
    b_val      = params.get("beta[1]", params.get("beta.1", 0.0))
    nu_val     = params.get("nu", params.get("eta", 5.0))

    cond_vol = res.conditional_volatility
    if hasattr(cond_vol, "iloc"):
        cond_var = float(cond_vol.iloc[-1] ** 2)
    else:
        cond_var = float(cond_vol[-1] ** 2)

    all_ret = np.concatenate([returns_train, returns_test])
    m = len(returns_test)
    var = np.zeros(m)

    for i in range(m):
        t = len(returns_train) + i
        cond_var = omega_val + a_val * (all_ret[t - 1] ** 2) + b_val * cond_var
        cond_var = max(cond_var, 1e-10)
        if dist_is_t(res):
            z = t_dist.ppf(alpha, nu_val)
        else:
            z = norm.ppf(alpha)
        var[i] = -np.sqrt(cond_var) * z

    return var


def dist_is_t(res):
    """Heuristic: check if GARCH result was estimated with Student-t distribution."""
    params = res.params
    return ("nu" in params) or ("eta" in params)


def compute_historical_var(returns_full, train_end, test_start, test_end,
                           alpha, window):
    """
    Rolling historical VaR for a forward test window.

    For each day in [test_start, test_end), VaR is the alpha-quantile
    of the `window` trading days immediately before that day.
    No look-ahead.
    """
    n_test = test_end - test_start
    var_arr = np.full(n_test, np.nan)
    for i in range(n_test):
        idx = test_start + i
        if idx < window:
            continue
        w = returns_full[max(0, idx - window):idx]
        q = np.percentile(w, alpha * 100.0)
        var_arr[i] = -q
    return var_arr


def build_xgb_features(returns, start_idx, end_idx, max_lag=5):
    """
    Build feature matrix and target vector for XGBoost quantile regression.
    Features at time t => predict target at time t+1.

    Features (10 + 2):
        - ret_lag{1..5}
        - abs_ret_lag{1..5}
        - hist_vol_20
        - hist_vol_5
    """
    feats = []
    targs = []
    min_idx = start_idx + max_lag
    if end_idx - min_idx < 30:
        return np.empty((0, 12)), np.empty(0)

    for t in range(min_idx, end_idx - 1):
        row = []
        for k in range(1, max_lag + 1):
            row.append(returns[t - k])
        for k in range(1, max_lag + 1):
            row.append(abs(returns[t - k]))
        row.append(np.std(returns[max(0, t - 20):t]) if t >= 20 else np.std(returns[:t]))
        row.append(np.std(returns[max(0, t - 5):t]) if t >= 5 else np.std(returns[:t]))
        feats.append(row)
        targs.append(max(-returns[t + 1], 0.0))

    return np.array(feats), np.array(targs)


def build_xgb_features_single(returns, idx, max_lag=5):
    """Feature vector for a single prediction at time idx."""
    row = []
    for k in range(1, max_lag + 1):
        row.append(returns[idx - k])
    for k in range(1, max_lag + 1):
        row.append(abs(returns[idx - k]))
    row.append(np.std(returns[max(0, idx - 20):idx]) if idx >= 20 else np.std(returns[:idx]))
    row.append(np.std(returns[max(0, idx - 5):idx]) if idx >= 5 else np.std(returns[:idx]))
    return np.array(row).reshape(1, -1)


def xgb_quantile_forecasts(returns, train_start, train_end, test_start, test_end,
                           alpha, xgb_params):
    """
    Train XGBoost quantile model on [train_start, train_end), predict on
    [test_start, test_end). Uses quantile_alpha = 1 - alpha.
    """
    X_train, y_train = build_xgb_features(returns, train_start, train_end)
    n_test = test_end - test_start

    if X_train.shape[0] < 30:
        return np.full(n_test, np.nan)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    quantile_alpha = 1.0 - alpha
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile_alpha,
        **xgb_params
    )
    model.fit(X_scaled, y_train)

    forecasts = np.zeros(n_test)
    for i in range(n_test):
        X_pred = build_xgb_features_single(returns, test_start + i)
        X_pred_scaled = scaler.transform(X_pred)
        pred = model.predict(X_pred_scaled)[0]
        forecasts[i] = max(pred, 1e-8)

    return forecasts


# ============================================================
# BACKTEST METRICS
# ============================================================

def kupiec_pof(breaches, alpha):
    """Kupiec Proportion-of-Failures test p-value."""
    n = len(breaches)
    x = int(np.sum(breaches))
    if n == 0:
        return np.nan
    if x == 0:
        lr = -2.0 * np.log((1.0 - alpha) ** n)
    elif x == n:
        lr = -2.0 * np.log(alpha ** n)
    else:
        p_hat = x / n
        lr = -2.0 * ((n - x) * np.log((1.0 - alpha) / (1.0 - p_hat))
                     + x * np.log(alpha / p_hat))
    return float(1.0 - chi2.cdf(lr, df=1))


def christoffersen_independence(breaches):
    """Christoffersen independence test p-value."""
    n = len(breaches)
    if n < 2:
        return np.nan
    b = np.asarray(breaches, dtype=int)
    n00 = int(np.sum((b[:-1] == 0) & (b[1:] == 0)))
    n01 = int(np.sum((b[:-1] == 0) & (b[1:] == 1)))
    n10 = int(np.sum((b[:-1] == 1) & (b[1:] == 0)))
    n11 = int(np.sum((b[:-1] == 1) & (b[1:] == 1)))

    if n01 == 0 or n11 == 0:
        return 1.0

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p   = (n01 + n11) / n

    num = (1.0 - p) ** (n00 + n10) * p ** (n01 + n11)
    den = (1.0 - p01) ** n00 * p01 ** n01 * (1.0 - p11) ** n10 * p11 ** n11
    if num == 0 or den == 0:
        return 1.0
    lr_ind = -2.0 * np.log(num / den)
    return float(1.0 - chi2.cdf(lr_ind, df=1))


def es_realized(returns, var_forecasts):
    """Realised Expected Shortfall: mean loss on breach days."""
    mask = returns < -var_forecasts
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(-returns[mask]))


# ============================================================
# ONE SIMULATION (ALL MODELS, BOTH ALPHAS, BOTH T_TEST)
# ============================================================

MODEL_NAMES = ["GARCH-t", "XGBoost-Q", "Historical-VaR", "GARCH-Normal"]


def run_single_simulation(sim_id):
    """
    Execute one Monte Carlo replication.

    Generates a single return path from the known GARCH-t DGP, then
    evaluates all models at both test-window sizes (T_test = 58, 250).
    Returns a list of dicts (one per model/alpha/T_test combination).
    """
    rng = np.random.default_rng(RNG_SEED + sim_id * 1000 + 1)

    # 1. Generate data
    returns, _, _ = simulate_garch_t(
        N_TOTAL, GARCH_OMEGA, GARCH_ALPHA, GARCH_BETA, GARCH_NU, rng
    )

    # --- Cache GARCH fits (same for both t_test values since train end shifts) ---
    garch_fits = {}   # (train_start, train_end) -> fitted result
    xgb_models = {}   # (train_start, train_end, alpha) -> (scaler, model)

    all_results = []

    for t_test in T_TEST_VALUES:
        train_end   = N_TOTAL - t_test        # exclusive
        test_start  = N_TOTAL - t_test
        test_end    = N_TOTAL
        model_train_start = max(0, train_end - T_TRAIN)

        returns_train_model = returns[model_train_start:train_end]
        returns_test        = returns[test_start:test_end]

        # --- Fit / retrieve GARCH models ---
        garch_key = (model_train_start, train_end)
        if garch_key not in garch_fits:
            garch_fits[garch_key] = {
                "t": fit_garch(returns_train_model, dist="t"),
                "normal": fit_garch(returns_train_model, dist="normal"),
            }

        for alpha in ALPHAS:
            expected = alpha

            # ---- GARCH-t ----
            res_t = garch_fits[garch_key]["t"]
            if res_t is not None:
                var_gt = garch_forecast_1step(
                    res_t, returns_train_model, returns_test, alpha
                )
            else:
                var_gt = np.full(t_test, np.nan)

            b_gt  = (returns_test < -var_gt).astype(int)
            all_results.append({
                "sim_id": sim_id, "model": "GARCH-t", "alpha": alpha,
                "t_test": t_test, "breach_rate": b_gt.mean(),
                "kupiec_pval": kupiec_pof(b_gt, expected),
                "christo_pval": christoffersen_independence(b_gt),
                "es_realized": es_realized(returns_test, var_gt),
            })

            # ---- GARCH-Normal ----
            res_n = garch_fits[garch_key]["normal"]
            if res_n is not None:
                var_gn = garch_forecast_1step(
                    res_n, returns_train_model, returns_test, alpha
                )
            else:
                var_gn = np.full(t_test, np.nan)

            b_gn = (returns_test < -var_gn).astype(int)
            all_results.append({
                "sim_id": sim_id, "model": "GARCH-Normal", "alpha": alpha,
                "t_test": t_test, "breach_rate": b_gn.mean(),
                "kupiec_pval": kupiec_pof(b_gn, expected),
                "christo_pval": christoffersen_independence(b_gn),
                "es_realized": es_realized(returns_test, var_gn),
            })

            # ---- XGBoost Quantile ----
            xgb_key = (model_train_start, train_end, alpha)
            if xgb_key not in xgb_models:
                # Train once, cache scaler+model
                X, y = build_xgb_features(
                    returns, model_train_start, train_end
                )
                if X.shape[0] >= 30:
                    scaler = StandardScaler()
                    X_s = scaler.fit_transform(X)
                    model = xgb.XGBRegressor(
                        objective="reg:quantileerror",
                        quantile_alpha=1.0 - alpha,
                        **XGB_PARAMS
                    )
                    model.fit(X_s, y)
                    xgb_models[xgb_key] = (scaler, model)
                else:
                    xgb_models[xgb_key] = None

            cached_xgb = xgb_models[xgb_key]
            if cached_xgb is not None:
                scaler, model = cached_xgb
                forecasts = np.zeros(t_test)
                for i in range(t_test):
                    Xp = build_xgb_features_single(returns, test_start + i)
                    Xp_s = scaler.transform(Xp)
                    pred = model.predict(Xp_s)[0]
                    forecasts[i] = max(pred, 1e-8)
            else:
                forecasts = np.full(t_test, np.nan)

            b_xgb = (returns_test < -forecasts).astype(int)
            all_results.append({
                "sim_id": sim_id, "model": "XGBoost-Q", "alpha": alpha,
                "t_test": t_test,
                "breach_rate": b_xgb.mean() if not np.isnan(b_xgb).all() else np.nan,
                "kupiec_pval": kupiec_pof(b_xgb, expected),
                "christo_pval": christoffersen_independence(b_xgb),
                "es_realized": es_realized(returns_test, forecasts),
            })

            # ---- Historical VaR ----
            var_hist = compute_historical_var(
                returns, train_end, test_start, test_end, alpha, T_TRAIN
            )
            b_hist = (returns_test < -var_hist).astype(int)
            all_results.append({
                "sim_id": sim_id, "model": "Historical-VaR", "alpha": alpha,
                "t_test": t_test, "breach_rate": b_hist.mean(),
                "kupiec_pval": kupiec_pof(b_hist, expected),
                "christo_pval": christoffersen_independence(b_hist),
                "es_realized": es_realized(returns_test, var_hist),
            })

    return all_results


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 64)
    print(" Monte Carlo Simulation: Structural Limitations of")
    print(" VaR Model Selection under Known GARCH-t DGP")
    print("=" * 64)
    print(f" DGP parameters:  omega={GARCH_OMEGA}, alpha={GARCH_ALPHA}, "
          f"beta={GARCH_BETA}, nu={GARCH_NU}")
    print(f" N_total={N_TOTAL}   N_sim={N_SIM}   T_train={T_TRAIN}")
    print(f" T_test values: {T_TEST_VALUES}")
    print("=" * 64)

    tasks = list(range(N_SIM))
    print(f"\nRunning {len(tasks)} simulations in parallel ...")
    t0 = time.time()

    all_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_simulation)(sim_id) for sim_id in tasks
    )

    elapsed = time.time() - t0
    print(f"\nSimulations completed in {elapsed / 60:.1f} minutes "
          f"({elapsed:.0f} s)")

    # Flatten
    flat = []
    for batch in all_results:
        flat.extend(batch)
    df = pd.DataFrame(flat)

    # Save raw
    raw_path = os.path.join(OUTPUT_DIR, "monte_carlo_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Raw results saved to {raw_path}  ({len(df)} rows)")

    # ---------- summary table ----------
    summary_rows = []
    for model in MODEL_NAMES:
        for alpha in ALPHAS:
            for t_test in T_TEST_VALUES:
                sub = df[(df["model"] == model) &
                         (df["alpha"] == alpha) &
                         (df["t_test"] == t_test)]
                if len(sub) == 0:
                    continue
                br   = sub["breach_rate"].dropna()
                kp   = sub["kupiec_pval"].dropna()
                cp   = sub["christo_pval"].dropna()
                esr  = sub["es_realized"].dropna()

                summary_rows.append({
                    "model": model, "alpha": alpha, "t_test": t_test,
                    "n_sim": len(sub),
                    "mean_breach_rate":     br.mean(),
                    "std_breach_rate":      br.std(),
                    "kupiec_rejection_rate": (kp < 0.05).mean(),
                    "christo_rejection_rate": (cp < 0.05).mean(),
                    "mean_es_realized":     esr.mean(),
                    "std_es_realized":      esr.std(),
                    "mean_kupiec_pval":     kp.mean(),
                    "mean_christo_pval":    cp.mean(),
                })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "monte_carlo_summary.csv")
    df_summary.round(6).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    print("\n" + "=" * 64)
    print(" SUMMARY TABLE")
    print("=" * 64)
    for t_test in T_TEST_VALUES:
        print(f"\n--- T_test = {t_test} ---")
        print(df_summary[df_summary["t_test"] == t_test]
              .round(4).to_string(index=False))

    # ---------- FIGURE: bar chart of rejection rates ----------
    colors = {
        "GARCH-t":        "#2E86AB",
        "XGBoost-Q":      "#A23B72",
        "Historical-VaR": "#F18F01",
        "GARCH-Normal":   "#C73E1D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, alpha in enumerate(ALPHAS):
        ax = axes[idx]
        label_alpha = f"{int((1 - alpha) * 100)}%"

        x_labels = []
        bar_vals = []
        cols     = []
        for t_test in T_TEST_VALUES:
            for model in MODEL_NAMES:
                row = df_summary[
                    (df_summary["model"] == model) &
                    (df_summary["alpha"] == alpha) &
                    (df_summary["t_test"] == t_test)
                ]
                if len(row) == 0:
                    continue
                x_labels.append(f"{model}\nn={t_test}")
                bar_vals.append(row["kupiec_rejection_rate"].values[0])
                cols.append(colors.get(model, "#999999"))

        x_pos = np.arange(len(x_labels))
        ax.bar(x_pos, bar_vals, color=cols, alpha=0.88,
               edgecolor="black", linewidth=0.4)
        ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=1,
                   label=r"$\alpha$=0.05 (size)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Kupiec Rejection Rate")
        ax.set_title(f"VaR {label_alpha}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(0.15, max(bar_vals) * 1.25))

    fig.suptitle("Monte Carlo: Empirical Kupiec Rejection Rates — Known GARCH-t DGP",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "monte_carlo_power.pdf")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    # ---------- FIGURE: empirical power curves ----------
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, t_test in enumerate(T_TEST_VALUES):
        ax = axes2[idx]
        sub = df[df["t_test"] == t_test]

        for alpha, ls, lw in [(0.05, "-", 1.8), (0.01, "--", 1.4)]:
            for model, color, marker in [("GARCH-t", colors["GARCH-t"], "o"),
                                          ("XGBoost-Q", colors["XGBoost-Q"], "s")]:
                s = sub[(sub["model"] == model) & (sub["alpha"] == alpha)]
                brs = s["breach_rate"].dropna().values
                kps = s["kupiec_pval"].dropna().values

                if len(brs) < 10:
                    continue

                # Sort by breach rate, bin for smooth power curve
                order = np.argsort(brs)
                brs_sorted = brs[order]
                kps_sorted = kps[order]

                n_bins = 20
                bin_edges = np.linspace(brs_sorted.min(), brs_sorted.max(), n_bins + 1)
                power = []
                centres = []
                for j in range(n_bins):
                    mask = (brs_sorted >= bin_edges[j]) & (brs_sorted < bin_edges[j + 1])
                    if mask.sum() < 5:
                        continue
                    centres.append(0.5 * (bin_edges[j] + bin_edges[j + 1]))
                    power.append((kps_sorted[mask] < 0.05).mean())

                if len(centres) > 2:
                    ax.plot(centres, power, color=color, linestyle=ls,
                            linewidth=lw, marker=marker, markersize=3,
                            label=f"{model}  " + r"$\alpha$=" + f"{alpha}")

        ax.axhline(y=0.80, color="gray", linestyle=":", linewidth=0.8,
                   label="80% power")
        ax.axvline(x=0.05, color="black", linestyle="--", linewidth=0.5,
                   alpha=0.5, label=r"H$_0$: p=" + f"{0.05}")
        ax.set_xlabel("Empirical Breach Rate")
        ax.set_ylabel("Kupiec Rejection Rate (Power)")
        ax.set_title(f"Empirical Power Curve — n={t_test}")
        ax.legend(fontsize=7, loc="upper left")

    fig2.suptitle("Kupiec Test Power: Known GARCH-t DGP",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    pcurve_path = os.path.join(FIGURES_DIR, "monte_carlo_power_curves.pdf")
    fig2.savefig(pcurve_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Power-curve figure saved to {pcurve_path}")

    # ---------- quick sanity check ----------
    print("\n--- Internal sanity check ---")
    gt = df[(df["model"] == "GARCH-t") & (df["alpha"] == 0.05) & (df["t_test"] == 250)]
    if len(gt) > 0:
        avg_br = gt["breach_rate"].dropna().mean()
        print(f"  GARCH-t (alpha=0.05, T_test=250): mean breach rate = {avg_br:.4f} "
              f"(expected {0.05:.4f})")
    else:
        print("  No valid GARCH-t rows to check.")

    print("\nDone.")
    print("=" * 64)


if __name__ == "__main__":
    main()
