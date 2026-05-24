"""
LIGHT-RUN SCRIPT — Quick backtest on a single ticker (FUNO11.MX) at both α=0.05 and 0.01.
Uses optimized variants: vectorized Historical VaR, fast XGBoost quantile (no tuning).
Saves output to output/light_run/.
"""
import os
import time
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from fibras.data_loader import load_ticker_data, load_fx_data
from fibras.regimes import assign_regimes_exante
from fibras.features import create_features, get_feature_columns
from fibras.garch_models import generate_all_garch_forecasts
from fibras.xgboost_models import (
    rolling_xgboost_quantile_fast,
    prepare_pure_xgboost_features,
    prepare_var_target,
)
from fibras.backtest import (
    compute_historical_var_series,
    var_from_volatility,
    expected_shortfall_realized,
    evaluate_model,
)

OUTPUT_DIR = "output/light_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKER = "FUNO11.MX"
WINDOW = 125
REFIT_FREQ = 20
ALPHAS = [0.05, 0.01]

print("=" * 70)
print(f"LIGHT RUN — Single Ticker: {TICKER}")
print(f"Both α levels: {[f'{100*(1-a):.0f}%' for a in ALPHAS]}")
print(f"Window={WINDOW}, Refit={REFIT_FREQ}")
print(f"Using FAST XGBoost variants (fixed params, no CV tuning)")
print("=" * 70)

# -----------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------
t0 = time.time()
print("\n[1/6] Loading data...")
df_ticker = load_ticker_data(TICKER)
df_fx = load_fx_data()
print(f"  {TICKER}: {len(df_ticker)} obs, period {df_ticker.index[0].date()} to {df_ticker.index[-1].date()}")
print(f"  FX: {len(df_fx)} obs")

# -----------------------------------------------------------------------
# 2. Regime assignment
# -----------------------------------------------------------------------
print("\n[2/6] Assigning ex-ante regimes...")
df_ticker = assign_regimes_exante(df_ticker, vol_window=20, lookback=500, threshold_percentile=90.0)
print(f"  Regime counts: {df_ticker['regime'].value_counts().to_dict()}")

# -----------------------------------------------------------------------
# 3. GARCH forecasts (alpha-independent)
# -----------------------------------------------------------------------
print("\n[3/6] Computing GARCH-t forecasts...")
t_garch = time.time()
garch_forecasts = generate_all_garch_forecasts(
    returns=df_ticker['log_return'],
    window=WINDOW,
    distributions=["t"],
    refit_freq=REFIT_FREQ,
    verbose=True
)
print(f"  GARCH done in {time.time() - t_garch:.1f}s")

# -----------------------------------------------------------------------
# 4. Prepare XGBoost feature sets (alpha-independent)
# -----------------------------------------------------------------------
print("\n[4/6] Preparing XGBoost features...")
feat_pure, pure_cols = prepare_pure_xgboost_features(df_ticker, df_fx, lags=20)
feat_pure = prepare_var_target(feat_pure)
print(f"  Feature matrix: {feat_pure.shape}, {len(pure_cols)} features")

# -----------------------------------------------------------------------
# 5. Per-alpha loop: compute forecasts, align, backtest
# -----------------------------------------------------------------------
print("\n[5/6] Computing model forecasts per alpha...")

all_results = []
all_aligned = {}

for ALPHA in ALPHAS:
    alpha_tag = "p95" if ALPHA == 0.05 else "p99"
    alpha_pct = f"{100*(1-ALPHA):.0f}%"
    print(f"\n  --- Alpha = {ALPHA:.2f} ({alpha_pct} VaR) ---")

    t_alpha = time.time()

    # XGBoost quantile (FAST, no tuning)
    t_xgb = time.time()
    xgb_var_forecast = rolling_xgboost_quantile_fast(
        df=feat_pure,
        feature_cols=pure_cols,
        alpha=ALPHA,
        target_col='target_var',
        window=WINDOW,
        refit_freq=REFIT_FREQ,
        n_estimators=50,
        verbose=False
    )
    xgb_time = time.time() - t_xgb
    print(f"    XGBoost quantile fast: {xgb_time:.1f}s")

    # Align forecasts
    returns_series = df_ticker['log_return']
    common_index = (
        returns_series.index
        .intersection(garch_forecasts['t'].dropna().index)
        .intersection(xgb_var_forecast.dropna().index)
        .intersection(feat_pure.index)
    )

    aligned = pd.DataFrame(index=common_index)
    aligned['return'] = df_ticker.loc[common_index, 'log_return']
    aligned['regime'] = df_ticker.loc[common_index, 'regime']

    # GARCH-t VaR (parametric)
    aligned['vol_garch_t'] = garch_forecasts['t'].loc[common_index]
    aligned['var_garch_t'] = var_from_volatility(aligned['vol_garch_t'], ALPHA, 't')

    # XGBoost quantile VaR (direct prediction)
    aligned['var_xgb_q'] = xgb_var_forecast.loc[common_index]

    # Historical VaR
    aligned['var_hist'] = compute_historical_var_series(
        returns_series, WINDOW, ALPHA, common_index
    )

    print(f"    Aligned observations: {len(aligned)}")
    regime_counts = aligned['regime'].value_counts(dropna=True)
    print(f"    Regime distribution: {regime_counts.to_dict()}")

    # Expected Shortfall (realized, ex-post)
    for model_var_col in ['var_garch_t', 'var_xgb_q', 'var_hist']:
        es_col = f"es_{model_var_col[4:]}"
        aligned[es_col] = expected_shortfall_realized(
            aligned['return'], aligned[model_var_col]
        )

    # Evaluate all models
    models = {
        'GARCH-t': 'var_garch_t',
        'XGBoost-Quantile': 'var_xgb_q',
        'Historical-VaR': 'var_hist',
    }

    for name, var_col in models.items():
        metrics = evaluate_model(aligned['return'], aligned[var_col], ALPHA)
        metrics['Model'] = name
        metrics['alpha'] = ALPHA
        metrics['alpha_label'] = alpha_pct
        metrics['ticker'] = TICKER
        all_results.append(metrics)

    # Print per-model summary
    for name, var_col in models.items():
        row = [r for r in all_results if r['Model'] == name and r['alpha'] == ALPHA][-1]
        print(f"    {name:20s} breach_rate={row['breach_rate']:.4f}  "
              f"kupiec_pval={row['kupiec_pval']:.4f}  "
              f"ES={row['es_realized']:.6f}" if not np.isnan(row['es_realized']) else
              f"    {name:20s} breach_rate={row['breach_rate']:.4f}  "
              f"kupiec_pval={row['kupiec_pval']:.4f}  ES=nan")

    aligned.to_csv(f"{OUTPUT_DIR}/aligned_forecasts_{alpha_tag}.csv")
    all_aligned[ALPHA] = aligned

    print(f"    Alpha loop time: {time.time() - t_alpha:.1f}s")

# -----------------------------------------------------------------------
# 6. Summary table
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[6/6] RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(all_results)
display_cols = ['alpha_label', 'Model', 'n_obs', 'breach_rate', 'expected_rate',
                'kupiec_pval', 'christo_pval', 'ci_lower', 'ci_upper', 'es_realized']
display_cols = [c for c in display_cols if c in results_df.columns]
print(results_df[display_cols].round(4).to_string(index=False))

results_df.to_csv(f"{OUTPUT_DIR}/light_run_results.csv", index=False)

print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
print(f"Results saved to {OUTPUT_DIR}/")

# -----------------------------------------------------------------------
# Power analysis note for n=58
# -----------------------------------------------------------------------
print("\n" + "-" * 70)
print("POWER ANALYSIS NOTE (for small test sets)")
print("-" * 70)
from fibras.power_analysis import kupiec_power_curve

for ALPHA in ALPHAS:
    res = kupiec_power_curve(n=58, alpha=ALPHA, n_sim=3000)
    # Find power at deviations from nominal rate
    for delta in [0.05, 0.10, 0.15]:
        true_rate = ALPHA + delta
        if true_rate > 0.20:
            continue
        power_val = np.interp(true_rate, res['true_rate'], res['power'])
        print(f"  n=58, α={ALPHA:.2f}, true_rate={true_rate:.2f}: power={power_val:.2%}")

print("\nDone.")
