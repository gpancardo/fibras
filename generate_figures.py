"""
Standalone figure-generation script for the FIBRA VaR paper.
Produces fig1, fig2, and fig3 in BOTH .png and .pdf formats.
Optimized for accessibility (color-blind friendly) and high-contrast B&W print.
Uses the exact same parameters as the modeling pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Configuration — matches the pipeline (run_analysis.py)
# ---------------------------------------------------------------------------
DATA_RAW = "data/raw"
OUTPUT = "output"
OUTPUT_FIGS = os.path.join(OUTPUT, "figures")
os.makedirs(OUTPUT_FIGS, exist_ok=True)

REGIME_VOL_WINDOW = 20       # rolling window for realised volatility
REGIME_LOOKBACK = 500        # rolling window for percentile threshold
REGIME_PERCENTILE = 90.0     # percentile threshold for High Vol

# Ajustes de Matplotlib optimizados para publicación académica y legibilidad
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "grid.alpha": 0.4,
    "grid.linestyle": ":",
})
plt.style.use("seaborn-v0_8-whitegrid")

# Paleta de colores Accesible / Contraste B&W (Inspirada en Okabe-Ito)
COLOR_BASE = "#000000"       # Negro puro para líneas principales
COLOR_ALT1 = "#0072B2"       # Azul oscuro (GARCH-t / n=58) -> Gris oscuro en B&W
COLOR_ALT2 = "#D55E00"       # Naranja rojizo (XGBoost / n=250) -> Gris medio en B&W
COLOR_LIGHT = "#999999"      # Gris para fondo/retornos
COLOR_SHADE = "#E69F00"      # Color de fondo para regímenes (fácilmente distinguible)

def load_fibra_series():
    """Load full FIBRA index returns from cached raw data."""
    df = pd.read_csv(os.path.join(DATA_RAW, "fibra_index.csv"),
                     index_col=0, parse_dates=True)
    df = df.dropna(subset=["log_return"])
    return df


def load_aligned_forecasts():
    """Load the aligned out-of-sample forecast file."""
    df = pd.read_csv(os.path.join(OUTPUT, "aligned_forecasts.csv"),
                     index_col=0, parse_dates=True)
    return df


# ---------------------------------------------------------------------------
# Figure 1 — Volatility regimes (Hatching para B&W y estilos de línea claros)
# ---------------------------------------------------------------------------
def generate_fig1_volatility_regimes():
    print("Generating fig1_volatility_regimes (PNG & PDF) ...")
    df = load_fibra_series()
    ret = df["log_return"]

    rv = ret.rolling(REGIME_VOL_WINDOW).std().shift(1)
    df["rv"] = rv

    rolling_threshold = rv.rolling(REGIME_LOOKBACK).quantile(REGIME_PERCENTILE / 100.0)
    df["threshold"] = rolling_threshold
    df["high_vol"] = (df["rv"] > df["threshold"]) & df["threshold"].notna()

    first_valid = df["threshold"].first_valid_index()
    plot_df = df.loc[first_valid:].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Línea 1: Continua y delgada
    ax.plot(plot_df.index, plot_df["rv"], color=COLOR_BASE, linewidth=0.8, linestyle="-",
            label=f"{REGIME_VOL_WINDOW}-day rolling realised volatility")
    
    # Línea 2: Discontinua (dashed) y más gruesa para contraste inmediato
    ax.plot(plot_df.index, plot_df["threshold"], color=COLOR_ALT1, linewidth=1.5,
            linestyle="--",
            label=f"{REGIME_PERCENTILE:.0f}th pctile ({REGIME_LOOKBACK}-day rolling)")

    # Zonas de régimen usando patrones de rayas (hatching) para impresión sin color
    high = plot_df[plot_df["high_vol"]]
    if not high.empty:
        starts = [high.index[0]]
        ends = []
        for i in range(1, len(high)):
            gap = (high.index[i] - high.index[i - 1]).days
            if gap > 5:
                ends.append(high.index[i - 1])
                starts.append(high.index[i])
        ends.append(high.index[-1])
        
        for s, e in zip(starts, ends):
            # hatch='//' crea líneas diagonales visibles en blanco y negro
            ax.axvspan(s, e, alpha=0.15, color=COLOR_SHADE, hatch="//", edgecolor=COLOR_SHADE,
                       label="High Vol Regime" if s == starts[0] else "")

    n_high = plot_df["high_vol"].sum()
    n_total = len(plot_df)
    ax.set_title(
        f"FIBRA Index — Realised Volatility and Ex-Ante Regimes "
        f"({n_high:,} High Vol / {n_total:,} total)"
    )
    ax.set_ylabel("Daily volatility (std. dev.)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "fig1_volatility_regimes.png"))
    fig.savefig(os.path.join(OUTPUT_FIGS, "fig1_volatility_regimes.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — VaR violations (Contraste estricto de formas y estilos de línea)
# ---------------------------------------------------------------------------
def generate_fig2_var_violations():
    print("Generating fig2_var_violations (PNG & PDF) ...")
    aligned = load_aligned_forecasts()

    fig, ax = plt.subplots(figsize=(12, 5))

    # Retornos en gris claro de fondo para no ensuciar las líneas de decisión
    ax.plot(aligned.index, aligned["return"], color=COLOR_LIGHT,
            linewidth=0.5, alpha=0.6, label="Daily log-return")

    # GARCH-t: Línea discontinua (dashed) azul/gris oscuro
    ax.plot(aligned.index, -aligned["var_garch_t"], color=COLOR_ALT1,
            linewidth=1.4, linestyle="--", label="VaR 95% (GARCH-t)")

    # XGBoost: Línea punto-raya (dashdot) naranja/gris medio
    ax.plot(aligned.index, -aligned["var_xgb_ensemble"], color=COLOR_ALT2,
            linewidth=1.4, linestyle="-.", label="VaR 95% (XGBoost Ens.)")

    # Brechas de GARCH-t: Cuadrados oscuros con borde
    gt_breaches = aligned["return"] < -aligned["var_garch_t"]
    gb = aligned[gt_breaches]
    if not gb.empty:
        ax.scatter(gb.index, gb["return"], color=COLOR_ALT1, s=35,
                   marker="s", zorder=5, edgecolors="k", linewidths=0.7,
                   label=f"GARCH-t breach ({gt_breaches.sum()})")

    # Brechas de XGBoost: Triángulos apuntando hacia arriba vacíos o con alto contraste
    xgb_breaches = aligned["return"] < -aligned["var_xgb_ensemble"]
    xb = aligned[xgb_breaches]
    if not xb.empty:
        ax.scatter(xb.index, xb["return"], color="#FFFFFF", s=40,
                   marker="^", zorder=5, edgecolors=COLOR_ALT2, linewidths=1.2,
                   label=f"XGBoost Ens. breach ({xgb_breaches.sum()})")

    # Régimen de volatilidad con patrón de líneas verticales ('||')
    hv = aligned[aligned["regime"] == "High Vol"]
    if not hv.empty:
        starts = [hv.index[0]]
        ends = []
        for i in range(1, len(hv)):
            if (hv.index[i] - hv.index[i - 1]).days > 5:
                ends.append(hv.index[i - 1])
                starts.append(hv.index[i])
        ends.append(hv.index[-1])
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=0.12, color=COLOR_SHADE, hatch="\\\\", edgecolor=COLOR_SHADE,
                       label="High Vol Regime" if s == starts[0] else "")

    ax.set_ylabel("Log-return")
    ax.set_title("FIBRA Index — 95% VaR Backtest: GARCH-t vs. XGBoost Ensemble")
    ax.legend(loc="upper right", ncol=2, frameon=True, facecolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "fig2_var_violations.png"))
    fig.savefig(os.path.join(OUTPUT_FIGS, "fig2_var_violations.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Kupiec power curve (Marcadores y cajas de texto legibles)
# ---------------------------------------------------------------------------
def kupiec_power_curve(n, alpha=0.05, true_rates=None, n_sim=3000):
    if true_rates is None:
        true_rates = np.linspace(0.025, 0.20, 20)
    rng = np.random.default_rng(42)
    power = []
    for p_true in true_rates:
        rejections = 0
        for _ in range(n_sim):
            breaches = rng.binomial(1, p_true, size=n)
            x = breaches.sum()
            p_hat = x / n if n > 0 else 0
            if p_hat == 0:
                lr = -2 * np.log((1 - alpha) ** n)
            elif p_hat == 1:
                lr = -2 * np.log(alpha ** n)
            else:
                lr = -2 * ((n - x) * np.log((1 - alpha) / (1 - p_hat))
                           + x * np.log(alpha / p_hat))
            p_val = 1 - chi2.cdf(lr, df=1)
            if p_val < alpha:
                rejections += 1
        power.append(rejections / n_sim)
    return true_rates, np.array(power)


def generate_fig3_power_curve():
    print("Generating fig3_power_curve (PNG & PDF) ...")
    true_rates, pow62 = kupiec_power_curve(62, n_sim=3000)
    _, pow250 = kupiec_power_curve(250, n_sim=3000)

    fig, ax = plt.subplots(figsize=(8, 5))

    # n=62: Línea continua con círculos pequeños
    ax.plot(true_rates, pow62, color=COLOR_ALT1, linewidth=1.8,
            linestyle="-", marker="o", markersize=4, label="n = 62")
    
    # n=250: Línea discontinua con cuadrados pequeños
    ax.plot(true_rates, pow250, color=COLOR_ALT2, linewidth=1.8,
            linestyle="--", marker="s", markersize=4, label="n = 250")

    # Líneas de referencia con alta distinción (Gris oscuro y negro con estilos claros)
    ax.axhline(y=0.05, color=COLOR_BASE, linestyle=":", linewidth=1.2,
               label="Significance level (5%)")
    ax.axvline(x=0.05, color=COLOR_LIGHT, linestyle="-.", linewidth=1.0,
               label="Nominal rate (5%)")

    # Ajuste de anotaciones para evitar solapamientos visuales en escala de grises
    for n, pwr, col, offset_y in [(62, pow62, COLOR_ALT1, 0.08), (250, pow250, COLOR_ALT2, -0.08)]:
        idx10 = np.searchsorted(true_rates, 0.10)
        if idx10 < len(pwr):
            ax.annotate(f"Power at 10%: {pwr[idx10]:.2f}",
                        xy=(0.10, pwr[idx10]),
                        xytext=(0.12, pwr[idx10] + offset_y),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.0),
                        fontsize=8.5, color=COLOR_BASE,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  alpha=0.9, edgecolor=COLOR_LIGHT))

    ax.set_xlabel("True breach rate")
    ax.set_ylabel("Power (rejection rate)")
    ax.set_title("Kupiec POF Test — Power Curve")
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.set_xlim(0.02, 0.20)
    ax.set_ylim(-0.02, 1.02)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FIGS, "fig3_power_curve.png"))
    fig.savefig(os.path.join(OUTPUT_FIGS, "fig3_power_curve.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_fig1_volatility_regimes()
    generate_fig2_var_violations()
    generate_fig3_power_curve()
    print("\n[SUCCESS] All figures saved to output/figures/ in BOTH .png and .pdf formats.")