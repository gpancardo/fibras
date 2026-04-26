"""
Visualization suite for FIBRA VaR backtesting results.
Generates publication-quality figures for the research note.
Includes ex-ante regime shading and FX conditional analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from fibras.power_analysis import kupiec_power_curve

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
sns.set_style("whitegrid")

INPUT_DIR = "output"
OUTPUT_DIR = "output/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load aligned forecasts and regime metrics."""
    aligned = pd.read_csv(f"{INPUT_DIR}/aligned_forecasts.csv", index_col=0, parse_dates=True)
    regime_metrics = pd.read_csv(f"{INPUT_DIR}/regime_metrics.csv", index_col=[0, 1])
    fx_metrics = pd.read_csv(f"{INPUT_DIR}/fx_conditional_metrics.csv", index_col=[0, 1])
    return aligned, regime_metrics, fx_metrics


def plot_returns_with_regimes(aligned: pd.DataFrame) -> None:
    """Figura 1: Retornos diarios con regímenes High Vol sombreados."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(aligned.index, aligned['return'], color='#2c3e50', alpha=0.7, linewidth=0.5, label='Daily log return')

    high_vol = aligned[aligned['regime'] == 'High Vol']
    if not high_vol.empty:
        dates = high_vol.index
        diff = np.diff(dates) > pd.Timedelta(days=5)
        starts = [dates[0]] + dates[1:][diff].tolist()
        ends = dates[:-1][diff].tolist() + [dates[-1]]
        for start, end in zip(starts, ends):
            ax.axvspan(start, end, alpha=0.2, color='#c0392b', label='High Vol' if start == starts[0] else "")

    ax.set_xlabel("")
    ax.set_ylabel("Log Return")
    ax.set_title("FIBRA Daily Returns with High Volatility Regimes (Ex-Ante)")
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_returns_regimes.png")
    plt.close()


def plot_var_breaches(aligned: pd.DataFrame, var_col: str, model_name: str) -> None:
    """Figura 2: VaR forecasts con marcadores de violación."""
    fig, ax = plt.subplots(figsize=(12, 5))

    returns = aligned['return']
    var = aligned[var_col]

    ax.plot(returns.index, returns, color='#7f8c8d', alpha=0.5, linewidth=0.5, label='Return')
    ax.plot(var.index, -var, color='#e74c3c', linestyle='--', linewidth=1.2, label=f'VaR 95%')

    breaches = returns < -var
    ax.scatter(returns[breaches].index, returns[breaches],
               color='#c0392b', s=8, alpha=0.8, label=f'Breach ({breaches.sum()})')

    ax.set_xlabel("")
    ax.set_ylabel("Log Return")
    ax.set_title(f"VaR Backtest: {model_name}")
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/fig2_var_breaches_{var_col.replace('var_', '')}.png"
    plt.savefig(filename)
    plt.close()


def plot_volatility_comparison(aligned: pd.DataFrame) -> None:
    """Figura 3: Comparación de pronósticos de volatilidad."""
    fig, ax = plt.subplots(figsize=(12, 5))

    vol_cols = ['vol_garch_normal', 'vol_garch_t', 'vol_xgb_pure', 'vol_xgb_ensemble']
    labels = ['GARCH-Normal', 'GARCH-t', 'XGBoost-Pure', 'XGBoost-Ensemble']
    colors = ['#2980b9', '#27ae60', '#8e44ad', '#d35400']
    alphas = [0.6, 0.6, 0.7, 0.8]
    linestyles = ['-', '--', '-.', ':']

    for col, label, color, alpha, ls in zip(vol_cols, labels, colors, alphas, linestyles):
        ax.plot(aligned.index, aligned[col], label=label, color=color, alpha=alpha, linestyle=ls, linewidth=1.2)

    ax.set_xlabel("")
    ax.set_ylabel("Forecasted Volatility (daily)")
    ax.set_title("Volatility Forecast Comparison")
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_volatility_comparison.png")
    plt.close()


def create_traffic_light_table(regime_metrics: pd.DataFrame) -> None:
    """Figura 4: Tabla semáforo de tasas de violación por régimen."""
    pivot = regime_metrics['breach_rate'].unstack(level='regime')
    pivot = pivot * 100

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')

    expected = 5.0
    cell_text = []
    cell_colors = []
    for model in pivot.index:
        row_text = [model]
        row_colors = ['white']
        for regime in pivot.columns:
            rate = pivot.loc[model, regime]
            if pd.isna(rate):
                row_text.append('--')
                row_colors.append('#ecf0f1')
            else:
                row_text.append(f"{rate:.2f}%")
                if abs(rate - expected) <= expected * 0.5:
                    row_colors.append('#a5d6a7')
                elif abs(rate - expected) <= expected * 1.0:
                    row_colors.append('#fff59d')
                else:
                    row_colors.append('#ef9a9a')
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    columns = ['Model'] + list(pivot.columns)
    table = ax.table(cellText=cell_text, colLabels=columns,
                     cellColours=cell_colors, colColours=['#bdc3c7']*len(columns),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    ax.set_title("VaR Breach Rates by Regime (Target: 5.00%)", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_traffic_light_table.png")
    plt.close()


def plot_power_curve(
    n: int = 400,
    alpha: float = 0.05,
    n_sim: int = 1000,
    output_dir: str = "output/figures"
) -> None:
    """
    Generate and save Kupiec power curve figure.

    Args:
        n: Sample size per simulation.
        alpha: VaR significance level.
        n_sim: Number of simulations per true rate.
        output_dir: Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    true_rates = np.linspace(0.02, 0.08, 13)
    power_df = kupiec_power_curve(n=n, alpha=alpha, true_rates=true_rates, n_sim=n_sim)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(power_df['true_rate'], power_df['power'],
            marker='o', color='#2980b9', linewidth=2, markersize=5,
            label='Kupiec Test Power')

    ax.axhline(y=alpha, color='#c0392b', linestyle='--', linewidth=1.2,
               label=f'Significance level ({alpha})')
    ax.axvline(x=alpha, color='#27ae60', linestyle='--', linewidth=1.2,
               label=f'Expected rate ({alpha})')

    for pct in [alpha - 0.01, alpha + 0.01]:
        power_at_pct = np.interp(pct, power_df['true_rate'], power_df['power'])
        ax.annotate(
            f'Power at {pct:.2f}: {power_at_pct:.2f}',
            xy=(pct, power_at_pct),
            xytext=(pct + 0.005, power_at_pct + 0.05),
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=0.8),
            fontsize=8, color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#bdc3c7')
        )

    ax.set_xlabel('True Breach Rate')
    ax.set_ylabel('Power (Rejection Rate)')
    ax.set_title('Kupiec POF Test — Power Curve (n={0})'.format(n))
    ax.legend(loc='lower right')
    ax.set_xlim(0.02, 0.08)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_power_curve.png")
    plt.close()


def generate_all_figures() -> None:
    """Genera todas las figuras para el paper."""
    print("Loading data...")
    aligned, regime_metrics, _ = load_data()

    print("Generating Figure 1: Returns with regimes...")
    plot_returns_with_regimes(aligned)

    print("Generating Figure 2: VaR breach plots...")
    for var_col in ['var_garch_normal', 'var_garch_t', 'var_xgb_pure', 'var_xgb_ensemble']:
        model_name = var_col.replace('var_', '').replace('_', ' ').title()
        plot_var_breaches(aligned, var_col, model_name)

    print("Generating Figure 3: Volatility comparison...")
    plot_volatility_comparison(aligned)

    print("Generating Figure 4: Traffic light table...")
    create_traffic_light_table(regime_metrics)

    print("Generating Figure 5: Kupiec power curve...")
    plot_power_curve()

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all_figures()