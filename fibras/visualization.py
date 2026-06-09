"""
Visualization suite for FIBRA VaR backtesting results.
Generates publication-quality figures for the research note.
Includes ex-ante regime shading, FX conditional analysis,
power curves, size distortion, rolling stability, and conservatism metrics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from fibras.power_analysis import kupiec_power_curve, size_distribution

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
    """Figura 1: Retornos diarios con regimenes High Vol sombreados."""
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
    plt.savefig(f"{OUTPUT_DIR}/fig1_returns_regimes.pdf", format='pdf')
    plt.close()


def plot_var_breaches(aligned: pd.DataFrame, var_col: str, model_name: str) -> None:
    """Figura 2: VaR forecasts con marcadores de violacion."""
    fig, ax = plt.subplots(figsize=(12, 5))

    returns = aligned['return']
    var = aligned[var_col]

    ax.plot(returns.index, returns, color='#7f8c8d', alpha=0.5, linewidth=0.5, label='Return')
    ax.plot(var.index, -var, color='#e74c3c', linestyle='--', linewidth=1.2, label='VaR 95%')

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
    filename = f"{OUTPUT_DIR}/fig2_var_breaches_{var_col.replace('var_', '')}.pdf"
    plt.savefig(filename, format='pdf')
    plt.close()


def plot_volatility_comparison(aligned: pd.DataFrame) -> None:
    """Figura 3: Comparacion de pronosticos de volatilidad."""
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
    plt.savefig(f"{OUTPUT_DIR}/fig3_volatility_comparison.pdf", format='pdf')
    plt.close()


def create_traffic_light_table(regime_metrics: pd.DataFrame) -> None:
    """Figura 4: Tabla semaforo de tasas de violacion por regimen."""
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
    plt.savefig(f"{OUTPUT_DIR}/fig4_traffic_light_table.pdf", format='pdf')
    plt.close()


# ─────────────────────────────────────────────────────────────────
# NEW FIGURES (Refinement Pipeline)
# ─────────────────────────────────────────────────────────────────

def plot_power_curve_small_sample(
    n_values: list = None,
    alpha: float = 0.05,
    output_dir: str = "output/figures"
) -> None:
    """
    Power curve figure for small-sample backtesting.

    Shows rejection probability of the Kupiec test for n=62, n=95, and
    optionally n=250 as a baseline. Uses the exact binomial test as comparison.
    """
    if n_values is None:
        n_values = [62, 95, 250]

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = {62: '#2980b9', 95: '#27ae60', 250: '#d35400'}
    linestyles = {62: '-', 95: '--', 250: '-.'}

    for n_val in n_values:
        df = kupiec_power_curve(n=n_val, alpha=alpha, n_sim=3000)
        color = colors.get(n_val, '#7f8c8d')
        ls = linestyles.get(n_val, '-')
        lw = 2.0 if n_val <= 95 else 1.5

        ax.plot(df['true_rate'], df['power_kupiec'],
                color=color, linestyle=ls, linewidth=lw,
                label=f'n = {n_val} (Kupiec)')

        ax.plot(df['true_rate'], df['power_exact'],
                color=color, linestyle=':', linewidth=1.0,
                alpha=0.6, label=f'n = {n_val} (exact binom.)' if n_val == n_values[0] else "")

    ax.axhline(y=0.05, color='#c0392b', linestyle=':', linewidth=1.0, label='Significance level ($\\alpha$ = 0.05)')
    ax.axvline(x=alpha, color='#7f8c8d', linestyle='--', linewidth=1.0, label=f'Nominal rate ({alpha})')

    ax.set_xlabel('True Breach Rate')
    ax.set_ylabel('Rejection Probability (Power)')
    ax.set_title('Kupiec POF Test — Power Curves for Small Samples')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/power_curve_small_sample.pdf", format='pdf')
    plt.close()


def plot_size_distribution(
    n_values: list = None,
    alpha: float = 0.05,
    n_sim: int = 10000,
    output_dir: str = "output/figures"
) -> None:
    """
    Size distortion histogram: distribution of Kupiec p-values under H0.

    Shows deviation from uniform distribution, highlighting the excess
    rejection rate at various sample sizes.
    """
    if n_values is None:
        n_values = [62, 95, 250, 500]

    os.makedirs(output_dir, exist_ok=True)

    n_plots = len(n_values)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, n_val in zip(axes, n_values):
        df = size_distribution(n=n_val, alpha=alpha, n_sim=n_sim)
        rejection_rate = (df['p_value'] < 0.05).mean()

        ax.hist(df['p_value'], bins=30, density=True, color='#2980b9', alpha=0.7, edgecolor='white')
        ax.axhline(y=1.0, color='#c0392b', linestyle='--', linewidth=1.2, label='Uniform (correct)')
        ax.axvline(x=0.05, color='#d35400', linestyle=':', linewidth=1.0, label='Nominal 5% level')

        ax.set_xlabel('p-value')
        ax.set_ylabel('Density')
        ax.set_title(f'n = {n_val}  (rejection rate = {rejection_rate:.1%})')
        ax.legend(loc='upper right', fontsize=7)

    fig.suptitle('Size Distortion: Kupiec POF p-values Under the Null (H$_0$: p = $\\alpha$)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/size_distortion.pdf", format='pdf')
    plt.close()


def plot_rolling_breach_rates(
    rolling_data: dict,
    alpha: float = 0.05,
    output_dir: str = "output/figures"
) -> None:
    """
    Rolling breach rate stability plot.

    Args:
        rolling_data: Dict mapping model_name -> pd.DataFrame from rolling_breach_rate().
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ['#2980b9', '#c0392b', '#27ae60', '#d35400']
    for idx, (model_name, df) in enumerate(rolling_data.items()):
        valid = df['rolling_breach_rate'].dropna()
        color = colors[idx % len(colors)]
        ax.plot(valid.index, valid.values, color=color, linewidth=1.0,
                alpha=0.8, label=f'{model_name}')

    ax.axhline(y=alpha, color='#7f8c8d', linestyle='--', linewidth=1.2, label=f'Nominal rate ({alpha})')
    ax.set_xlabel('')
    ax.set_ylabel('Breach Rate (rolling 30-day)')
    ax.set_title('Rolling Breach Rate Stability')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim(-0.02, 0.35)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rolling_breach_rates.pdf", format='pdf')
    plt.close()


def plot_conservatism_bars(
    conservatism_data: dict,
    output_dir: str = "output/figures"
) -> None:
    """
    Horizontal bar chart of conservatism metric C = p_hat - alpha.

    Args:
        conservatism_data: Dict mapping model_name -> conservatism C value.
    """
    os.makedirs(output_dir, exist_ok=True)

    names = list(conservatism_data.keys())
    values = list(conservatism_data.values())

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = []
    for v in values:
        if v < -0.02:
            colors.append('#2980b9')      # strongly conservative
        elif v < 0:
            colors.append('#a5d6a7')      # conservative
        elif v < 0.05:
            colors.append('#fff59d')      # mildly anti-conservative
        else:
            colors.append('#c0392b')      # strongly anti-conservative

    bars = ax.barh(names, values, color=colors, edgecolor='white', height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() + 0.002 if val >= 0 else bar.get_x() + bar.get_width() - 0.025,
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center', fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='#2c3e50', linewidth=1.5)
    ax.set_xlabel('Conservatism C = p̂ − α')
    ax.set_title('Conservatism Metric by Model')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/conservatism_bars.pdf", format='pdf')
    plt.close()


def plot_binomial_ci_table(
    overall_df: pd.DataFrame,
    output_dir: str = "output/figures"
) -> None:
    """
    Generate a visual table of exact binomial confidence intervals.

    Args:
        overall_df: DataFrame with columns: n_obs, n_breaches, breach_rate,
                    expected_rate, ci_lower_cp, ci_upper_cp.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')

    columns = ['Model', 'n', 'Breaches', 'Rate', 'Target', '95% CI', 'C']
    cell_text = []
    cell_colors = []

    for model_name, row in overall_df.iterrows():
        n = int(row.get('n_obs', 0))
        x = int(row.get('n_breaches', 0))
        rate = row.get('breach_rate', float('nan'))
        expected = row.get('expected_rate', float('nan'))
        ci_lo = row.get('ci_lower_cp', row.get('ci_lower', float('nan')))
        ci_hi = row.get('ci_upper_cp', row.get('ci_upper', float('nan')))
        c_val = row.get('conservatism_c', float('nan'))

        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not pd.isna(ci_lo) else "--"
        rate_str = f"{rate:.3f}" if not pd.isna(rate) else "--"
        c_str = f"{c_val:+.3f}" if not pd.isna(c_val) else "--"

        cell_text.append([str(model_name), str(n), str(x), rate_str, str(expected), ci_str, c_str])

        if not pd.isna(c_val):
            if c_val < -0.02:
                cell_colors.append(['#d5f5e3'] * 7)
            elif c_val > 0.02:
                cell_colors.append(['#fadbd8'] * 7)
            else:
                cell_colors.append(['white'] * 7)
        else:
            cell_colors.append(['#ecf0f1'] * 7)

    table = ax.table(cellText=cell_text, colLabels=columns,
                     cellColours=cell_colors, colColours=['#bdc3c7'] * len(columns),
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.8)

    ax.set_title('Binomial Confidence Intervals (Clopper-Pearson, 95%)', fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/binomial_ci_table.pdf", format='pdf')
    plt.close()


# ─────────────────────────────────────────────────────────────────
# ORIGINAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def plot_power_curve(
    n: int = 400,
    alpha: float = 0.05,
    n_sim: int = 1000,
    output_dir: str = "output/figures"
) -> None:
    """Legacy wrapper: generate Kupiec power curve figure."""
    os.makedirs(output_dir, exist_ok=True)
    true_rates = np.linspace(0.02, 0.08, 13)
    power_df = kupiec_power_curve(n=n, alpha=alpha, true_rates=true_rates, n_sim=n_sim)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(power_df['true_rate'], power_df['power_kupiec'],
            marker='o', color='#2980b9', linewidth=2, markersize=5,
            label='Kupiec Test Power')

    ax.axhline(y=alpha, color='#c0392b', linestyle='--', linewidth=1.2,
               label=f'Significance level ({alpha})')
    ax.axvline(x=alpha, color='#27ae60', linestyle='--', linewidth=1.2,
               label=f'Expected rate ({alpha})')

    ax.set_xlabel('True Breach Rate')
    ax.set_ylabel('Power (Rejection Rate)')
    ax.set_title(f'Kupiec POF Test — Power Curve (n={n})')
    ax.legend(loc='lower right')
    ax.set_xlim(0.02, 0.08)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_power_curve_legacy.pdf", format='pdf')
    plt.close()


def plot_pvalue_staircase(
    n: int = 62,
    alpha: float = 0.05,
    observed_counts: dict = None,
    output_dir: str = "output/figures"
) -> None:
    """
    Kupiec p-value as a function of breach count (the \"staircase\".

    Shows that at small n, the p-value function is highly discontinuous:
    a single additional breach can flip the test from \"accept\" to \"reject.\"

    Args:
        n: Sample size.
        alpha: Nominal breach rate.
        observed_counts: Dict mapping model_name -> breach_count.
        output_dir: Directory for the output figure.
    """
    from scipy.stats import chi2

    os.makedirs(output_dir, exist_ok=True)

    x_max = int(np.ceil(n * 0.25))  # show up to ~25% breach rate
    x_vals = np.arange(0, x_max + 1)
    p_vals = np.empty(len(x_vals))

    eps = 1e-300
    for i, x in enumerate(x_vals):
        p_hat = x / n
        if p_hat == 0:
            lr = -2 * np.log(max((1 - alpha) ** n, eps))
        elif p_hat == 1:
            lr = -2 * np.log(max(alpha ** n, eps))
        else:
            lr = -2 * ((n - x) * np.log(np.maximum((1 - alpha) / np.maximum(1 - p_hat, eps), eps))
                        + x * np.log(np.maximum(alpha / np.maximum(p_hat, eps), eps)))
        p_vals[i] = 1 - chi2.cdf(lr, df=1)

    fig, ax = plt.subplots(figsize=(9, 5))

    markerline, stemlines, baseline = ax.stem(x_vals, p_vals, linefmt='#bdc3c7', markerfmt='o', basefmt=' ')
    plt.setp(markerline, color='#2980b9', markersize=6)

    ax.axhline(y=0.05, color='#c0392b', linestyle='--', linewidth=1.2,
               label='5\\% significance level')
    ax.axvline(x=n * alpha, color='#7f8c8d', linestyle=':', linewidth=1.0,
               label=f'Expected count ({n * alpha:.1f})')

    if observed_counts:
        colors_model = ['#d35400', '#27ae60', '#8e44ad', '#2980b9', '#c0392b']
        for idx, (model_name, x_obs) in enumerate(observed_counts.items()):
            c = colors_model[idx % len(colors_model)]
            y_obs = np.interp(x_obs, x_vals, p_vals)
            ax.plot(x_obs, y_obs, marker='D', markersize=10, color=c,
                    markeredgecolor='k', markeredgewidth=0.5, linestyle='None',
                    zorder=10, label=f'{model_name} ({x_obs} breaches)')

    ax.set_xlabel('Number of breaches $x$')
    ax.set_ylabel('Kupiec $p$-value')
    ax.set_title(f'Kupiec POF Test: $p$-value Staircase ($n = {n}$, $\\alpha = {alpha}$)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-0.5, x_max + 0.5)
    ax.set_ylim(-0.02, 1.02)

    ax.text(0.98, 0.92, 'Accept $H_0$', transform=ax.transAxes,
            ha='right', fontsize=9, color='#27ae60', fontstyle='italic')
    ax.text(0.98, 0.02, 'Reject $H_0$', transform=ax.transAxes,
            ha='right', fontsize=9, color='#c0392b', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pvalue_staircase.pdf", format='pdf')
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
