"""
Main execution script for the FIBRA VaR Model Comparison study.
Incorporates ex-ante regimes, USD/MXN factor, fixed rolling windows,
hyperparameter tuning for XGBoost, and conditional FX stress analysis.

Run this script from the project root after installing dependencies.
Outputs are saved to the `output/` directory.
"""

import os
import warnings
import pandas as pd
import numpy as np

from fibras.data_loader import load_fibra_data, load_fx_data, load_ticker_data
from fibras.regimes import assign_regimes_exante
from fibras.features import (
    create_features,
    add_garch_features,
    get_feature_columns
)
from fibras.garch_models import generate_all_garch_forecasts
from fibras.xgboost_models import (
    rolling_xgboost_tuned,
    prepare_pure_xgboost_features,
    prepare_ensemble_features
)
from fibras.backtest import (
    var_from_volatility,
    expected_shortfall_from_volatility,
    evaluate_model,
    evaluate_by_regime,
    evaluate_conditional_fx,
    diebold_mariano_test
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_TUNED_XGB = True           # True = RandomizedSearchCV; False = fixed params
ALPHA = 0.05                   # VaR confidence level (5% = 95% VaR)
WINDOW = 125                   # Rolling window for both GARCH and XGBoost
REFIT_FREQ = 20                # Refit frequency for models
OUTPUT_DIR = "output"       # Directory for results
REGIME_VOL_WINDOW = 20         # Window for calculating realized volatility
REGIME_LOOKBACK = 500          # Lookback period for calculating volatility percentile
REGIME_PERCENTILE = 90.0       # Percentile threshold for high volatility regime

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)


# ============================================================================
# 0. LOAD SHARED DATA (FX)
# ============================================================================

print("=" * 60)
print("FIBRA VaR Model Comparison with FX Stress Analysis")
print("=" * 60)

print("\nLoading FX data (shared)...")
df_fx = load_fx_data()
print(f"  FX period: {df_fx.index[0].date()} to {df_fx.index[-1].date()}")
print(f"  FX observations: {len(df_fx)}")


# ============================================================================
# MEJORA 1: MULTI-ASSET VALIDATION
# ============================================================================

print("\n" + "=" * 60)
print("MEJORA 1: VALIDACION MULTI-ACTIVO")
print("=" * 60)

MULTI_TICKERS = ["FIBRATC14.MX", "FUNO11.MX", "FIBRAPL14.MX", "^MXX", "EWW"]

multi_results = []
for ticker in MULTI_TICKERS:
    print(f"\n--- Procesando {ticker} ---")
    try:
        df_ticker = load_ticker_data(ticker)
        print(f"  Datos: {len(df_ticker)} observaciones")

        df_ticker = assign_regimes_exante(df_ticker, vol_window=20, lookback=500, threshold_percentile=90.0)

        df_feat_ticker = create_features(df_ticker, df_fx, lags=20)
        feat_cols = get_feature_columns(df_feat_ticker)

        ticker_returns = df_ticker['log_return']
        garch_t_forecasts = generate_all_garch_forecasts(
            ticker_returns, window=WINDOW, distributions=["t"],
            refit_freq=REFIT_FREQ, verbose=False
        )

        xgb_forecast = rolling_xgboost_tuned(
            df=df_feat_ticker, feature_cols=feat_cols, target_col='target_vol',
            window=WINDOW, refit_freq=REFIT_FREQ, n_iter_search=10, verbose=False
        )

        common_idx = (
            ticker_returns.index
            .intersection(garch_t_forecasts['t'].dropna().index)
            .intersection(xgb_forecast.dropna().index)
        )
        aligned_t = pd.DataFrame(index=common_idx)
        aligned_t['return'] = df_ticker.loc[common_idx, 'log_return']
        aligned_t['regime'] = df_ticker.loc[common_idx, 'regime']
        aligned_t['vol_garch_t'] = garch_t_forecasts['t'].loc[common_idx]
        aligned_t['vol_xgb_pure'] = xgb_forecast.loc[common_idx]

        aligned_t['var_garch_t'] = var_from_volatility(aligned_t['vol_garch_t'], ALPHA, 't')
        aligned_t['var_xgb_pure'] = var_from_volatility(aligned_t['vol_xgb_pure'], ALPHA, 'normal')

        for model_name, var_col in [('GARCH-t', 'var_garch_t'), ('XGBoost-Pure', 'var_xgb_pure')]:
            metrics = evaluate_model(aligned_t['return'], aligned_t[var_col], ALPHA)
            metrics['ticker'] = ticker
            metrics['model'] = model_name
            multi_results.append(metrics)

        print(f"  Observaciones alineadas: {len(aligned_t)}")
        for model_name in ['GARCH-t', 'XGBoost-Pure']:
            r = [m for m in multi_results if m['ticker'] == ticker and m['model'] == model_name][0]
            print(f"  {model_name}: breach_rate={r['breach_rate']:.4f}, kupiec_pval={r['kupiec_pval']:.4f}")

    except Exception as e:
        print(f"  ERROR procesando {ticker}: {e}")

if multi_results:
    multi_df = pd.DataFrame(multi_results)
    multi_df = multi_df[['ticker', 'model', 'n_obs', 'breach_rate', 'kupiec_pval', 'christo_pval', 'ci_lower', 'ci_upper']]
    multi_df = multi_df.round(4)
    multi_df.to_csv(f"{OUTPUT_DIR}/multi_asset_results.csv", index=False)
    print(f"\n--- Resultados Multi-Activo (guardados en {OUTPUT_DIR}/multi_asset_results.csv) ---")
    print(multi_df.to_string())
else:
    print("\nNo se generaron resultados multi-activo.")


# ============================================================================
# 1. DATA LOADING (ORIGINAL FIBRA ANALYSIS)
# ============================================================================

print("\n" + "=" * 60)
print("ORIGINAL ANALYSIS: FIBRA Index Detailed")
print("=" * 60)

print("\nLoading FIBRA data...")
df_fibra = load_fibra_data()

print(f"  FIBRA period: {df_fibra.index[0].date()} to {df_fibra.index[-1].date()}")
print(f"  FIBRA observations: {len(df_fibra)}")


# ============================================================================
# 2. EX-ANTE REGIME ASSIGNMENT
# ============================================================================

print("\n" + "-" * 40)
print("Assigning ex-ante regimes (rolling vol percentile)...")
df_fibra = assign_regimes_exante(
    df_fibra,
    vol_window=REGIME_VOL_WINDOW,
    lookback=REGIME_LOOKBACK,
    threshold_percentile=REGIME_PERCENTILE
)
regime_counts = df_fibra['regime'].value_counts(dropna=True)
print(f"  Regime distribution:\n{regime_counts}")


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n" + "-" * 40)
print("Engineering features (no leakage, includes FX)...")
df_features = create_features(df_fibra, df_fx, lags=20)
print(f"  Feature matrix shape: {df_features.shape}")


# ============================================================================
# 4. GARCH FORECASTS (FIXED ROLLING WINDOW)
# ============================================================================

print("\n" + "-" * 40)
print(f"Computing rolling GARCH forecasts (window={WINDOW})...")
garch_forecasts = generate_all_garch_forecasts(
    returns=df_fibra['log_return'],
    window=WINDOW,
    distributions=["normal", "t"],
    refit_freq=REFIT_FREQ,
    verbose=True
)


# ============================================================================
# 5. XGBOOST FORECASTS
# ============================================================================

print("\n" + "-" * 40)
print("Preparing XGBoost feature sets...")

# Pure XGBoost
feat_pure, pure_cols = prepare_pure_xgboost_features(df_fibra, df_fx, lags=20)

# Ensemble (GARCH-t augmented)
feat_ensemble, ensemble_cols = prepare_ensemble_features(
    df_fibra, df_fx, garch_forecasts['t'], dist_name='t', lags=20
)

print(f"  Pure XGBoost features: {len(pure_cols)}")
print(f"  Ensemble XGBoost features: {len(ensemble_cols)}")

# Rolling forecasts
if USE_TUNED_XGB:
    print("\nComputing TUNED XGBoost forecasts...")
    print("  Pure XGBoost...")
    pure_xgb_forecast = rolling_xgboost_tuned(
        df=feat_pure,
        feature_cols=pure_cols,
        target_col='target_vol',
        window=WINDOW,
        refit_freq=REFIT_FREQ,
        n_iter_search=10,
        verbose=True
    )
    print("  Ensemble XGBoost...")
    ensemble_xgb_forecast = rolling_xgboost_tuned(
        df=feat_ensemble,
        feature_cols=ensemble_cols,
        target_col='target_vol',
        window=WINDOW,
        refit_freq=REFIT_FREQ,
        n_iter_search=10,
        verbose=True
    )
else:
    from fibras.xgboost_models import rolling_xgboost_simple
    print("\nComputing SIMPLE XGBoost forecasts...")
    pure_xgb_forecast = rolling_xgboost_simple(
        df=feat_pure,
        feature_cols=pure_cols,
        window=WINDOW,
        refit_freq=REFIT_FREQ,
        verbose=True
    )
    ensemble_xgb_forecast = rolling_xgboost_simple(
        df=feat_ensemble,
        feature_cols=ensemble_cols,
        window=WINDOW,
        refit_freq=REFIT_FREQ,
        verbose=True
    )


# ============================================================================
# 6. ALIGN FORECASTS AND COMPUTE VaR
# ============================================================================

print("\n" + "-" * 40)
print("Aligning forecasts and computing VaR series...")

common_index = (
    df_fibra.index
    .intersection(garch_forecasts['normal'].dropna().index)
    .intersection(garch_forecasts['t'].dropna().index)
    .intersection(pure_xgb_forecast.dropna().index)
    .intersection(ensemble_xgb_forecast.dropna().index)
    .intersection(df_features.index)
)

aligned = pd.DataFrame(index=common_index)
aligned['return'] = df_fibra.loc[common_index, 'log_return']
aligned['regime'] = df_fibra.loc[common_index, 'regime']
aligned['fx_return'] = df_fx.loc[common_index, 'fx_return']

aligned['vol_garch_normal'] = garch_forecasts['normal'].loc[common_index]
aligned['vol_garch_t'] = garch_forecasts['t'].loc[common_index]
aligned['vol_xgb_pure'] = pure_xgb_forecast.loc[common_index]
aligned['vol_xgb_ensemble'] = ensemble_xgb_forecast.loc[common_index]

aligned['var_garch_normal'] = var_from_volatility(aligned['vol_garch_normal'], ALPHA, 'normal')
aligned['var_garch_t'] = var_from_volatility(aligned['vol_garch_t'], ALPHA, 't')
aligned['var_xgb_pure'] = var_from_volatility(aligned['vol_xgb_pure'], ALPHA, 'normal')
aligned['var_xgb_ensemble'] = var_from_volatility(aligned['vol_xgb_ensemble'], ALPHA, 'normal')

print(f"  Aligned observations: {len(aligned)}")
print(f"  Regime counts in aligned data:\n{aligned['regime'].value_counts(dropna=False)}")

# Expected Shortfall for all models
for col, dist, df in [
    ('vol_garch_normal', 'normal', None),
    ('vol_garch_t', 't', None),
    ('vol_xgb_pure', 'normal', None),
    ('vol_xgb_ensemble', 'normal', None),
]:
    aligned[f'es_{col[4:]}'] = expected_shortfall_from_volatility(
        aligned[col], ALPHA, dist, df_t=df
    )

# Historical VaR baseline (rolling empirical quantile)
hist_var = pd.Series(index=aligned.index, dtype=float)
hist_es = pd.Series(index=aligned.index, dtype=float)
for i, date in enumerate(aligned.index):
    idx = df_fibra.index.get_loc(date)
    if idx >= WINDOW:
        window_rets = df_fibra['log_return'].iloc[idx - WINDOW:idx]
        hist_var.iloc[i] = -window_rets.quantile(ALPHA)
        hist_es.iloc[i] = -window_rets[window_rets <= window_rets.quantile(ALPHA)].mean()
aligned['var_hist'] = hist_var
aligned['es_hist'] = hist_es

# XGBoost VaR using t-quantile (tail-adjustment check)
aligned['var_xgb_pure_t'] = var_from_volatility(aligned['vol_xgb_pure'], ALPHA, 't')


# ============================================================================
# MEJORA 2: DIEBOLD-MARIANO TEST DE PRECISION DE VOLATILIDAD
# ============================================================================

print("\n" + "=" * 60)
print("MEJORA 2: DIEBOLD-MARIANO TEST")
print("=" * 60)

actual_vol = aligned['return'].abs()

dm_results = []
comparisons = [
    ('vol_garch_t', 'vol_xgb_pure', 'GARCH-t vs XGBoost-Pure'),
    ('vol_garch_t', 'vol_xgb_ensemble', 'GARCH-t vs XGBoost-Ensemble'),
]
for f1, f2, label in comparisons:
    result = diebold_mariano_test(
        actual=actual_vol,
        forecast1=aligned[f1],
        forecast2=aligned[f2],
        loss='mse'
    )
    result['comparison'] = label
    dm_results.append(result)

dm_df = pd.DataFrame(dm_results)
dm_df = dm_df[['comparison', 'dm_stat', 'p_value']].round(4)
dm_df.to_csv(f"{OUTPUT_DIR}/dm_test_results.csv", index=False)
print(dm_df.to_string())
print(f"\nResults saved to {OUTPUT_DIR}/dm_test_results.csv")


# ============================================================================
# 7. OVERALL BACKTESTING
# ============================================================================

print("\n" + "=" * 60)
print("OVERALL MODEL PERFORMANCE")
print("=" * 60)

models = {
    'GARCH-Normal': 'var_garch_normal',
    'GARCH-t': 'var_garch_t',
    'XGBoost-Pure': 'var_xgb_pure',
    'XGBoost-Ensemble': 'var_xgb_ensemble'
}

overall_results = {}
for name, var_col in models.items():
    metrics = evaluate_model(aligned['return'], aligned[var_col], ALPHA)
    overall_results[name] = metrics

overall_df = pd.DataFrame(overall_results).T
overall_df.index.name = 'Model'
overall_df = overall_df.round(4)

print(overall_df.to_string())


# ============================================================================
# 8. REGIME-SPECIFIC BACKTESTING
# ============================================================================

print("\n" + "=" * 60)
print("REGIME-SPECIFIC PERFORMANCE")
print("=" * 60)

regime_results = {}
for name, var_col in models.items():
    regime_df = evaluate_by_regime(
        aligned['return'],
        aligned[var_col],
        aligned['regime'],
        ALPHA
    )
    if not regime_df.empty:
        regime_df['model'] = name
        regime_results[name] = regime_df

if regime_results:
    combined_regime = pd.concat(regime_results.values(), ignore_index=True)
    combined_regime = combined_regime.set_index(['model', 'regime'])
    print(combined_regime.round(4).to_string())
else:
    combined_regime = pd.DataFrame()
    print("No regime-specific results available.")


# ============================================================================
# 9. CONDITIONAL FX STRESS ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("CONDITIONAL FX STRESS ANALYSIS (MXN Depreciation > 2%)")
print("=" * 60)

fx_results = {}
for name, var_col in models.items():
    fx_df = evaluate_conditional_fx(
        aligned['return'],
        aligned[var_col],
        aligned['fx_return'],
        threshold=0.02,
        alpha=ALPHA
    )
    if not fx_df.empty:
        fx_df['model'] = name
        fx_results[name] = fx_df

if fx_results:
    combined_fx = pd.concat(fx_results.values(), ignore_index=True)
    combined_fx = combined_fx.set_index(['model', 'condition'])
    print(combined_fx.round(4).to_string())
else:
    combined_fx = pd.DataFrame()
    print("No FX conditional results available.")


# ============================================================================
# 10. EXPORT RESULTS
# ============================================================================

print("\n" + "-" * 40)
print(f"Saving results to {OUTPUT_DIR}/")

aligned.to_csv(f"{OUTPUT_DIR}/aligned_forecasts.csv")
overall_df.to_csv(f"{OUTPUT_DIR}/overall_metrics.csv")

if not combined_regime.empty:
    combined_regime.to_csv(f"{OUTPUT_DIR}/regime_metrics.csv")
if not combined_fx.empty:
    combined_fx.to_csv(f"{OUTPUT_DIR}/fx_conditional_metrics.csv")

print("Export complete.")

# ES comparison table
es_models = {
    'GARCH-Normal': 'es_garch_normal',
    'GARCH-t': 'es_garch_t',
    'XGBoost-Pure': 'es_xgb_pure',
    'XGBoost-Ensemble': 'es_xgb_ensemble',
    'Historical': 'es_hist',
}
es_rows = []
for name, col in es_models.items():
    series = aligned[col].dropna()
    es_rows.append({
        'Model': name,
        'Mean_ES': series.mean(),
        'Std_ES': series.std(),
        'ES_over_VaR': (series / aligned[f'var_{col[3:]}' if col != 'es_hist' else aligned['var_hist']].dropna()).mean(),
    })
es_df = pd.DataFrame(es_rows).round(4)
es_df.to_csv(f"{OUTPUT_DIR}/es_comparison.csv", index=False)
print("\nES comparison:")
print(es_df.to_string())

# Historical VaR metrics
hist_metrics = evaluate_model(aligned['return'], aligned['var_hist'], ALPHA)
print(f"\nHistorical VaR (baseline): breach_rate={hist_metrics['breach_rate']:.4f}, kupiec_pval={hist_metrics['kupiec_pval']:.4f}")

# XGBoost tail-adjusted (t-quantile) metrics
xgb_t_metrics = evaluate_model(aligned['return'], aligned['var_xgb_pure_t'], ALPHA)
print(f"\nXGBoost-Pure (t-quantile, df=5): breach_rate={xgb_t_metrics['breach_rate']:.4f}, kupiec_pval={xgb_t_metrics['kupiec_pval']:.4f}")


# ============================================================================
# 11. INTERPRETATION SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("INTERPRETATION SUMMARY")
print("=" * 60)

if not overall_df.empty:
    best_model = overall_df['kupiec_pval'].idxmax()
    print(f"Best model by Kupiec test (overall): {best_model}")
    print(f"  Breach rate: {overall_df.loc[best_model, 'breach_rate']:.2%}")
    print(f"  95% CI: [{overall_df.loc[best_model, 'ci_lower']:.2%}, {overall_df.loc[best_model, 'ci_upper']:.2%}]")

if not combined_fx.empty:
    print("\nFX Stress Performance (MXN > +2%):")
    for model in combined_fx.index.get_level_values('model').unique():
        try:
            rate = combined_fx.loc[(model, 'FX_Stressed'), 'breach_rate']
            pval = combined_fx.loc[(model, 'FX_Stressed'), 'kupiec_pval']
            print(f"  {model}: breach rate = {rate:.2%}, Kupiec p = {pval:.4f}")
        except KeyError:
            pass

print("\nAnalysis complete.")


# ============================================================================
# MEJORA 3: POWER CURVE DEL TEST DE KUPIEC
# ============================================================================

print("\n" + "=" * 60)
print("MEJORA 3: CURVA DE POTENCIA DEL TEST DE KUPIEC")
print("=" * 60)

from fibras.visualization import plot_power_curve
plot_power_curve(n=400, alpha=0.05, output_dir=f"{OUTPUT_DIR}/figures")
print(f"Curva de potencia guardada en {OUTPUT_DIR}/figures/fig_power_curve.png")