"""
Lightweight profiler for the FIBRA VaR pipeline.
Runs each major component in isolation and reports timings.
"""
import time
import pandas as pd
import numpy as np
import cProfile
import pstats
import io

from fibras.data_loader import load_fibra_data, load_fx_data
from fibras.regimes import assign_regimes_exante
from fibras.features import create_features, add_garch_features, get_feature_columns
from fibras.garch_models import generate_all_garch_forecasts
from fibras.xgboost_models import (
    rolling_xgboost_tuned,
    rolling_xgboost_quantile,
    prepare_pure_xgboost_features,
    prepare_ensemble_features,
    prepare_var_target,
)
from fibras.backtest import (
    compute_historical_var_series,
    var_from_volatility,
    expected_shortfall_realized,
    evaluate_model,
)

WINDOW = 125
REFIT_FREQ = 20

print("=" * 60)
print("PIPELINE PROFILER — Component Timings")
print("=" * 60)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
t0 = time.time()
df_fibra = load_fibra_data()
df_fx = load_fx_data()
t1 = time.time()
print(f"\n[LOAD] Data loading: {t1 - t0:.2f}s")
print(f"  FIBRA rows: {len(df_fibra)}, FX rows: {len(df_fx)}")

# ---------------------------------------------------------------------------
# Regime assignment
# ---------------------------------------------------------------------------
t0 = time.time()
df_fibra = assign_regimes_exante(df_fibra, vol_window=20, lookback=500, threshold_percentile=90.0)
t1 = time.time()
print(f"\n[REGIME] Ex-ante regime assignment: {t1 - t0:.2f}s")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
t0 = time.time()
df_features = create_features(df_fibra, df_fx, lags=20)
t1 = time.time()
print(f"\n[FEATURES] create_features(): {t1 - t0:.2f}s")
print(f"  Feature matrix shape: {df_features.shape}")

# ---------------------------------------------------------------------------
# GARCH forecasts (both dists)
# ---------------------------------------------------------------------------
t0 = time.time()
garch_forecasts = generate_all_garch_forecasts(
    returns=df_fibra['log_return'],
    window=WINDOW,
    distributions=["normal", "t"],
    refit_freq=REFIT_FREQ,
    verbose=False
)
t1 = time.time()
print(f"\n[GARCH] Both distributions (window={WINDOW}): {t1 - t0:.2f}s")

# ---------------------------------------------------------------------------
# XGBoost feature prep (alpha-independent)
# ---------------------------------------------------------------------------
t0 = time.time()
feat_pure, pure_cols = prepare_pure_xgboost_features(df_fibra, df_fx, lags=20)
feat_ensemble, ensemble_cols = prepare_ensemble_features(
    df_fibra, df_fx, garch_forecasts['t'], dist_name='t', lags=20
)
feat_pure = prepare_var_target(feat_pure)
feat_ensemble = prepare_var_target(feat_ensemble)
t1 = time.time()
print(f"\n[XGB_PREP] Feature set preparation: {t1 - t0:.2f}s")
print(f"  Pure features: {len(pure_cols)}, Ensemble features: {len(ensemble_cols)}")

# ---------------------------------------------------------------------------
# XGBoost tuned (volatility forecast)
# ---------------------------------------------------------------------------
t0 = time.time()
xgb_vol_forecast = rolling_xgboost_tuned(
    df=df_features,
    feature_cols=get_feature_columns(df_features),
    target_col='target_vol',
    window=WINDOW,
    refit_freq=REFIT_FREQ,
    n_iter_search=10,
    verbose=False
)
t1 = time.time()
print(f"\n[XGB_TUNED] Volatility forecast (RandomizedSearchCV, n_iter=10): {t1 - t0:.2f}s")
refit_count = (len(df_features) - WINDOW) // REFIT_FREQ + 1
estimates = refit_count * 10 * 3  # n_iter × cv_splits
print(f"  ~{refit_count} refits × {10} iter × {3} CV folds = ~{estimates} XGBoost fits")

# ---------------------------------------------------------------------------
# XGBoost quantile (one alpha, to measure)
# ---------------------------------------------------------------------------
t0 = time.time()
xgb_q_pure = rolling_xgboost_quantile(
    df=feat_pure,
    feature_cols=pure_cols,
    alpha=0.05,
    target_col='target_var',
    window=WINDOW,
    refit_freq=REFIT_FREQ,
    n_iter_search=10,
    verbose=False
)
t1 = time.time()
print(f"\n[XGB_QUANTILE] alpha=0.05 (RandomizedSearchCV, n_iter=10): {t1 - t0:.2f}s")

# ---------------------------------------------------------------------------
# Historical VaR
# ---------------------------------------------------------------------------
common_idx = (df_fibra.index
              .intersection(garch_forecasts['t'].dropna().index)
              .intersection(xgb_vol_forecast.dropna().index)
              .intersection(df_features.index))
t0 = time.time()
hist_var = compute_historical_var_series(
    df_fibra['log_return'], WINDOW, 0.05, common_idx
)
t1 = time.time()
print(f"\n[HIST_VAR] compute_historical_var_series (n={len(common_idx)} dates): {t1 - t0:.4f}s")

# ---------------------------------------------------------------------------
# Aligned evaluation
# ---------------------------------------------------------------------------
t0 = time.time()
aligned = pd.DataFrame(index=common_idx)
aligned['return'] = df_fibra.loc[common_idx, 'log_return']
aligned['regime'] = df_fibra.loc[common_idx, 'regime']
aligned['var_hist'] = hist_var
for name, var_col in [('hist', 'var_hist')]:
    metrics = evaluate_model(aligned['return'], aligned[var_col], 0.05)
t1 = time.time()
print(f"\n[EVAL] Backtesting (1 model): {t1 - t0:.4f}s")
print(f"  Aligned obs: {len(aligned)}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ESTIMATED TOTAL TIME FOR FULL PIPELINE")
print("=" * 60)
est_xgb = (len(df_features) - WINDOW) // REFIT_FREQ + 1
print(f"  XGB Tuned (vol):        already timed above")
print(f"  XGB Quantile ×2 alphas ×2 models: 4× more than single quantile above")
print(f"  GARCH:                  already timed above")
print(f"  Total refits per XGB model: ~{est_xgb}")
print(f"\n  Dominant cost: XGBoost with RandomizedSearchCV")
print(f"  Each refit does {10} random param sets × {3} CV folds = 30 fits of 100-tree XGBoost")
