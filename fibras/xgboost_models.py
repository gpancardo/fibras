"""
XGBoost-based volatility forecasting models.
Implements rolling-window XGBoost regressors with hyperparameter tuning via TimeSeriesSplit.
Includes pure ML and GARCH-X ensemble variants.
CPU-only training; no GPU required.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from typing import List, Optional

# Grid reducido pero relevante para series financieras
PARAM_GRID = {
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_lambda': [0.1, 1.0, 10.0]
}


def rolling_xgboost_tuned(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'target_vol',
    window: int = 500,
    refit_freq: int = 20,
    n_iter_search: int = 10,
    verbose: bool = False
) -> pd.Series:
    """
    Rolling window XGBoost with hyperparameter tuning at each refit.

    Uses TimeSeriesSplit (3 splits) to avoid look-ahead bias during tuning.
    The model is refit every `refit_freq` days; between refits, the latest model
    is used for prediction.

    Args:
        df: DataFrame containing features and target.
        feature_cols: List of column names to use as predictors.
        target_col: Name of target column.
        window: Rolling estimation window size.
        refit_freq: Re-estimate model every N steps.
        n_iter_search: Number of parameter settings sampled in RandomizedSearchCV.
        verbose: Print progress every 100 steps.

    Returns:
        Series of forecasted volatility aligned with df index.
        Values before the first window are NaN.
    """
    forecasts = pd.Series(index=df.index, dtype=float)
    current_model = None

    for i in range(window, len(df)):
        # Check if we need to refit
        if current_model is None or (i - window) % refit_freq == 0:
            train = df.iloc[i-window:i]
            X_train = train[feature_cols]
            y_train = train[target_col]

            # TimeSeriesSplit with 3 splits for temporal cross-validation
            tscv = TimeSeriesSplit(n_splits=3)

            model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                verbosity=0,
                n_jobs=1
            )

            search = RandomizedSearchCV(
                model,
                param_distributions=PARAM_GRID,
                n_iter=n_iter_search,
                cv=tscv,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=1
            )
            search.fit(X_train, y_train)
            current_model = search.best_estimator_

        # Predict for current day
        X_test = df.iloc[[i]][feature_cols]
        pred = current_model.predict(X_test)[0]
        forecasts.iloc[i] = max(pred, 0)

        if verbose and i % 100 == 0:
            print(f"  XGBoost tuned step {i}/{len(df)}")

    return forecasts


def rolling_xgboost_simple(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'target_vol',
    window: int = 500,
    refit_freq: int = 20,
    verbose: bool = False
) -> pd.Series:
    """
    Rolling window XGBoost without hyperparameter tuning (fixed params).
    Useful for baseline comparison or when computational budget is tight.
    """
    forecasts = pd.Series(index=df.index, dtype=float)
    current_model = None

    fixed_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': 1
    }

    for i in range(window, len(df)):
        if current_model is None or (i - window) % refit_freq == 0:
            train = df.iloc[i-window:i]
            X_train = train[feature_cols]
            y_train = train[target_col]

            model = xgb.XGBRegressor(**fixed_params)
            model.fit(X_train, y_train)
            current_model = model

        X_test = df.iloc[[i]][feature_cols]
        pred = current_model.predict(X_test)[0]
        forecasts.iloc[i] = max(pred, 0)

        if verbose and i % 100 == 0:
            print(f"  XGBoost simple step {i}/{len(df)}")

    return forecasts


def prepare_pure_xgboost_features(
    df_fibra: pd.DataFrame,
    df_fx: pd.DataFrame,
    lags: int = 20
) -> tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature set for pure XGBoost (no GARCH inputs).
    """
    from fibras.features import create_features, get_feature_columns
    feat_df = create_features(df_fibra, df_fx, lags=lags)
    feature_cols = get_feature_columns(feat_df)
    return feat_df, feature_cols


def prepare_ensemble_features(
    df_fibra: pd.DataFrame,
    df_fx: pd.DataFrame,
    garch_forecast: pd.Series,
    dist_name: str = 't',
    lags: int = 20
) -> tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature set for GARCH-X ensemble (includes GARCH forecasts as features).
    """
    from fibras.features import create_features, add_garch_features, get_feature_columns
    feat_df = create_features(df_fibra, df_fx, lags=lags)
    feat_df = add_garch_features(feat_df, garch_forecast, dist_name)
    feature_cols = get_feature_columns(feat_df)
    return feat_df, feature_cols


def rolling_garch_x_ensemble(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'target_vol',
    garch_cols: List[str] = ['garch_vol', 'garch_ratio'],
    window: int = 500,
    refit_freq: int = 20,
    n_iter_search: int = 10,
    verbose: bool = False
) -> pd.Series:
    """
    Rolling window GARCH-X Ensemble model with hyperparameter tuning.

    Combines base features with GARCH-t forecasts and ratios.

    Args:
        df: DataFrame containing features, GARCH columns, and target.
        feature_cols: List of base feature column names.
        target_col: Name of target column.
        garch_cols: List of GARCH-related columns to include.
        window: Rolling estimation window size.
        refit_freq: Re-estimate model every N steps.
        n_iter_search: Number of parameter settings sampled in RandomizedSearchCV.
        verbose: Print progress every 100 steps.

    Returns:
        Series of forecasted volatility aligned with df index.
    """
    all_features = feature_cols + garch_cols
    return rolling_xgboost_tuned(
        df=df,
        feature_cols=all_features,
        target_col=target_col,
        window=window,
        refit_freq=refit_freq,
        n_iter_search=n_iter_search,
        verbose=verbose
    )


if __name__ == "__main__":
    from fibras.data_loader import load_fibra_data, load_fx_data
    from fibras.features import create_features, get_feature_columns

    print("Cargando datos...")
    df_fibra = load_fibra_data()
    df_fx = load_fx_data()

    print("Creando features...")
    feat_df = create_features(df_fibra, df_fx, lags=10)
    feature_cols = get_feature_columns(feat_df)

    print(f"Features: {len(feature_cols)} columnas")
    print(f"Filas: {len(feat_df)}")

    # Prueba rápida con ventana reducida
    test_window = 200
    if len(feat_df) > test_window + 10:
        print(f"\nProbando rolling XGBoost simple (window={test_window})...")
        forecasts = rolling_xgboost_simple(
            feat_df,
            feature_cols=feature_cols,
            window=test_window,
            refit_freq=50,
            verbose=True
        )
        print(f"Forecasts generados: {forecasts.notna().sum()}")