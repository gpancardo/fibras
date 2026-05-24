"""
Ingeniería de features con estricto respeto a la información disponible en t-1.
Incluye el retorno del USD/MXN como factor externo.

Todas las features se calculan usando únicamente datos hasta t-1.
El target es |r_t| (volatilidad realizada del día t).
"""

import pandas as pd
import numpy as np
from typing import Optional


def create_features(
    df_fibra: pd.DataFrame,
    df_fx: pd.DataFrame,
    lags: int = 20
) -> pd.DataFrame:
    """
    Construye el dataset de modelado sin leakage.

    Args:
        df_fibra: DataFrame con columnas 'log_return', 'price', 'volume'.
        df_fx: DataFrame con columna 'fx_return'.
        lags: Número de rezagos para retornos absolutos.

    Returns:
        DataFrame con features y target 'target_vol'.
    """
    # Alinear fechas
    common_idx = df_fibra.index.intersection(df_fx.index)
    df = df_fibra.loc[common_idx].copy()
    df['fx_return'] = df_fx.loc[common_idx, 'fx_return']

    # Target: valor absoluto del retorno del día t (shift(-1) para alinear)
    df['target_vol'] = df['log_return'].abs().shift(-1)

    # Features de retornos propios (hasta t-1)
    for lag in range(1, lags + 1):
        df[f'abs_ret_lag{lag}'] = df['log_return'].abs().shift(lag)

    for lag in range(1, 6):
        df[f'sq_ret_lag{lag}'] = (df['log_return'] ** 2).shift(lag)

    # Volatilidad histórica (ventana terminando en t-1)
    df['hist_vol_20'] = df['log_return'].rolling(20).std().shift(1)
    df['hist_vol_60'] = df['log_return'].rolling(60).std().shift(1)

    # Features de volumen
    if 'volume' in df.columns:
        df['log_volume_change'] = np.log(df['volume'] / df['volume'].shift(1)).shift(1)

    # Factor USD/MXN (retorno y volatilidad)
    df['fx_abs_ret'] = df['fx_return'].abs().shift(1)
    df['fx_vol_20'] = df['fx_return'].rolling(20).std().shift(1)

    # Dummies de día de la semana (exógenas, sin leakage)
    df['dow'] = df.index.dayofweek
    dow_dummies = pd.get_dummies(df['dow'], prefix='dow')
    df = pd.concat([df, dow_dummies], axis=1)

    # Features de GARCH-t
    if 'garch_t_forecast' in df.columns:
        df['garch_t_ratio'] = df['garch_t_forecast'] / df['hist_vol_20']

    # Drop metadata columns that may carry legitimate NaN (regime, etc.)
    # before NaN removal so they don't shrink the sample needlessly
    drop_cols = ['regime', 'fx_return_raw']
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    # Eliminar filas con NaN en features o target
    df = df.dropna()

    return df


def add_garch_features(
    df: pd.DataFrame,
    garch_forecast: pd.Series,
    dist_name: str = 't'
) -> pd.DataFrame:
    """
    Añade el forecast GARCH como feature.

    Args:
        df: DataFrame con features base.
        garch_forecast: Serie con pronóstico de volatilidad GARCH (index aligned, NaN
            en fechas sin refit).
        dist_name: Nombre de la distribución ('normal' o 't').

    Returns:
        DataFrame con columnas adicionales: garch_vol_{dist_name}, garch_hist_ratio_{dist_name}.
    """
    df = df.copy()
    col_name = f'garch_vol_{dist_name}'
    df[col_name] = garch_forecast

    if 'hist_vol_20' in df.columns:
        df[f'garch_hist_ratio_{dist_name}'] = (
            df[col_name] / df['hist_vol_20'].replace(0, np.nan)
        )

    return df


def get_feature_columns(
    df: pd.DataFrame,
    exclude_patterns: Optional[list] = None
) -> list:
    """
    Retorna lista de columnas de features, excluyendo las no predictivas.

    Args:
        df: DataFrame completo.
        exclude_patterns: Lista de substrings a excluir.

    Returns:
        Lista de nombres de columnas.
    """
    if exclude_patterns is None:
        exclude_patterns = ['price', 'volume', 'log_return', 'target_vol',
                            'regime', 'regime_numeric', 'dow', 'fx_return', 'usdmxn']

    feature_cols = []
    for col in df.columns:
        exclude = False
        for pat in exclude_patterns:
            if pat in col:
                exclude = True
                break
        if not exclude:
            feature_cols.append(col)

    return feature_cols


if __name__ == "__main__":
    from fibras.data_loader import load_fibra_data, load_fx_data

    print("Cargando datos...")
    df_fibra = load_fibra_data()
    df_fx = load_fx_data()

    print("Creando features...")
    df_feat = create_features(df_fibra, df_fx, lags=10)

    print(f"Dimensiones: {df_feat.shape}")
    print(f"Columnas: {df_feat.columns.tolist()}")
    print(f"\nPrimeras 5 filas:")
    print(df_feat.head())