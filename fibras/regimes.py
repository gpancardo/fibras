"""
Definición de regímenes de volatilidad en tiempo real (ex-ante).
Usa la volatilidad histórica móvil para clasificar cada día como
'Alta Volatilidad' (High Vol) o 'Volatilidad Normal' (Normal) sin información futura.
"""

import pandas as pd
import numpy as np


def assign_regimes_exante(
    df: pd.DataFrame,
    vol_window: int = 20,
    lookback: int = 500,
    threshold_percentile: float = 90.0
) -> pd.DataFrame:
    """
    Asigna regímenes basados en la volatilidad realizada móvil.

    Para cada día t, se calcula la volatilidad realizada de los últimos `vol_window`
    días (hasta t-1). Luego se compara con el percentil `threshold_percentile`
    de la distribución de volatilidades observadas en los últimos `lookback` días
    (también hasta t-1). Si la volatilidad actual supera ese umbral, se clasifica
    como 'High Vol', de lo contrario 'Normal'.

    Este método evita el look-ahead bias porque toda la información usada para
    clasificar el día t está disponible en t-1.

    Args:
        df: DataFrame con columna 'log_return'.
        vol_window: Ventana para calcular volatilidad realizada (ej. 20 días).
        lookback: Ventana para calcular el umbral del percentil (ej. 500 días).
        threshold_percentile: Percentil para definir alta volatilidad (ej. 90).

    Returns:
        DataFrame con columnas adicionales:
            'regime': 'Normal' o 'High Vol'
            'regime_numeric': 0 (Normal) o 1 (High Vol)
    """
    df = df.copy()
    returns = df['log_return']

    # Volatilidad realizada móvil (desviación estándar de los últimos vol_window días)
    # shift(1) para usar solo información hasta t-1
    realized_vol = returns.rolling(vol_window).std().shift(1)

    # Umbral móvil: percentil de la volatilidad en los últimos `lookback` días
    rolling_threshold = realized_vol.rolling(lookback).quantile(threshold_percentile / 100.0)

    # Asignación de régimen
    df['regime'] = 'Normal'
    mask_high = realized_vol > rolling_threshold
    df.loc[mask_high, 'regime'] = 'High Vol'

    # Los primeros días no tienen suficiente historia para calcular el umbral
    min_period = vol_window + lookback
    df.loc[:df.index[min_period], 'regime'] = pd.NA

    df['regime_numeric'] = (df['regime'] == 'High Vol').astype(int)
    return df


def get_regime_counts(df: pd.DataFrame) -> pd.Series:
    """Retorna el conteo de observaciones por régimen (excluyendo NA)."""
    return df['regime'].value_counts(dropna=True)


if __name__ == "__main__":
    from fibras.data_loader import load_fibra_data

    df = load_fibra_data()
    df = assign_regimes_exante(df)

    print("Conteo de regímenes:")
    print(get_regime_counts(df))
    print(f"\nPrimeras fechas con régimen asignado: {df['regime'].first_valid_index()}")