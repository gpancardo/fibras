"""
Descarga robusta de datos para FIBRAs y factores macro (USD/MXN).
Incluye caché local para garantizar reproducibilidad exacta.
"""

import os
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional

# Configuración
DATA_RAW_DIR = "data/raw"
INDEX_TICKER = "FIBRATC14.MX"
FX_TICKER = "USDMXN=X"
DEFAULT_START = "2018-01-01"

os.makedirs(DATA_RAW_DIR, exist_ok=True)


def _save_and_hash(df: pd.DataFrame, name: str) -> None:
    """Guarda DataFrame en CSV y genera hash MD5 para trazabilidad."""
    path = os.path.join(DATA_RAW_DIR, f"{name}.csv")
    df.to_csv(path, index=True)
    with open(path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    with open(os.path.join(DATA_RAW_DIR, "checksum.md5"), "a") as f:
        f.write(f"{file_hash}  {name}.csv\n")


def download_index(start: str = DEFAULT_START, end: Optional[str] = None) -> pd.DataFrame:
    """Descarga FIBRATC14 con auto_adjust=True."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = yf.download(INDEX_TICKER, start=start, end=end, progress=False, auto_adjust=True)
    df = df[['Close', 'Volume']].copy()
    df.columns = ['price', 'volume']
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    _save_and_hash(df, "fibra_index")
    return df


def download_usdmxn(start: str = DEFAULT_START, end: Optional[str] = None) -> pd.DataFrame:
    """Descarga tipo de cambio USD/MXN."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = yf.download(FX_TICKER, start=start, end=end, progress=False, auto_adjust=True)
    df = df[['Close']].copy()
    df.columns = ['fx_price']
    df['fx_return'] = np.log(df['fx_price'] / df['fx_price'].shift(1))
    df = df.dropna()
    _save_and_hash(df, "usdmxn")
    return df


def load_fibra_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Carga los datos del índice FIBRA. Si use_cache=True, lee de CSV local.
    """
    cache_path = os.path.join(DATA_RAW_DIR, "fibra_index.csv")
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = download_index()
    return df


def load_fx_data(use_cache: bool = True) -> pd.DataFrame:
    """Carga datos de USD/MXN."""
    cache_path = os.path.join(DATA_RAW_DIR, "usdmxn.csv")
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        df = download_usdmxn()
    return df


def load_ticker_data(ticker: str, start: str = DEFAULT_START, end: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Descarga y prepara datos de un ticker específico.

    Args:
        ticker: Símbolo del activo (e.g., 'FIBRATC14.MX').
        start: Fecha de inicio (formato 'YYYY-MM-DD').
        end: Fecha de fin (formato 'YYYY-MM-DD').
        use_cache: Si True, lee de CSV local si existe.

    Returns:
        DataFrame con columnas: 'price', 'volume', 'log_return'.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    cache_name = ticker.replace('.', '_')
    cache_path = os.path.join(DATA_RAW_DIR, f"{cache_name}.csv")

    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No se encontraron datos para el ticker {ticker}.")

    df = df[['Close', 'Volume']].copy()
    df.columns = ['price', 'volume']
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()

    _save_and_hash(df, cache_name)
    return df


if __name__ == "__main__":
    df_idx = load_fibra_data(use_cache=False)
    df_fx = load_fx_data(use_cache=False)
    print(f"Índice: {df_idx.shape}")
    print(f"USD/MXN: {df_fx.shape}")