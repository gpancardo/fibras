"""
Robust data downloader for FIBRAs and macro factors (USD/MXN).

Includes local caching with MD5 verification for exact reproducibility.
Each CSV has a sidecar .md5 file; an aggregate checksum.md5 is also maintained.
On cache load, the hash is verified; on mismatch, data is re-downloaded.
"""

import os
import hashlib
import sys
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np

# --- Configuration ---
DATA_RAW_DIR = "data/raw"
INDEX_TICKER = "FIBRATC14.MX"
FX_TICKER = "USDMXN=X"
DEFAULT_START = "2018-01-01"

os.makedirs(DATA_RAW_DIR, exist_ok=True)


# ============================================================
# HASHING AND VERIFICATION
# ============================================================

def _file_md5(path: str) -> str:
    """Return MD5 hex digest of file at *path*."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _sidecar_path(name: str) -> str:
    return os.path.join(DATA_RAW_DIR, f"{name}.csv.md5")


def _write_sidecar(name: str, file_hash: str) -> None:
    with open(_sidecar_path(name), "w") as f:
        f.write(file_hash)


def _verify_hash(name: str) -> bool:
    """Return True iff CSV exists and its MD5 matches the stored sidecar."""
    csv_path = os.path.join(DATA_RAW_DIR, f"{name}.csv")
    md5_path = _sidecar_path(name)
    if not os.path.exists(csv_path) or not os.path.exists(md5_path):
        return False
    return _file_md5(csv_path) == open(md5_path).read().strip()


def _regenerate_checksum_file() -> None:
    """Scan all .csv files in DATA_RAW_DIR and write a clean checksum.md5.

    Paths are stored relative to the project root so that
    ``md5sum -c data/raw/checksum.md5`` works from the repo root.
    """
    lines = []
    for fname in sorted(os.listdir(DATA_RAW_DIR)):
        if not fname.endswith(".csv") or fname == "checksum.md5":
            continue
        rel_path = os.path.join(DATA_RAW_DIR, fname)
        h = _file_md5(rel_path)
        lines.append(f"{h}  {rel_path}")
    cs_path = os.path.join(DATA_RAW_DIR, "checksum.md5")
    with open(cs_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def verify_all() -> dict:
    """Check all cached CSVs. Return dict of {name: True/False}."""
    results = {}
    for fname in sorted(os.listdir(DATA_RAW_DIR)):
        if not fname.endswith(".csv") or fname == "checksum.md5":
            continue
        name = fname[:-4]  # strip .csv
        results[name] = _verify_hash(name)
    return results


# ============================================================
# SAVE + DOWNLOAD HELPERS
# ============================================================

def _save_and_hash(df: pd.DataFrame, name: str) -> None:
    """Save DataFrame as CSV, write sidecar .md5, update aggregate checksum."""
    csv_path = os.path.join(DATA_RAW_DIR, f"{name}.csv")
    df.to_csv(csv_path, index=True)
    file_hash = _file_md5(csv_path)
    _write_sidecar(name, file_hash)
    _regenerate_checksum_file()


def download_index(start: str = DEFAULT_START,
                   end: Optional[str] = None) -> pd.DataFrame:
    """Download FIBRATC14.MX with auto_adjust=True."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = yf.download(INDEX_TICKER, start=start, end=end,
                     progress=False, auto_adjust=True)
    df = df[["Close", "Volume"]].copy()
    df.columns = ["price", "volume"]
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df = df.dropna()
    _save_and_hash(df, "fibra_index")
    return df


def download_usdmxn(start: str = DEFAULT_START,
                    end: Optional[str] = None) -> pd.DataFrame:
    """Download USD/MXN exchange rate."""
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = yf.download(FX_TICKER, start=start, end=end,
                     progress=False, auto_adjust=True)
    df = df[["Close"]].copy()
    df.columns = ["fx_price"]
    df["fx_return"] = np.log(df["fx_price"] / df["fx_price"].shift(1))
    df = df.dropna()
    _save_and_hash(df, "usdmxn")
    return df


# ============================================================
# PUBLIC LOAD FUNCTIONS
# ============================================================

def load_fibra_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load FIBRA index data.

    If *use_cache* is True and a valid (hash-verified) cache exists,
    read from disk. Otherwise download fresh data.
    """
    cache_path = os.path.join(DATA_RAW_DIR, "fibra_index.csv")
    if use_cache and os.path.exists(cache_path) and _verify_hash("fibra_index"):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    return download_index()


def load_fx_data(use_cache: bool = True) -> pd.DataFrame:
    """Load USD/MXN data with optional cache + hash verification."""
    cache_path = os.path.join(DATA_RAW_DIR, "usdmxn.csv")
    if use_cache and os.path.exists(cache_path) and _verify_hash("usdmxn"):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    return download_usdmxn()


def load_ticker_data(ticker: str,
                     start: str = DEFAULT_START,
                     end: Optional[str] = None,
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Download and prepare data for a specific ticker.

    Cache name is derived by replacing '.' with '_'
    (e.g. 'FIBRATC14.MX' -> 'FIBRATC14_MX').
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    cache_name = ticker.replace(".", "_")
    cache_path = os.path.join(DATA_RAW_DIR, f"{cache_name}.csv")

    if use_cache and os.path.exists(cache_path) and _verify_hash(cache_name):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for ticker {ticker}.")

    df = df[["Close", "Volume"]].copy()
    df.columns = ["price", "volume"]
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df = df.dropna()
    _save_and_hash(df, cache_name)
    return df


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if "--verify" in sys.argv:
        results = verify_all()
        print(f"{'File':<30} {'Status':<10}")
        print("-" * 40)
        ok = True
        for name, valid in results.items():
            status = "OK" if valid else "MISMATCH / MISSING"
            if not valid:
                ok = False
            print(f"{name:<30} {status:<10}")
        if ok:
            print("\nAll cached files verified successfully.")
        else:
            print("\nSome files failed verification. Re-run to re-download.")
        sys.exit(0 if ok else 1)

    # Default: download index + FX
    df_idx = load_fibra_data(use_cache=False)
    df_fx = load_fx_data(use_cache=False)
    print(f"Index:  {df_idx.shape}")
    print(f"USD/MXN: {df_fx.shape}")
