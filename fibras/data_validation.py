"""
Data validation layer for the FIBRA VaR pipeline.

Performs explicit checks before modeling:
  - Missing values
  - Non-trading days alignment
  - Return calculation consistency
  - Outlier detection (extreme returns)

Output: data_validation_report.json
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _check_missing(df: pd.DataFrame, ticker: str) -> dict:
    """Check for missing values in returns and prices."""
    results = {"ticker": ticker, "checks": []}

    for col in ["log_return", "price", "volume"]:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            n_total = len(df)
            results["checks"].append({
                "check": f"missing_{col}",
                "n_missing": int(n_miss),
                "n_total": n_total,
                "pct_missing": round(float(n_miss / n_total * 100), 2) if n_total else 0,
                "status": "PASS" if n_miss == 0 else "FAIL"
            })

    return results


def _check_non_trading_alignment(df: pd.DataFrame, ticker: str) -> dict:
    """Check for expected trading-day spacing (no weekends)."""
    if len(df) < 2:
        return {"ticker": ticker, "checks": [{"check": "non_trading_days", "status": "SKIP"}]}

    dates = pd.Series(df.index)
    diffs = dates.diff().dropna()
    # Weekends: > 1 day but < 4 days (Fri-Mon gap)
    gaps = diffs[diffs > pd.Timedelta(days=1)]
    long_gaps = gaps[gaps > pd.Timedelta(days=4)]

    long_gap_dates = []
    for gap_val, gap_idx in zip(long_gaps, long_gaps.index):
        if pd.notna(gap_val):
            target_date = df.index[gap_idx]
            if hasattr(target_date, 'date'):
                long_gap_dates.append(str(target_date.date()))
            else:
                long_gap_dates.append(str(target_date))
        if len(long_gap_dates) >= 5:
            break

    return {
        "ticker": ticker,
        "checks": [{
            "check": "non_trading_days",
            "total_obs": len(df),
            "n_weekend_gaps": int((gaps <= pd.Timedelta(days=4)).sum()),
            "n_long_gaps": int((gaps > pd.Timedelta(days=4)).sum()),
            "long_gap_dates": long_gap_dates,
            "status": "PASS" if len(long_gaps) == 0 else "WARN"
        }]
    }


def _check_return_consistency(df: pd.DataFrame, ticker: str) -> dict:
    """Check that log_return matches price differences."""
    if "log_return" not in df.columns or "price" not in df.columns:
        return {"ticker": ticker, "checks": [{"check": "return_consistency", "status": "SKIP"}]}

    computed = np.log(df["price"] / df["price"].shift(1))
    diff = (df["log_return"] - computed).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()

    return {
        "ticker": ticker,
        "checks": [{
            "check": "return_consistency",
            "max_abs_diff": round(float(max_diff), 10),
            "mean_abs_diff": round(float(mean_diff), 10),
            "status": "PASS" if max_diff < 1e-8 else "FAIL"
        }]
    }


def _check_outliers(df: pd.DataFrame, ticker: str, threshold: float = 5.0) -> dict:
    """Detect extreme returns (> threshold standard deviations from mean)."""
    if "log_return" not in df.columns:
        return {"ticker": ticker, "checks": [{"check": "outliers", "status": "SKIP"}]}

    r = df["log_return"].dropna()
    mean_r = float(r.mean())
    std_r = float(r.std())
    z_scores = (r - mean_r) / std_r
    extreme = z_scores[z_scores.abs() > threshold]
    n_extreme = len(extreme)

    return {
        "ticker": ticker,
        "checks": [{
            "check": "outliers",
            "threshold_sigma": threshold,
            "mean_return": round(mean_r, 6),
            "std_return": round(std_r, 6),
            "n_extreme": int(n_extreme),
            "pct_extreme": round(float(n_extreme / len(r) * 100), 3),
            "extreme_dates": [str(d.date()) for d in extreme.index[:10]],
            "extreme_values": [round(float(v), 6) for v in extreme.values[:10]],
            "status": "WARN" if n_extreme > 0 else "PASS"
        }]
    }


def validate_data(
    df_fibra: pd.DataFrame,
    df_fx: pd.DataFrame,
    ticker: str = "FIBRATC14.MX",
    output_dir: str = "output"
) -> dict:
    """
    Run full data validation suite and save report.

    Args:
        df_fibra: FIBRA data with log_return, price, volume.
        df_fx: FX data with fx_return.
        ticker: Ticker label.
        output_dir: Directory for report output.

    Returns:
        Full validation report as dict.
    """
    report = {
        "pipeline": "FIBRA VaR Backtesting",
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "sections": []
    }

    report["sections"].append(_check_missing(df_fibra, ticker))
    report["sections"].append(_check_non_trading_alignment(df_fibra, ticker))
    report["sections"].append(_check_return_consistency(df_fibra, ticker))
    report["sections"].append(_check_outliers(df_fibra, ticker))

    # FX checks
    if df_fx is not None and len(df_fx) > 0:
        fx_report = _check_missing(df_fx, "USD/MXN")
        fx_report["checks"] = [c for c in fx_report["checks"] if "fx_return" in c.get("check", "")]
        if fx_report["checks"]:
            report["sections"].append(fx_report)

        # Alignment check: FIBRA vs FX index overlap
        overlap = df_fibra.index.intersection(df_fx.index)
        report["sections"].append({
            "ticker": "Alignment",
            "checks": [{
                "check": "fibra_fx_alignment",
                "n_fibra": len(df_fibra),
                "n_fx": len(df_fx),
                "n_overlap": len(overlap),
                "pct_overlap": round(len(overlap) / max(len(df_fibra), 1) * 100, 1),
                "status": "PASS" if len(overlap) > 0.9 * len(df_fibra) else "WARN"
            }]
        })

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "data_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report
