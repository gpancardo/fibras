"""
GARCH volatility forecasting models.
Implements rolling-window GARCH(1,1) estimation with Normal and Student-t innovations.
Uses a fixed rolling window to avoid look-ahead bias and ensure fair comparison.
CPU-only, no GPU required.
"""

import pandas as pd
import numpy as np
from arch import arch_model
from typing import Literal, Optional

Distribution = Literal["normal", "t"]


def fit_single_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: Distribution = "normal"
) -> Optional[pd.Series]:
    """
    Fit a GARCH(p,q) model on a single window and return conditional volatility.

    Args:
        returns: Series of log returns (decimal, not percent).
        p: ARCH order.
        q: GARCH order.
        dist: Innovation distribution ('normal' or 't').

    Returns:
        Series of conditional volatility (same index as returns) or None if fit fails.
    """
    if len(returns) < 100:
        return None

    scaled = returns * 100

    try:
        model = arch_model(scaled, vol='Garch', p=p, q=q, dist=dist)
        res = model.fit(disp='off', show_warning=False)
        cond_vol = res.conditional_volatility / 100
        return cond_vol
    except Exception:
        return None


def rolling_garch_forecast(
    returns: pd.Series,
    window: int = 500,
    dist: Distribution = "normal",
    refit_freq: int = 20
) -> pd.Series:
    """
    Rolling GARCH forecast with optional Student-t distribution.

    Args:
        returns: Series of log returns.
        window: Rolling window size.
        dist: Innovation distribution ('normal' or 't').
        refit_freq: Frequency of model re-estimation.

    Returns:
        Series of conditional volatility indexed by the last observation
        of each window. Values at non-refit dates are NaN.
    """
    vol_forecasts = pd.Series(index=returns.index, dtype=float)

    for start in range(0, len(returns) - window, refit_freq):
        end = start + window
        window_returns = returns.iloc[start:end]
        cond_vol = fit_single_garch(window_returns, dist=dist)

        if cond_vol is not None:
            vol_forecasts.iloc[end] = cond_vol.iloc[-1]

    return vol_forecasts


def generate_all_garch_forecasts(
    returns: pd.Series,
    window: int = 500,
    distributions: list[Distribution] = ["normal", "t"],
    refit_freq: int = 20,
    verbose: bool = False
) -> dict[str, pd.Series]:
    """
    Compute rolling-window GARCH forecasts for multiple distributions.

    Args:
        returns: Log returns.
        window: Rolling window size.
        distributions: List of distributions to compute.
        refit_freq: Re-estimate frequency.
        verbose: Print progress.

    Returns:
        Dictionary of forecasts for each distribution.
    """
    forecasts = {}
    for dist in distributions:
        if verbose:
            print(f"Computing GARCH-{dist} (window={window})...")
        forecasts[dist] = rolling_garch_forecast(
            returns=returns,
            window=window,
            dist=dist,
            refit_freq=refit_freq
        )
    return forecasts


if __name__ == "__main__":
    from fibras.data_loader import load_fibra_data

    df = load_fibra_data()
    returns = df['log_return']

    forecasts = generate_all_garch_forecasts(returns, window=500, verbose=True)

    print(f"\nNormal forecast shape: {forecasts['normal'].shape}")
    print(f"t-dist forecast shape: {forecasts['t'].shape}")
    print(f"NaN count (normal): {forecasts['normal'].isna().sum()}")
    print(f"NaN count (t): {forecasts['t'].isna().sum()}")