"""
Value-at-Risk computation and backtesting framework.
Includes Kupiec POF test, Christoffersen independence test, and exact binomial confidence intervals.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, t as t_dist, chi2, binom
from typing import Literal, Dict, Optional

DistType = Literal["normal", "t"]


def var_from_volatility(
    volatility_forecast: pd.Series,
    alpha: float = 0.05,
    dist: DistType = "normal",
    df_t: Optional[float] = None
) -> pd.Series:
    """
    Convert volatility forecast to Value-at-Risk.

    VaR_α = -σ * z_α   (assuming zero mean return)

    Args:
        volatility_forecast: Forecasted daily volatility (standard deviation).
        alpha: Significance level (e.g., 0.05 for 95% VaR).
        dist: Distribution for quantile ('normal' or 't').
        df_t: Degrees of freedom for t-distribution; defaults to 5 if None.

    Returns:
        Series of VaR estimates (positive values representing loss magnitude).
    """
    if dist == "normal":
        z = norm.ppf(alpha)
    elif dist == "t":
        dof = df_t if df_t is not None else 5.0
        z = t_dist.ppf(alpha, dof)
    else:
        raise ValueError("dist must be 'normal' or 't'")

    var = -volatility_forecast * z
    return var


def expected_shortfall_from_volatility(
    volatility_forecast: pd.Series,
    alpha: float = 0.05,
    dist: DistType = "normal",
    df_t: Optional[float] = None
) -> pd.Series:
    """
    Convert volatility forecast to Expected Shortfall.

    For normal: ES_α = σ * φ(z_α) / α, where z_α = Φ⁻¹(α)
    For Student-t: ES_α = σ * [f_t(t_α, ν) / α] * (ν + t_α²) / (ν - 1),
      where t_α = F_t⁻¹(α, ν) and ν > 1.

    ES values are positive (loss magnitude), consistent with VaR.

    Args:
        volatility_forecast: Forecasted daily volatility (standard deviation).
        alpha: Significance level (e.g., 0.05 for 95% ES).
        dist: Distribution for tail quantile ('normal' or 't').
        df_t: Degrees of freedom for t-distribution; defaults to 5 if None.

    Returns:
        Series of ES estimates (positive values representing loss magnitude).
    """
    if dist == "normal":
        z = norm.ppf(alpha)
        es_factor = norm.pdf(z) / alpha
    elif dist == "t":
        dof = df_t if df_t is not None else 5.0
        if dof <= 1:
            raise ValueError("df_t must be > 1 for Expected Shortfall to exist.")
        z = t_dist.ppf(alpha, dof)
        es_factor = (t_dist.pdf(z, dof) / alpha) * (dof + z**2) / (dof - 1)
    else:
        raise ValueError("dist must be 'normal' or 't'")

    es = volatility_forecast * es_factor
    return es


def backtest_series(
    returns: pd.Series,
    var_forecast: pd.Series,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Align returns and VaR, flag breaches.

    Args:
        returns: Actual log returns.
        var_forecast: VaR forecast (positive loss magnitude).
        alpha: Significance level used for VaR.

    Returns:
        DataFrame with columns: return, var, breach (1 if return < -var else 0).
    """
    df = pd.DataFrame({
        'return': returns,
        'var': var_forecast
    }).dropna()

    df['breach'] = (df['return'] < -df['var']).astype(int)
    return df


def kupiec_pof(breaches: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """
    Kupiec Proportion of Failures test.
    H0: Observed breach rate equals expected rate alpha.

    Args:
        breaches: Binary series of breach indicators.
        alpha: Expected breach rate.

    Returns:
        Dictionary with 'statistic' (LR) and 'p_value'.
    """
    n = len(breaches)
    x = breaches.sum()
    p_hat = x / n if n > 0 else 0

    if n == 0:
        return {'statistic': np.nan, 'p_value': np.nan}

    if p_hat == 0:
        lr_pof = -2 * np.log((1 - alpha) ** n)
    elif p_hat == 1:
        lr_pof = -2 * np.log(alpha ** n)
    else:
        lr_pof = -2 * (
            (n - x) * np.log((1 - alpha) / (1 - p_hat)) +
            x * np.log(alpha / p_hat)
        )

    p_value = 1 - chi2.cdf(lr_pof, df=1)
    return {'statistic': lr_pof, 'p_value': p_value}


def christoffersen_independence(breaches: pd.Series) -> Dict[str, float]:
    """
    Christoffersen test for independence of breaches.
    H0: Breaches occur independently (no clustering).

    Args:
        breaches: Binary series of breach indicators.

    Returns:
        Dictionary with 'statistic' (LR) and 'p_value'.
    """
    breaches_clean = breaches.dropna()
    if len(breaches_clean) < 2:
        return {'statistic': np.nan, 'p_value': np.nan}

    n = len(breaches_clean)
    n00 = ((breaches_clean.shift(1) == 0) & (breaches_clean == 0)).sum()
    n01 = ((breaches_clean.shift(1) == 0) & (breaches_clean == 1)).sum()
    n10 = ((breaches_clean.shift(1) == 1) & (breaches_clean == 0)).sum()
    n11 = ((breaches_clean.shift(1) == 1) & (breaches_clean == 1)).sum()

    if n01 == 0 or n11 == 0:
        return {'statistic': 0.0, 'p_value': 1.0}

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / n

    lr_ind = -2 * np.log(
        ((1 - p) ** (n00 + n10) * p ** (n01 + n11)) /
        ((1 - p01) ** n00 * p01 ** n01 * (1 - p11) ** n10 * p11 ** n11)
    )
    p_value = 1 - chi2.cdf(lr_ind, df=1)
    return {'statistic': lr_ind, 'p_value': p_value}


def binomial_confidence_interval(
    breaches: pd.Series,
    alpha: float = 0.05,
    conf_level: float = 0.95
) -> Dict[str, float]:
    """
    Exact Clopper-Pearson confidence interval for breach rate.

    Args:
        breaches: Binary series of breach indicators.
        alpha: Expected breach rate (for reference, not used in calculation).
        conf_level: Confidence level for interval (e.g., 0.95).

    Returns:
        Dictionary with 'lower' and 'upper' bounds.
    """
    n = len(breaches)
    x = breaches.sum()

    if n == 0:
        return {'lower': np.nan, 'upper': np.nan}

    if x == 0:
        lower = 0.0
        upper = 1 - (alpha / 2) ** (1 / n)  # Aproximación
    elif x == n:
        lower = (alpha / 2) ** (1 / n)
        upper = 1.0
    else:
        lower = binom.ppf((1 - conf_level) / 2, n, x / n) / n
        upper = binom.ppf(1 - (1 - conf_level) / 2, n, x / n) / n

    return {'lower': lower, 'upper': upper}


def evaluate_model(
    returns: pd.Series,
    var_forecast: pd.Series,
    alpha: float = 0.05
) -> Dict:
    """
    Compute all backtest metrics for a given VaR series.

    Args:
        returns: Actual log returns.
        var_forecast: VaR forecast series.
        alpha: Expected breach rate.

    Returns:
        Dictionary with metrics including confidence intervals.
    """
    bt_df = backtest_series(returns, var_forecast, alpha)
    breaches = bt_df['breach']

    kupiec = kupiec_pof(breaches, alpha)
    christo = christoffersen_independence(breaches)
    ci = binomial_confidence_interval(breaches, alpha)

    return {
        'n_obs': len(bt_df),
        'n_breaches': int(breaches.sum()),
        'breach_rate': breaches.mean(),
        'expected_rate': alpha,
        'ci_lower': ci['lower'],
        'ci_upper': ci['upper'],
        'kupiec_stat': kupiec['statistic'],
        'kupiec_pval': kupiec['p_value'],
        'christo_stat': christo['statistic'],
        'christo_pval': christo['p_value']
    }


def evaluate_by_regime(
    returns: pd.Series,
    var_forecast: pd.Series,
    regimes: pd.Series,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Evaluate VaR performance separately for each regime.

    Args:
        returns: Log returns.
        var_forecast: VaR forecasts.
        regimes: Series with regime labels (same index as returns).
        alpha: Expected breach rate.

    Returns:
        DataFrame with one row per regime containing all evaluation metrics.
    """
    results = []
    for regime_name in regimes.dropna().unique():
        mask = (regimes == regime_name) & returns.notna() & var_forecast.notna()
        sub_returns = returns[mask]
        sub_var = var_forecast[mask]

        if len(sub_returns) < 30:
            continue

        metrics = evaluate_model(sub_returns, sub_var, alpha)
        metrics['regime'] = regime_name
        results.append(metrics)

    return pd.DataFrame(results)


def evaluate_conditional_fx(
    returns: pd.Series,
    var_forecast: pd.Series,
    fx_returns: pd.Series,
    threshold: float = 0.02,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Evaluate VaR performance conditional on large FX moves.

    Splits the sample into days where MXN depreciated by > threshold
    (fx_return > threshold) vs. other days.

    Args:
        returns: Log returns of FIBRA.
        var_forecast: VaR forecasts.
        fx_returns: Daily FX returns (USD/MXN).
        threshold: Threshold for large depreciation (e.g., 0.02 = 2%).
        alpha: Expected breach rate.

    Returns:
        DataFrame with metrics for 'FX_Stressed' and 'FX_Calm'.
    """
    common_idx = returns.index.intersection(var_forecast.index).intersection(fx_returns.index)
    aligned_returns = returns.loc[common_idx]
    aligned_var = var_forecast.loc[common_idx]
    aligned_fx = fx_returns.loc[common_idx]

    stressed_mask = aligned_fx > threshold
    calm_mask = ~stressed_mask

    results = []
    for label, mask in [('FX_Stressed', stressed_mask), ('FX_Calm', calm_mask)]:
        if mask.sum() < 10:
            continue
        metrics = evaluate_model(aligned_returns[mask], aligned_var[mask], alpha)
        metrics['condition'] = label
        results.append(metrics)

    return pd.DataFrame(results)


def regime_specific_metrics(
    returns: pd.Series,
    var_forecast: pd.Series,
    regimes: pd.Series,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute backtesting metrics for specific regimes.

    Args:
        returns: Series of realized returns.
        var_forecast: Series of VaR forecasts.
        regimes: Series of regime labels (e.g., 'High Vol', 'Normal').
        alpha: VaR confidence level.

    Returns:
        DataFrame with metrics for each regime.
    """
    metrics = []
    for regime in regimes.unique():
        mask = regimes == regime
        regime_returns = returns[mask]
        regime_var = var_forecast[mask]

        breaches = (regime_returns < -regime_var).astype(int)
        breach_rate = breaches.mean()
        kupiec_stat = -2 * np.log((1 - alpha) ** (len(breaches) - breaches.sum()) * alpha ** breaches.sum())
        kupiec_stat += 2 * np.log((1 - breach_rate) ** (len(breaches) - breaches.sum()) * breach_rate ** breaches.sum())

        metrics.append({
            'Regime': regime,
            'Breach Rate': breach_rate,
            'Kupiec POF': kupiec_stat,
            'Num Breaches': breaches.sum(),
            'Total Days': len(breaches)
        })

    return pd.DataFrame(metrics)


def logistic_regression_effect(
    breaches: pd.Series,
    regimes: pd.Series
) -> pd.DataFrame:
    """
    Perform logistic regression to test regime effect on breaches.

    Args:
        breaches: Series of breach indicators (1 if breach, 0 otherwise).
        regimes: Series of regime labels (e.g., 'High Vol', 'Normal').

    Returns:
        DataFrame with logistic regression results.
    """
    from sklearn.linear_model import LogisticRegression

    X = pd.get_dummies(regimes, drop_first=True)
    y = breaches

    model = LogisticRegression()
    model.fit(X, y)

    coef = model.coef_[0]
    p_values = np.exp(coef) / (1 + np.exp(coef))  # Approximation for odds ratio

    return pd.DataFrame({
        'Regime': X.columns,
        'Coefficient': coef,
        'Odds Ratio': p_values
    })


def diebold_mariano_test(
    actual: pd.Series,
    forecast1: pd.Series,
    forecast2: pd.Series,
    loss: str = 'mse',
    h: int = 1
) -> Dict[str, float]:
    """
    Perform Diebold-Mariano test for predictive accuracy.

    Args:
        actual: Realized values (e.g., realized volatility).
        forecast1: Forecasts from model 1.
        forecast2: Forecasts from model 2.
        loss: Loss function ('mse' or 'mae').
        h: Forecast horizon (default 1).

    Returns:
        Dictionary with DM statistic and p-value.
    """
    if loss == 'mse':
        e1 = (actual - forecast1) ** 2
        e2 = (actual - forecast2) ** 2
    elif loss == 'mae':
        e1 = np.abs(actual - forecast1)
        e2 = np.abs(actual - forecast2)
    else:
        raise ValueError("Loss function must be 'mse' or 'mae'.")

    d = e1 - e2
    mean_d = np.mean(d)
    n = len(d)

    # Newey-West adjustment for autocorrelation
    gamma = np.array([np.cov(d[:-i], d[i:])[0, 1] for i in range(1, h)])
    var_d = np.var(d) + 2 * np.sum(gamma)

    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - t_dist.cdf(np.abs(dm_stat), df=n-1))

    return {"dm_stat": dm_stat, "p_value": p_value}


if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    returns = pd.Series(np.random.normal(0, 0.01, n))
    var_forecast = pd.Series(np.full(n, 0.01645))

    metrics = evaluate_model(returns, var_forecast, alpha=0.05)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")