"""
Value-at-Risk computation and backtesting framework.
Includes Kupiec POF test, Christoffersen independence test, exact binomial
confidence intervals, rolling stability analysis, conservatism metrics,
and pooled cross-asset evaluation.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, t as t_dist, chi2, binom
from typing import Literal, Dict, Optional, Union

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


def expected_shortfall_realized(
    returns: pd.Series,
    var_forecast: pd.Series,
) -> Optional[float]:
    """
    ES realizado (ex-post): media de los retornos que violan el VaR.

    Esta es una medida realizada, no predictiva: calcula la pérdida promedio
    condicional en los días donde el retorno fue peor que el VaR pronosticado.

    Args:
        returns: Log returns (negativos en pérdidas).
        var_forecast: VaR forecast (positive loss magnitude).

    Returns:
        ES as positive magnitude (mean of -returns on breach days), or NaN if no breaches.
    """
    common_idx = returns.index.intersection(var_forecast.index)
    r = returns.loc[common_idx]
    v = var_forecast.loc[common_idx]

    breach_mask = r < -v
    if breach_mask.sum() == 0:
        return float('nan')

    breach_losses = -r[breach_mask]
    return float(breach_losses.mean())


def compute_historical_var_series(
    returns: pd.Series,
    window: int,
    alpha: float,
    aligned_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Compute rolling historical VaR as empirical quantile of past returns.

    For each date in aligned_index, VaR is the α-quantile of returns
    in the `window` trading days immediately before that date.
    No model estimation required.

    Args:
        returns: Full series of log returns.
        window: Rolling window size (e.g., 125).
        alpha: VaR significance level (e.g., 0.05 for 95% VaR).
        aligned_index: Dates for which to compute VaR.

    Returns:
        Series of VaR estimates (positive loss magnitudes), indexed by aligned_index.
    """
    # OPTIMIZATION: vectorized pandas rolling quantile instead of Python loop
    rolling_quant = returns.rolling(window).quantile(alpha).shift(1)
    var_series = -rolling_quant.reindex(aligned_index)
    return var_series


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


def exact_binomial_pvalue(
    breaches: pd.Series,
    alpha: float = 0.05
) -> float:
    """
    Exact binomial (two-sided) p-value for the null H0: p = alpha.

    Uses the Clopper-Pearson principle: sum of binomial probabilities for
    all outcomes at least as extreme as the observed count.
    """
    n = len(breaches)
    x = int(breaches.sum())

    if n == 0:
        return float('nan')

    p_vals = binom.pmf(np.arange(0, n + 1), n, alpha)
    observed_pmf = binom.pmf(x, n, alpha)

    p_value = p_vals[p_vals <= observed_pmf].sum()
    return float(min(p_value, 1.0))


def binomial_ci_proper(
    breaches: pd.Series,
    alpha: float = 0.05,
    conf_level: float = 0.95
) -> Dict[str, float]:
    """
    Exact Clopper-Pearson confidence interval for the binomial proportion
    using the beta distribution relationship.

    Args:
        breaches: Binary series of breach indicators.
        alpha: Expected breach rate (for reference, not used).
        conf_level: Confidence level for interval.

    Returns:
        Dictionary with 'lower' and 'upper' bounds.
    """
    from scipy.stats import beta as beta_dist

    n = len(breaches)
    x = int(breaches.sum())

    if n == 0:
        return {'lower': float('nan'), 'upper': float('nan')}

    tail_prob = (1.0 - conf_level) / 2.0

    if x == 0:
        lower = 0.0
        upper = 1.0 - tail_prob ** (1.0 / n)
    elif x == n:
        lower = tail_prob ** (1.0 / n)
        upper = 1.0
    else:
        lower = beta_dist.ppf(tail_prob, x, n - x + 1)
        upper = beta_dist.ppf(1.0 - tail_prob, x + 1, n - x)

    return {'lower': float(lower), 'upper': float(upper)}


def expected_breach_distribution(
    n: int,
    alpha: float = 0.05
) -> Dict[str, float]:
    """Expected count, variance, and standard deviation of breach events under H0."""
    expected_count = n * alpha
    variance = n * alpha * (1.0 - alpha)
    std_dev = np.sqrt(variance)
    return {
        'expected_count': expected_count,
        'variance': variance,
        'std_dev': std_dev,
        'cv': std_dev / expected_count if expected_count > 0 else float('inf')
    }


def conservatism_metric(
    breach_rate: float,
    alpha: float = 0.05
) -> float:
    """
    Conservatism metric: C = p̂ - α

    Negative C → conservative (fewer breaches than expected).
    Positive C → anti-conservative (more breaches than expected).
    """
    return breach_rate - alpha


def rolling_breach_rate(
    returns: pd.Series,
    var_forecast: pd.Series,
    window: int = 30
) -> pd.DataFrame:
    """
    Compute rolling breach rate over a backward-looking window.

    Returns DataFrame with rolling breach rate, count, total, and VaR mean.
    """
    common_idx = returns.index.intersection(var_forecast.index)
    r = returns.loc[common_idx]
    v = var_forecast.loc[common_idx]

    breach = (r < -v).astype(int)
    rolling_rate = breach.rolling(window).mean()
    rolling_count = breach.rolling(window).sum()
    rolling_var_mean = v.rolling(window).mean()

    out = pd.DataFrame({
        'return': r,
        'var': v,
        'breach': breach,
        'rolling_breach_rate': rolling_rate,
        'rolling_breach_count': rolling_count,
        'rolling_var_mean': rolling_var_mean,
    }, index=common_idx)
    return out


def pooled_breach_rate(
    models_data: list,
    alpha: float = 0.05
) -> Dict:
    """
    Compute pooled breach rate across multiple assets or models.

    Args:
        models_data: List of dicts with 'n_obs', 'n_breaches', 'name'.
        alpha: Expected breach rate.

    Returns:
        Dict with pooled statistics.
    """
    total_obs = sum(d['n_obs'] for d in models_data)
    total_breaches = sum(d['n_breaches'] for d in models_data)
    p_pool = total_breaches / total_obs if total_obs > 0 else float('nan')

    from scipy.stats import binomtest
    try:
        bt = binomtest(total_breaches, total_obs, p=alpha, alternative='two-sided')
        exact_p = bt.pvalue
    except Exception:
        exact_p = float('nan')

    exp_dist = expected_breach_distribution(total_obs, alpha)
    c = conservatism_metric(p_pool, alpha)

    return {
        'pooled_n_obs': total_obs,
        'pooled_n_breaches': total_breaches,
        'pooled_breach_rate': p_pool,
        'expected_rate': alpha,
        'conservatism_c': c,
        'exact_binomial_pvalue': exact_p,
        'expected_count': exp_dist['expected_count'],
        'expected_std_dev': exp_dist['std_dev'],
    }


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
    ci_proper = binomial_ci_proper(breaches, alpha)
    ci = binomial_confidence_interval(breaches, alpha)
    exact_p = exact_binomial_pvalue(breaches, alpha)
    exp_dist = expected_breach_distribution(len(bt_df), alpha)
    c_value = conservatism_metric(breaches.mean(), alpha)
    es = expected_shortfall_realized(returns, var_forecast)

    return {
        'n_obs': len(bt_df),
        'n_breaches': int(breaches.sum()),
        'breach_rate': breaches.mean(),
        'expected_rate': alpha,
        'ci_lower': ci['lower'],
        'ci_upper': ci['upper'],
        'ci_lower_cp': ci_proper['lower'],
        'ci_upper_cp': ci_proper['upper'],
        'kupiec_stat': kupiec['statistic'],
        'kupiec_pval': kupiec['p_value'],
        'christo_stat': christo['statistic'],
        'christo_pval': christo['p_value'],
        'exact_binomial_pval': exact_p,
        'expected_n_breaches': exp_dist['expected_count'],
        'expected_std_dev': exp_dist['std_dev'],
        'conservatism_c': c_value,
        'es_realized': es
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