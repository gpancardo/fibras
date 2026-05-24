"""
Power analysis and size distortion tools for VaR backtesting.

Provides:
- kupiec_power_curve: Rejection probability as function of true breach rate.
- size_distribution: Distribution of Kupiec p-values under the null (H0: p = alpha).
- block_bootstrap_ci: Empirical confidence interval via block bootstrap.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_power_curve(
    n: int = 400,
    alpha: float = 0.05,
    true_rates: np.ndarray = None,
    n_sim: int = 3000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate the power curve for the Kupiec POF test.

    Args:
        n: Number of observations per simulation.
        alpha: VaR confidence level.
        true_rates: Array of true breach rates to test.
        n_sim: Number of simulations per true rate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: true_rate, power, power_exact.
    """
    if true_rates is None:
        true_rates = np.linspace(0.005, 0.25, 50)

    from scipy.stats import binomtest, binom as binom_dist

    # Precompute the LR statistic for every possible count x = 0..n under H0
    x_vals = np.arange(0, n + 1)
    p_hat_vals = x_vals / n
    # Kupiec LR: handle boundaries
    lr_vals = np.empty(n + 1)
    mask_0 = (p_hat_vals == 0)
    mask_1 = (p_hat_vals == 1)
    mask_mid = (~mask_0) & (~mask_1)

    eps = 1e-300
    lr_vals[mask_0] = -2 * np.log(max((1 - alpha) ** n, eps))
    lr_vals[mask_1] = -2 * np.log(max(alpha ** n, eps))
    lr_vals[mask_mid] = -2 * (
        (n - x_vals[mask_mid]) * np.log(np.maximum((1 - alpha) / np.maximum(1 - p_hat_vals[mask_mid], eps), eps)) +
        x_vals[mask_mid] * np.log(np.maximum(alpha / np.maximum(p_hat_vals[mask_mid], eps), eps))
    )
    p_kupiec = 1 - chi2.cdf(lr_vals, df=1)
    rej_kupiec_mask = p_kupiec < 0.05

    # Exact binomial rejection mask for each x
    rej_exact_mask = np.zeros(n + 1, dtype=bool)
    for idx, x in enumerate(x_vals):
        try:
            bt = binomtest(int(x), n, p=alpha, alternative='two-sided')
            rej_exact_mask[idx] = bt.pvalue < 0.05
        except Exception:
            pass

    power = []
    power_exact = []

    for p_true in true_rates:
        # Exact expected power = sum over x of Binom(x | n, p_true) * reject_mask[x]
        pmf = binom_dist.pmf(x_vals, n, p_true)
        power.append(float(np.dot(pmf, rej_kupiec_mask)))
        power_exact.append(float(np.dot(pmf, rej_exact_mask)))

    return pd.DataFrame({
        "true_rate": true_rates,
        "power_kupiec": power,
        "power_exact": power_exact
    })


def size_distribution(
    n: int = 58,
    alpha: float = 0.05,
    n_sim: int = 10000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate distribution of Kupiec p-values under the NULL hypothesis (H0: p = alpha).

    Under correct specification, p-values should be uniformly distributed.
    Deviation from uniformity indicates size distortion.

    Args:
        n: Sample size (e.g., 58, 71).
        alpha: Nominal breach rate.
        n_sim: Number of simulation draws.
        seed: Random seed.

    Returns:
        DataFrame with n_sim rows, column 'p_value'.
    """
    # Use exact binomial distribution for speed (n is small)
    from scipy.stats import binom as binom_dist

    x_vals = np.arange(0, n + 1)
    p_hat_vals = x_vals / n

    eps = 1e-300
    lr_vals = np.empty(n + 1)
    mask_0 = (p_hat_vals == 0)
    mask_1 = (p_hat_vals == 1)
    mask_mid = (~mask_0) & (~mask_1)

    lr_vals[mask_0] = -2 * np.log(max((1 - alpha) ** n, eps))
    lr_vals[mask_1] = -2 * np.log(max(alpha ** n, eps))
    lr_vals[mask_mid] = -2 * (
        (n - x_vals[mask_mid]) * np.log(np.maximum((1 - alpha) / np.maximum(1 - p_hat_vals[mask_mid], eps), eps)) +
        x_vals[mask_mid] * np.log(np.maximum(alpha / np.maximum(p_hat_vals[mask_mid], eps), eps))
    )
    p_vals_per_x = 1 - chi2.cdf(lr_vals, df=1)

    # Sample counts from binomial under H0, map to p-values
    rng = np.random.default_rng(seed)
    counts = rng.binomial(n, alpha, size=n_sim)
    p_values = p_vals_per_x[counts]

    return pd.DataFrame({"p_value": p_values})


def block_bootstrap_ci(
    breaches: pd.Series,
    block_size: int = 5,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42
) -> dict:
    """
    Empirical confidence interval for breach rate via block bootstrap.

    Resamples blocks of consecutive breach indicators to preserve time-series
    dependence, then computes the empirical distribution of the breach rate.

    Args:
        breaches: Binary breach indicator series.
        block_size: Number of consecutive observations per block.
        n_bootstrap: Number of bootstrap resamples.
        alpha: VaR level (for reference).
        seed: Random seed.

    Returns:
        Dict with 'lower', 'upper', 'mean', 'median' breach rates.
    """
    n = len(breaches)
    if n < block_size:
        return {'lower': float('nan'), 'upper': float('nan'),
                'mean': float('nan'), 'median': float('nan')}

    values = breaches.values.astype(float)
    n_blocks = int(np.ceil(n / block_size))

    rng = np.random.default_rng(seed)
    boot_rates = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        block_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        resampled = np.concatenate([
            values[bi:bi + block_size] for bi in block_indices
        ])[:n]
        boot_rates[i] = resampled.mean()

    lower = float(np.percentile(boot_rates, 2.5))
    upper = float(np.percentile(boot_rates, 97.5))
    mean = float(boot_rates.mean())
    median = float(np.median(boot_rates))

    return {
        'lower': lower,
        'upper': upper,
        'mean': mean,
        'median': median
    }


if __name__ == "__main__":
    import sys

    if '--size' in sys.argv:
        print("Running size distortion simulation (n=58, alpha=0.05, 10000 sims)...")
        df = size_distribution(n=58, alpha=0.05, n_sim=10000)
        rejection_rate = (df['p_value'] < 0.05).mean()
        print(f"  Simulated rejection rate under H0: {rejection_rate:.4f} (nominal 0.05)")
        print(f"  P-value quantiles: {df['p_value'].quantile([0.25, 0.5, 0.75]).to_dict()}")

    elif '--power' in sys.argv:
        print("Running power curve (n=58, alpha=0.05)...")
        df = kupiec_power_curve(n=58, alpha=0.05, n_sim=3000)
        print(df.to_string())

    elif '--bootstrap' in sys.argv:
        rng = np.random.default_rng(42)
        breaches = pd.Series(rng.binomial(1, 0.10, size=58))
        result = block_bootstrap_ci(breaches, block_size=5)
        print("Block bootstrap CI:")
        for k, v in result.items():
            print(f"  {k}: {v:.4f}")

    else:
        print("Usage: python power_analysis.py [--size | --power | --bootstrap]")
