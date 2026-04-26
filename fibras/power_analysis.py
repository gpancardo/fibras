import numpy as np
import pandas as pd
from scipy.stats import chi2

def kupiec_power_curve(n=400, alpha=0.05, true_rates=None, n_sim=1000):
    """
    Simulate the power curve for the Kupiec test.

    Args:
        n: Number of observations per simulation.
        alpha: VaR confidence level.
        true_rates: Array of true breach rates to test.
        n_sim: Number of simulations per true rate.

    Returns:
        DataFrame with true rates and power values.
    """
    if true_rates is None:
        true_rates = np.linspace(0.02, 0.08, 13)

    rng = np.random.default_rng(42)
    power = []

    for p_true in true_rates:
        rejections = 0
        for _ in range(n_sim):
            breaches = rng.binomial(1, p_true, size=n)
            breach_rate = breaches.mean()
            kupiec_stat = -2 * np.log((1 - alpha) ** (n - breaches.sum()) * alpha ** breaches.sum())
            kupiec_stat += 2 * np.log((1 - breach_rate) ** (n - breaches.sum()) * breach_rate ** breaches.sum())
            p_value = 1 - chi2.cdf(kupiec_stat, df=1)
            if p_value < alpha:
                rejections += 1
        power.append(rejections / n_sim)

    return pd.DataFrame({"true_rate": true_rates, "power": power})

if __name__ == "__main__":
    results = kupiec_power_curve()
    print("Power curve results:")
    print(results.to_string())