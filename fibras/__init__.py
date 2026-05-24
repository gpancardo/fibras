"""
FIBRA VaR — Ex-Ante Regimes and Machine Learning Overfitting.
"""

from fibras.backtest import (
    var_from_volatility,
    expected_shortfall_from_volatility,
    expected_shortfall_realized,
    compute_historical_var_series,
    backtest_series,
    kupiec_pof,
    christoffersen_independence,
    binomial_confidence_interval,
    binomial_ci_proper,
    exact_binomial_pvalue,
    expected_breach_distribution,
    conservatism_metric,
    rolling_breach_rate,
    pooled_breach_rate,
    evaluate_model,
    evaluate_by_regime,
    evaluate_conditional_fx,
    diebold_mariano_test,
)
from fibras.data_loader import load_fibra_data, load_fx_data, load_ticker_data
from fibras.data_validation import validate_data
from fibras.features import create_features, add_garch_features, get_feature_columns
from fibras.garch_models import generate_all_garch_forecasts, rolling_garch_forecast
from fibras.power_analysis import (
    kupiec_power_curve,
    size_distribution,
    block_bootstrap_ci,
)
from fibras.regimes import assign_regimes_exante
from fibras.xgboost_models import (
    rolling_xgboost_tuned,
    rolling_xgboost_simple,
    prepare_pure_xgboost_features,
    prepare_ensemble_features,
)
