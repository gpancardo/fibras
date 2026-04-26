# FIBRA VaR: Ex-Ante Regimes and Machine Learning Calibration

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

**A rigorous backtesting study comparing GARCH models and gradient-boosted trees for Value-at-Risk forecasting on Mexican REITs, with a methodological focus on ex-ante vs. ex-post regime definitions.**

**Author:** Germán Pancardo

---

## Abstract

This study compares classical GARCH models against gradient-boosted trees (XGBoost) for Value-at-Risk (VaR) forecasting on the S&P/BMV FIBRAS Index (`FIBRATC14.MX`) spanning 2018 to present. All models use identical rolling-window information sets, strict zero-leakage feature engineering, and rigorous out-of-sample statistical testing. The **key methodological contribution** is a comparison of ex-ante regimes (rolling realized volatility percentiles, available at time *t*) against ex-post regimes (fixed crisis dates, known only after the fact), and a power-curve analysis to interpret backtest results under finite-sample constraints. Results show that GARCH-t provides more reliably calibrated VaR forecasts, while XGBoost systematically overestimates risk. The study documents the inherent tension between methodological rigour and statistical power in short emerging-market REIT series.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data](#data)
- [Configuration](#configuration)
- [Methodology](#methodology)
- [Outputs](#outputs)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [Citation](#citation)

---

## Overview

This project provides a complete, reproducible pipeline for VaR backtesting with regime-aware evaluation. It was designed for an academic study on Mexican Real Estate Investment Trusts (FIBRAs) but the framework is general and applies to any liquid time series.

**Core research question:** Can machine learning models outperform GARCH in short, emerging-market return series when evaluated honestly — using ex-ante (forward-looking) regime definitions instead of ex-post (look-ahead) crisis dates?

The pipeline covers:

- Data downloading with caching and MD5 checksums
- Zero-leakage feature engineering (lag structure enforced at `t-1`)
- Ex-ante regime assignment from rolling realized volatility
- Rolling-window GARCH(1,1) forecasting (Normal and Student-t)
- Rolling-window XGBoost forecasting (pure and GARCH-X ensemble)
- VaR computation and full backtesting suite
- Multi-asset validation across FIBRAs, IPC, and Mexico ETF
- Diebold-Mariano test for volatility forecast precision comparison
- Kupiec test power curve for finite-sample significance context
- FX-conditional stress analysis
- Publication-quality figures

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<username>/fibras.git
cd fibras
pip install pandas numpy scipy scikit-learn matplotlib seaborn yfinance arch xgboost
```

**Requirements:** Python 3.10 or higher.

The `arch` package is used for GARCH estimation, `xgboost` for gradient-boosted trees, `yfinance` for market data, and `scikit-learn` for cross-validation. All other packages are standard scientific Python libraries.

---

## Quick Start

```bash
python -m fibras.run_analysis
```

Results are written to `output/` upon completion.

---

## Project Structure

```
fibras/                      ← Python package root (run as: python -m fibras)
│   ├── __init__.py          Package entry point
│   ├── run_analysis.py      Main execution script — full pipeline + improvements
│   ├── data_loader.py       Downloads and caches ticker data + USD/MXN
│   ├── regimes.py          Ex-ante regime assignment (rolling volatility percentile)
│   ├── features.py        Zero-leakage feature engineering (lags at t-1)
│   ├── garch_models.py     GARCH(1,1) with Normal and Student-t innovations
│   ├── xgboost_models.py   XGBoost (pure and GARCH-X ensemble) with hyperparameter tuning
│   ├── backtest.py        VaR computation, Kupiec POF, Christoffersen, CI, Diebold-Mariano
│   ├── power_analysis.py    Kupiec test power curve simulation
│   └── visualization.py   Publication-quality figures (incl. power curve)
├── data/raw/               Cached raw data + MD5 checksums
├── output/                  Backtesting results and figures
│   └── figures/            Publication-quality plots
├── paper/                  LaTeX paper draft and references
├── poster/                 A0 poster layout
├── context.md              Full study specification
└── README.md              This file
```

---

## Data

| Asset | Ticker | Source | Period |
|-------|-------|--------|--------|
| FIBRAS Index ETF | `FIBRATC14.MX` | S&P/BMV via yfinance | 2018-01-01 to present |
| Fibra Uno (liquid FIBRA) | `FUNO11.MX` | BMV via yfinance | 2018-01-01 to present |
| Prologis México | `FIBRAPL14.MX` | BMV via yfinance | 2018-01-01 to present |
| IPC (BMV benchmark) | `^MXX` | BMV via yfinance | 2018-01-01 to present |
| iShares MSCI Mexico ETF | `EWW` | NYSE via yfinance | 2018-01-01 to present |
| USD/MXN FX | `USDMXN=X` | Yahoo Finance via yfinance | 2018-01-01 to present |

- Daily closing prices, `auto_adjust=True` (splits and dividends adjusted)
- Returns: log-return $r_t = \ln(P_t / P_{t-1})$
- Raw CSV files are cached locally with MD5 checksums for reproducibility
- If the cache is present, subsequent runs load from disk automatically

---

## Configuration

Key parameters in `run_analysis.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALPHA` | `0.05` | VaR confidence level (5% = 95% VaR) |
| `WINDOW` | `125` | Rolling estimation window (trading days) |
| `REFIT_FREQ` | `20` | Re-estimate models every N trading days |
| `USE_TUNED_XGB` | `True` | Use `RandomizedSearchCV` for XGBoost (vs. fixed params) |

To change the VaR level, modify `ALPHA` (e.g., `0.01` for 99% VaR) and re-run.

---

## Methodology

### Regime Definitions

**Ex-ante (High Vol / Normal)**  
Each day *t* is classified as *High Vol* if the realized 20-day volatility exceeds the 90th percentile of the last 500 realized volatilities. All information is available at *t-1*; no look-ahead bias.

**Ex-post (Crisis / Normal)**  
Fixed crisis windows are known only after the fact (COVID: Mar–Jun 2020; Mexican election: May–Jun 2024). This is the standard industry practice and the baseline the study critiques.

### Models

| Model | Distribution | Rolling Window | Re-estimate |
|-------|------------|-------------|-----------|
| GARCH(1,1) — Normal | Normal | 125 days | Every 20 days |
| GARCH(1,1) — Student-t | Student-t (df estimated) | 125 days | Every 20 days |
| XGBoost — Pure | — | 125 days | Every 20 days |
| XGBoost — GARCH-X Ensemble | — | 125 days | Every 20 days |

XGBoost variants are tuned via 3-fold `TimeSeriesSplit` with `RandomizedSearchCV` (10 iterations) over `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and `reg_lambda`.

### VaR Calculation

$$\text{VaR}_t^{(m)}(\alpha) = -\hat{\sigma}_t^{(m)} \cdot z_\alpha$$

where $z_\alpha$ is the $\alpha$-quantile of the Standard Normal (GARCH-Normal and XGBoost) or Student-t with estimated degrees of freedom (GARCH-t). Zero expected return is assumed.

### Backtesting Tests

| Test | Null Hypothesis |
|------|----------------|
| Kupiec POF | Breach rate equals the expected rate $\alpha$ |
| Christoffersen Independence | Breaches occur independently (no clustering) |
| Clopper-Pearson CI | Exact confidence interval for breach rate |
| Diebold-Mariano | Forecast errors from two models have equal expected loss |
| Kupiec Power Curve | Finite-sample rejection rates for known deviations |

### Multi-Asset Validation

The pipeline runs GARCH-t and XGBoost-Pure (tuned) on five assets — `FIBRATC14.MX`, `FUNO11.MX`, `FIBRAPL14.MX`, `^MXX`, `EWW` — to confirm results are not specific to a single ticker. A summary table is exported to `output/multi_asset_results.csv`.

### Diebold-Mariano Test

Forecast accuracy is compared pairwise via the Diebold-Mariano test using squared-error loss. Realized volatility is proxied by $|r_t|$. The test uses a simple Newey-West autocorrelation correction (truncation at forecast horizon $h=1$). Results are saved to `output/dm_test_results.csv`.

### Kupiec Power Curve

The finite-sample power of the Kupiec POF test is estimated via Monte Carlo simulation ($n=400$, $n_{\text{sim}}=1000$, `random_state=42`). The proportion of rejections is computed over true breach rates ranging from 2% to 8%. The resulting curve (`output/figures/fig_power_curve.png`) shows that deviations of $\pm1$–$2$ percentage points from the nominal 5% rate yield low rejection probabilities, contextualizing non-significant p-values.

---

## Outputs

All results are saved to `output/`:

| File | Description |
|------|-------------|
| `aligned_forecasts.csv` | Full aligned series: returns, VaR forecasts, regimes, FX returns |
| `overall_metrics.csv` | Table 1 — overall backtesting metrics by model |
| `regime_metrics.csv` | Tables 2–3 — regime-specific backtesting metrics |
| `fx_conditional_metrics.csv` | FX stress analysis (MXN depreciation > 2%) |
| `multi_asset_results.csv` | GARCH-t vs XGBoost-Pure results across 5 assets |
| `dm_test_results.csv` | Diebold-Mariano test: GARCH-t vs XGBoost forecast precision |

Metrics include: number of observations, number of breaches, breach rate, Clopper-Pearson 95% CI, Kupiec POF statistic and p-value, Christoffersen statistic and p-value.

---

## Visualization

Generate all figures from saved outputs:

```bash
python -m fibras.visualization
```

Figures are saved to `output/figures/`. The power curve is generated automatically by `run_analysis.py` and by `generate_all_figures()`. To compute the power curve data standalone:

```bash
python -m fibras.power_analysis
```

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
yfinance>=0.2.0
arch>=5.0.0
xgboost>=2.0.0
```

---

## Citation

> Pancardo, G. (2026). *Is Your Backtest Lying? Ex-Ante Regimes Expose Machine Learning Overfitting in Mexican REIT Value-at-Risk*. Working Paper.

---

## Author

**Germán Pancardo**  
[ ghpancardo@gmail.com ]

For the full study specification, see `context.md`.