# Finite-Sample Limitations of VaR Backtesting under Ex-Ante Volatility Regimes: Evidence from Mexican FIBRAs

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

**Reproducible pipeline for the accompanying paper.** Compares GARCH and XGBoost VaR models under ex-ante volatility regimes on Mexican FIBRA data, with exact finite-sample power analysis of standard backtests.

**Author:** Germán Pancardo

---

## Abstract

Ex-ante volatility regime definitions consume approximately 520 trading days for classification, reducing the usable out-of-sample evaluation window to 62--95 observations for Mexican FIBRAs. At n = 62, the Kupiec test rejects a correctly specified model 7.6% of the time (nominal 5%) and has only 43% power to detect a model breaching at double the nominal rate. The Christoffersen test is uninformative below n ≈ 100. This pipeline implements the proposed alternative framework: exact binomial p-values, Clopper--Pearson confidence intervals, and a conservatism metric C = p̂ − α. Applied to FIBRA data, GARCH(1,1)-t produces a breach rate of 4.8% (C = −0.002) while XGBoost-Pure produces 11.3% (C = +0.063). The pattern holds across five equity tickers: GARCH-t pooled rate 3.6%, XGBoost pooled rate 9.7% (p < 0.0001). When evaluation windows fall below 100 observations, standard backtests lack discriminatory power and directional conservatism across assets provides more reliable guidance than isolated p-values.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducibility](#reproducibility)
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

This pipeline provides a complete, reproducible backtesting framework for Value-at-Risk model validation under strict ex-ante (no-look-ahead) conditions. It was designed for the accompanying paper but the framework is general and applies to any liquid time series.

**Core contribution:** Quantifies the finite-sample degradation of standard backtests when ex-ante regime definitions compress the evaluation window below n = 100, and provides a practical alternative based on exact binomial inference.

The pipeline covers:

- Data downloading with MD5 checksum verification on every cache load
- Zero-leakage feature engineering (all lags enforced at t-1)
- Ex-ante regime assignment from rolling realized volatility
- Rolling-window GARCH(1,1) forecasting (Normal and Student-t)
- Rolling-window XGBoost forecasting (pure and GARCH-X ensemble)
- VaR computation and full backtesting suite
- Exact binomial p-values, Clopper-Pearson confidence intervals, conservatism metric C = p̂ − α
- Kupiec POF and Christoffersen independence tests (for comparability)
- Block bootstrap confidence intervals for breach rates
- Diebold-Mariano test for volatility forecast comparison
- Exact finite-sample power curves and size distortion analysis
- Multi-asset validation across five equity tickers
- Sensitivity analysis (varying estimation window and refit frequency)
- Data validation layer (missing values, alignment, outliers)
- Monte Carlo simulation of structural limitations (standalone)
- Publication-quality PDF figures

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<username>/fibras.git
cd fibras
pip install pandas numpy scipy scikit-learn matplotlib seaborn yfinance arch xgboost
```

**Requirements:** Python 3.10 or higher.

The `arch` package is used for GARCH estimation, `xgboost` for gradient-boosted trees, `yfinance` for market data, and `scikit-learn` for cross-validation.

---

## Quick Start

Run the full analysis pipeline:

```bash
python -m fibras.run_analysis
```

All results and figures are written to `output/` upon completion (~10--15 minutes single-threaded).

To verify data integrity:

```bash
python -m fibras.data_loader --verify
```

---

## Reproducibility

The pipeline is fully deterministic:

- **Global seed:** 42 (set before any stochastic operation)
- **Data caching:** Every downloaded CSV has a sidecar `.csv.md5` file containing its MD5 hash. The `checksum.md5` aggregate file enables batch verification via `md5sum -c`. On every cache load, the pipeline verifies the hash against the sidecar; a mismatch triggers automatic re-download.
- **Deterministic GARCH:** The `arch` package is seeded via the global numpy seed.
- **Deterministic XGBoost:** `random_state=42` is set in all model calls.
- **Git hash:** The current commit is logged to `output/pipeline_meta.json` on every run.

To verify all cached data:

```bash
python -m fibras.data_loader --verify
```

To check the aggregate checksum file externally (run from project root):

```bash
md5sum -c data/raw/checksum.md5
```

---

## Project Structure

```
fibras/                      Python package root (run as: python -m fibras)
├── __init__.py
├── run_analysis.py          Main execution script (full pipeline)
├── data_loader.py           Data download, caching, MD5 verification
├── regimes.py              Ex-ante regime assignment (rolling volatility percentile)
├── features.py             Zero-leakage feature engineering
├── garch_models.py         GARCH(1,1) with Normal and Student-t innovations
├── xgboost_models.py       XGBoost (pure and ensemble) with tuning
├── backtest.py             VaR computation, Kupiec, Christoffersen, Clopper-Pearson, DM test
├── power_analysis.py       Exact finite-sample power curves
├── data_validation.py      Pre-modeling data checks
└── visualization.py        Publication-quality figures (PDF)
data/raw/                   Cached raw data + .md5 sidecars + checksum.md5
src/
└── monte_carlo_simulation.py  Standalone Monte Carlo (GARCH-t DGP)
output/
├── aligned_forecasts.csv
├── overall_metrics.csv
├── regime_metrics.csv
├── multi_asset_results.csv
├── conservatism_metrics.csv
├── sensitivity_analysis.csv
├── dm_test_results.csv
├── breaches_*.csv
├── pipeline_meta.json
├── data_validation_report.json
├── monte_carlo_raw.csv
├── monte_carlo_summary.csv
└── figures/                Publication-quality PDF figures
paper/                      LaTeX source + compiled PDF
├── paper.tex
├── title.tex
├── references.bib
└── paper.pdf
poster/                     A0 poster layout
PIPELINE_DIAGNOSIS.md       Technical diagnosis and optimization notes
context.md                  Full study specification (superseded by paper)
```

---

## Data

| Asset | Ticker | Source | Period |
|-------|--------|--------|--------|
| FIBRAS Index ETF | `FIBRATC14.MX` | S&P/BMV via yfinance | 2018-01-01 to present |
| Fibra Uno | `FUNO11.MX` | BMV via yfinance | 2018-01-01 to present |
| Prologis México | `FIBRAPL14.MX` | BMV via yfinance | 2018-01-01 to present |
| IPC (BMV benchmark) | `^MXX` | BMV via yfinance | 2018-01-01 to present |
| iShares MSCI Mexico ETF | `EWW` | NYSE via yfinance | 2018-01-01 to present |
| USD/MXN FX | `USDMXN=X` | Yahoo Finance via yfinance | 2018-01-01 to present |

- Daily closing prices, `auto_adjust=True` (splits and dividends adjusted)
- Returns: log-return $r_t = \ln(P_t / P_{t-1})$
- Raw CSV files are cached with MD5 sidecar checksums; every cache load verifies integrity
- If the cache is missing or corrupted, data is re-downloaded automatically
- Data validation (missing values, alignment, outliers) runs before any model fitting

---

## Configuration

Key parameters in `run_analysis.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALPHA` | `0.05` | VaR confidence level (5% = 95% VaR) |
| `WINDOW` | `125` | Rolling estimation window (trading days) |
| `REFIT_FREQ` | `20` | Re-estimate models every N trading days |

To change the VaR level, modify `ALPHA` (e.g., `0.01` for 99% VaR) and re-run.

---

## Methodology

### Regime Definition

**Ex-ante (High Vol / Normal):** Each day t is classified as High Vol if the realized 20-day volatility exceeds the 90th percentile of the last 500 realized volatilities. All information is available at t-1; no look-ahead bias.

### Models

| Model | Distribution | Rolling Window | Re-estimate |
|-------|------------|----------------|-------------|
| GARCH(1,1) — Normal | Normal | 125 days | Every 20 days |
| GARCH(1,1) — Student-t | Student-t (df estimated) | 125 days | Every 20 days |
| XGBoost — Pure | — | 125 days | Every 20 days |
| XGBoost — GARCH-X Ensemble | — | 125 days | Every 20 days |

XGBoost variants use tuned hyperparameters via 3-fold TimeSeriesSplit with RandomizedSearchCV (10 iterations).

### VaR Calculation

$$\text{VaR}_t^{(m)}(\alpha) = -\hat{\sigma}_t^{(m)} \cdot z_\alpha$$

where $z_\alpha$ is the $\alpha$-quantile of the Standard Normal (GARCH-Normal and XGBoost) or Student-t with estimated degrees of freedom (GARCH-t). Zero expected return is assumed.

### Backtesting Framework

| Tool | Role | Properties |
|------|------|------------|
| Conservatism metric $C = \hat{p} - \alpha$ | Primary decision variable | Valid at any n; negative = conservative, positive = anti-conservative |
| Exact binomial test | Primary inference | Non-asymptotic; controls size even at n < 100 |
| Clopper-Pearson CI | Uncertainty quantification | Exact 95% interval for breach rate |
| Kupiec POF test | Secondary context | Asymptotic $\chi^2_1$ approximation; unreliable at n < 100 |
| Christoffersen test | Secondary context | Negligible power at n < 100 |

### Power Analysis

Rejection probabilities are computed exactly from the binomial distribution (not Monte Carlo). For each possible breach count $x \in \{0, \dots, n\}$, the Kupiec likelihood-ratio statistic and the exact binomial p-value are evaluated against the 5% threshold. Power at a given true breach rate $p_{\text{true}}$ is:

$$\text{Power}(p_{\text{true}}) = \sum_{x=0}^{n} \binom{n}{x} \, p_{\text{true}}^{x} \, (1-p_{\text{true}})^{n-x} \cdot \mathbf{1}\{\text{test rejects at count } x\}$$

---

## Outputs

All results are saved to `output/`:

| File | Description |
|------|-------------|
| `aligned_forecasts.csv` | Full aligned series: returns, VaR forecasts, regimes, FX returns |
| `overall_metrics.csv` | Backtesting metrics by model (breach rate, C, exact p, CP-CI, Kupiec, Christoffersen) |
| `regime_metrics.csv` | Regime-specific backtesting metrics |
| `conservatism_metrics.csv` | Per-model conservatism C |
| `multi_asset_results.csv` | GARCH-t vs XGBoost across 5 tickers with pooled rates |
| `dm_test_results.csv` | Diebold-Mariano test results |
| `sensitivity_analysis.csv` | Window/refit sensitivity for GARCH-t |
| `breaches_*.csv` | Per-model breach vectors |
| `pipeline_meta.json` | Seed, git hash, timestamp |
| `data_validation_report.json` | Pre-modeling validation results |
| `monte_carlo_raw.csv` | Full Monte Carlo simulation data (3200 rows) |
| `monte_carlo_summary.csv` | Aggregated Monte Carlo results |
| `figures/` | Publication-quality PDF figures |

---

## Visualization

Generate all figures from saved outputs:

```bash
python -m fibras.visualization
```

Figures are saved to `output/figures/` as PDF files. The power curves are also generated automatically by `run_analysis.py`.

Standalone power curve computation:

```bash
python -m fibras.power_analysis
```

Monte Carlo simulation (standalone, ~1 minute):

```bash
python src/monte_carlo_simulation.py
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

> Pancardo, G. (2026). *Finite-Sample Limitations of VaR Backtesting under Ex-Ante Volatility Regimes: Evidence from Mexican FIBRAs*. Working Paper.

---

## Author

**Germán Pancardo**  
ghpancardo@gmail.com

For the full study specification, see `context.md`. The final paper draft is in `paper/paper.pdf`.
