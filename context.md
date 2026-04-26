# context.md — FIBRA VaR: Ex‑Ante Regimes, Finite-Sample Power, and Machine Learning Calibration

## 1. Project Overview

This document describes the complete specification for a short paper (9/10 target) and an academic poster (10/10 target) on Value‑at‑Risk (VaR) backtesting for Mexican FIBRAs. The study compares classical GARCH models against gradient‑boosted trees (XGBoost) under **ex‑ante** and **ex‑post** regime definitions, providing a methodological contribution on the impact of look‑ahead bias in market regime classification. All models use identical rolling‑window information sets, strict feature‑engineering rules (no data leakage), and rigorous out‑of‑sample statistical tests.

**Core message:**  
Ex‑ante regime definitions (rolling volatility percentiles) impose strict data requirements that materially constrain testable sample sizes in short financial series. Within this honest framework, GARCH‑t produces more reliably calibrated VaR forecasts, while XGBoost systematically overestimates risk. The finite‑sample power of the Kupiec test limits what can be claimed with statistical confidence; the power curve analysis provides a diagnostic for interpreting non‑significant results. The study documents the inherent tension between methodological rigour and statistical power in emerging‑market risk management, rather than arguing for or against any model class a priori.

---

## 2. Deliverables

- **Short paper** (~12 pages, LaTeX or Markdown) formatted for *Revista Mexicana de Economía y Finanzas*.
- **Poster** (A0 vertical, high‑impact visual layout).
- **Reproducible code** on GitHub (Python, requirements.txt, cached data with checksums).

---

## 3. Data

- **Ticker:** `FIBRATC14.MX` (S&P/BMV FIBRAS Index ETF).
- **FX:** `USDMXN=X` (for exploratory analysis).
- **Period:** 2018‑01‑01 to latest available.
- **Sampling:** Daily closing prices adjusted (`auto_adjust=True`).
- **Return:** Log‑return $r_t = \ln(P_t/P_{t-1})$.
- **Cache:** Save raw CSVs with MD5 checksums in `data/raw/`.

---

## 4. Feature Engineering (Zero Leakage)

All features for day $t$ use **only information up to $t-1$**.

**Target:** $y_{t} = |r_t|$ (absolute return, proxy for realised volatility).

**Features:**
- $|r_{t-k}|$ for $k=1,\ldots,20$
- $r_{t-k}^2$ for $k=1,\ldots,5$
- 20‑day historical volatility: $\sigma_{20,t-1} = \text{std}(r_{t-20},\ldots,r_{t-1})$
- 60‑day historical volatility (same rule)
- $\Delta \log(\text{volume}_{t-1})$
- Day‑of‑week dummies (Monday,…)
- For ensemble models: GARCH‑t forecast $\hat{\sigma}_{\text{GARCH-t},t}$ and ratio $\hat{\sigma}_{\text{GARCH-t},t} / \sigma_{20,t-1}$.
  - Implementation note: these two features are added post-hoc via `add_garch_features()` in `features.py`, after `create_features()` has built the base feature matrix. `rolling_garch_forecast()` returns a `pd.Series` indexed by the original returns timeline (NaN at non-refit dates), not a standalone `DataFrame`. The ratio is computed in `add_garch_features()` where $\sigma_{20,t-1}$ is available.

**FX features (exploratory):**
- $|\text{fx\_ret}_{t-1}|$
- 20‑day FX volatility: $\text{std}(\text{fx\_ret}_{t-20},\ldots,\text{fx\_ret}_{t-1})$

These are **not** used in the main model comparison, only for an auxiliary analysis.

---

## 5. Regime Definitions

### 5.1 Ex‑Ante Regime (High Vol / Normal)

For each day $t$:
1. Compute rolling realised volatility: $\text{rv}_{t-1} = \text{std}(r_{t-20},\ldots,r_{t-1})$.
2. Compute the 90th percentile of $\text{rv}$ over the previous 500 days: $\text{threshold}_{t-1} = Q_{0.90}(\text{rv}_{t-500},\ldots,\text{rv}_{t-1})$.
3. If $\text{rv}_{t-1} > \text{threshold}_{t-1}$, label day $t$ as **High Vol**, else **Normal**.

Note: The first $20+500$ days will have missing regime labels and are excluded from the aligned backtest.

### 5.2 Ex‑Post Regime (Crisis / Normal)

Define fixed crisis periods *known after the fact*:
- COVID crash: 2020‑03‑01 to 2020‑06‑30
- Mexican general election volatility: 2024‑05‑01 to 2024‑06‑30

Any day inside these intervals is **Crisis**, else **Normal**.

---

## 6. Models

All models produce 1‑day ahead volatility forecast $\hat{\sigma}_t$.

### 6.1 GARCH(1,1) – Normal
- Distribution: Normal.
- Estimation: rolling window of **125** observations.
- Re‑estimation frequency: every **20** trading days.
- Forecast: conditional volatility at the last fitted point.

### 6.2 GARCH(1,1) – Student‑t
- Same as above, but with Student‑t innovations (degrees of freedom estimated).

### 6.3 Pure XGBoost
- Features: all base features (no GARCH input).
- Rolling window: 125 observations.
- Re‑fit every 20 days.
- **Hyperparameter tuning:** inside each window, 3‑fold `TimeSeriesSplit` with `RandomizedSearchCV` (10 iterations) over grid: `max_depth` [2,3,4], `learning_rate` [0.01,0.05,0.1], `subsample` [0.7,0.8,0.9], `colsample_bytree` [0.7,0.8,0.9], `reg_lambda` [0.1,1,10]. Use `neg_mean_squared_error` scoring.
- Predict $\hat{\sigma}_t = \max(0, \text{prediction})$.

### 6.4 GARCH‑X Ensemble
- Features: base features **plus** the GARCH‑t forecast and its ratio to historical vol.
- Same tuning protocol as pure XGBoost.

### 6.5 Historical VaR Baseline (for robustness)
- Forecast $\hat{\sigma}_t$ is the rolling 125‑day standard deviation of $r_{t-1}, r_{t-2}, \ldots$, shifted to avoid leakage.
- Alternatively, directly compute VaR as the empirical 5% quantile of the last 125 returns (no volatility model). Both variants acceptable.

---

## 7. VaR Calculation

For a given model $m$, $\text{VaR}_t^{(m)}(\alpha) = - \hat{\sigma}_t^{(m)} \cdot z_\alpha$, where $z_\alpha$ is the $\alpha$‑quantile of the assumed distribution (Standard Normal for GARCH‑Normal/XGBoost, Student‑t with estimated df for GARCH‑t). Assume zero expected return.

Confidence levels: primary $\alpha=5\%$, robustness $\alpha=1\%$.

---

## 8. Backtesting Metrics

Compute over the common aligned out‑of‑sample period (all models produce forecasts):

### 8.1 Breach indicator
$\text{I}_t = 1$ if $r_t < -\text{VaR}_t$ else 0.

### 8.2 Kupiec POF test
$H_0$: breach rate $= \alpha$. Report likelihood‑ratio statistic and p‑value.

### 8.3 Christoffersen Independence test
$H_0$: breaches are independent. Report LR statistic and p‑value.

### 8.4 Exact binomial confidence interval
95% Clopper‑Pearson interval for the observed breach rate.

### 8.5 Error decomposition
- Mean error: $\text{ME} = \frac{1}{T}\sum (r_t + \text{VaR}_t)$.
- Mean error on breach days: $\text{ME}_{\text{breach}} = \frac{1}{N_{\text{breach}}}\sum_{t:\text{breach}} (r_t + \text{VaR}_t)$.

### 8.6 Regime‑specific metrics
Repeat all above for subsets defined by ex‑ante regime and ex‑post regime.

### 8.7 Formal regime effect test
For each model, logistic regression:
$\text{logit}(P(\text{breach}_t=1)) = \beta_0 + \beta_1 \cdot \text{HighVol}_t$
Report $\beta_1$, p‑value, and odds ratio.

---

## 9. Robustness Checks

1. **VaR level:** re‑run entire backtest with $\alpha=1\%$. Main conclusion (GARCH‑t > XGBoost) must hold or be discussed.
2. **Window length:** change rolling window from 500 to 250 and 750. Show that qualitative ranking of models remains stable.
3. **Baseline model:** include Historical VaR. If it performs similarly or better than XGBoost, report it as evidence of ML over‑engineering.

---

## 10. Conditional FX Analysis (Exploratory)

Define stress threshold: daily MXN depreciation > 1%. Split the aligned sample into “FX Stressed” and “FX Calm”. Compute breach rates and Kupiec tests for each model in each subgroup. Report as a table but explicitly state that the small sample precludes causal claims; it is only suggestive of the ML’s sensitivity to exchange‑rate noise.

---

## 11. Paper Structure

1. **Abstract** (~150 words): problem, ex‑ante vs ex‑post regime finding, main model ranking, implication.
2. **Introduction**: motivate VaR backtesting bias, relevance of FIBRAs, contribution (methodological warning).
3. **Data and Methodology**:
   - Data description.
   - Regime definitions (ex‑ante, ex‑post) with formulas.
   - Models (explain each concisely).
   - VaR and backtesting metrics.
4. **Results**:
   - Overall performance (Table 1).
   - Regime‑specific: ex‑ante (Table 2) and ex‑post (Table 3) with contrast commentary.
   - Logistic regression results for regime effect (Table 4).
   - Error decomposition (Table 5).
   - Robustness: 1% VaR and alternate windows (summarise in text, tables in appendix).
   - FX conditional (brief paragraph, table in appendix).
5. **Discussion**:
   - Why GARCH‑t works and XGBoost overfits (data scarcity, tail parameterisation).
   - The danger of ex‑post regimes in model validation.
   - Practical recommendations for risk managers.
6. **Conclusion**: summary, limitations, future work (individual FIBRAs, expected shortfall).
7. **References**.
8. **Appendix**: full robustness tables, code availability, data checksums.

---

## 12. Poster Design (A0 vertical)

### Headline (top 10%)
**“Is Your Backtest Lying? Ex‑Ante Regimes Expose Machine Learning Overfitting in Mexican REIT VaR”**

### Main Graphic (centre, ~50%)
One large plot containing:
- Daily FIBRA returns (grey line, α=0.5)
- GARCH‑t 95% VaR (green dashed)
- XGBoost‑Ensemble 95% VaR (orange dotted)
- High‑Vol regime shading (red semi‑transparent)
- Legend clearly visible.

### Left column (20% width)
- Problem: Risk managers use crisis dates to validate models (look‑ahead).
- Method: Ex‑ante regimes from rolling realised volatility (formula). Four models compared. Tests: Kupiec, Christoffersen, logistic regression.

### Right column (20% width)
- Key results: Traffic‑light table with ex‑ante vs ex‑post verdicts. GARCH‑t ✅ | XGBoost 🔴 under ex‑ante; XGBoost appears 🟡 under ex‑post → false confidence.
- One‑sentence takeaway: “Ex‑post regimes hide model failures; use ex‑ante definitions to avoid catastrophic VaR breaches.”

### Bottom strip (5%)
Author, affiliation, GitHub QR code, contact.

---

## 13. Implementation Checklist

- [ ] Data download script with caching and MD5 hashes.
- [ ] Feature engineering module: ensure all shifts end at $t-1$.
- [ ] Ex‑ante regime assignment function.
- [ ] Ex‑post regime assignment function.
- [x] GARCH forecasting with fixed rolling window, refit frequency.
  - `rolling_garch_forecast()` returns a `pd.Series` indexed by the original returns timeline (NaN at non-refit dates), not a `DataFrame`.
- [x] XGBoost forecasting with `TimeSeriesSplit` tuning inside each window.
- [ ] VaR calculation and backtesting module (including all tests).
- [ ] Logistic regression function for regime effect.
- [ ] Error decomposition function.
- [ ] Main analysis script that runs everything and exports CSVs.
- [x] Script rerun with `ALPHA=0.01` and alternative window sizes.
- [ ] Visualization script (combined plot, traffic‑light table).
- [ ] LaTeX/Markdown paper templated with placeholder tables replaced by real outputs.
- [ ] Poster design (Canva/TikZ) incorporating final figures.

---

## 14. Code Organisation
├── fibras/                  Python package
│   ├── __init__.py        Package entry point
│   ├── run_analysis.py   Main execution script
│   ├── data_loader.py    Data download + caching (MD5)
│   ├── regimes.py        Ex-ante regime assignment
│   ├── features.py       Zero-leakage feature engineering
│   ├── garch_models.py  GARCH(1,1) — returns pd.Series (not DataFrame)
│   ├── xgboost_models.py  XGBoost + GARCH-X ensemble
│   ├── backtest.py      VaR backtesting suite
│   ├── power_analysis.py  Kupiec power curve
│   └── visualization.py  Publication figures
├── data/raw/              Cached data + checksum.md5
├── output/                Results + figures
│   └── figures/
├── paper/
│   ├── paper.md
│   └── references.bib
├── poster/
│   └── poster.pdf
└── README.md

This context provides all necessary specifications to build the study from scratch with high rigor. No new models are needed; the strength comes from the clean comparison protocol, the ex‑ante vs ex‑post identification, and the thorough statistical framework.