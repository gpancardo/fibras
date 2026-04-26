# Poster Design Instructions — "Is Your Backtest Lying?"

**Target**: A0 vertical (841 mm × 1189 mm), designed in Canva or PowerPoint.
**Audience**: Quantitatively literate academics and practitioners.

---

## 1. Title Section (Top 10% of poster)

**Background**: Dark navy (`#1a2332`) or charcoal (`#2d2d2d`)

**Title text** (white, bold, ~72pt):
> Is Your Backtest Powered?
> *Ex-Ante Regimes and the Limits of ML Calibration in Mexican REIT Value-at-Risk*

**Subtitle** (grey `#b0b0b0`, ~28pt):
> FIBRA VaR Study | 95% Confidence Level | Rolling Ex-Ante Regimes

**Institution line**: Author Name, Affiliation, email | GitHub QR code (bottom-right of title bar)

---

## 2. Three-Column Layout

### Left Column (25% width) — Problem & Method

**Background**: White or very light grey (`#f5f5f5`)

**Header** "The Problem" (navy, ~30pt, bold)

Bullet points (~22pt):
- FIBRAs = Mexican REITs, growing market, underdeveloped risk management
- Standard practice: validate VaR models using known crisis dates → **look-ahead bias**
- Crisis labels applied *after the fact* cannot inform real-time risk decisions

**Header** "Our Approach" (navy, ~30pt, bold)

Bullet points (~22pt):
- **Ex-ante regimes**: rolling 20-day vol vs. its 90th percentile over past 500 days
- If recent vol > threshold → **High Vol**; else → **Normal**
- All features use info up to $t-1$ only (zero leakage)

**Models compared**:
| Model | Key Detail |
|-------|-----------|
| GARCH(1,1)-Normal | Normal innovations |
| GARCH(1,1)-**t** | Student-$t$ (df estimated) |
| XGBoost-Pure | Base features only |
| XGBoost-Ensemble | + GARCH-t forecast as feature |

- Rolling window: 125 days, refit every 20
- VaR: 95%, zero-mean assumption
- Tests: Kupiec POF, Christoffersen, Diebold-Mariano

**Icon/illustration**: Small diagram of rolling window shifting forward

---

### Centre Column (50% width) — The Key Graphic

**Main plot** (occupies ~60% of this column's height):

Plot FIBRATC14.MX aligned data:
- **X-axis**: Date (2020–2026)
- **Y-axis**: Daily return (grey line, `alpha=0.6`)
- **GARCH-t 95% VaR**: Green dashed line (`#2ecc71`, width 2)
- **XGBoost-Ensemble 95% VaR**: Orange dotted line (`#e67e22`, width 2)
- **Shading**: Red semi-transparent (`#e74c3c`, `alpha=0.15`) for High Vol ex-ante periods
- Breach marks: small red triangles at the bottom for each breach day

**Data source**: `aligned_forecasts.csv` — columns `return`, `var_garch_t`, `var_xgb_ensemble`, `regime`

**Annotation box** (inside the plot, white background with border):
```
GARCH-t:    5.2% breach rate (p = 0.95) ✓
XGBoost:   12.1% breach rate (p = 0.04) ✗
```

**Below the plot** (remaining ~40% of centre column):

**Mini multi-asset table** (traffic-light coloured):
| Asset | GARCH-t | XGBoost-Pure |
|-------|---------|-------------|
| FIBRATC14 | 🟢 5.3% | 🟡 10.5% |
| FUNO11 | 🟢 2.8% | 🟡 9.7% |
| FIBRAPL14 | 🟢 1.4% | 🔴 13.9%* |
| ^MXX | 🟢 2.8% | 🟢 5.6% |
| EWW | 🟡 0.0% | 🟢 5.6% |

* = Kupiec p < 0.05

Legend: 🟢 Green = breach rate within [3%, 7%] | 🟡 Yellow = within [2%, 8%] | 🔴 Red = outside

**Power curve thumbnail** (small figure, ~8cm wide, placed to the right of the table):

Use `output/figures/fig_power_curve.png` with caption:
> Kupiec power at n=58: even 7% true breach rate detected only ~15% of the time

---

### Right Column (25% width) — Key Findings & Limits

**Background**: White or very light grey, matching left column

**Header** "Key Findings" (navy, ~30pt, bold)

Numbered list (~22pt):
1. **GARCH-t is more reliably calibrated** across all five assets (breach rates 0–5.3%)
2. **XGBoost breach rates ~2x nominal** (10–14%), and Kupiec rejects at 5% level for FIBRATC14 and FIBRAPL14
3. **Diebold-Mariano**: no significant forecast accuracy difference — additional complexity does not yield detectable improvements
4. **Only 4 High-Vol days** in aligned sample → stress regime inference impossible
5. **All 58 days** fell in FX_Calm → no FX stress analysis possible

**Header** "Limitations" (red-tinted navy, ~26pt, bold)

Bullet list (~20pt):
- Only 58 out-of-sample days (methodological purity vs. data availability)
- High-Vol regime not separately evaluable
- Only 95% VaR tested
- Only 5 Mexican assets
- Low statistical power in all tests

**Header** "Take-Home Message" (white text, green `#27ae60` background box)

> **For Mexican FIBRAs, current evidence favours simpler models.**
> Ex-ante regimes impose real data constraints — report power curves alongside p-values, and interpret non-significant results in light of the sample size that rigour leaves behind.

---

## 3. Bottom Strip (5% height)

**Dark background** (same as title bar)

Content (white text, ~20pt, distributed horizontally):
- Author Name | email@example.com
- GitHub: github.com/example/fibras-var
- QR code to repository
- Conference name / date placeholder

---

## 4. Visual Style Guide

### Colours
| Element | Colour | Hex |
|---------|--------|-----|
| Title background | Dark navy | `#1a2332` |
| Title text | White | `#ffffff` |
| GARCH-t line | Green | `#2ecc71` |
| XGBoost line | Orange | `#e67e22` |
| High Vol shade | Red | `#e74c3c` (alpha 0.15) |
| Breach markers | Dark red | `#c0392b` |
| Table green | Light green | `#d5f5e3` |
| Table yellow | Light yellow | `#fef9e7` |
| Table red | Light red | `#fadbd8` |
| Take-home box bg | Green | `#27ae60` |
| Secondary text | Grey | `#7f8c8d` |

### Typography
- Title: Bold sans-serif (e.g., Montserrat, Lato, or Helvetica Neue)
- Body: Clean sans-serif (Open Sans, Roboto)
- Numbers/tables: Monospace (Courier New, Fira Code) for alignment
- Minimum readable size: 18pt body, 14pt footnotes

### Figures
- Use `output/figures/fig_power_curve.png` as-is
- Generate main VaR plot using `visualization.py` (or manually in matplotlib) with the specs above
- Ensure all figures are 300 DPI minimum for print

---

## 5. Checklist Before Printing

- [ ] Main plot generated and placed in centre column
- [ ] Multi-asset traffic-light table formatted with correct colours
- [ ] Power curve thumbnail resized and placed
- [ ] All numbers match paper tables exactly
- [ ] QR code linked to GitHub repository
- [ ] Text fits within column widths (no overflow)
- [ ] Font sizes sufficient for reading from 1–2 metres
- [ ] Colour palette checked for print contrast
- [ ] PDF exported at 300 DPI minimum
- [ ] Title readable from 3 metres away
