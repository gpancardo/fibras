# Submission Strategy — FIBRA VaR Paper

## Target 1: Journal of Risk (Risk.net)
**Status: Primary target. Submit first.**

| Field | Detail |
|-------|--------|
| Publisher | Infopro Digital (Risk Journals) |
| WoS Indexing | **SSCI** (Social Sciences Citation Index) |
| Scopus CiteScore | 0.9 |
| JCR Impact Factor | 0.5 |
| APC | **None** (subscription model) |
| Est. acceptance | ~50-60% |
| Est. review time | 3-6 months |
| Submission portal | https://editorialexpress.com/risk |
| Format | Blind PDF (no author info) + separate title page PDF |

**Strengths of fit:**
- VaR backtesting is core to the journal's scope
- Journal explicitly lists "ML in risk management" as a topic
- Methods are correct, paper is transparent about limitations
- Power analysis + multi-asset validation strengthen the applied contribution
- Already formatted to guidelines (legends, abstract length, key messages, blind-ready)

**Weaknesses to address before submission:**
- Rewrite Literature Review and Key Messages in your own words (add academic voice)
- Add remaining DOIs to references.bib
- Verify word count after your rewrites, update in title.tex

---

## Target 2: Journal of Risk Model Validation (Risk.net)
**Status: Fallback if Journal of Risk rejects (not desk-reject). Same submission system.**

| Field | Detail |
|-------|--------|
| Publisher | Infopro Digital (Risk Journals) |
| WoS Indexing | Scopus |
| APC | **None** (subscription model) |
| Est. acceptance | ~60-70% |
| Est. review time | 3-4 months |
| Submission portal | Same as JOR (select from dropdown) |

**Why switch:**
- Narrower scope (model validation) matches this paper's core message well
- Likely higher acceptance rate than JOR
- Same publisher, same formatting guidelines
- Paper is literally about *when backtest validation fails*

**What changes needed:**
- Minimal — frame Introduction around "validation reliability" rather than "risk management"
- Everything else stays the same

**When to switch:**
- If JOR desk-rejects (fast answer, usually 1-2 weeks)
- OR after peer review rejection with manageable reviewer comments

---

## Target 3: REMEF (Revista Mexicana de Economía y Finanzas)
**Status: Final fallback. Only if both Risk Journals reject.**

| Field | Detail |
|-------|--------|
| Publisher | IMEF (Instituto Mexicano de Ejecutivos de Finanzas) |
| WoS Indexing | **ESCI** (Emerging Sources Citation Index) |
| Scopus CiteScore | 0.8 |
| APC | **None** (SciELO OA, funded by CONACYT) |
| Est. acceptance | ~30-40% |
| Est. review time | 3-6 months |
| Language | Spanish or English |
| Submission portal | https://www.remef.org.mx |

**Why it's a fallback:**
- Q4 in SJR (lower prestige than JOR Q3)
- Regional focus (Mexico/Latin America) — narrower audience
- Less international visibility despite Scopus/WoS indexing

**What changes needed:**
- Add a Spanish abstract (required for Mexican audience)
- Shift emphasis toward Mexican FIBRA market (local relevance angle)
- Can keep English main text or translate to Spanish
- REMEF's formatting is more flexible — convert from LaTeX to their OJS template

**When to switch:**
- Only after JOR and JRMV reject
- Or if reviewer feedback at Risk Journals is overwhelmingly negative

---

## Decision Flowchart

```
Submit to Journal of Risk
         |
    ├── Desk reject (1-2 wk) → Submit to J. Risk Model Validation
    │                              |
    │                         ├── Accept → done
    │                         └── Reject → REMEF
    │
    ├── Peer review → revise → accept → done
    │
    └── Peer review → reject
              |
         Address reviewer comments
              |
         Submit to J. Risk Model Validation
              |
         ├── Accept → done
         └── Reject → REMEF
```

---

## Notes

- **No APC at any target** — all three are subscription-based or publicly funded OA
- **Same paper, different jackets** — no major content rewrite needed between JOR and JRMV; REMEF needs Spanish abstract and regional reframing
- **Timeline estimate:** 6-12 months for a final acceptance across all three attempts
- **For CV purposes:** Journal of Risk > Journal of Risk Model Validation > REMEF (this is the ranking for MFE admissions and DS/ML roles)
