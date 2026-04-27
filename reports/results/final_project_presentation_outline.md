# Final Project Presentation Outline

## Slide 1: Project Question

**Message:** Can event-aligned fundamentals, market context, and Capital IQ Key Developments sentiment improve quarterly filing-event classification for Consumer Staples v1?

Include:

- Universe: Consumer Staples v1, 34 tickers.
- Unit of observation: one quarterly filing event.
- Final label: 63-trading-day sector-relative sign label.

## Slide 2: Final Data Stack

**Message:** The final package combines fundamentals, market context, and Capital IQ Key Developments sentiment.

Include:

- Fundamentals: SEC-derived quarterly accounting data.
- Market: event-time market context.
- Sentiment: zero-shot FinBERT + Capital IQ Key Developments feature engineering.
- Framing: within-sector relative sentiment.

## Slide 3: Label and Validation Contract

**Message:** The active benchmark uses the canonical 63-day sector-relative label and a 2024 holdout.

Include:

- Active config: `configs/event_panel_v2_quarterly_63d_sector_relative.yaml`.
- Label artifact: `outputs/quarterly/labels/label_map_excess_63d.parquet`.
- Holdout start: 2024-01-01.
- Validation: 5-fold purged expanding window with embargo.

## Slide 4: Fundamentals Integrity Fix

**Message:** The staged rebuild removed the major unit-contamination blocker before final reporting.

Include:

| Consumer Staples v1 metric | before | after |
|---|---:|---:|
| Feature ratio sanity flags | 2,966 | 63 |
| Panel ratio sanity flags | 2,402 | 53 |
| Amount relationship flags | 806 | 0 |

Speaker note: The rebuild did not run modeling, tuning, SHAP, Optuna, sentiment scoring, or label changes.

## Slide 5: Data Integrity Audit

**Message:** Final data-layer integrity passed with no failures.

Include:

| status | count |
|---|---:|
| PASS | 98 |
| REVIEW | 2 |
| INFO | 2 |
| LIMITED | 1 |
| FAIL | 0 |

Speaker note: Review items are high news-count tickers and low-coverage ticker-years; the limited item is missing validation artifacts for one older experiment family.

## Slide 6: Sentiment Coverage

**Message:** Capital IQ Key Developments provides usable 30d and 63d coverage.

Include:

| window | overall | 2024 holdout |
|---|---:|---:|
| 7d | 60.11% | 55.47% |
| 30d | 88.18% | 86.13% |
| 63d | 93.77% | 92.70% |

Suggested figure: coverage table or concise bar chart.

## Slide 7: Modeling Ladder

**Message:** The within-sector adjusted sentiment rung produced the strongest 2024 holdout AUC.

Include:

| rung | feature set | selected model | holdout AUC |
|---|---|---|---:|
| A | Quarterly core | xgboost | 0.5501 |
| B | Quarterly core + market | xgboost | 0.5020 |
| C | Core + market + Capital IQ raw sentiment | logistic regression | 0.5958 |
| D | Core + market + Capital IQ within-sector adjusted sentiment | logistic regression | 0.6038 |

Suggested figure: `outputs/quarterly/modeling/final/figures/ladder_holdout_auc.png`.

## Slide 8: Stability Check

**Message:** The holdout improvement was meaningful, but not stable across pseudo-holdout years.

Include:

- Core + market selected-model holdout AUC: 0.5020.
- Within-sector adjusted sentiment selected-model holdout AUC: 0.6038.
- Bootstrap 2024 AUC-delta interval: -0.0930 to 0.1307.
- Cross-year pseudo-holdout results were mixed.

Suggested figures:

- `outputs/quarterly/modeling/final/figures/year_holdout_auc_delta.png`
- `outputs/quarterly/modeling/final/figures/bootstrap_auc_delta_interval.png`

## Slide 9: Why Tuning Was Not Run

**Message:** Tuning was intentionally deferred because the locked untuned result was not stable enough.

Include:

- No Optuna or hyperparameter search was run for final packaging.
- The locked ladder already showed 2024 lift but mixed pseudo-holdout years.
- Tuning after observing the holdout would increase holdout-selection risk.

## Slide 10: Final Verdict

**Message:** Capital IQ Key Developments sentiment is a promising but fragile feature layer.

Include:

- Stronger 2024 holdout AUC with within-sector adjusted sentiment.
- Fragile stability across pseudo-holdout years.
- Evidence supports future testing, not broad sector generalization.
- Final interpretation: holdout improvement was meaningful, but not stable across pseudo-holdout years.

## Slide 11: Limitations

**Message:** The final claim is intentionally bounded.

Include:

- Consumer Staples v1 only.
- Capital IQ Key Developments are event-text, not general news.
- Uneven coverage remains for some ticker-years.
- Some high-news-count ticker behavior needs review.
- Bootstrap uncertainty crosses zero.

## Slide 12: Future Work

**Message:** The next step is broader, predeclared validation.

Include:

- Replicate on additional sectors.
- Save full fold maps, purge audits, and prediction artifacts for every experiment.
- Investigate event categories and recency decay.
- Add CV-only tuning only under a predeclared protocol with one final holdout read.
