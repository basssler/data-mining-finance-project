# Final Project Results One Page

## Objective

Test whether fundamentals, market context, and Capital IQ Key Developments sentiment improve quarterly filing-event classification for **Consumer Staples v1, 34 tickers**, using the locked **63-trading-day sector-relative sign label**.

## Final Setup

| item | final setting |
|---|---|
| Universe | Consumer Staples v1, 34 tickers |
| Data layers | fundamentals, market, Capital IQ Key Developments sentiment |
| Active label | 63-trading-day sector-relative sign label |
| Holdout | 2024 |
| Validation | 5-fold purged expanding window with embargo |
| Sentiment method | zero-shot FinBERT + Capital IQ Key Developments feature engineering |
| Sentiment framing | within-sector relative sentiment |

## Integrity Readiness

| audit status | count |
|---|---:|
| PASS | 98 |
| REVIEW | 2 |
| INFO | 2 |
| LIMITED | 1 |
| FAIL | 0 |

The fundamentals rebuild sharply reduced unit-related ratio and amount flags. For Consumer Staples v1, feature ratio sanity flags fell from 2,966 to 63, panel ratio sanity flags fell from 2,402 to 53, and amount relationship flags fell from 806 to 0.

## Sentiment Coverage

| window | overall | 2024 holdout |
|---|---:|---:|
| 7d | 60.11% | 55.47% |
| 30d | 88.18% | 86.13% |
| 63d | 93.77% | 92.70% |

Coverage is good enough for a 30d/63d feature-layer test, but still uneven by ticker-year.

## Final Model Ladder

| rung | feature set | selected model | holdout AUC |
|---|---|---|---:|
| A | Quarterly core | xgboost | 0.5501 |
| B | Quarterly core + market | xgboost | 0.5020 |
| C | Core + market + Capital IQ raw sentiment | logistic regression | 0.5958 |
| D | Core + market + Capital IQ within-sector adjusted sentiment | logistic regression | 0.6038 |

## Verdict

Capital IQ Key Developments sentiment is a **promising but fragile feature layer**. The within-sector adjusted sentiment selected model improved 2024 holdout AUC to **0.6038** versus **0.5020** for the core + market selected model. However, the improvement was not stable across pseudo-holdout years, and the 2024 bootstrap AUC-delta interval was **-0.0930 to 0.1307**.

Final interpretation: holdout improvement was meaningful, but not stable across pseudo-holdout years.

## Why No Tuning

Tuning was not run because the locked untuned ladder already showed fragility. Additional tuning after observing the 2024 result would weaken the final evidence by increasing holdout-selection risk.

## Bottom Line

Ready for final report and presentation packaging, with a conservative conclusion: promising Consumer Staples v1 evidence, not broad sector generalization.
