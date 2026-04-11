# Event Panel V2 Universe V2 Benchmark

## Scope

- Locked setup unchanged: `event_panel_v2`, `5-trading-day excess return sign`, logistic regression, random forest, XGBoost.
- This report compares the locked 34-name benchmark against the Phase 5 expanded large-cap cross-sector universe.

## Panel Comparison

| Panel | Rows | Tickers | Event Date Range | Event Counts | Selected Model |
|---|---:|---:|---|---|---|
| 34-name locked panel | 1,109 | 34 | 2015-07-31 to 2024-12-19 | 10-K=278, 10-Q=831 | random_forest |
| universe_v2 expanded panel | 4,908 | 126 | 2014-12-29 to 2024-12-20 | 10-K=1,226, 10-Q=3,682 | logistic_regression |

## Per-Model Comparison

| Model | 34-Name CV AUC | Universe_v2 CV AUC | 34-Name CV Log Loss | Universe_v2 CV Log Loss | 34-Name Holdout AUC | Universe_v2 Holdout AUC | 34-Name Holdout Log Loss | Universe_v2 Holdout Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.5181 | 0.5013 | 0.7510 | 0.7014 | 0.5051 | 0.5068 | 0.7409 | 0.6945 |
| random_forest | 0.5260 | 0.4860 | 0.6989 | 0.7005 | 0.5237 | 0.4859 | 0.6945 | 0.6979 |
| xgboost | 0.5244 | 0.4960 | 0.8119 | 0.7468 | 0.5290 | 0.5091 | 0.7569 | 0.7100 |

## Feature Exclusions

- Explicit exclusions carried forward: `["gross_margin", "current_filing_sentiment_available"]`
- Auto all-missing exclusions on expanded universe: `["sec_sentiment_score", "sec_positive_prob", "sec_negative_prob", "sec_neutral_prob", "sec_sentiment_abs", "sec_sentiment_change_prev", "sec_positive_change_prev", "sec_negative_change_prev", "sec_chunk_count", "sec_log_chunk_count"]`
- Auto constant exclusions on expanded universe: `[]`

## Decision

- Did more names improve stability or signal? `mixed_or_no`. Expanded-universe best CV AUC moved from `0.5260` to `0.5013`, and best holdout AUC moved from `0.5237` to `0.5068`.
- Did the same primary model remain best? `no`. Current locked winner: `random_forest`. Expanded-universe winner: `logistic_regression`.
- Should the expanded universe become the new default research universe? `not_yet`.
- Recommendation: keep the wider universe as a ready scaling path, but do not promote it to the default unless the rerun shows a clean improvement on both CV and 2024 holdout.

## Interpretation

- This Phase 5 report is the scale test only. It does not change the observation unit, label horizon, feature families, or model set.
- If the expanded universe wins cleanly, it becomes the better pre-external-data anchor because it adds cross-sectional breadth without changing the locked method stack.
