# Event Panel V2 Segment Benchmark V1

## Scope

- Modeling unit: coarse financial segments built from the enriched profile scores.
- Label framing: `5-trading-day excess return sign`.
- Holdout boundary: `2024-01-01`.
- Segment-specific models were only run when the segment cleared minimum rows, ticker breadth, date breadth, and holdout-row safeguards.

## Per-Segment Model Results

| Segment | Rows | Tickers | Dates | Holdout Rows | Model | Mean CV AUC | Holdout AUC | Mean CV Log Loss | Holdout Log Loss | Selected |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| financially_weak | 391 | 32 | 353 | 65 | logistic_regression | 0.5188 | 0.5429 | 0.8240 | 0.7609 |  |
| financially_weak | 391 | 32 | 353 | 65 | random_forest | 0.5272 | 0.5943 | 0.6907 | 0.6819 |  |
| financially_weak | 391 | 32 | 353 | 65 | xgboost | 0.5465 | 0.5571 | 0.7927 | 0.7868 | yes |
| growth_fragile | 177 | 28 | 166 | 24 | logistic_regression | 0.3131 | 0.5903 | 1.6312 | 0.9795 |  |
| growth_fragile | 177 | 28 | 166 | 24 | random_forest | 0.3504 | 0.5139 | 0.7174 | 0.6936 |  |
| growth_fragile | 177 | 28 | 166 | 24 | xgboost | 0.4316 | 0.5347 | 0.9171 | 0.7610 | yes |
| mixed_other | 499 | 34 | 403 | 44 | logistic_regression | 0.5237 | 0.3646 | 0.8150 | 0.9400 |  |
| mixed_other | 499 | 34 | 403 | 44 | random_forest | 0.5367 | 0.3937 | 0.6953 | 0.7370 |  |
| mixed_other | 499 | 34 | 403 | 44 | xgboost | 0.5632 | 0.4417 | 0.8219 | 0.9228 | yes |

## Skipped Segments

| Segment | Rows | Tickers | Dates | Reasons |
|---|---:|---:|---:|---|
| financially_strong | 42 | 9 | 42 | rows<120, dates<80, holdout_rows<20 |

## Interpretation

- The goal here is not to force every profile into its own model. The goal is to test whether coarse, economically distinct groups are stable enough to justify separate fits.
- Any segment skipped here should be treated as analysis-only until more rows or broader universe coverage are available.
