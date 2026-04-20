# Financially Weak Follow-Up V1

## Scope

- This is the direct apples-to-apples follow-up requested after the profile analysis.
- Both models are evaluated on the same `financially_weak` 2024 holdout subset.
- Comparison target: does a dedicated segment model outperform the enriched full-panel model on that subset?

## Holdout Subset

- Rows: `65`
- Tickers: `28`

## Model Comparison

| Training Scope | Model | Backend | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC |
|---|---|---|---:|---:|---:|---:|---:|
| full_panel | xgboost | cpu | 0.5324 | 0.7425 | 0.5000 | 0.5333 | 0.0035 |
| financially_weak_only | xgboost | cpu | 0.5590 | 0.7966 | 0.5429 | 0.6333 | 0.0372 |

## Interpretation

- On the financially weak subset, the dedicated segment model beat the enriched full-panel model on holdout AUC.
- This follow-up is narrower than the benchmark report: it tests segment specialization only on the subset where segmentation looked most promising.
