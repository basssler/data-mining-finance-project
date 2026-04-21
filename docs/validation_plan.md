# WRDS Stock Prediction Validation Plan

## Objective
Evaluate short-horizon stock-direction models without lookahead leakage.

## Current Policy
- Train/validation period: 2015-01-01 through 2023-12-31
- Final holdout period: 2024-01-01 through 2024-12-31
- Fold style: expanding walk-forward
- Embargo gap: 5 calendar days between train end and validation start
- Supported targets:
  - `label_up_5d`
  - `label_up_21d`

## Rules
- No random shuffling.
- No future rows inside feature engineering windows.
- No cross-date peer statistics.
- Final 2024 holdout remains untouched until model selection is complete.

## Current Implementation
`src/modeling/splits.py` materializes fold metadata to `reports/results/stock_prediction_splits.json`.
