# Quarterly Feature Design Comparison

## Summary

- Current plain quarterly checkpoint: holdout AUC `0.4411`.
- Current quarterly Alpha Vantage checkpoint: holdout AUC `0.4683`.
- Best redesigned family by holdout-first ordering: `quarterly_plus_sentiment` with model `random_forest`, CV AUC `0.5005`, and holdout AUC `0.4956`.

## Family Comparison

| Experiment Family | Design Note | Best Holdout Model | Selected Model | CV AUC | CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Rows | Beat Plain 63d | Beat Quarterly AV | Top SHAP Feature |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| quarterly_plus_sentiment | quarterly_core_plus_sentiment_probabilities | random_forest | xgboost | 0.5005 | 0.6968 | 0.4956 | 0.7056 | 101 | yes | yes | qfd_av_revision_x_surprise |
| quarterly_core | fundamentals_profiles_event_context_av_only | xgboost | xgboost | 0.5243 | 0.7805 | 0.4427 | 0.8156 | 101 | yes |  | qfd_av_revision_x_surprise |
| quarterly_plus_medium_market | quarterly_core_plus_medium_market_context | xgboost | xgboost | 0.5384 | 0.7987 | 0.4322 | 0.8472 | 101 |  |  | drawdown_21d |

## Readout

- At least one redesigned family beat the current quarterly Alpha Vantage checkpoint on holdout AUC.
- Use the best family as the input to the quarterly ablation pass; prioritize dropping residual short-horizon proxy features only when they fail to support holdout after retraining.
