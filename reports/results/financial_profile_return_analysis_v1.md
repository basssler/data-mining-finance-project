# Financial Profile Return Analysis V1

## Scope

- Panel: `event_panel_v2` with the enriched financial-profile features.
- Label framing: `5-trading-day excess return sign`.
- Holdout boundary: `2024-01-01`.
- Goal: test whether financial profiles differ in average post-event behavior before splitting the model by segment.

## Fine Profile Results

| financial_profile | rows | tickers | mean_excess_return | median_excess_return | hit_rate | holdout_rows | holdout_mean_excess_return | holdout_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixed_profile | 453 | 34 | -0.0008 | -0.0011 | 0.4834 | 42 | -0.0004 | 0.5238 |
| distressed_weak_quality | 354 | 30 | 0.0011 | 0.0001 | 0.5056 | 59 | 0.0007 | 0.4746 |
| high_growth_fragile | 177 | 28 | -0.0016 | -0.0020 | 0.4689 | 24 | -0.0018 | 0.5000 |
| short_term_healthy_levered | 51 | 10 | 0.0012 | 0.0047 | 0.5490 | 0 | n/a | n/a |
| mature_defensive | 14 | 5 | 0.0034 | 0.0045 | 0.7143 | 1 | 0.0268 | 1.0000 |
| stable_compounder | 8 | 1 | 0.0061 | 0.0133 | 0.7500 | 0 | n/a | n/a |

## Coarse Segment Results

| coarse_segment | rows | tickers | mean_excess_return | median_excess_return | hit_rate | holdout_rows | holdout_mean_excess_return | holdout_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixed_other | 499 | 34 | 0.0003 | 0.0005 | 0.5110 | 44 | 0.0016 | 0.5455 |
| financially_weak | 391 | 32 | 0.0005 | -0.0007 | 0.4962 | 65 | 0.0004 | 0.4615 |
| growth_fragile | 177 | 28 | -0.0016 | -0.0020 | 0.4689 | 24 | -0.0018 | 0.5000 |
| financially_strong | 42 | 9 | 0.0007 | 0.0025 | 0.5714 | 4 | 0.0009 | 0.5000 |

## Interpretation

- This report is descriptive only. It does not prove tradable segment alpha, but it does test whether the groups are economically different enough to justify segmented models.
- Coarse segments are the safer modeling unit because several fine profiles are too sparse for stable fold-by-fold training.
