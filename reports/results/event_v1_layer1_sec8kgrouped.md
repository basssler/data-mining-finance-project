# event_v1_layer1_sec8kgrouped

- Best model: `hist_gradient_boosting`
- Panel path: `C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\modeling\event_v1\event_v1_layer1_sec8kgrouped_panel.parquet`
- Candidate feature count: `38`
- CV AUC: `0.5079`
- CV log loss: `0.7188`
- 2024 holdout AUC: `0.5209`
- 2024 holdout log loss: `0.7044`

## Comparison vs event_v1_layer1

| Metric | event_v1_layer1 | event_v1_layer1_sec8kgrouped |
|---|---:|---:|
| Mean CV AUC | 0.5027 | 0.5079 |
| Mean CV log loss | 0.6959 | 0.7188 |
| 2024 holdout AUC | 0.5205 | 0.5209 |
| 2024 holdout log loss | 0.6935 | 0.7044 |

## Win Criteria

| Criterion | Result |
|---|---|
| CV AUC > 0.5027 | PASS |
| CV log loss < 0.6959 | FAIL |
| 2024 holdout AUC >= 0.5205 | PASS |
| 2024 holdout log loss <= 0.6935 | FAIL |

- Overall result: FAIL

## Comparison vs event_v1_layer1_secfilings

| Metric | event_v1_layer1_secfilings | event_v1_layer1_sec8kgrouped |
|---|---:|---:|
| Mean CV AUC | 0.5019 | 0.5079 |
| Mean CV log loss | 0.6958 | 0.7188 |
| 2024 holdout AUC | 0.5238 | 0.5209 |
| 2024 holdout log loss | 0.6933 | 0.7044 |
