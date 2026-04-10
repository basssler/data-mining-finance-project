# event_v1_layer1_secfilings

- Best model: `random_forest`
- Panel path: `C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\modeling\event_v1\event_v1_layer1_secfilings_panel.parquet`
- Candidate feature count: `36`
- CV AUC: `0.5019`
- CV log loss: `0.6958`
- 2024 holdout AUC: `0.5238`
- 2024 holdout log loss: `0.6933`

## Comparison vs event_v1_layer1

| Metric | event_v1_layer1 | event_v1_layer1_secfilings |
|---|---:|---:|
| Mean CV AUC | 0.5027 | 0.5019 |
| Mean CV log loss | 0.6959 | 0.6958 |
| 2024 holdout AUC | 0.5205 | 0.5238 |
| 2024 holdout log loss | 0.6935 | 0.6933 |

## Win Criteria

| Criterion | Result |
|---|---|
| CV AUC > 0.5027 | FAIL |
| CV log loss < 0.6959 | PASS |
| 2024 holdout AUC >= 0.5205 | PASS |
| 2024 holdout log loss <= 0.6935 | PASS |

- Overall result: FAIL
