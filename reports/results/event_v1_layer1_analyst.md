# event_v1_layer1_analyst

- Best model: `random_forest`
- Panel path: `C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\modeling\event_v1\event_v1_layer1_analyst_panel.parquet`
- Candidate feature count: `33`
- CV AUC: `0.5046`
- CV log loss: `0.6952`
- 2024 holdout AUC: `0.5191`
- 2024 holdout log loss: `0.6934`

## Comparison vs event_v1_layer1

| Metric | event_v1_layer1 | event_v1_layer1_analyst |
|---|---:|---:|
| Mean CV AUC | 0.5027 | 0.5046 |
| Mean CV log loss | 0.6959 | 0.6952 |
| 2024 holdout AUC | 0.5205 | 0.5191 |
| 2024 holdout log loss | 0.6935 | 0.6934 |

## Win Criteria

| Criterion | Result |
|---|---|
| CV AUC > 0.5027 | PASS |
| CV log loss < 0.6959 | PASS |
| 2024 holdout AUC >= 0.5205 | FAIL |
| 2024 holdout log loss <= 0.6935 | PASS |

- Overall result: FAIL
