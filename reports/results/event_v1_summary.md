# Event V1 Summary

| Run | Best Model | CV AUC | CV Log Loss | Holdout AUC | Holdout Log Loss |
|---|---|---:|---:|---:|---:|
| event_v1_layer1 | hist_gradient_boosting | 0.5056 | 0.6970 | 0.5180 | 0.6955 |
| event_v1_layer1_layer2 | hist_gradient_boosting | 0.5086 | 0.7196 | 0.5175 | 0.7058 |
| event_v1_full | hist_gradient_boosting | 0.5087 | 0.7257 | 0.5028 | 0.7104 |

## Interpretation

- Layer 2 v2 real win versus event_v1_layer1: no.
- Full panel real win versus event_v1_layer1_layer2: no.
- A run is only treated as promising when CV AUC improves, CV log loss improves, and the 2024 holdout does not reverse the direction of the improvement.
- Threshold metrics such as F1 and recall were not treated as sufficient on their own.
