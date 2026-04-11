# Phase 6 SEC Sentiment Test V1

## Dataset Choice

- Used artifact: `C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\interim\features\layer3_sec_sentiment_features.parquet`
- Why it was chosen: it is the existing filing-level SEC sentiment artifact already used by the event-panel builder, is available locally, and is joined at the filing-event level by accession number.
- Timing safety: the event panel attaches same-filing sentiment only after the filing becomes available via `effective_model_date`; no daily forward-filled sentiment layer is used in v2.

## Structural Finding

- The locked `event_panel_v2` Phase 4 anchor already contains the SEC filing sentiment feature columns at full coverage.
- As a result, this Phase 6 panel is not a new additive merge in practice; it is an explicit rerun and freeze-point for the already-embedded SEC sentiment path.

## Comparison Against Phase 4 Anchor

- Row count: Phase 4 `1,109` vs Phase 6 `1,109`
- Feature count: Phase 4 `72` vs Phase 6 `72`
- Selected primary model: Phase 4 `random_forest` vs Phase 6 `random_forest`
- Best CV AUC: Phase 4 `0.5260` vs Phase 6 `0.5260`
- Best holdout AUC: Phase 4 `0.5237` vs Phase 6 `0.5237`

## Decision

- Final decision: **FREEZE**
- Rationale: SEC filing sentiment is not newly improving the locked benchmark here because it was already present in the baseline panel. Keep the code and artifacts for reference, but do not treat this phase as proof of a new additive lift.
- Promotion status: do not promote as a separate new dataset layer from this Phase 6 run.
- Rejection status: do not reject the sentiment path outright either, because the path is already part of the event-panel baseline and remains timing-safe.

