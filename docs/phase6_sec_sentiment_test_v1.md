# Phase 6 SEC Sentiment Test V1

## Dataset Choice

- Used artifact: `C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\interim\features\layer3_sec_sentiment_features.parquet`
- Why it was chosen: it is the existing filing-level SEC sentiment artifact already used by the event-panel builder, is available locally, and is joined at the filing-event level by accession number.
- Timing safety: the event panel attaches same-filing sentiment only after the filing becomes available via `effective_model_date`; no daily forward-filled sentiment layer is used in v2.

## Structural Finding

- Benchmark parity with the canonical enriched `event_panel_v2` artifact: `exact_match`.
- Stored panel column comparison: Phase 4 `78` vs Phase 6 `72`. Sentiment-column count in Phase 6 panel: `13`.
- This phase should be read as a documented SEC sentiment path check against the canonical enriched benchmark, not as a separate promoted anchor by default.

## Comparison Against Phase 4 Anchor

- Row count: Phase 4 `1,109` vs Phase 6 `1,109`
- Stored panel columns: Phase 4 `78` vs Phase 6 `72`
- Selected primary model: Phase 4 `xgboost` vs Phase 6 `xgboost`
- Best CV AUC: Phase 4 `0.5415` vs Phase 6 `0.5415`
- Best holdout AUC: Phase 4 `0.5388` vs Phase 6 `0.5388`

## Decision

- Final decision: **FREEZE**
- Rationale: the Phase 6 benchmark currently matches the canonical enriched benchmark exactly, so it serves as a reproducibility confirmation rather than evidence for a new promoted layer.
- Promotion status: keep the canonical enriched `event_panel_v2` benchmark as the main anchor.
- Rejection status: keep the SEC sentiment path documented and timing-safe, but do not present this report as a separate benchmark win.

