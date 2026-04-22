# Stock Direction Prediction: Executive Summary

## Main Result

The current repo-wide anchor is the enriched `event_panel_v2` benchmark.

- Selected model: `xgboost`
- CV AUC: `0.5415`
- 2024 holdout AUC: `0.5388`

This is the canonical benchmark to cite unless a regenerated benchmark CSV clearly beats it.

## Phase Outcomes

| Phase | What | Current Result |
|---|---|---|
| **4** | Enriched event-driven benchmark | **Main anchor**: XGBoost, CV AUC `0.5415`, holdout AUC `0.5388` |
| **5** | Universe expansion to `event_panel_v2_universe_v2` | Not promoted: selected-model CV AUC `0.5021`, holdout AUC `0.5282` |
| **6** | SEC sentiment reproducibility | Exact match to canonical anchor on current artifacts |
| **6B** | Alpha Vantage earnings additive test | Complete but mixed: selected-model CV AUC `0.5262`, holdout AUC `0.5580`; stays `FREEZE` |

## What To Say Clearly

- The benchmark story is modest but real: the enriched event-panel setup remains above random, not transformational.
- The canonical benchmark is the enriched `event_panel_v2` result, not the broader-universe or additive variants.
- Phase 6B is no longer an incomplete run. The manifest is complete, but the additive evidence is mixed, so it is not promoted.

## Bottom Line

The repo now supports one clean headline:

`event_panel_v2` with enriched features is the main benchmark, while the derivative tests help bound what did and did not improve it.
