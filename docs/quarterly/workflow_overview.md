# Quarterly AUC Improvement Workflow

## Goal
Use this workflow as the canonical repo path for the live quarterly event system.

The repo default is no longer the daily `5`-day lane. The quarterly system now defines the live modeling unit, label, validation policy, benchmark ladder, and promotion review process.

## Phase Order
1. lock repo state
2. rebuild the event-level panel
3. fix tradability timing
4. redesign the target
5. upgrade validation
6. expand accounting coverage
7. engineer stronger quarterly features
8. add event-aware market features
9. add event-specific sentiment features
10. tune stronger models
11. promote only stable winners

## Repo Implementation
- Canonical benchmark ladder manifest: `configs/quarterly/experiments/benchmark_ladder.yaml`
- Canonical benchmark log: `docs/quarterly/benchmark_log.md`
- Canonical experiment log: `docs/quarterly/experiment_log.md`
- Canonical promoted-model registry: `docs/quarterly/promoted_models.md`
- Canonical quarterly panel outputs: `outputs/quarterly/panels/quarterly_event_master.parquet`, `outputs/quarterly/panels/quarterly_event_panel_base.parquet`, `outputs/quarterly/panels/quarterly_event_panel_features.parquet`
- Generated ladder diagnostics: `outputs/quarterly/diagnostics/benchmark_ladder.csv`

## Operating Rules
- Change one major layer at a time.
- Keep the daily `5`-day benchmark as a historical comparator, not the repo default.
- Keep the holdout fixed while quarterly experiments move.
- Save config, metrics, predictions, and feature inventory for every promoted quarterly run.
- Do not promote a quarterly model on holdout AUC alone.

## Current Live Definition

- Unit of observation: one quarterly filing event
- Live label: `event_v2_63d_sign`
- Live validation: `5`-fold purged expanding-window CV with `2024` holdout
- Live baseline config: `configs/event_panel_v2_quarterly.yaml`
- Active candidate config: `configs/event_panel_v2_quarterly_stability_core_additive.yaml`
- Promotion registry: `docs/quarterly/promoted_models.md`

## Current Ladder Mapping
- Steps 2, 5, 6, 7, and 8 already map to checked-in quarterly configs and benchmark artifacts.
- Steps 3, 4, and 9 are intentionally left open until the repo has dedicated label, validation, and champion-promotion artifacts.

## Command
Refresh the quarterly workflow scaffold and ladder diagnostics with:

```powershell
.venv\Scripts\python.exe -m src.quarterly_workflow --write-artifacts
```
