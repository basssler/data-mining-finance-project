# Quarterly Event Panel Research Repo

This repository now treats the quarterly event-driven workflow as the canonical project path.

The live modeling unit is `one row = one quarterly filing event`. The live label is `event_v2_63d_sign`, which classifies the `63`-trading-day excess return sign relative to a sector equal-weight benchmark excluding the focal stock. The live validation policy is a `5`-fold purged expanding-window CV with a fixed `2024-01-01` holdout boundary.

## Canonical Workflow

- Primary quarterly baseline config: `configs/event_panel_v2_quarterly.yaml`
- Active quarterly candidate under review: `configs/event_panel_v2_quarterly_stability_core_additive.yaml`
- Quarterly workflow manifest: `configs/quarterly/experiments/benchmark_ladder.yaml`
- Quarterly benchmark log: `docs/quarterly/benchmark_log.md`
- Quarterly experiment log: `docs/quarterly/experiment_log.md`
- Quarterly benchmark registry: `docs/quarterly/promoted_models.md`

No quarterly champion is promoted yet. The current candidate under review is `event_panel_v2_quarterly_stability_core_additive`, which selects `xgboost` at mean CV AUC `0.5262` and 2024 holdout AUC `0.4880`. It is the active promotion-track run because it sits on the stability ladder, but it has not cleared promotion criteria.

## Legacy Daily Baseline

The older daily `5`-trading-day workflow is preserved as a frozen comparator, not the live repo default.

- Legacy config namespace: `configs/daily/`
- Legacy docs namespace: `docs/daily/`
- Frozen baseline config copy: `configs/daily/legacy_event_panel_v2_primary.yaml`
- Historical benchmark artifact: `reports/results/event_panel_v2_primary_benchmark.csv`

The root-level `configs/event_panel_v2_primary.yaml` remains in place for compatibility with existing scripts and reports, but it should be treated as a legacy daily artifact.

## Repo Layout

- `src/`: training, panel construction, reporting, and research utilities
- `configs/daily/`: legacy daily baseline registry
- `configs/quarterly/`: canonical quarterly manifests and indexes
- `docs/daily/`: frozen daily workflow notes
- `docs/quarterly/`: live quarterly workflow docs
- `reports/results/`: checked-in benchmark outputs
- `outputs/quarterly/`: generated quarterly diagnostics and future promotion artifacts

## Common Runs

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Regenerate the live quarterly baseline:

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly.yaml
```

Refresh the active quarterly candidate:

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly_stability_core_additive.yaml
```

Refresh quarterly scaffold docs and diagnostics:

```powershell
.venv\Scripts\python.exe -m src.quarterly_workflow --write-artifacts
```

Run the legacy daily comparator only when a historical comparison is required:

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_primary.yaml
```

## Current Weaknesses

- No quarterly run has cleared promotion yet.
- The quarterly holdout remains weak relative to the historical daily baseline.
- The benchmark ladder still lacks a dedicated improved-label rung and a dedicated promotion memo.

For regeneration order and artifact expectations, use `docs/benchmark_regeneration_runbook.md`.
