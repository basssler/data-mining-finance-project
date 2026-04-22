# Event Panel V2 Finance Research Repo

This repository contains the current event-driven benchmark workflow for short-horizon stock-direction experiments on a 34-name Consumer Staples universe, plus derivative tests for universe expansion, SEC sentiment reproducibility, and Alpha Vantage earnings features.

The canonical benchmark in the checked-in artifacts is the enriched `event_panel_v2` result in `reports/results/event_panel_v2_primary_benchmark.csv`. On the current repo state, that anchor selects `xgboost` with CV AUC `0.5415` and 2024 holdout AUC `0.5388`.

## What Is In Scope

- Event-based panel construction with point-in-time joins
- Locked 5-trading-day excess-return-sign label
- Three-model benchmark matrix: `logistic_regression`, `random_forest`, `xgboost`
- Derivative comparisons:
  - Phase 5 universe expansion
  - Phase 6 SEC sentiment reproducibility
  - Phase 6B Alpha Vantage earnings additive test

## Repo Layout

- `src/`: panel builders, training scripts, benchmark reporters, and analysis utilities
- `configs/`: YAML configs for the locked benchmark variants
- `docs/`: workflow notes, decision memos, and experiment documentation
- `reports/results/`: benchmark CSV and markdown outputs
- `reports/`: class-facing summaries
- `data/raw/`, `data/interim/`, `data/processed/`: source, intermediate, and modeled artifacts
- `artifacts/`: feature-analysis and benchmark-adjacent outputs

## Environment

Use the repo-local virtual environment, not the system Python.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

All script examples below assume the working directory is the repo root and `.venv` is active.

## Common Runs

Regenerate the canonical enriched benchmark:

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_primary.yaml
```

Regenerate derivative benchmark CSVs:

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_universe_v2.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_sec_sentiment_v1.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_phase6b_alpha_vantage.yaml
```

Regenerate report narratives from the checked-in benchmark artifacts:

```powershell
.venv\Scripts\python.exe src\report_event_panel_v2_universe_v2.py
.venv\Scripts\python.exe src\report_event_panel_v2_sec_sentiment_v1.py
.venv\Scripts\python.exe src\report_event_panel_v2_phase6b_alpha_vantage.py
```

## Current Interpretation

- `event_panel_v2` is the main anchor.
- `event_panel_v2_universe_v2` is a scale test and is not promoted on the current results.
- `event_panel_v2_sec_sentiment_v1` currently reproduces the canonical benchmark rather than replacing it.
- `event_panel_v2_phase6b_alpha_vantage` is a completed additive test with mixed results: higher selected-model holdout AUC than the canonical anchor, but lower selected-model CV AUC, so it is not promoted as the default.

For the artifact regeneration order, see `docs/benchmark_regeneration_runbook.md`.
