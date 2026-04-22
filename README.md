# Event Panel V2 Finance Research Repo

This repository contains the current event-driven benchmark workflow for short-horizon stock-direction experiments on a 34-name Consumer Staples universe, plus derivative tests for universe expansion, SEC sentiment reproducibility, Alpha Vantage earnings features, and a parallel quarterly 63-trading-day excess-return-sign lane built on the same `event_panel_v2` base.

The canonical benchmark in the checked-in artifacts is the enriched `event_panel_v2` result in `reports/results/event_panel_v2_primary_benchmark.csv`. On the current repo state, that anchor selects `xgboost` with CV AUC `0.5415` and 2024 holdout AUC `0.5388`.

## What Is In Scope

- Event-based panel construction with point-in-time joins
- Locked 5-trading-day excess-return-sign label
- Parallel quarterly 63-trading-day excess-return-sign benchmark lane
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

Run the parallel quarterly 63-trading-day excess-return-sign lane on the same `event_panel_v2` base:

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly.yaml
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

- `event_panel_v2` with the locked 5-trading-day excess-return-sign setup remains the main anchor in the historical benchmark and decision docs.
- `event_panel_v2_quarterly` is an additive parallel 63-trading-day benchmark lane with separate outputs built from the same base panel.
- `event_panel_v2_universe_v2` is a scale test and is not promoted on the current results.
- `event_panel_v2_sec_sentiment_v1` currently reproduces the canonical benchmark rather than replacing it.
- `event_panel_v2_phase6b_alpha_vantage` is a completed additive test with mixed results: higher selected-model holdout AUC than the canonical anchor, but lower selected-model CV AUC, so it is not promoted as the default.

### 5-Day Anchor vs Quarterly Lane

Treat this as a parallel experiment, not a benchmark replacement. The existing 5-day anchor uses the `5-trading-day excess return sign` label on `1,109` active rows with `137` 2024 holdout rows; it selects `xgboost` with CV AUC `0.5415` and 2024 holdout AUC `0.5388`. The new quarterly lane uses the `63-trading-day excess return sign` label on `1,073` active rows with `101` 2024 holdout rows; it also selects `xgboost`, but at CV AUC `0.5270` and 2024 holdout AUC `0.4411`.

The horizon change fixes part of the original framing problem, but an obvious feature/target mismatch still remains. In the quarterly lane, the feature set is still heavily populated by short- and medium-horizon market state inputs such as `volume_ratio_20d`, `vol_ratio_21d_63d`, `realized_vol_63d`, `overnight_gap_1d`, `drawdown_21d`, and `rel_return_5d`, alongside filing-timed sentiment features. That leaves the quarterly target at risk of being driven more by near-term trading regime proxies than by genuinely quarterly information arrival, so the lane should be read as a useful side-by-side test rather than the new default benchmark.

For the artifact regeneration order, see `docs/benchmark_regeneration_runbook.md`.
