# Data Mining Finance Project

This repo is an event-driven finance modeling project. The current default workflow is the quarterly filing-event lane: one modeling row represents one quarterly filing event, features describe information available before the event becomes tradable, and the model predicts the sign of future excess return against a sector equal-weight benchmark.

The historical daily workflow is still in the repo for comparison, but new testing should start with the quarterly workflow.

## What The Project Does

The project builds machine-learning panels from public-company financial data, market data, SEC filings, earnings metadata, and sentiment features. It then trains and compares classification models such as logistic regression, random forest, XGBoost, LightGBM, and CatBoost.

The live quarterly validation policy is a purged expanding-window cross validation with a fixed 2024 holdout period. This matters because event data is time ordered, and ordinary random train/test splits would leak future information into the training set.

## Repo Layout

- `src/`: Python source for data cleaning, panel building, feature engineering, training, evaluation, and reports.
- `configs/`: executable YAML configs for model runs.
- `configs/quarterly/`: quarterly workflow manifests, experiment configs, and benchmark indexes.
- `configs/daily/`: frozen legacy daily configs.
- `data/raw/`: raw source inputs such as fundamentals, SEC filing text, Alpha Vantage files, and universe data.
- `data/interim/`: cleaned and joined intermediate panels used by the training scripts.
- `data/processed/`: processed modeling panels from earlier workflows.
- `outputs/quarterly/`: generated quarterly panels, labels, diagnostics, validation outputs, and model artifacts.
- `reports/results/`: benchmark CSV and Markdown outputs.
- `docs/`: workflow notes, data dictionaries, validation notes, and experiment logs.
- `tests/`: pytest coverage for panel building, labels, features, and training helpers.

## Conceptual Layers

The code is organized around feature layers. A model config chooses which panel and feature set to use.

- Raw data layer: source files under `data/raw/`, including fundamentals, prices, SEC filing text, Capital IQ universe data, and Alpha Vantage earnings data.
- Layer 1, accounting fundamentals: balance sheet, income statement, cash-flow, profitability, liquidity, solvency, growth, and financial-health features.
- Layer 2, market context: pre-event returns, volatility, volume, beta, shocks, drawdowns, and sector-relative market features.
- Layer 3, text and event signals: SEC filing sentiment, grouped filing events, analyst signals, and external/news sentiment features.
- Quarterly event layer: converts filings into tradable events, assigns event timing, builds event identifiers, and keeps one row per filing event.
- Quarterly feature-design layer: adds deltas, cross-sectional context, Alpha Vantage surprise/revision features, event-aware market features, and event-specific sentiment features.
- Label layer: creates future excess-return labels such as 10-day, 21-day, and 63-day sign, thresholded, or quantile targets.
- Modeling layer: trains configured model families, applies purged time-series validation, writes benchmarks, SHAP summaries, concentration diagnostics, and validation artifacts.

## Requirements

Use Python 3.10 or newer. The project has been developed on Windows PowerShell, but the Python commands are standard.

Install dependencies from:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

## Data Needed

Most tester-friendly runs expect the checked-in or locally generated parquet artifacts to already exist. The most important files are:

- `data/interim/event_panel_v2_quarterly_feature_design.parquet`
- `data/interim/prices/prices_with_labels.parquet`
- `outputs/quarterly/labels/label_map_excess_21d.parquet`
- `outputs/quarterly/labels/label_map_excess_63d.parquet`
- `outputs/quarterly/panels/quarterly_event_panel_features.parquet`

If those files are present, you can train immediately. If they are missing, use the rebuild steps below.

## Quick Start For Testers

From the repo root:

```powershell
cd C:\Users\maxba\Documents\GitHub\data-mining-finance-project
.\.venv\Scripts\Activate.ps1
```

Run the test suite:

```powershell
python -m pytest
```

Run the main quarterly benchmark:

```powershell
python src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly.yaml
```

Run the active quarterly stability candidate:

```powershell
python src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly_stability_core_additive.yaml
```

After training, inspect:

- `reports/results/event_panel_v2_quarterly_benchmark.md`
- `reports/results/event_panel_v2_quarterly_benchmark.csv`
- `reports/results/event_panel_v2_quarterly_stability_core_additive_benchmark.md`
- `reports/results/event_panel_v2_quarterly_stability_core_additive_benchmark.csv`
- `outputs/quarterly/validation/`

## Rebuild Quarterly Artifacts

Use this path when you want to regenerate the quarterly panels and diagnostics before training.

1. Build the quarterly event panel artifacts:

```powershell
python src\build_quarterly_event_panel.py
```

This writes the quarterly master/base/feature panels under `outputs/quarterly/panels/` and diagnostics under `outputs/quarterly/diagnostics/`.

2. Build the quarterly feature-design panel:

```powershell
python src\build_event_panel_v2_quarterly_feature_design.py
```

This reads `data/interim/event_panel_v2_phase6b_alpha_vantage.parquet`, joins market/fundamental/event metadata, and writes `data/interim/event_panel_v2_quarterly_feature_design.parquet`.

3. Refresh quarterly workflow docs and ladder diagnostics:

```powershell
python -m src.quarterly_workflow --write-artifacts
```

This updates quarterly workflow artifacts such as `outputs/quarterly/diagnostics/benchmark_ladder.csv`.

4. Train a benchmark:

```powershell
python src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly.yaml
```

## Optional Model Runs

Run the tuned pass-2 quarterly experiment:

```powershell
python src\train_event_panel_v2.py --config configs\quarterly\quarterly_tuned_model_upgrade_pass2_v1.yaml
```

Run the legacy daily comparator only when you need historical comparison:

```powershell
python src\train_event_panel_v2.py --config configs\event_panel_v2_primary.yaml
```

## How To Read Results

The benchmark Markdown files summarize the selected model, cross-validation metrics, holdout metrics, and interpretation notes. The CSV files are easier to compare across models and configs.

Key metrics:

- `auc_roc`: ranking quality for the binary target. Higher is better.
- `log_loss`: probability calibration and confidence penalty. Lower is better.
- `holdout`: performance on the fixed 2024 evaluation period.
- `cv`: purged expanding-window cross-validation performance before the holdout.

Do not promote a model from holdout AUC alone. The quarterly workflow also checks stability, feature concentration, and whether gains survive across folds.

## Common Troubleshooting

- Missing parquet file: run the rebuild steps or confirm the required data artifact exists in `data/interim/` or `outputs/quarterly/`.
- Import error for `xgboost`, `lightgbm`, `catboost`, `shap`, or `transformers`: rerun `python -m pip install -r requirements.txt` inside the virtual environment.
- CUDA or GPU warnings from XGBoost: configs are set to prefer GPU only when the local stack supports it cleanly and otherwise fall back to CPU.
- Long runtime: start with `python -m pytest` or `configs/event_panel_v2_quarterly.yaml` before running tuned configs with many trials.
- PowerShell activation error: use the `Set-ExecutionPolicy` command shown in the requirements section.

## Useful Docs

- `docs/quarterly/workflow_overview.md`: quarterly workflow phase order and operating rules.
- `docs/quarterly/benchmark_log.md`: benchmark history.
- `docs/quarterly/experiment_log.md`: experiment notes.
- `docs/quarterly/promoted_models.md`: promotion registry.
- `docs/benchmark_regeneration_runbook.md`: regeneration order and artifact expectations.
- `docs/data_dictionary.md`: data and feature definitions.
- `docs/validation_plan.md`: validation design notes.
