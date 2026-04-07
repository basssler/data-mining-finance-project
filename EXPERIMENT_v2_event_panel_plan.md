# EXPERIMENT_v2_event_panel_plan

## Objective

Implement the next experiment iteration for the supervised stock-direction repo **without breaking the locked Layer 1 benchmark path**.

This iteration should:

- preserve the existing Layer 1 benchmark exactly
- keep all new work modular and versioned
- avoid leakage and false progress
- test whether **problem framing + validation upgrades + stronger controls** unlock signal before adding more model complexity
- use the current repo structure rather than refactoring the whole codebase

Official project context already follows a layered design:

- **Layer 1** = financial statement features
- **Layer 2** = market-derived features
- **Layer 3** = sentiment features

The next iteration should extend that design cleanly rather than overwrite it.

---

## Locked Benchmark Boundary

### Do not modify these files

These files are part of the locked benchmark path and should remain untouched:

- `src/universe.py`
- `src/edgar_pull.py`
- `src/fundamentals_clean.py`
- `src/feature_engineering.py`
- `src/prices.py`
- `src/panel_builder.py`
- `src/train_baseline.py`
- `src/market_features.py`
- `src/train_layer_comparison.py`
- `src/sec_filing_text_pull.py`
- `src/sec_sentiment_features.py`
- `src/sec_sentiment_prepare.py`
- `src/panel_builder_layer3.py`
- `src/train_layer3_comparison.py`
- `src/sec_sentiment_features_mda.py`
- `src/sec_sentiment_prepare_mda.py`
- `src/panel_builder_layer3_mda.py`
- `src/train_layer3_mda_comparison.py`

### Do not overwrite these existing benchmark datasets

- `data/interim/features/layer1_financial_features.parquet`
- `data/interim/prices/prices_with_labels.parquet`
- `data/processed/modeling/layer1_modeling_panel.parquet`
- `data/interim/features/layer2_market_features.parquet`
- `data/interim/features/layer3_sec_sentiment_features.parquet`
- `data/processed/modeling/layer1_layer3_modeling_panel.parquet`
- `data/interim/features/layer3_sec_sentiment_mda_features.parquet`
- `data/processed/modeling/layer1_layer3_mda_modeling_panel.parquet`

### Do not overwrite any existing benchmark output files

If the repo currently writes metrics, predictions, logs, or comparisons for the locked baseline path, those outputs must remain unchanged.

---

## Branch and Experiment Naming

### Recommended branch

- `exp/event-panel-v1`

### Experiment family name

Use **event_v1** as the namespace for all new code, data artifacts, logs, and reports.

---

## Core Diagnosis Behind This Iteration

The current score pattern suggests the repo does **not** have a simple estimator selection problem.

Observed pattern:

- AUCs clustered around ~0.50 to ~0.51
- log loss clustered around ~0.693 to ~0.698
- some higher F1 / recall values likely caused by threshold behavior or base-rate effects rather than true discrimination

Working interpretation:

1. **Label/problem framing is probably noisy** for daily rows with 5-day forward sign.
2. **Grain mismatch is likely real** because fundamentals and filing sentiment move more like event data than daily-flow data.
3. **Validation needs to be hardened** because overlapping forward-return labels can create optimistic results if splits are not properly purged.
4. **Layer 2 likely needs redesign as a control layer**, not just more generic technical indicators.
5. **Layer 3 likely needs event-aware features** like deltas, surprise, decay, and interactions rather than static score averages.

This experiment lane should therefore start with:

- validation hardening
- label/path redesign
- stronger market controls
- then event-driven sentiment

Not with new model classes.

---

## High-Level Build Order

Implement the new lane in this exact order:

1. **Validation harness**
2. **New label module**
3. **New panel builder using Layer 1 + new labels only**
4. **Train and sanity check Layer 1 under the new label/validation setup**
5. **Add redesigned Layer 2 controls**
6. **Retrain and compare**
7. **Add event-driven Layer 3 sentiment features**
8. **Retrain and compare again**

This order is mandatory because it isolates whether any improvement comes from:

- the label reframe
- the market control redesign
- the event-driven text layer

---

## New Files to Add

Create the following files. Do **not** replace existing files.

### Validation / evaluation

- `src/validation_event_v1.py`
- `src/evaluate_event_v1.py`

### Labels

- `src/labels_event_v1.py`

### Market features

- `src/market_features_v2.py`

### Sentiment features

- `src/sec_sentiment_event_v1.py`

### Panel building

- `src/panel_builder_event_v1.py`

### Training

- `src/train_event_v1.py`

### Optional helper/config files if needed

Only if helpful and repo-safe:

- `src/config_event_v1.py`
- `src/io_event_v1.py`
- `src/utils_event_v1.py`

Do not create an unnecessary framework. Keep helpers minimal.

---

## New Dataset Names

Use versioned, non-conflicting outputs.

### Labels

- `data/interim/labels/labels_event_v1.parquet`

### Layer 2 features

- `data/interim/features/layer2_market_features_v2.parquet`

### Layer 3 features

- `data/interim/features/layer3_sec_sentiment_event_v1.parquet`

### Panels

- `data/processed/modeling/event_v1_layer1_panel.parquet`
- `data/processed/modeling/event_v1_layer1_layer2_panel.parquet`
- `data/processed/modeling/event_v1_full_panel.parquet`

### Predictions / metrics / reports

- `reports/results/event_v1_layer1_metrics.json`
- `reports/results/event_v1_layer1_layer2_metrics.json`
- `reports/results/event_v1_full_metrics.json`
- `reports/results/event_v1_layer1_predictions.parquet`
- `reports/results/event_v1_layer1_layer2_predictions.parquet`
- `reports/results/event_v1_full_predictions.parquet`
- `reports/results/event_v1_summary.md`

If the repo does not currently have `reports/results/`, create it.

---

## Implementation Contract by File

## 1) `src/validation_event_v1.py`

### Goal

Create a **reusable, immutable evaluation harness** for the new experiment lane.

### Requirements

Implement functions that:

- generate expanding-time splits
- apply a **purge / embargo style separation** between train and validation when labels overlap
- support a final untouched holdout window
- return deterministic split metadata for reproducibility

### Minimum behavior

- handle the repo’s daily-row panel format
- assume the target uses a **5-trading-day forward horizon** unless otherwise passed
- prevent train rows whose label windows overlap with validation rows
- support a configurable embargo buffer
- expose split start/end dates clearly

### Suggested API

```python
make_event_v1_splits(
    df: pd.DataFrame,
    date_col: str,
    horizon_days: int = 5,
    n_splits: int = 5,
    embargo_days: int = 5,
    holdout_start: str | None = None,
) -> dict
```

### Expected output

A structure containing:

- train indices per fold
- validation indices per fold
- final holdout indices
- human-readable fold date ranges

### Acceptance criteria

- no leakage between train and validation under overlapping forward windows
- deterministic splits from the same input
- validation logic can be imported and reused by all training scripts in the new lane

---

## 2) `src/evaluate_event_v1.py`

### Goal

Centralize scoring for the experimental lane.

### Requirements

Implement utilities that compute and save:

- AUC
- F1
- precision
- recall
- log loss
- positive prediction rate
- class balance diagnostics
- optional per-fold summaries

### Suggested API

```python
evaluate_classification_run(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict
```

Also add a helper to write JSON and markdown summaries.

### Acceptance criteria

- all event_v1 training scripts use the same scoring function
- output schema is stable and machine-readable
- metrics are directly comparable across event_v1 stages

---

## 3) `src/labels_event_v1.py`

### Goal

Create the **new experimental target path**.

### First label to implement

Implement **excess-return sign over 5 trading days**.

Preferred order of construction:

1. stock 5-day forward return
2. minus sector benchmark 5-day forward return if sector benchmark is available
3. otherwise minus market benchmark 5-day forward return

If the repo already has a benchmark series or easy way to derive sector/market controls, use that. If not, fall back cleanly to a market benchmark and document the fallback.

### Optional secondary label

Also support an optional **no-trade-zone** mode that drops observations whose excess forward return is near zero.

Suggested parameter:

```python
neutral_band_bps: int | None = None
```

If `neutral_band_bps` is provided, rows with absolute excess forward return below the band are dropped from the experimental sample.

### Requirements

The module should:

- generate forward returns using only future prices relative to time `t`
- align labels to the prediction timestamp correctly
- document exact label math in code comments
- avoid contaminating feature construction with forward information

### Suggested API

```python
build_event_v1_labels(
    prices_df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
    close_col: str = "close",
    horizon_days: int = 5,
    benchmark_mode: str = "market",
    neutral_band_bps: int | None = None,
) -> pd.DataFrame
```

### Output columns

At minimum:

- `ticker`
- `date`
- `forward_return_5d`
- `benchmark_forward_return_5d`
- `excess_forward_return_5d`
- `target_event_v1`
- optional diagnostic flag columns for dropped neutral rows

### Acceptance criteria

- label math is transparent and auditable
- output dates align with panel rows cleanly
- no leakage through benchmark computation or shifting

---

## 4) `src/panel_builder_event_v1.py`

### Goal

Build the new experimental panels **without changing existing panel builders**.

### Phase 1 requirement

First, build a panel using:

- existing Layer 1 features
- new event_v1 labels
- no new Layer 2 or Layer 3 features yet

### Later requirements

Then support additional panel modes:

- Layer 1 + Layer 2 v2
- Layer 1 + Layer 2 v2 + Layer 3 event_v1

### Suggested API

```python
build_event_v1_panel(
    layer1_path: str,
    labels_path: str,
    layer2_path: str | None = None,
    layer3_path: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame
```

### Requirements

- preserve the repo’s join keys and alignment conventions
- keep timestamps strictly causal
- do not backfill future information
- drop or flag rows with unusable target values
- print or save panel diagnostics:
  - row count
  - ticker count
  - date range
  - target balance
  - feature count
  - missingness summary

### Acceptance criteria

- panel build is reproducible
- existing benchmark panels remain untouched
- event_v1 panels are saved under new names only

---

## 5) `src/train_event_v1.py`

### Goal

Train models for the experimental lane using the **shared validation harness** and **shared evaluator**.

### Phase 1 requirement

Support training on:

- `event_v1_layer1_panel.parquet`

### Phase 2 requirement

Support training on:

- `event_v1_layer1_layer2_panel.parquet`

### Phase 3 requirement

Support training on:

- `event_v1_full_panel.parquet`

### Initial model set

Keep the model set simple and aligned with the current repo:

- logistic regression
- random forest
- hist gradient boosting

Do not add new model families in the first pass.

### Requirements

- use `validation_event_v1.py` for all splits
- use `evaluate_event_v1.py` for all metrics
- write outputs to versioned event_v1 report paths only
- preserve the ability to compare layer-by-layer performance
- keep preprocessing fit only on training folds

### Suggested behavior

Support a command-line or config switch like:

```bash
python -m src.train_event_v1 --panel event_v1_layer1
python -m src.train_event_v1 --panel event_v1_layer1_layer2
python -m src.train_event_v1 --panel event_v1_full
```

### Acceptance criteria

- one training script handles all event_v1 panel variants
- no event_v1 run modifies benchmark outputs
- results are saved in stable, comparable formats

---

## 6) `src/market_features_v2.py`

### Goal

Redesign Layer 2 to be a **market control layer**, not just a pile of technical indicators.

### Feature priorities

Focus on controls with high upside relative to implementation complexity:

#### Relative performance features

- 5-day stock return minus market return
- 10-day stock return minus market return
- 21-day stock return minus market return
- if feasible, sector-relative versions of the same

#### Regime / risk features

- rolling 21-day realized volatility
- rolling 63-day realized volatility
- volatility ratio: short vol / long vol
- rolling beta to market if feasible

#### Shock / state features

- overnight gap
- 1-day absolute return shock
- rolling drawdown from 21-day high
- return z-score vs trailing window

#### Volume / attention controls

- volume / 20-day average volume
- log volume
- abnormal volume flag

### Explicitly avoid in v2

Do **not** spend time first on low-fit, high-noise additions like:

- huge libraries of technical indicators
- distribution-fitting parameters unless already easy
- complicated signal stacking
- feature explosion without control logic

### Suggested API

```python
build_market_features_v2(
    prices_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
) -> pd.DataFrame
```

### Acceptance criteria

- features are strictly causal
- output is cleanly joinable by ticker/date
- features are documented and named clearly

---

## 7) `src/sec_sentiment_event_v1.py`

### Goal

Create an **event-driven Layer 3 feature set** from existing SEC sentiment artifacts.

### Do not redo the entire SEC ingestion stack

Reuse what already exists where possible.

The new module should sit on top of current sentiment outputs and derive **event-aware features**.

### Feature priorities

Build these first:

#### Filing-event anchored features

- latest filing sentiment score as of date `t`
- days since latest filing
- filing sentiment decay feature

#### Change / surprise features

- sentiment delta vs previous filing
- absolute sentiment change
- positive-to-negative regime flip flag
- negative-to-positive regime flip flag

#### Uncertainty / intensity features

- absolute sentiment magnitude
- uncertainty proxy if available from existing outputs
- dispersion / disagreement proxy if available

#### Interaction features

- sentiment delta × realized volatility
- sentiment magnitude × days since filing
- negative sentiment × abnormal volume

### Explicitly avoid in v1

Do **not** first build:

- full new NLP pipelines
- full news ingestion
- retraining FinBERT
- broad text expansion outside the existing SEC artifacts

### Suggested API

```python
build_sec_sentiment_event_v1(
    sentiment_df: pd.DataFrame,
    ticker_col: str = "ticker",
    filing_date_col: str = "filing_date",
    score_col: str = "sentiment_score",
    panel_dates_df: pd.DataFrame | None = None,
) -> pd.DataFrame
```

### Acceptance criteria

- features are generated from existing SEC sentiment outputs where possible
- features are date-safe and forward-fill only from past filings
- no future filing information leaks into earlier rows

---

## Training and Comparison Sequence

Run the following sequence and save outputs separately.

### Run 1

**Layer 1 features + event_v1 labels**

Purpose:
- test whether label/validation redesign alone improves signal

Input panel:
- `data/processed/modeling/event_v1_layer1_panel.parquet`

### Run 2

**Layer 1 + Layer 2 v2 + event_v1 labels**

Purpose:
- test whether stronger market controls improve discriminative power

Input panel:
- `data/processed/modeling/event_v1_layer1_layer2_panel.parquet`

### Run 3

**Layer 1 + Layer 2 v2 + Layer 3 event_v1 + event_v1 labels**

Purpose:
- test whether event-driven sentiment adds lift once the framing and controls are stronger

Input panel:
- `data/processed/modeling/event_v1_full_panel.parquet`

---

## Decision Rules for Interpreting Results

Treat an experiment as promising only if **AUC and log loss both improve**, not just F1/recall.

### Do not call it a real win if

- AUC is flat but F1 rises
- recall rises while probability calibration worsens
- performance improves in one fold but not across the aggregate
- improvement disappears on final holdout
- improvement is tiny and unstable relative to fold noise

### Minimum practical interpretation rule

A result is only worth keeping if:

- validation improvement is consistent across folds
- holdout does not reverse the direction of improvement
- gains are present in **AUC and log loss**, not only threshold metrics

---

## Anti-Leakage Guardrails

These rules are mandatory.

1. **No future data in feature construction**
   - all features at date `t` must use only information available on or before `t`

2. **No overlapping-label leakage across train and validation**
   - use purge / embargo logic for 5-day forward labels

3. **No fitting preprocessors on full data**
   - scalers, imputers, winsorization thresholds must be fit on training folds only

4. **No silent data overwrites**
   - event_v1 outputs must write to new file names only

5. **No benchmark file edits**
   - locked benchmark scripts and outputs must remain unchanged

6. **No model sprawl before problem framing is tested**
   - do not introduce XGBoost or new architectures in the first pass of this lane

---

## What to Implement First This Week

### Priority order

#### First

- `src/validation_event_v1.py`
- `src/labels_event_v1.py`
- `src/panel_builder_event_v1.py`

#### Then

- build `event_v1_layer1_panel.parquet`
- run `src/train_event_v1.py` on that panel

#### Only after that

- `src/market_features_v2.py`
- rebuild and rerun

#### Only after that

- `src/sec_sentiment_event_v1.py`
- rebuild and rerun

---

## Codex Execution Instructions

Use the following operating rules while implementing:

1. Do not touch the locked benchmark path.
2. Add new files only under the names specified above unless a minimal helper file is needed.
3. Prefer small, repo-compatible functions over a framework rewrite.
4. Keep all dataset and report names versioned and non-conflicting.
5. Write code that is explicit, debuggable, and easy to audit for leakage.
6. Add concise comments where label alignment or leakage prevention logic is non-obvious.
7. Preserve existing import patterns and repo style where reasonable.
8. Do not optimize for elegance at the cost of reproducibility.

---

## Definition of Success for This Iteration

This iteration is successful if it produces one of the following outcomes:

### Outcome A

A cleaner label/validation setup materially improves Layer 1 discrimination.

### Outcome B

Layer 2 v2 meaningfully improves over event_v1 Layer 1.

### Outcome C

Event-driven Layer 3 sentiment provides incremental lift after stronger controls are added.

### Outcome D

The repo now has a clean experimental lane that safely rejects bad ideas without contaminating the benchmark path.

Even if A, B, and C fail, D is still a valid and valuable success state because it makes future experimentation faster, safer, and more trustworthy.

---

## Final Instruction

Start with the evaluation harness and the new label path.

Do **not** begin by adding more sentiment features or more models.

The first question this repo needs answered is:

**Does a better validation and label framing setup reveal real signal that the current daily 5-day direction setup is washing out?**

