# Event Panel V2 Workflow

This document turns the event-panel redesign plan into an implementation workflow you can actually run inside the repo.

Current repo scope now includes two benchmark lanes on top of the same `event_panel_v2` foundation:
- the original locked 5-trading-day excess-return-sign lane that remains the historical primary in the existing lock and decision docs
- an additive quarterly 63-trading-day excess-return-sign lane with its own config and outputs

This workflow document should treat the quarterly lane as an extension of the event-panel benchmark surface, not as a rewrite of the earlier promotion history.

The workflow is based on six phases:
1. leakage and timestamp audit
2. event-based panel rebuild
3. label/horizon comparison
4. frozen-model benchmark rerun
5. universe expansion
6. external dataset sourcing only after the redesigned panel proves useful

---

## 1. Branch strategy

### Recommended branch
Create a new branch:

```bash
git checkout exp/event-panel-v1
# or your latest committed branch that already contains EXP-006 through EXP-009

git pull
git checkout -b exp/event-panel-v2
```

### Why this branch
- `event_v1` stays preserved
- EXP-008 and EXP-009 stay logged and reproducible
- the redesign is methodologically different enough that it should not live on the same branch as the current daily-panel experiments

### Rule
Do **all event-panel redesign work** on `exp/event-panel-v2`.
Do not mix it into `main` until the redesign has produced a clean decision.

---

## 2. Where to put this file

Put this file here in the repo:

```text
docs/event_panel_v2_workflow.md
```

That should become the source-of-truth execution doc for the redesign branch.

---

## 3. How to work with Codex / Claude / ChatGPT

## Do not use one giant prompt

Use:
- **one master markdown file** as the source of truth
- **one focused implementation prompt per phase**
- **one commit per phase**

That gives you:
- cleaner diffs
- easier rollback
- better reproducibility
- less scope drift
- easier debugging when a phase breaks

### Correct workflow

1. Update `docs/event_panel_v2_workflow.md`
2. Start a fresh prompt for **only one phase**
3. Let the coding agent modify only the files needed for that phase
4. Run the code locally
5. Review outputs
6. Commit
7. Move to the next phase

### Wrong workflow
- one giant prompt that asks for all six phases at once
- redesign + new datasets + new models in the same change
- changing labels, panels, and evaluation rules at the same time

---

## 4. GPU training policy

Use the GPU **only where it actually helps**.

### Recommended anchor models for V2
- Logistic Regression
- Random Forest
- XGBoost

### GPU note
- Logistic Regression: CPU
- Random Forest: CPU
- XGBoost: GPU

That is fine. You do **not** need every model on the GPU.

### Recommended XGBoost GPU config
For XGBoost 2.x:

```python
xgb.XGBClassifier(
    tree_method="hist",
    device="cuda",
    eval_metric="logloss",
    random_state=42,
)
```

For older XGBoost versions:

```python
xgb.XGBClassifier(
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    eval_metric="logloss",
    random_state=42,
)
```

### Practical recommendation
In the redesigned workflow, treat XGBoost as the main nonlinear benchmark and the one that uses your GPU.

---

## 5. Implementation sequence

## Phase 1 — Leakage and timestamp audit

### Goal
Prove the current setup is temporally valid before redesigning it.

### Expected outputs
- `docs/leakage_audit_v1.md`
- `data/audit/leakage_audit_sample.csv`
- one yes/no conclusion: `current panel is temporally valid`

### Files likely involved
- panel-building logic
- label-building logic
- any timestamp alignment utilities
- one new audit script, for example:
  - `src/audit/leakage_audit_v1.py`

### Commit name
```bash
git commit -m "Add leakage audit for event panel v2 redesign"
```

### Prompt for coding agent
```text
Read docs/event_panel_v2_workflow.md and implement Phase 1 only.

Goal:
Create a manual leakage and timestamp audit for the existing panel.

Requirements:
- Sample 15 to 20 rows across train and pre-holdout periods
- For each sampled row, output:
  - ticker
  - feature timestamp
  - filing/news availability timestamp
  - label start date
  - label end date
  - any forward-fill boundary used
  - sector/benchmark timing reference
- Explicitly verify after-close handling
- Flag suspicious missingness and any possible post-label leakage
- Do not change modeling logic yet

Deliverables:
- docs/leakage_audit_v1.md
- data/audit/leakage_audit_sample.csv
- clear pass/fail summary

Constraints:
- modify only files needed for the audit
- no panel redesign yet
- no new datasets
- no model changes
```

---

## Phase 2 — Build the event-based panel

### Goal
Replace the daily forward-filled observation unit with an event-based row design.

### New row unit
One row = one ticker-event

### Start with these event types
- 10-Q filing date
- 10-K filing date
- earnings announcement date if available

### Expected outputs
- `src/panel_builder_event_v2.py`
- `data/interim/event_panel_v2.parquet`
- `docs/event_panel_spec_v2.md`

### Design rules
- fundamentals must be latest available **as of event date**
- market features must end at `t-1`
- sentiment/news features must be timestamp-safe relative to the event
- keep the old daily panel as a benchmark only, not the new primary experiment panel

### Commit name
```bash
git commit -m "Build event-based panel v2"
```

### Prompt for coding agent
```text
Read docs/event_panel_v2_workflow.md and implement Phase 2 only.

Goal:
Build the new event-based panel where one row equals one ticker-event.

Event types for v1:
- 10-Q filing date
- 10-K filing date
- earnings announcement date if available

Requirements:
- create src/panel_builder_event_v2.py
- create data/interim/event_panel_v2.parquet
- create docs/event_panel_spec_v2.md
- use only information available on or before the event date
- market windows must end at t-1
- stop daily forward-fill as the main research panel
- keep the existing daily panel unchanged as a comparison baseline

Constraints:
- no label redesign yet
- no model changes yet
- no new datasets yet
- keep assumptions explicit and auditable
```

---

## Phase 3 — Test label horizons

### Goal
Find out whether the main bottleneck is the prediction horizon.

### Labels to test
- 5-trading-day excess return sign
- 21-trading-day excess return sign
- optional high-signal label: top/bottom quantiles only, drop the middle band

### Metrics
- AUC
- log loss
- precision / recall
- Spearman rank correlation or rank IC

### Expected outputs
- `docs/label_comparison_v1.md`
- comparison table for event panel versus current daily panel

### Commit name
```bash
git commit -m "Add event panel label horizon comparison"
```

### Prompt for coding agent
```text
Read docs/event_panel_v2_workflow.md and implement Phase 3 only.

Goal:
Run the event-based setup with multiple label variants and compare them.

Label variants:
- 5-day excess return sign
- 21-day excess return sign
- optional top/bottom quantile label with middle band dropped

Requirements:
- keep the same core evaluation discipline
- report AUC, log loss, precision, recall, and rank IC if feasible
- compare event panel results against the current daily-panel baseline
- write docs/label_comparison_v1.md

Constraints:
- no new datasets
- no universe expansion yet
- no new model families
```

---

## Phase 4 — Freeze methods and rerun benchmark matrix

### Goal
Stop model churn and test the redesigned data with only the three anchor models.

### Fixed model set
- Logistic Regression
- Random Forest
- XGBoost

### Expected outputs
- one fixed training config for event panel v2
- benchmark result table across label variants

### Current additive scope
- Keep the original 5-trading-day lane as the historical anchor for the existing lock docs.
- Allow parallel configs that reuse the same `event_panel_v2` base when they represent separate benchmark lanes rather than a replacement benchmark.
- The current example is the quarterly 63-trading-day excess-return-sign lane in `configs/event_panel_v2_quarterly.yaml`.

### Suggested files
- `src/train_event_v2.py`
- `configs/event_v2_training.yaml` or similar if you use config files
- `reports/results/event_v2_benchmark_matrix.md`

### Commit name
```bash
git commit -m "Freeze event panel v2 model benchmark matrix"
```

### Prompt for coding agent
```text
Read docs/event_panel_v2_workflow.md and implement Phase 4 only.

Goal:
Create a fixed benchmark training workflow for the event-based panel using only three anchor models.

Models:
- Logistic Regression
- Random Forest
- XGBoost

Requirements:
- XGBoost should support GPU if available
- keep evaluation policy fixed
- generate a benchmark matrix across available label variants
- avoid introducing any other algorithms

Constraints:
- no new datasets
- no universe expansion yet
- no feature redesign beyond what already exists in the event panel
```

---

## Phase 5 — Expand the universe only if the event panel shows life

### Goal
Increase cross-sectional variation only after the event-based setup is validated.

### Expansion order
- start with 80 to 150 liquid large-cap names across sectors
- move beyond that only if the engineering and joins are stable

### Expected outputs
- `docs/universe_v2.md`
- new ticker list
- rerun of best event-based configuration on expanded universe

### Decision gate
Only do this if the event panel at least matches or slightly beats the daily panel on one label variant.

### Commit name
```bash
git commit -m "Expand event panel v2 universe to cross-sector large caps"
```

### Prompt for coding agent
```text
Read docs/event_panel_v2_workflow.md and implement Phase 5 only.

Goal:
Expand the event-based panel universe from the current narrow universe to a broader large-cap cross-sector universe.

Requirements:
- target 80 to 150 liquid large-cap names first
- preserve the existing event-panel alignment rules
- create docs/universe_v2.md
- regenerate the event panel and rerun the best configuration from earlier phases

Constraints:
- only do this after the event-based panel has shown at least some life
- no new datasets yet
- no method changes
```

---

## Phase 6 — Only then add external datasets

### Goal
Add new information only after the redesigned event panel proves it can capture signal.

### Dataset priority order
1. analyst estimate revisions
2. earnings surprise and guidance data
3. transcript-based management tone
4. target price changes / recommendation changes
5. high-quality ticker-date news feeds

### Rule
Judge every new external dataset against the **event-based panel**, not the old daily forward-filled panel.

### Commit name
```bash
git commit -m "Add external dataset layer to event panel v2"
```

### Prompt template for any future dataset
```text
Read docs/event_panel_v2_workflow.md and implement one new external dataset layer only.

Goal:
Add a single new dataset to the event-based panel and test whether it improves the locked benchmark.

Requirements:
- build one feature artifact
- merge onto event_panel_v2
- keep labels, validation, and models fixed
- write comparison results versus the no-external-data event panel baseline

Constraints:
- one dataset only
- no new methods
- no unrelated feature changes
```

---

## 6. Repo structure recommendation

Recommended additions:

```text
docs/
  event_panel_v2_workflow.md
  event_panel_spec_v2.md
  leakage_audit_v1.md
  label_comparison_v1.md
  universe_v2.md

src/
  audit/
    leakage_audit_v1.py
  panel_builder_event_v2.py
  train_event_v2.py

data/
  audit/
    leakage_audit_sample.csv
  interim/
    event_panel_v2.parquet

reports/
  results/
    event_v2_benchmark_matrix.md
```

---

## 7. Local execution workflow

For each phase:

```bash
# 1. make sure you are on the redesign branch
git checkout exp/event-panel-v2

# 2. pull latest changes if needed
git pull

# 3. run only the relevant script(s)
python -m src.audit.leakage_audit_v1
python -m src.panel_builder_event_v2
python -m src.train_event_v2 --panel event_panel_v2 --label 5d
python -m src.train_event_v2 --panel event_panel_v2 --label 21d

# 4. inspect outputs
# 5. commit before moving on
```

If you are using XGBoost on GPU, make sure your environment sees CUDA before you trust runtime speed.

---

## 8. Decision gates

### After Phase 1
If leakage or timestamp drift exists, fix that before anything else.

### After Phase 2
If the event panel is not cleaner and more interpretable than the daily panel, stop and simplify.

### After Phase 3
If 21-day materially beats 5-day, consider reframing the thesis around post-event short-to-medium horizon reaction.

### After Phase 4
If the redesigned event panel still shows no lift with frozen methods, do not add more algorithms.

If a quarterly lane is run, evaluate it as a parallel benchmark track with separate outputs rather than back-editing the earlier 5-day lock history.

### After Phase 5
If expanding the universe does not help, scale was not the main bottleneck.

### After Phase 6
Only keep external datasets that improve the event-based panel under the same locked evaluation discipline.

---

## 9. Final operating rule

The workflow is:

**master MD -> one focused implementation prompt -> local run -> review -> commit -> next phase**

Not:

**one giant prompt -> many moving parts -> unclear causality**
