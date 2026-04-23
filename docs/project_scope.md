# Project Scope

## Project Title
Quarterly Event-Driven Excess-Return Classification

## Research Question
Can an event-driven quarterly filing panel predict the sign of 63-trading-day excess returns better than frozen daily baselines, and which quarterly feature families are stable enough to promote?

## Project Type
Supervised learning classification project.

## Prediction Target
Binary classification on the live quarterly label:
- Label = 1 if `63`-trading-day excess return sign is positive
- Label = 0 otherwise
- Benchmark mode = `sector_equal_weight_ex_self`

## Unit of Observation
One row = one quarterly filing event for one ticker.

## Canonical Universe
`34` large-cap Consumer Staples names stored in `src/universe.py`.

Why this remains canonical:
- quarterly accounting features stay comparable within-sector
- event timing stays auditable
- the benchmark history remains reproducible across the same 2015-2024 window

## Time Horizon
2015-2024

## Current Feature Families

### Baseline Quarterly Event Features
- point-in-time accounting snapshot features
- event timing fields
- filing availability fields

### Expanded Quarterly Design Features
- accounting deltas and profile composites
- Alpha Vantage earnings estimate and surprise features where coverage is available
- selected quarterly interaction or additive stability features depending on config

### Non-Canonical Families
- short-horizon daily market controls are legacy or comparison-only
- daily `5`-day label artifacts are preserved only for historical comparison
- sentiment layers remain experimental and are not promoted

## Models to Compare
- Logistic Regression
- Random Forest
- XGBoost

## Validation Strategy
Purged expanding-window validation:
- `5` folds
- `5` embargo days
- minimum training window = `252` dates
- final holdout starts `2024-01-01`
- no random split

## Evaluation Metrics
- AUC-ROC
- Log loss
- Precision / Recall
- rank IC
- worst-fold and concentration stability diagnostics for promotion review

## Deliverables
- reproducible quarterly benchmark configs
- quarterly benchmark registry and experiment log
- checked-in benchmark CSV and markdown artifacts
- quarterly diagnostics under `outputs/quarterly/`
- frozen legacy daily comparator

## In Scope
- quarterly event panel construction
- label / validation / feature ladder comparisons
- stability-aware benchmark promotion
- preserving the daily benchmark only as a legacy baseline

## Out of Scope
- live trading
- production deployment
- portfolio construction
- real-time streaming
- deleting the historical daily baseline

## MVP Definition
The project is successful at minimum if:
1. a quarterly event panel is reproducible from checked-in configs
2. the live target and validation scheme are unambiguous in repo docs
3. the active quarterly candidate and frozen daily baseline are clearly separated
4. a contributor can identify which quarterly run is baseline, comparison-only, or candidate
