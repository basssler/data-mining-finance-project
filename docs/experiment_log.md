# Experiment Log

## Purpose
Track every meaningful modeling run, data change, and decision.

---

## Experiment Template

### Experiment ID
EXP-001

### Date
2026-03-18

### Objective
Train first baseline model using Layer 1 financial features only.

### Dataset Version
modeling_table_v1.csv

### Universe
30-40 Large cap S&P 500 companies from a single sector. 
Why? focused sector subset allows for us better comparability of finance rations, keeps it manageable, and preserves the strong market coverage for later sentiment modeling. 

### Date Range
2015-2024

### Feature Set
Layer 1 only
- current_ratio
- debt_to_equity
- net_margin
- roa
- roe
- revenue_growth_yoy

### Target
label = 1 if 5-day forward return > 0, else 0

### Validation
TimeSeriesSplit
- n_splits = 5
- gap = 5
- final holdout = last 6 months

### Model
Logistic Regression

### Preprocessing
- missing values imputed
- winsorization applied
- RobustScaler used

### Hyperparameters
- C = 1.0
- penalty = l2
- solver = lbfgs

### Results
- AUC-ROC:
- F1-score:
- Precision:
- Recall:
- Log loss:

### Observations
- Model handled baseline features reasonably well
- Class balance was [balanced / imbalanced]
- Some features may need winsorization or transformation

### Problems
- Missing values in ROE
- Some tickers had incomplete filing history

### Decision
Keep as baseline and compare against Random Forest next.

### Next Step
Run Layer 1 Random Forest on same split for apples-to-apples comparison.