# Experiment Log

## Purpose
Track every meaningful modeling run, data change, and decision.

---

## Data Pipeline Update

### Experiment ID
EXP-000

### Date
2026-03-18

### Objective
Build the first Layer 1 fundamentals pipeline checkpoint before modeling.

### Dataset Version
- `data/raw/fundamentals/raw_fundamentals.parquet`
- `data/interim/fundamentals/fundamentals_quarterly_clean.parquet`
- `data/interim/features/layer1_financial_features.parquet`

### Universe
Fixed starter universe for testing:
- AAPL
- MSFT
- AMZN
- GOOGL
- META

### Date Range
2015-01-01 to 2024-12-31

### Feature Set
Layer 1 financial-statement-only features engineered:
- current_ratio
- quick_ratio
- cash_ratio
- working_capital_to_total_assets
- debt_to_equity
- debt_to_assets
- long_term_debt_ratio
- gross_margin
- operating_margin
- net_margin
- roa
- roe
- asset_turnover
- inventory_turnover
- receivables_turnover
- revenue_growth_qoq
- revenue_growth_yoy
- earnings_growth_qoq
- earnings_growth_yoy
- cfo_to_net_income
- accruals_ratio

### Validation
Not run yet. This checkpoint is focused on data ingestion, cleaning, and feature construction only.

### Model
Not run yet.

### Preprocessing
- Pulled raw fundamentals from SEC EDGAR using `edgartools`
- Consolidated long-format concept data into one raw parquet file
- Cleaned fundamentals into one row per `ticker + period_end`
- Fixed a reshape bug where metadata-level deduplication was dropping valid concept values
- Engineered Layer 1 accounting features with safe division and lag-based growth calculations

### Hyperparameters
Not applicable yet.

### Results
- Raw fundamentals rows: 20,134
- Clean quarterly fundamentals rows: 201
- Tickers in clean data: 5
- Clean fundamentals date range: 2015-03-28 to 2024-09-30
- Engineered feature columns: 21

### Observations
- Core fundamentals now have realistic missingness after the cleaning fix
- `gross_margin` is fully missing because gross profit is not yet available in the raw pull
- `inventory_turnover` and `quick_ratio` are sparser because inventory is not consistently reported across all firms
- Growth features have expected early-period missingness because they require lagged values

### Problems
- Git status/branch checks from this sandbox are blocked by a repository ownership warning
- The current universe is only a small starter set, not the final project universe
- Some ratios remain sparse because the current raw concept set is intentionally limited

### Decision
Pause after the Layer 1 feature-engineering milestone and use this as a clean checkpoint for the test branch.

### Next Step
Build the next Layer 1 step: either the modeling panel assembly or the first baseline training script.

---

## Layer 1 Benchmark Lock

### Experiment ID
EXP-001

### Date
2026-04-04

### Objective
Finalize the Layer 1 baseline benchmark using financial-statement-only features on the full Consumer Staples universe.

### Dataset Version
- `data/interim/fundamentals/fundamentals_quarterly_clean.parquet`
- `data/interim/features/layer1_financial_features.parquet`
- `data/interim/prices/prices_with_labels.parquet`
- `data/processed/modeling/layer1_modeling_panel.parquet`

### Universe
34 Consumer Staples names stored in `src/universe.py`

### Date Range
2015-01-01 to 2024-12-31

### Feature Set
Initial engineered Layer 1 set:
- current_ratio
- quick_ratio
- cash_ratio
- working_capital_to_total_assets
- debt_to_equity
- debt_to_assets
- long_term_debt_ratio
- gross_margin
- operating_margin
- net_margin
- roa
- roe
- asset_turnover
- inventory_turnover
- receivables_turnover
- revenue_growth_qoq
- revenue_growth_yoy
- earnings_growth_qoq
- earnings_growth_yoy
- cfo_to_net_income
- accruals_ratio

Final modeling features after filtering:
- current_ratio
- quick_ratio
- cash_ratio
- working_capital_to_total_assets
- debt_to_equity
- debt_to_assets
- long_term_debt_ratio
- operating_margin
- net_margin
- roa
- roe
- asset_turnover
- revenue_growth_qoq
- earnings_growth_qoq
- cfo_to_net_income
- accruals_ratio

### Target
`label = 1 if 5-day forward return > 0, else 0`

### Validation
Single time-based holdout split:
- Train dates before `2023-02-10`
- Test dates on or after `2023-02-10`

### Models
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
- Leakage-safe fundamentals alignment using `filing_date <= trading date`
- Median imputation
- Standard scaling for logistic regression
- Drop features with more than 20% missing values in training data
- Clip ratio outliers using train-set 1st and 99th percentiles

### Hyperparameters
- Logistic Regression:
  - `solver = lbfgs`
  - `max_iter = 1000`
  - `class_weight = balanced`
- Random Forest:
  - `n_estimators = 300`
  - `max_depth = 8`
  - `min_samples_leaf = 20`
  - `class_weight = balanced_subsample`
- HistGradientBoosting:
  - `max_depth = 6`
  - `learning_rate = 0.05`
  - `max_iter = 300`
  - `min_samples_leaf = 50`

### Results
- Modeling panel rows: `68,524`
- Tickers: `34`
- Best model: `Random Forest`
- Best AUC-ROC: `0.5138`

Full comparison:
- Logistic Regression:
  - AUC-ROC: `0.4926`
  - F1-score: `0.4040`
  - Precision: `0.4964`
  - Recall: `0.3405`
  - Log loss: `0.6941`
- Random Forest:
  - AUC-ROC: `0.5138`
  - F1-score: `0.3892`
  - Precision: `0.5220`
  - Recall: `0.3103`
  - Log loss: `0.6933`
- HistGradientBoosting:
  - AUC-ROC: `0.5029`
  - F1-score: `0.6236`
  - Precision: `0.5029`
  - Recall: `0.8205`
  - Log loss: `0.6966`

### Observations
- Layer 1 data pipeline is now fully working end-to-end.
- Financial-statement-only features produce only weak signal for 5-day direction prediction.
- Random Forest performed best, but only slightly above random guessing.
- The result is still useful because it creates a defensible baseline for comparing later feature layers.
- Fully missing or very sparse features did not materially change the final conclusion when removed.

### Problems
- `gross_margin` remained unusable because gross profit was not part of the raw pull.
- Some inventory- and receivables-based features were too sparse to keep in the final benchmark model.
- Short-horizon stock direction may simply be weakly related to slow-moving quarterly fundamentals.

### Decision
Lock Layer 1 here as the official baseline benchmark and move to Layer 2 market features next.

### Next Step
Build Layer 2 market features and compare Layer 1 versus Layer 1 + Layer 2 on the same time-based evaluation setup.

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
