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
