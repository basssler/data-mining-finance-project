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

## Layer 2 V1 Comparison

### Experiment ID
EXP-002

### Date
2026-04-04

### Objective
Test whether adding a first set of daily market features improves the locked Layer 1 benchmark.

### Dataset Version
- `data/interim/features/layer2_market_features.parquet`
- `data/processed/modeling/layer1_modeling_panel.parquet`
- `outputs/comparison/layer_comparison_metrics.json`

### Universe
34 Consumer Staples names stored in `src/universe.py`

### Date Range
2015-01-01 to 2024-12-31

### Feature Set
Layer 2 v1 market features:
- return_5d
- return_21d
- volatility_21d
- volume_ratio_20d
- rsi_14

Comparison setups:
- Layer 1 only
- Layer 1 + Layer 2 core
- Layer 1 + Layer 2 full

### Target
`label = 1 if 5-day forward return > 0, else 0`

### Validation
Same time-based holdout used for Layer 1:
- Train dates before `2023-02-10`
- Test dates on or after `2023-02-10`

### Models
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
- Merge Layer 2 features onto the daily Layer 1 modeling panel by `ticker + date`
- Use the same missingness filter and outlier clipping rules as Layer 1
- Keep comparison on the exact same holdout split for fairness

### Results
Best AUC-ROC by setup:
- Layer 1 only: `0.5138`
- Layer 1 + Layer 2 core: `0.5043`
- Layer 1 + Layer 2 full: `0.5002`

Best model by setup:
- Layer 1 only: `Random Forest`
- Layer 1 + Layer 2 core: `HistGradientBoosting`
- Layer 1 + Layer 2 full: `HistGradientBoosting`

### Observations
- Layer 2 v1 did not improve on the locked Layer 1 benchmark.
- The core market subset performed slightly better than the full market set.
- `volume_ratio_20d` and `rsi_14` may be hurting rather than helping in the current specification.
- The first market feature pass likely needs better feature design rather than simply more indicators.

### Problems
- The initial market feature set appears too noisy for the current models and target horizon.
- The added features may not yet capture the strongest short-horizon market structure.

### Decision
Keep Layer 1 as the best benchmark for now and run one more focused Layer 2 refinement pass.

### Next Step
Build a cleaner Layer 2 v2 feature set focused on momentum, reversal, and normalized volatility behavior, then rerun the same comparison.

---

## Layer 2 V2 Refinement

### Experiment ID
EXP-003

### Date
2026-04-04

### Objective
Run one more focused Layer 2 pass using a cleaner market feature redesign and compare it against the locked Layer 1 benchmark.

### Dataset Version
- `data/interim/features/layer2_market_features.parquet`
- `data/processed/modeling/layer1_modeling_panel.parquet`
- `outputs/comparison/layer_comparison_metrics.json`

### Universe
34 Consumer Staples names stored in `src/universe.py`

### Date Range
2015-01-01 to 2024-12-31

### Feature Set
Layer 2 v2 added:
- return_1d
- return_5d
- return_10d
- return_21d
- volatility_5d
- volatility_21d
- volatility_ratio_5d_21d
- volume_ratio_20d
- volume_zscore_20d
- rsi_14

Comparison setups:
- Layer 1 only
- Layer 1 + Layer 2 core
- Layer 1 + Layer 2 full

### Target
`label = 1 if 5-day forward return > 0, else 0`

### Validation
Same time-based holdout used previously:
- Train dates before `2023-02-10`
- Test dates on or after `2023-02-10`

### Models
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
- Merge Layer 2 features onto the daily Layer 1 modeling panel by `ticker + date`
- Use the same missingness filter and outlier clipping rules as Layer 1
- Keep the same holdout split for fair comparison

### Results
Best AUC-ROC by setup:
- Layer 1 only: `0.5138`
- Layer 1 + Layer 2 core: `0.5074`
- Layer 1 + Layer 2 full: `0.5047`

Best model by setup:
- Layer 1 only: `Random Forest`
- Layer 1 + Layer 2 core: `Random Forest`
- Layer 1 + Layer 2 full: `HistGradientBoosting`

### Observations
- The focused Layer 2 redesign still did not beat the locked Layer 1 benchmark.
- The core Layer 2 subset again performed better than the full market set, but still underperformed Layer 1 alone.
- This strengthens the conclusion that simple daily market indicators are not enough in the current setup.
- Further Layer 2 work should focus on market-relative or sector-relative features rather than more generic technical indicators.

### Problems
- Two Layer 2 attempts failed to improve on the Layer 1 benchmark.
- The current market features may be too noisy or too generic for the 5-day prediction target.

### Decision
Log Layer 2 here as not yet successful, pause further Layer 2 iteration, and discuss redesign options with teammate before continuing.

### Next Step
Move attention to Layer 3 planning while keeping open the option to redesign Layer 2 using market-adjusted or sector-relative features after team discussion.

---

## Layer 3A Full-Filing SEC Sentiment

### Experiment ID
EXP-004

### Date
2026-04-04

### Objective
Test whether SEC filing sentiment from full 10-K and 10-Q text improves the locked Layer 1 benchmark.

### Dataset Version
- `data/interim/sentiment/sec_filing_sentiment.parquet`
- `data/interim/features/layer3_sec_sentiment_features.parquet`
- `data/processed/modeling/layer1_layer3_modeling_panel.parquet`
- `outputs/comparison/layer3_comparison_metrics.json`

### Universe
34 Consumer Staples names stored in `src/universe.py`

### Date Range
2015-01-01 to 2024-12-31

### Feature Set
Layer 3A full-filing SEC sentiment features:
- sec_sentiment_score
- sec_positive_prob
- sec_negative_prob
- sec_neutral_prob
- sec_sentiment_abs
- sec_sentiment_change_prev
- sec_positive_change_prev
- sec_negative_change_prev
- sec_chunk_count
- sec_log_chunk_count
- sec_is_10k
- sec_is_10q

Comparison setups:
- Layer 1 only
- Layer 1 + Layer 3

### Target
`label = 1 if 5-day forward return > 0, else 0`

### Validation
Same time-based holdout used previously:
- Train dates before `2023-02-10`
- Test dates on or after `2023-02-10`

### Models
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
- Pulled raw 10-K and 10-Q filing text from SEC EDGAR
- Scored filings with FinBERT using long-document chunking
- Dropped amendments for the first sentiment pass
- Prepared filing-level sentiment features
- Leakage-safe sentiment alignment using `filing_date <= trading date`
- Used the same missingness filter and outlier clipping rules as Layer 1

### Results
Filing sentiment scoring:
- Original filings scored: `1,322`
- 10-K filings: `330`
- 10-Q filings: `992`
- Average chunk count per filing: `80.17`

Best AUC-ROC by setup:
- Layer 1 only: `0.5138`
- Layer 1 + Layer 3: `0.5129`

Best model by setup:
- Layer 1 only: `Random Forest`
- Layer 1 + Layer 3: `HistGradientBoosting`

### Observations
- Full-filing SEC sentiment nearly matched the locked Layer 1 benchmark and performed much better than the simple Layer 2 market attempts.
- `HistGradientBoosting` improved materially when Layer 3 features were added, which suggests the filing sentiment features are adding some non-linear signal.
- Full-filing sentiment was heavily neutral on average, which is expected for long, boilerplate-heavy SEC documents.
- The continuous FinBERT probability features were useful to keep; the dominant sentiment label had no variation and was not useful.

### Problems
- Full 10-K and 10-Q filings dilute management tone with a large amount of neutral legal and accounting text.
- Layer 3A did not clearly beat the locked Layer 1 benchmark, so it is promising but not yet a winning improvement.
- Some 10-K filings incorporate parts of MD&A by reference, which can weaken section-level narrative extraction.

### Decision
Keep Layer 3A full-filing sentiment as the first official Layer 3 baseline and move to an MD&A-only refinement pass.

### Next Step
Extract and score MD&A-only text from 10-K and 10-Q filings, then compare Layer 1 versus Layer 1 + Layer 3 MD&A sentiment.

---

## Layer 3B MD&A SEC Sentiment

### Experiment ID
EXP-005

### Date
2026-04-04

### Objective
Test whether MD&A-only SEC filing sentiment improves on both the full-filing Layer 3 baseline and the locked Layer 1 benchmark.

### Dataset Version
- `data/interim/sentiment/sec_filing_sentiment_mda.parquet`
- `data/interim/features/layer3_sec_sentiment_mda_features.parquet`
- `data/processed/modeling/layer1_layer3_mda_modeling_panel.parquet`
- `outputs/comparison/layer3_mda_comparison_metrics.json`

### Universe
34 Consumer Staples names stored in `src/universe.py`

### Date Range
2015-01-01 to 2024-12-31

### Feature Set
Layer 3B MD&A SEC sentiment features:
- mda_sentiment_score
- mda_positive_prob
- mda_negative_prob
- mda_neutral_prob
- mda_sentiment_abs
- mda_sentiment_change_prev
- mda_positive_change_prev
- mda_negative_change_prev
- mda_chunk_count
- mda_log_chunk_count
- mda_text_length
- mda_log_text_length
- mda_is_10k
- mda_is_10q

Comparison setups:
- Layer 1 only
- Layer 1 + Layer 3 full filing
- Layer 1 + Layer 3 MD&A

### Target
`label = 1 if 5-day forward return > 0, else 0`

### Validation
Same time-based holdout used previously:
- Train dates before `2023-02-10`
- Test dates on or after `2023-02-10`

### Models
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
- Extracted MD&A from 10-K and 10-Q filings using item-boundary rules
- Fixed the first MD&A extractor so it skips table-of-contents-style false matches and prefers longer plausible spans
- Scored extracted MD&A text with FinBERT using long-document chunking on CUDA
- Prepared filing-level MD&A sentiment features from successfully scored filings only
- Leakage-safe MD&A sentiment alignment using `filing_date <= trading date`
- Used the same missingness filter and outlier clipping rules as Layer 1

### Results
MD&A extraction and scoring:
- Filings processed: `1,322`
- Filings with MD&A scored: `1,229`
- Filings without usable MD&A: `93`
- Average MD&A chunk count: `28.26`
- Average MD&A text length: `68,823`

Best AUC-ROC by setup:
- Layer 1 only: `0.5138`
- Layer 1 + Layer 3 full filing: `0.5129`
- Layer 1 + Layer 3 MD&A: `0.5127`

Best model by setup:
- Layer 1 only: `Random Forest`
- Layer 1 + Layer 3 full filing: `HistGradientBoosting`
- Layer 1 + Layer 3 MD&A: `Random Forest`

### Observations
- The MD&A extractor fix worked. Coverage improved from a failed first pass to `1,229` scored filings, which is enough to treat MD&A as a real modeling branch.
- MD&A sentiment was more focused and less neutral than the full-filing pass, which is directionally what we wanted from a management-narrative refinement.
- MD&A sentiment improved the logistic-regression result materially versus the full-filing Layer 3 setup, which suggests the MD&A features are cleaner and more linearly interpretable.
- Even so, neither full-filing nor MD&A Layer 3 beat the locked Layer 1 benchmark.

### Problems
- One ticker had no usable scored MD&A feature rows in the filing-level MD&A table, which created about `3%` missingness after daily alignment.
- MD&A sentiment still did not produce a clear AUC improvement over Layer 1, so the added complexity is not yet justified by benchmark performance alone.
- SEC filing sentiment appears promising, but still diluted relative to what a stronger event-driven text source might provide.

### Decision
Keep Layer 1 as the official best benchmark. Record Layer 3 as more promising than Layer 2, but not yet an outperforming layer.

### Next Step
Pause SEC-filing sentiment iteration here and discuss with teammate whether the next text source should be earnings call transcripts or a more event-specific SEC filing set such as 8-Ks.

---

## Current Project Takeaways

### Benchmark Hierarchy
- Best overall benchmark so far: Layer 1 only, `Random Forest`, AUC-ROC `0.5138`
- Closest challenger: Layer 1 + Layer 3 full-filing SEC sentiment, AUC-ROC `0.5129`
- Next closest: Layer 1 + Layer 3 MD&A SEC sentiment, AUC-ROC `0.5127`
- Layer 2 simple market features never beat Layer 1 in the current setup

### What We Learned
- Layer 1 fundamentals alone are weak but stable enough to serve as the official project baseline.
- Simple daily technical indicators were not useful enough to improve that baseline, so Layer 2 likely needs a redesign around market-relative or sector-relative features.
- SEC filing sentiment is more promising than Layer 2, especially after narrowing to MD&A, but still did not produce a clear benchmark win.
- For this 5-day prediction target, richer event-driven text such as earnings call transcripts may be a better next sentiment source than more broad full-document sentiment.

### Team Discussion Framing
- The project is not blocked by bad infrastructure anymore. The pipelines for Layer 1, Layer 2, and Layer 3 all run end-to-end.
- The core research question now is feature usefulness, not code stability.
- The next redesign decision should be strategic: improve Layer 2 with relative market signals, or push Layer 3 toward more event-driven text.

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
