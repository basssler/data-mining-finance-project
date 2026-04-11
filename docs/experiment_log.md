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

## Event V1 Experimental Lane

### Experiment ID
EXP-006

### Date
2026-04-07

### Objective
Validate the new additive `event_v1` lane and test whether event-aware framing, redesigned Layer 2 controls, or event-driven SEC sentiment can beat the event-aware baseline.

### Dataset Version
- `data/interim/labels/labels_event_v1.parquet`
- `data/interim/features/layer2_market_features_v2.parquet`
- `data/interim/features/layer3_sec_sentiment_event_v1.parquet`
- `data/processed/modeling/event_v1_layer1_panel.parquet`
- `data/processed/modeling/event_v1_layer1_layer2_panel.parquet`
- `data/processed/modeling/event_v1_full_panel.parquet`
- `reports/results/event_v1_layer1_metrics.json`
- `reports/results/event_v1_layer1_layer2_metrics.json`
- `reports/results/event_v1_full_metrics.json`
- `reports/results/event_v1_summary.md`

### Universe
34 Consumer Staples names stored in `src/universe.py`

### Date Range
2015-01-01 to 2024-12-31

### Feature Set
Comparison setups:
- `event_v1_layer1`: Layer 1 fundamentals only on the new event-aware target
- `event_v1_layer1_layer2`: Layer 1 + redesigned Layer 2 v2 market/event controls
- `event_v1_full`: Layer 1 + Layer 2 v2 + event-driven SEC sentiment features

Layer 2 v2 features:
- rel_return_5d
- rel_return_10d
- rel_return_21d
- realized_vol_21d
- realized_vol_63d
- vol_ratio_21d_63d
- beta_63d_to_sector
- overnight_gap_1d
- abs_return_shock_1d
- drawdown_21d
- return_zscore_21d
- volume_ratio_20d
- log_volume
- abnormal_volume_flag

Event-driven SEC sentiment features:
- sec_event_score_latest
- sec_event_abs_latest
- sec_event_days_since_filing
- sec_event_decay_30d
- sec_event_score_decayed
- sec_event_delta_prev
- sec_event_abs_delta_prev
- sec_event_neg_to_pos_flip
- sec_event_pos_to_neg_flip
- sec_event_uncertainty

Interaction features:
- sec_event_delta_x_vol21
- sec_event_magnitude_x_days_since
- sec_event_negative_x_abnormal_volume

### Target
`target_event_v1 = 1 if 5-day excess return > 0, else 0`

Excess return definition:
- stock 5-day forward return minus leave-one-out equal-weight 5-day forward return of the other Consumer Staples names in the universe

### Validation
Purged expanding-window validation on pre-holdout data with one untouched 2024 holdout:
- first `756` trading dates used as minimum training warmup
- `5` contiguous expanding validation folds before holdout
- purge gap of `9` trading dates before each validation block and before the 2024 holdout
- final holdout = all trading dates in calendar year 2024

### Models
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
- Built event-aware labels from `adj_close` using 5-day forward excess returns
- Used the existing Layer 1 modeling panel as a read-only base
- Median imputation
- Standard scaling for logistic regression only
- Drop features with more than `20%` missingness in train folds only
- Clip numeric features using train-fold `1%` and `99%` quantiles
- Select best model by mean CV AUC-ROC with mean CV log loss as tie-breaker
- Refit the selected model on all pre-holdout data and evaluate once on 2024

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
Labeling and panel construction:
- Event label rows: `85,418`
- Event label target balance: class_0=`49.87%`, class_1=`50.13%`
- `event_v1` panel rows: `68,524`
- Tickers: `34`
- Event panel date range: `2015-07-31` to `2024-12-31`

Best run by setup:
- `event_v1_layer1`: `Random Forest`
  - CV AUC-ROC: `0.5027`
  - CV log loss: `0.6959`
  - 2024 holdout AUC-ROC: `0.5205`
  - 2024 holdout log loss: `0.6935`
- `event_v1_layer1_layer2`: `HistGradientBoosting`
  - CV AUC-ROC: `0.5086`
  - CV log loss: `0.7196`
  - 2024 holdout AUC-ROC: `0.5175`
  - 2024 holdout log loss: `0.7058`
- `event_v1_full`: `HistGradientBoosting`
  - CV AUC-ROC: `0.5087`
  - CV log loss: `0.7257`
  - 2024 holdout AUC-ROC: `0.5028`
  - 2024 holdout log loss: `0.7104`

### Observations
- The `event_v1` lane is stable and reproducible end-to-end. Local reruns reproduced the saved metrics.
- Event-aware reframing alone did not create a stronger baseline than the locked Layer 1 benchmark.
- Layer 2 v2 produced a small CV AUC lift, but calibration worsened and the holdout did not support calling it a real improvement.
- Event-driven SEC sentiment added almost no incremental discrimination and clearly worsened holdout log loss.
- Threshold metrics such as F1 and recall improved in some runs, but the probability-ranking metrics still indicate weak signal.

### Problems
- `gross_margin` remains fully missing in the inherited Layer 1 feature base.
- Several Layer 1 accounting features are still too sparse to be consistently useful.
- The current event-aware Layer 2 and Layer 3 additions do not justify promotion beyond experimental status.

### Decision
Keep the locked Layer 1 benchmark as the official project benchmark. Keep `event_v1_layer1` as the clean event-aware experimental baseline. Do not advance Layer 2 v2 or the current event-driven SEC sentiment features as winning additions.

### Next Step
Test one higher-information event dataset on top of `event_v1_layer1`, with analyst estimate revisions / earnings surprise as the highest-priority next candidate. Other strong follow-ups are 8-K or guidance event flags and earnings-call transcript Q&A features.

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

---

## Experiment ID
EXP-007

### Date
2026-04-10

### Objective
Test one single additive analyst-event dataset on top of `event_v1_layer1` without changing the existing `event_v1` label, validation, trainer, or evaluation logic.

### Dataset Version
`analyst_event_v1`

Built from:
- `data/raw/analyst/analyst_ratings_processed.csv`
- `data/interim/analyst/layer3_analyst_event_v1.parquet`
- `data/modeling/event_v1/event_v1_layer1_analyst_panel.parquet`

### Universe
Same `event_v1` Consumer Staples universe used by the existing Layer 1 event-aware benchmark:
- `34` tickers
- `68,524` ticker-date rows after target filtering

### Date Range
Modeling panel:
- `2015-07-31` to `2024-12-31`

Preferred analyst source coverage actually available:
- `2009-02-14` to `2020-06-11`

### Feature Set
Base features:
- Existing `event_v1_layer1` fundamentals only

Additive analyst daily features:
- `analyst_event_count_1d`
- `analyst_event_count_5d`
- `analyst_upgrade_count_5d`
- `analyst_downgrade_count_5d`
- `analyst_reiterate_count_5d`
- `analyst_pt_up_count_5d`
- `analyst_pt_down_count_5d`
- `analyst_net_revision_score_5d`
- `analyst_mean_sentiment_1d`
- `analyst_mean_sentiment_5d`
- `analyst_sentiment_std_5d`
- `analyst_days_since_event`

Processed-source schema actually available:
- ticker / symbol field: `stock`
- timestamp / date field: `date`
- action / rating text field: none structured, derived from `title`
- sentiment / label field: none structured
- target price fields: none structured

Analyst parsing assumptions:
- parsed only direct single-name analyst titles from `title`
- used deterministic mappings for `upgrade`, `downgrade`, `reiterate`, and `initiate`
- mapped rating text to sentiment using a small transparent `{-1, 0, +1}` scheme
- derived price-target direction from `raises/lowers/cuts price target` text and `from X to Y` text when present
- shifted headlines at or after `4:00 PM America/New_York` to the next tradable panel date

### Target
Unchanged `event_v1` target:
- `target_event_v1 = 1 if 5-day excess return > 0, else 0`

### Validation
Unchanged `event_v1` validation policy:
- purged expanding-window CV on pre-holdout data
- `5` folds
- minimum training warmup = `756` trading dates
- embargo / purge gap = `5` days in config and `9` purged trading dates in realized splits
- final holdout = calendar year `2024`

### Models
Unchanged model family:
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
Unchanged train/eval behavior:
- median imputation
- standard scaling only for logistic regression
- train-fold missingness filter at `20%`
- train-fold clipping at `1%` / `99%`
- select best model by mean CV AUC with mean CV log loss tie-break
- refit on all pre-holdout data and evaluate once on the `2024` holdout

Analyst-layer construction:
- built daily ticker-date features aligned using only information available on or before date `t`
- merged only analyst feature columns onto `event_v1_layer1`
- kept existing `event_v1` artifacts unchanged

### Hyperparameters
No changes from `train_event_v1.py`.

Best model:
- `Random Forest`
  - `n_estimators = 300`
  - `max_depth = 8`
  - `min_samples_leaf = 20`
  - `class_weight = balanced_subsample`

### Results
Analyst source and feature coverage:
- Raw processed analyst rows: `1,400,469`
- Direct event rows kept after deterministic parsing: `143,359`
- Direct analyst event availability window: `2009-08-10` to `2020-06-11`
- Daily analyst feature rows: `68,524`
- Daily rows with same-day analyst events: `1,165`

Best run:
- Panel: `event_v1_layer1_analyst`
- Best model: `Random Forest`
- Mean CV AUC: `0.5046`
- Mean CV log loss: `0.6952`
- 2024 holdout AUC: `0.5191`
- 2024 holdout log loss: `0.6934`

Comparison vs `event_v1_layer1`:
- `event_v1_layer1` mean CV AUC: `0.5027`
- `event_v1_layer1_analyst` mean CV AUC: `0.5046`
- `event_v1_layer1` mean CV log loss: `0.6959`
- `event_v1_layer1_analyst` mean CV log loss: `0.6952`
- `event_v1_layer1` 2024 holdout AUC: `0.5205`
- `event_v1_layer1_analyst` 2024 holdout AUC: `0.5191`
- `event_v1_layer1` 2024 holdout log loss: `0.6935`
- `event_v1_layer1_analyst` 2024 holdout log loss: `0.6934`

Explicit win criteria:
- CV AUC `> 0.5027`: `PASS`
- CV log loss `< 0.6959`: `PASS`
- 2024 holdout AUC `>= 0.5205`: `FAIL`
- 2024 holdout log loss `<= 0.6935`: `PASS`

Overall outcome:
- `FAIL`

Holdout usable analyst features after the unchanged train-fold missingness filter:
- `analyst_event_count_1d`
- `analyst_event_count_5d`
- `analyst_upgrade_count_5d`
- `analyst_downgrade_count_5d`
- `analyst_reiterate_count_5d`
- `analyst_pt_up_count_5d`
- `analyst_pt_down_count_5d`
- `analyst_net_revision_score_5d`

### Observations
- The additive analyst layer produced a real but small CV improvement on both AUC and log loss.
- The 2024 holdout AUC still fell below the required benchmark threshold, so this was not a complete win.
- The processed analyst source is much thinner than expected. It contains only `title`, `date`, and `stock`, so all action, sentiment, and price-target logic had to be derived from title text.
- The source ends on `2020-06-11`, which means the 2024 holdout has no recent analyst-event counts or sentiment activity from this dataset.
- The analyst sentiment features were too sparse to survive the unchanged `20%` train-fold missingness rule, and `analyst_days_since_event` was just over the cutoff in the holdout refit.

### Problems
- No local `raw_analyst_ratings.csv` was available, so the implementation had to rely on the thinner processed CSV only.
- The processed source coverage stops more than three years before the `2024` holdout.
- Structured action, sentiment, and target-price columns were absent, so the feature layer could only use transparent headline parsing.
- `gross_margin` remains fully missing in the inherited Layer 1 base.

### Decision
Do not promote `event_v1_layer1_analyst` as a winning addition over `event_v1_layer1`. Keep it as an additive experiment only.

### Next Step
If analyst data remains a priority, recover a richer analyst source with post-2020 coverage and structured action / rating / target-price fields. Otherwise move to a higher-information event dataset with full holdout coverage, such as earnings surprise / estimate revision, 8-K guidance flags, or earnings-call features.

---

## Experiment ID
EXP-008

### Date
2026-04-10

### Objective
Test whether a minimal structured SEC filing-event metadata layer improves over `event_v1_layer1`, using only free SEC EDGAR / `data.sec.gov` metadata and no text parsing.

### Dataset Version
`sec_filing_events_v1`

Built from:
- `data/interim/sec/sec_universe_mapping_v1.parquet`
- `data/interim/sec/sec_filing_metadata_v1.parquet`
- `data/interim/sec/layer3_sec_filing_events_v1.parquet`
- `data/modeling/event_v1/event_v1_layer1_secfilings_panel.parquet`

### Universe
Same `event_v1` Consumer Staples universe used by the existing Layer 1 event-aware benchmark:
- `34` tickers
- `68,524` ticker-date rows after target filtering

Universe SEC mapping table columns:
- `ticker`
- `cik`
- `company_name`
- `sec_lookup_ticker`
- `mapping_source`
- `manual_override_note`

Manual overrides used:
- `BF-B` mapped with SEC lookup ticker `BF.B`

### Date Range
Modeling panel:
- `2015-07-31` to `2024-12-31`

Structured SEC filing metadata kept after filtering and effective-date alignment:
- `2015-07-30` to `2024-12-30`

### Feature Set
Base features:
- Existing `event_v1_layer1` fundamentals only

Additive structured SEC filing-event features:
- `sec_is_8k_today`
- `sec_is_10q_today`
- `sec_is_10k_today`
- `sec_filing_count_1d`
- `sec_filing_count_5d`
- `sec_after_close_filing_count_1d`
- `sec_pre_market_filing_count_1d`
- `sec_8k_decay_3d`
- `sec_8k_decay_5d`
- `sec_10q_decay_3d`
- `sec_10k_decay_3d`
- `sec_days_since_any_filing`
- `sec_days_since_8k`
- `sec_days_since_10q`
- `sec_days_since_10k`

SEC metadata assumptions:
- used SEC submissions metadata only
- used exact original form types `8-K`, `10-Q`, and `10-K`
- excluded amendments such as `8-K/A`, `10-Q/A`, and `10-K/A` in v1
- used `acceptanceDateTime` when available and converted timestamps from UTC to `America/New_York`
- filings before market open were available same trading day
- filings during market hours were treated as available same trading day
- filings after market close were shifted to the next trading day
- if timestamp had been missing, the rule would have shifted the filing to the next trading day conservatively to avoid leakage
- no sentiment, no full-text parsing, no MD&A parsing, and no 8-K item parsing

### Target
Unchanged `event_v1` target:
- `target_event_v1 = 1 if 5-day excess return > 0, else 0`

### Validation
Unchanged `event_v1` validation policy:
- purged expanding-window CV on pre-holdout data
- `5` folds
- minimum training warmup = `756` trading dates
- embargo / purge gap = `5` days in config and `9` purged trading dates in realized splits
- final holdout = calendar year `2024`

### Models
Unchanged model family:
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
Unchanged train/eval behavior:
- median imputation
- standard scaling only for logistic regression
- train-fold missingness filter at `20%`
- train-fold clipping at `1%` / `99%`
- select best model by mean CV AUC with mean CV log loss tie-break
- refit on all pre-holdout data and evaluate once on the `2024` holdout

SEC filing-event construction:
- fetched universe-level SEC submissions histories from `data.sec.gov`
- built a structured metadata table with `ticker`, `cik`, `form_type`, `filing_date`, and `filing_timestamp`
- aligned filings to effective model dates using trading dates from the locked Layer 1 panel
- merged only SEC filing-event feature columns onto `event_v1_layer1`
- kept existing `event_v1` artifacts unchanged

### Hyperparameters
No changes from `train_event_v1.py`.

Best model:
- `Random Forest`
  - `n_estimators = 300`
  - `max_depth = 8`
  - `min_samples_leaf = 20`
  - `class_weight = balanced_subsample`

### Results
SEC mapping and metadata coverage:
- Universe mapping rows: `34`
- Manual overrides used: `1`
- Structured filing metadata rows after filtering/alignment: `4,627`
- Filings by form type:
  - `8-K`: `3,518`
  - `10-Q`: `831`
  - `10-K`: `278`
- Timing bucket counts:
  - `pre_market`: `2,069`
  - `market_hours`: `1,274`
  - `after_close`: `1,284`

Daily SEC feature coverage:
- Daily SEC feature rows: `68,524`
- Rows with any filing today: `3,980`
- Rows with `8-K` today: `3,378`
- Rows with `10-Q` today: `830`
- Rows with `10-K` today: `278`

Best run:
- Panel: `event_v1_layer1_secfilings`
- Best model: `Random Forest`
- Mean CV AUC: `0.5019`
- Mean CV log loss: `0.6958`
- 2024 holdout AUC: `0.5238`
- 2024 holdout log loss: `0.6933`

Comparison vs `event_v1_layer1`:
- `event_v1_layer1` mean CV AUC: `0.5027`
- `event_v1_layer1_secfilings` mean CV AUC: `0.5019`
- `event_v1_layer1` mean CV log loss: `0.6959`
- `event_v1_layer1_secfilings` mean CV log loss: `0.6958`
- `event_v1_layer1` 2024 holdout AUC: `0.5205`
- `event_v1_layer1_secfilings` 2024 holdout AUC: `0.5238`
- `event_v1_layer1` 2024 holdout log loss: `0.6935`
- `event_v1_layer1_secfilings` 2024 holdout log loss: `0.6933`

Explicit win criteria:
- CV AUC `> 0.5027`: `FAIL`
- CV log loss `< 0.6959`: `PASS`
- 2024 holdout AUC `>= 0.5205`: `PASS`
- 2024 holdout log loss `<= 0.6935`: `PASS`

Overall outcome:
- `FAIL`

Holdout usable SEC filing-event features after the unchanged train-fold missingness filter:
- `sec_is_8k_today`
- `sec_is_10q_today`
- `sec_is_10k_today`
- `sec_filing_count_1d`
- `sec_filing_count_5d`
- `sec_after_close_filing_count_1d`
- `sec_pre_market_filing_count_1d`
- `sec_8k_decay_3d`
- `sec_8k_decay_5d`
- `sec_10q_decay_3d`
- `sec_10k_decay_3d`
- `sec_days_since_any_filing`
- `sec_days_since_8k`
- `sec_days_since_10q`
- `sec_days_since_10k`

### Observations
- The structured SEC filing-event layer produced the strongest `2024` holdout result seen so far versus the clean `event_v1_layer1` baseline.
- Holdout AUC and holdout log loss both improved relative to `event_v1_layer1`.
- The improvement was not stable enough in CV because mean CV AUC slipped below the required `0.5027` threshold.
- The feature set remained fully structured and low-complexity: form-type flags, counts, timing buckets, recency, and decay.
- Timestamp coverage from SEC submissions metadata was strong enough that the conservative missing-time fallback was effectively a safeguard rather than a dominant path.

### Problems
- The CV ranking metric did not clear the required baseline threshold, so this is not a full benchmark win.
- Filing-time alignment still depends on SEC acceptance timestamps and a market-hours assumption; this is structured and reasonable, but still an assumption.
- Amendments were excluded in v1 by design, so any signal specific to amendment timing is intentionally not captured here.
- `gross_margin` remains fully missing in the inherited Layer 1 base.

### Decision
Keep `event_v1_layer1_secfilings` as a promising but not winning additive experiment. Do not replace `event_v1_layer1` as the benchmark yet.

### Next Step
Stay within structured SEC metadata first. Strong next tests are:
- add amendment-aware counts as a separate SEC metadata experiment
- split `8-K` timing effects more explicitly without item parsing
- combine this filing-event layer with one other high-quality additive event dataset only after keeping the current event_v1 evaluation policy fixed

---

## Experiment ID
EXP-009

### Date
2026-04-10

### Objective
Test whether grouped `8-K` item-category features improve over `event_v1_layer1` and over the coarse SEC form-type event layer from EXP-008, while staying strictly inside structured SEC metadata only.

### Dataset Version
`sec_8k_grouped_events_v1`

Built from:
- reused `data/interim/sec/sec_universe_mapping_v1.parquet`
- reused `data/interim/sec/sec_filing_metadata_v1.parquet`
- refetched structured `8-K` `items` metadata from SEC `data.sec.gov` submissions JSON for the project universe only
- wrote `data/interim/sec/layer3_sec_8k_grouped_events_v1.parquet`
- wrote `data/modeling/event_v1/event_v1_layer1_sec8kgrouped_panel.parquet`

### Universe
Same `event_v1` Consumer Staples universe used by the locked Layer 1 benchmark:
- `34` tickers
- `68,524` ticker-date rows after target filtering

### Date Range
Modeling panel:
- `2015-07-31` to `2024-12-31`

Structured retained `8-K` item rows after reuse of EXP-008 timing-aligned metadata:
- `2015-07-30` to `2024-12-30`

### Feature Set
Base features:
- existing `event_v1_layer1` fundamentals only

Additive grouped `8-K` features:
- `sec_8k_earnings_results_today`
- `sec_8k_guidance_outlook_today`
- `sec_8k_leadership_governance_today`
- `sec_8k_financing_securities_today`
- `sec_8k_material_agreement_mna_today`
- `sec_8k_regulatory_legal_today`
- `sec_8k_earnings_results_decay_3d`
- `sec_8k_guidance_outlook_decay_3d`
- `sec_8k_leadership_governance_decay_3d`
- `sec_8k_financing_securities_decay_3d`
- `sec_8k_material_agreement_mna_decay_3d`
- `sec_8k_regulatory_legal_decay_3d`
- `sec_8k_item_count_1d`
- `sec_8k_item_count_5d`
- `sec_days_since_8k_earnings_results`
- `sec_days_since_8k_guidance_outlook`
- `sec_days_since_8k_leadership_governance`

Explicit item-to-category mapping used in code:
- `earnings_results`: `2.02`
- `guidance_outlook`: `7.01`
- `leadership_governance`: `5.02`, `5.03`, `5.07`
- `financing_securities`: `2.03`, `3.02`, `3.03`
- `material_agreement_mna`: `1.01`, `1.02`, `2.01`
- `regulatory_legal`: `1.03`, `4.01`, `4.02`

Explicitly skipped as too broad or ambiguous for v1:
- broad catch-all items such as `8.01` and `9.01`
- section-level integer item codes such as `1`, `2`, `4`, `5`, `7`, `8`, `9`, and `12`
- clean but out-of-scope items such as `2.05`, `2.06`, `5.01`, `5.04`, and `5.05`

Important assumptions:
- `7.01` was treated as a structured proxy for guidance / outlook style public disclosures because no text disambiguation was allowed in this experiment
- no full-text parsing, no sentiment, no MD&A extraction, and no exhibit parsing were used
- only exact original `8-K` filings were used for grouped-item features; amendments stayed excluded

### Target
Unchanged `event_v1` target:
- `target_event_v1 = 1 if 5-day excess return > 0, else 0`

### Validation
Unchanged `event_v1` validation policy:
- purged expanding-window CV on pre-holdout data
- `5` folds
- minimum training warmup = `756` trading dates
- embargo / purge gap = `5` days in config and `9` purged trading dates in realized splits
- final holdout = calendar year `2024`

### Models
Unchanged model family:
- Logistic Regression
- Random Forest
- HistGradientBoosting

### Preprocessing
Unchanged train/eval behavior:
- median imputation
- standard scaling only for logistic regression
- train-fold missingness filter at `20%`
- train-fold clipping at `1%` / `99%`
- select best model by mean CV AUC with mean CV log loss tie-break
- refit on all pre-holdout data and evaluate once on the `2024` holdout

Grouped `8-K` construction details:
- reused EXP-008 universe mapping and timing-aligned SEC metadata
- pulled `8-K` `items` arrays from SEC submissions JSON for the project universe only
- reused EXP-008 timing convention:
  - SEC UTC acceptance timestamps converted to `America/New_York`
  - filings after market close shifted to the next trading day
  - filings before market open available same trading day
  - if timestamp was missing in EXP-008 metadata, the filing had already been shifted conservatively to the next trading day
- merged only grouped `8-K` feature columns onto `event_v1_layer1`
- kept labels, validation, model selection logic, preprocessing, and holdout policy unchanged

### Hyperparameters
No changes from `train_event_v1.py`.

Best model:
- `HistGradientBoosting`
  - `max_depth = 6`
  - `learning_rate = 0.05`
  - `max_iter = 300`
  - `min_samples_leaf = 50`

### Results
Structured `8-K` item coverage:
- raw exact `8-K` submission rows fetched: `10,708`
- `8-K` filing rows with at least one parseable decimal item code: `9,457`
- retained structured item rows after reuse of EXP-008 timing-aligned metadata: `7,420`
- mapped category item rows retained: `3,651`

Mapped item frequencies retained in-window:
- `2.02`: `1,103`
- `7.01`: `821`
- `5.02`: `738`
- `5.07`: `280`
- `1.01`: `289`
- `2.03`: `196`
- `5.03`: `123`
- `1.02`: `64`
- `2.01`: `23`
- `3.02`: `2`
- `3.03`: `5`
- `4.01`: `3`
- `4.02`: `4`

Daily grouped-feature event frequencies:
- `sec_8k_earnings_results_today`: `1,102`
- `sec_8k_guidance_outlook_today`: `804`
- `sec_8k_leadership_governance_today`: `1,027`
- `sec_8k_financing_securities_today`: `201`
- `sec_8k_material_agreement_mna_today`: `317`
- `sec_8k_regulatory_legal_today`: `7`

Best run:
- Panel: `event_v1_layer1_sec8kgrouped`
- Best model: `HistGradientBoosting`
- Mean CV AUC: `0.5079`
- Mean CV log loss: `0.7188`
- 2024 holdout AUC: `0.5209`
- 2024 holdout log loss: `0.7044`

Comparison vs `event_v1_layer1`:
- `event_v1_layer1` mean CV AUC: `0.5027`
- `event_v1_layer1_sec8kgrouped` mean CV AUC: `0.5079`
- `event_v1_layer1` mean CV log loss: `0.6959`
- `event_v1_layer1_sec8kgrouped` mean CV log loss: `0.7188`
- `event_v1_layer1` 2024 holdout AUC: `0.5205`
- `event_v1_layer1_sec8kgrouped` 2024 holdout AUC: `0.5209`
- `event_v1_layer1` 2024 holdout log loss: `0.6935`
- `event_v1_layer1_sec8kgrouped` 2024 holdout log loss: `0.7044`

Comparison vs `event_v1_layer1_secfilings`:
- `event_v1_layer1_secfilings` mean CV AUC: `0.5019`
- `event_v1_layer1_sec8kgrouped` mean CV AUC: `0.5079`
- `event_v1_layer1_secfilings` mean CV log loss: `0.6958`
- `event_v1_layer1_sec8kgrouped` mean CV log loss: `0.7188`
- `event_v1_layer1_secfilings` 2024 holdout AUC: `0.5238`
- `event_v1_layer1_sec8kgrouped` 2024 holdout AUC: `0.5209`
- `event_v1_layer1_secfilings` 2024 holdout log loss: `0.6933`
- `event_v1_layer1_sec8kgrouped` 2024 holdout log loss: `0.7044`

Explicit win criteria:
- CV AUC `> 0.5027`: `PASS`
- CV log loss `< 0.6959`: `FAIL`
- 2024 holdout AUC `>= 0.5205`: `PASS`
- 2024 holdout log loss `<= 0.6935`: `FAIL`

Overall outcome:
- `FAIL`

Holdout usable grouped `8-K` features after the unchanged train-fold missingness filter:
- `sec_8k_earnings_results_today`
- `sec_8k_guidance_outlook_today`
- `sec_8k_leadership_governance_today`
- `sec_8k_financing_securities_today`
- `sec_8k_material_agreement_mna_today`
- `sec_8k_regulatory_legal_today`
- `sec_8k_earnings_results_decay_3d`
- `sec_8k_guidance_outlook_decay_3d`
- `sec_8k_leadership_governance_decay_3d`
- `sec_8k_item_count_1d`
- `sec_8k_item_count_5d`
- `sec_days_since_8k_earnings_results`
- `sec_days_since_8k_guidance_outlook`
- `sec_days_since_8k_leadership_governance`

### Observations
- Grouping `8-K` items materially improved mean CV AUC over both `event_v1_layer1` and the coarser EXP-008 SEC layer.
- The same grouped refinement damaged probability calibration badly enough that both CV log loss and holdout log loss failed the locked thresholds.
- The locked selection rule chose `HistGradientBoosting` because it won on mean CV AUC, even though the resulting probabilities were much worse calibrated than the best EXP-008 run.
- The grouped layer did not beat the coarser SEC filing-event layer on holdout AUC or holdout log loss.
- The most active grouped signals were earnings results, leadership / governance, and the coarse `7.01` guidance proxy. Financing and regulatory/legal coverage were thin.

### Problems
- `guidance_outlook` depends on `7.01`, which is a structured but noisy proxy for outlook-related public disclosures because v1 intentionally avoids any text disambiguation.
- Broad but common items such as `8.01` and `9.01` were skipped by design, which keeps the mapping auditable but leaves a large share of raw `8-K` activity unused.
- `regulatory_legal` coverage is extremely sparse, and its decay feature was far too missing to survive the unchanged `20%` train-fold filter.
- The grouped refinement improved rank ordering but produced much worse log loss than both the Layer 1 baseline and the EXP-008 SEC coarse layer.
- `gross_margin` remains fully missing in the inherited Layer 1 base.

### Decision
Do not promote `event_v1_layer1_sec8kgrouped` over either `event_v1_layer1` or `event_v1_layer1_secfilings`. Keep it as a controlled SEC-path refinement only.

### Next Step
If the SEC path stays active, keep it structured and simpler:
- test a narrower grouped layer built only from the strongest clean categories such as `2.02` earnings results and `5.02` / `5.07` governance events
- test a hybrid SEC metadata panel that keeps the EXP-008 coarse filing timing/count features and adds only the strongest grouped `8-K` flags
- do not expand into text NLP until the structured SEC metadata path is exhausted

---

## Experiment ID
EXP-010

### Date
2026-04-11

### Objective
Run the clean Phase 5 universe-expansion test on `event_panel_v2` using the locked Phase 4 setup only:
- wider universe
- same `5`-day excess-return-sign label
- same three anchor models
- no sentiment / no Layer 3 features in the final panel build

### Dataset Version
Official Phase 5 artifacts:
- `data/reference/universe_v2_tickers.csv`
- `docs/universe_v2.md`
- `data/interim/sec/sec_universe_mapping_v2.parquet`
- `data/interim/sec/sec_filing_metadata_v2.parquet`
- `data/interim/event_panel_v2_universe_v2.parquet`
- `docs/event_panel_spec_v2_universe_v2.md`
- `reports/results/event_panel_v2_universe_v2_benchmark.csv`
- `reports/results/event_panel_v2_universe_v2_benchmark.md`

Intermediate non-official artifacts created during failed or aborted attempts:
- `data/interim/sentiment/sec_filing_sentiment_universe_v2.parquet`
- `data/interim/features/layer3_sec_sentiment_features_universe_v2.parquet`

### Universe
`universe_v2` cross-sector large-cap expansion:
- `126` tickers in the successful run
- source list stored in `data/reference/universe_v2_tickers.csv`

### Date Range
Expanded event panel:
- `event_date`: `2014-12-29` to `2024-12-20`
- `effective_model_date`: `2015-01-02` to `2024-12-23`

### Feature Set
Locked feature families only:
- Layer 1 fundamentals snapshot features
- Layer 2 market features ending at `t-1`
- event-context timing features

Explicit no-sentiment setup for the official rerun:
- passed an empty filing-level sentiment parquet into `panel_builder_event_v2.py`
- this left all SEC sentiment columns fully missing and they were automatically excluded in training

Explicit feature exclusions carried forward:
- `gross_margin`
- `current_filing_sentiment_available`

Auto all-missing exclusions in the successful expanded-universe benchmark:
- `sec_sentiment_score`
- `sec_positive_prob`
- `sec_negative_prob`
- `sec_neutral_prob`
- `sec_sentiment_abs`
- `sec_sentiment_change_prev`
- `sec_positive_change_prev`
- `sec_negative_change_prev`
- `sec_chunk_count`
- `sec_log_chunk_count`

### Target
Locked Phase 4 primary label:
- `5`-trading-day excess return sign

### Validation
Locked Phase 4 evaluation policy:
- purged expanding-window CV
- `5` folds
- `2024` final holdout unchanged
- same holdout boundary and model-selection rule as the 34-name anchor

### Models
Locked Phase 4 model set only:
- Logistic Regression
- Random Forest
- XGBoost

### Preprocessing
Unchanged training logic:
- median imputation
- standard scaling for logistic regression
- train-fold missingness filtering
- train-fold clipping
- same benchmark-mode label construction
- same locked holdout policy

Phase 5-specific implementation notes:
- universe expansion required a manual SEC CIK override for `MMC`
- successful clean rerun built SEC metadata first, then rebuilt the expanded event panel with no sentiment input, then trained the locked benchmark matrix

### Hyperparameters
No changes from the locked Phase 4 setup.

### Results
Expanded panel structure:
- Rows: `4,908`
- Tickers: `126`
- Event counts:
  - `10-Q`: `3,682`
  - `10-K`: `1,226`

Per-model benchmark results:
- Logistic Regression:
  - Mean CV AUC: `0.5013`
  - Mean CV log loss: `0.7014`
  - 2024 holdout AUC: `0.5068`
  - 2024 holdout log loss: `0.6945`
- Random Forest:
  - Mean CV AUC: `0.4860`
  - Mean CV log loss: `0.7005`
  - 2024 holdout AUC: `0.4859`
  - 2024 holdout log loss: `0.6979`
- XGBoost:
  - Mean CV AUC: `0.4960`
  - Mean CV log loss: `0.7468`
  - 2024 holdout AUC: `0.5091`
  - 2024 holdout log loss: `0.7100`

Selected primary model:
- `logistic_regression`

Comparison versus the locked 34-name anchor:
- 34-name best CV AUC: `0.5260`
- expanded-universe best CV AUC: `0.5013`
- 34-name best holdout AUC: `0.5237`
- expanded-universe best holdout AUC: `0.5068`

### Observations
- The clean no-sentiment universe expansion run completed successfully after two minimal blocking fixes:
  - manual SEC mapping override for `MMC`
  - event-timing validation adjustment for two valid non-trading-day / pre-panel-start filing rows
- Expanding from `34` to `126` names did not improve the locked benchmark.
- The selected model changed from `random_forest` in the 34-name anchor to `logistic_regression` in the expanded-universe run.
- The successful official Phase 5 result is the no-sentiment rerun, not the earlier aborted sentiment-bearing build attempts.

### Problems
- The first expanded-universe SEC metadata build failed because `MMC` was missing from the manual override table.
- The first clean panel rebuild failed because the validator treated any pre-market or market-hours filing not landing on the same calendar date as an error, even when the raw date was not a tradable session or was before the modeled price panel began.
- Intermediate sentiment / Layer 3 artifacts were created during earlier failed attempts, but they are not part of the official Phase 5 conclusion.

### Decision
Do not promote the expanded universe as the new default research universe. Keep the locked 34-name event-panel anchor as the primary setup for now.

### Next Step
If universe expansion is revisited later, keep the same locked Phase 4 method stack and test whether broader scale only makes sense after stronger features or additional external datasets justify it.
