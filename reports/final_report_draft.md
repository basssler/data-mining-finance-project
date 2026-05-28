# Sentiment-Augmented Financial Prediction: A Data Mining Approach to Market Signal Extraction

**INEG 41403 Data Mining Final Project Report**  
**Student:** [Name]  
**Course:** INEG 41403 Data Mining  
**Project Repository:** `data-mining-finance-project`  
**Date:** [Submission Date]

---

## Abstract

This project studies whether sentiment features from financial text improve short-term stock price movement prediction when they are added to financial statement features and market-based features. The target is a supervised learning classification problem: given information available before a prediction date or event date, predict whether a stock will move up over a short future window.

The way I approached this was to build the project as a layered data mining pipeline instead of treating it like a black-box stock picker. Layer 1 uses financial statement ratios from SEC-derived fundamentals. Layer 2 adds market behavior such as momentum, volatility, volume, risk proxies, and relative strength. Layer 3 adds NLP-derived sentiment features from FinBERT scores on financial news, Capital IQ Key Developments, and SEC filing text. The main research question is whether each new layer adds marginal predictive value compared with the previous layer.

The current version establishes the pipeline design, feature engineering logic, data integrity checks, leakage-aware validation framework, and several locked modeling artifacts. Real results exist in the repository, including a final Capital IQ sentiment ladder and an active 63-trading-day sector-relative benchmark. The most encouraging result is that the Capital IQ within-sector adjusted sentiment layer reached a 2024 holdout AUC of 0.6038 compared with 0.5020 for the core plus market benchmark. However, the lift was not stable across pseudo-holdout years, and the 2024 bootstrap AUC-delta interval crossed zero. Preliminary results should therefore be interpreted cautiously. The strongest completed contribution is the pipeline design, feature engineering logic, and leakage-aware validation framework rather than a finished trading model.

---

## 1. Introduction

At first, this looks like a stock prediction problem, but underneath it is really a data alignment and validation problem. A model can look impressive if it accidentally sees information from the future, uses a filing before it was publicly available, backfills a price, or mixes up ticker/date joins. The biggest risk in this project is leakage.

The project is built around a simple idea: financial fundamentals, market behavior, and text sentiment may each contain some information about future returns. The question is not whether any one feature can predict the market perfectly. The more realistic question is whether adding sentiment to a financial and market feature set improves out-of-sample classification under a fair validation setup.

The layered design matters because it makes the research question measurable. If Layer 1 only uses fundamentals, then Layer 2 should only get credit for improvement after adding market information. Layer 3 should only get credit if sentiment improves performance beyond the financial and market baseline. That structure also makes the project easier to debug because it separates data issues, feature engineering choices, and model behavior.

The project should not be read as a completed hedge fund model. It is a serious data mining project that builds a point-in-time supervised learning pipeline and evaluates whether sentiment adds marginal predictive value. The current results are useful, but they are not strong enough to claim a stable trading edge.

---

## 2. Research Question and Objective

The main research question is:

**Does incorporating NLP-derived sentiment features into financial statement and market-based models meaningfully improve short-term stock price movement prediction?**

The objective is to answer that question using a fair comparison between feature layers:

| Layer | Feature set | Purpose |
|---|---|---|
| Layer 1 | Financial statement features only, mainly ratios from SEC-derived quarterly filings | Establish a fundamentals-only baseline |
| Layer 2 | Layer 1 plus market-derived features such as momentum, volatility, volume, risk proxies, and relative strength | Test whether price/volume context improves the baseline |
| Layer 3 | Layer 2 plus sentiment features from FinBERT-scored financial text | Test whether sentiment adds marginal information |

The output of the modeling task is a predicted probability that a stock will move up over the selected future horizon. The project then compares model performance, SHAP feature importance, and a simple cumulative-return-style interpretation. The cumulative return view is not meant to be a production trading strategy. It is only a presentation-friendly way to show what the classifier's predictions might imply.

---

## 3. Data Sources

The project combines structured financial data, market data, and text-based sentiment data.

| Data type | Main inputs | Role in project |
|---|---|---|
| SEC-derived fundamentals | Quarterly filing data and financial statement values | Build ratios such as liquidity, leverage, profitability, growth, and accrual-related features |
| Price and volume data | Adjusted prices, returns, volume, volatility, and sector-relative return context | Build momentum, volatility, volume, risk, and relative strength features |
| SEC filing text | SEC filing text and MD&A-style text where available | Build filing sentiment features with FinBERT |
| Capital IQ Key Developments / news-like event text | Prepared Capital IQ Key Developments rows and FinBERT scores | Build event-text sentiment, news count, and sector-adjusted sentiment features |
| Ticker/date identifiers | Ticker, event date, filing date, period end, and source identifiers | Join all feature layers into a supervised learning panel |

The active final package in the repository focuses on a Consumer Staples v1 universe of 34 tickers. That matters because the results should not be generalized to the full market yet. The final results are sector-limited and should be treated as a proof of pipeline design plus a narrow empirical test.

The repository also contains an expanded universe path, older event-panel experiments, SEC sentiment experiments, Capital IQ sentiment experiments, and quarterly validation artifacts. I used the checked-in result files rather than inventing final numbers.

---

## 4. Feature Engineering

The main feature engineering decision was to make the feature sets additive. This means each feature layer is supposed to answer a specific question:

- Layer 1: What can financial statements tell us by themselves?
- Layer 2: What changes when market behavior is added?
- Layer 3: What changes when sentiment is added?

The pipeline inputs and outputs can be summarized like this:

| Stage | Description |
|---|---|
| Inputs | SEC filing data, price/volume data, financial news or filing text, ticker/date identifiers |
| Process | clean data -> engineer features -> align by date -> lag features -> split chronologically -> train models -> evaluate on future data -> compare feature layers -> interpret with SHAP |
| Outputs | predicted probability of upward movement, model comparison table, SHAP feature importance, and simple cumulative return comparison |

### 4.1 Layer 1: Financial Statement Features

Layer 1 uses financial statement ratios and accounting-based variables. Examples from the repository include:

- Current ratio
- Quick ratio
- Cash ratio
- Working capital to total assets
- Debt to equity
- Debt to assets
- Long-term debt ratio
- Operating margin
- Net margin
- Return on assets
- Return on equity
- Asset turnover
- Revenue growth
- Earnings growth
- CFO to net income
- Accruals ratio
- Free cash flow features
- Profitability, liquidity, solvency, growth quality, and overall financial health profile scores

The main reason for using ratios instead of raw accounting values is that raw values are hard to compare across companies. A large company and a smaller company may have very different revenue or asset levels, but ratios let the model compare financial structure more fairly.

The project also includes a fundamentals integrity rebuild. This mattered because early financial data can have unit contamination or amount-scale problems. The final summary reports that ratio sanity flags dropped sharply after the staged rebuild:

| Universe | Metric | Before | After |
|---|---:|---:|---:|
| Universe V2 | Feature ratio sanity flags | 10,514 | 181 |
| Universe V2 | Panel ratio sanity flags | 8,260 | 158 |
| Universe V2 | Amount relationship flags | 2,374 | 3 |
| Universe V1 | Feature ratio sanity flags | 2,966 | 63 |
| Universe V1 | Panel ratio sanity flags | 2,402 | 53 |
| Universe V1 | Amount relationship flags | 806 | 0 |

That is one of the stronger completed parts of the project because it shows that the data quality problem was not ignored.

### 4.2 Layer 2: Market Features

Layer 2 adds market-derived features. These are based on price and volume behavior before the prediction point. Examples in the repository include:

- Relative return over 5, 10, and 21 days
- Realized volatility over 21 and 63 days
- Volatility ratio between short and longer windows
- Beta to sector over 63 days
- Overnight gap
- Absolute return shock
- Drawdown over 21 days
- Return z-score
- Volume ratio over 20 days
- Log volume
- Abnormal volume flag
- Event-aware pre-event return and pre-event volume features

The purpose of Layer 2 is to test whether market context improves the fundamentals baseline. In financial prediction this is important because fundamentals may describe company quality, but market behavior can show how investors are already reacting.

One result to be honest about is that market features did not always help. In the Capital IQ final ladder, the core plus market selected model had a higher mean CV AUC than the core-only model, but its 2024 holdout AUC was lower:

| Layer comparison | Selected model | Mean CV AUC | Holdout AUC |
|---|---|---:|---:|
| Quarterly core | XGBoost | 0.4878 | 0.5501 |
| Quarterly core + market | XGBoost | 0.5304 | 0.5020 |

So the market layer showed mixed evidence. This is a useful result because it shows why a layered design is better than just throwing every feature into one model.

### 4.3 Layer 3: Sentiment Features

Layer 3 adds sentiment features from financial text. The repository includes multiple sentiment paths:

- SEC filing sentiment features
- MD&A-style filing sentiment features
- Capital IQ Key Developments sentiment features
- Sector-adjusted sentiment features
- Sentiment means over different lookback windows
- News count and text coverage features
- Sentiment momentum features
- Event-specific filing sentiment features

The text is scored with FinBERT-style financial sentiment outputs. The model produces probabilities such as positive, negative, and neutral probabilities. Those probabilities are then turned into features like:

- Sentiment score
- Positive probability
- Negative probability
- Neutral probability
- Absolute sentiment
- Sentiment change from previous filing/event
- Positive and negative sentiment changes
- Text chunk count
- Log text chunk count
- News count over 30 or 63 days
- Sector-adjusted sentiment means

The Capital IQ Key Developments path helped with one of the biggest practical issues: sparse news coverage. The final package reports:

| Coverage window | Overall coverage | 2024 holdout coverage |
|---|---:|---:|
| 7 days | 60.11% | 55.47% |
| 30 days | 88.18% | 86.13% |
| 63 days | 93.77% | 92.70% |

This means the 30-day and 63-day windows are much more usable than the 7-day window for the current panel.

---

## 5. Preprocessing and Temporal Alignment

The main challenge was not just training a model, but making sure the data was lined up correctly. This project helped me understand that a model can look good for the wrong reason if the timing logic is wrong.

The preprocessing design includes the following rules:

| Step | Approach |
|---|---|
| Missing values | Drop features with too much training-set missingness; impute remaining missing values inside the training fold |
| Price data | Forward-fill price-related continuity where needed, never backfill from future prices |
| Fundamentals | Use filing availability or effective dates so a model only sees fundamentals that would have been known at the time |
| Sector medians | Use sector median imputation only from training data, not from validation or holdout data |
| Outliers | Winsorize or clip financial outliers at the 1st and 99th percentiles based on training data |
| Scaling | Fit scalers only on training folds; the assignment design calls for `RobustScaler`, while some older scripts use `StandardScaler` |
| Categorical variables | Encode categorical variables inside CV folds so validation categories do not leak into training preprocessing |
| Feature lagging | Lag all features by at least one trading day when the feature could otherwise include same-day information |
| Joins | Join by ticker/date/event identifiers and audit row counts so sentiment does not silently add or remove events |

The current repo includes strong alignment checks. The data layer integrity audit reports:

| Audit status | Count |
|---|---:|
| PASS | 98 |
| REVIEW | 2 |
| INFO | 2 |
| LIMITED | 1 |
| FAIL | 0 |

The audit also reports that adding Capital IQ sentiment did not change the panel row count:

| Panel check | Value |
|---|---:|
| Base row count | 1,108 |
| Sentiment-enriched row count | 1,108 |
| Row count difference | 0 |
| Duplicate event rows | 0 |
| Market rows with market-as-of date after event date | 0 |

The two review items were suspiciously high news counts for selected tickers and low coverage in some ticker-years. Those do not invalidate the whole project, but they should be documented because they could affect the sentiment layer.

---

## 6. Algorithms and Model Design

The project uses a supervised binary classification setup. The target is whether the stock's future movement is positive under the selected label definition. The models are compared using the same feature layer and validation policy.

### 6.1 Logistic Regression

Logistic Regression is the main interpretable linear baseline. It is useful because it gives a simple comparison point before using tree ensembles. It is also less likely to overfit small samples than a highly flexible model, although it can still overfit if preprocessing or feature selection leaks information.

In the final Capital IQ sentiment ladder, Logistic Regression was the selected model for the raw sentiment and within-sector adjusted sentiment rungs. That is interesting because the sentiment-added feature set did not need the most complex model to show a holdout lift.

### 6.2 Random Forest

Random Forest is a bagged tree ensemble. It trains many decision trees on bootstrap samples and averages their predictions. It can capture nonlinear relationships and feature interactions without requiring the same scaling assumptions as linear models.

Random Forest appears in many repository artifacts. It was especially useful in earlier sentiment and event-specific comparisons, including the Phase 9 event-specific sentiment run.

### 6.3 BaggingClassifier

BaggingClassifier is the more general version of the bagging idea. Random Forest is basically a specialized bagging approach using decision trees plus feature randomness. I include BaggingClassifier as a useful comparison model because it can test whether variance reduction by bootstrap aggregation helps even without the extra random feature selection used by Random Forest.

The current locked final result files focus mainly on Logistic Regression, Random Forest, XGBoost, and some gradient boosting variants. BaggingClassifier should be treated as a model-design candidate unless it is added to the final benchmark matrix and rerun under the same walk-forward validation.

### 6.4 AdaBoost or Gradient Boosting

Boosting builds models sequentially, where each new learner focuses more on mistakes from the previous learners. The repository includes HistGradientBoosting in older layer-comparison scripts and XGBoost in the main event-panel benchmark scripts. AdaBoost would be another boosting baseline, but the current final locked artifacts mainly report gradient boosting and XGBoost-style models.

Boosting can be helpful for this project because financial signals may be nonlinear. However, boosting can also overfit badly if the time split or feature timing is wrong. That is why validation design matters more than model complexity.

### 6.5 Lasso Regression

Lasso Regression uses L1 regularization, which can shrink some coefficients to zero. Since the main task is classification, the practical version for this project is an L1-regularized logistic model or a Lasso-style auxiliary baseline. The reason to include it is feature selection. With many financial ratios, market features, and sentiment columns, Lasso can help identify whether a small subset of features carries most of the signal.

No final Lasso performance number should be claimed unless it is run under the same chronological validation policy.

### 6.6 Ridge Regression

Ridge Regression uses L2 regularization, which shrinks coefficients but usually keeps all features. In a classification version, this is similar to L2-regularized Logistic Regression. Ridge-style regularization is useful when many features are correlated, which happens often with accounting ratios and rolling market features.

Ridge is less aggressive than Lasso and is usually a good stable baseline when feature sets are noisy.

### 6.7 XGBoost

XGBoost is a gradient boosted tree algorithm. It is one of the stronger algorithms in the repository and appears in the active quarterly benchmark configuration. The active 63-day sector-relative benchmark selected XGBoost as the primary candidate model, although it was marked `candidate_only` because the reproducibility threshold failed.

This is a good example of why the report should not overstate the project. XGBoost gave the best holdout AUC in that active benchmark, but the promotion metadata says it should not be treated as a fully stable champion yet.

---

## 7. Validation Strategy

Normal random k-fold cross-validation is wrong for this project. Random k-fold would shuffle rows across time, which means the training set could contain data from after the validation set. In finance, that is a major problem because market regimes change and because future data can leak through preprocessing, rolling windows, sector statistics, labels, or duplicated event records.

The correct approach is chronological validation. The intended structure is walk-forward or expanding-window validation:

1. Train on earlier dates.
2. Leave a gap or embargo before validation.
3. Validate on later dates.
4. Expand the training window.
5. Repeat for multiple folds.
6. Keep a final holdout period that is not used for model selection.

`TimeSeriesSplit` is the scikit-learn concept closest to this setup, but the project needs more than plain `TimeSeriesSplit`. The repository uses custom purged expanding-window logic because the panel has multiple rows per date, a forward-return label horizon, and an embargo requirement.

The active final quarterly config uses:

| Validation setting | Current value in active artifact |
|---|---|
| Fold style | Purged expanding-window / walk-forward |
| Number of folds | 5 |
| Embargo | 5 trading dates in saved validation artifacts |
| Label horizon | 63 trading days |
| Holdout start | 2024-01-01 |
| Minimum training dates | 252 |
| Final holdout | 2024 holdout in the saved artifact |

The prompt design also calls for a final 6-month true holdout set and around 63 trading days per test fold. The saved active artifact uses a full 2024 holdout instead of a 6-month holdout, and the saved fold summaries show validation blocks around 82-83 unique dates in the active 63-day benchmark. That is not something I want to hide. If the exact assignment requirement is a 6-month holdout and approximately 63 trading days per validation fold, the split generator should be adjusted and rerun before claiming that exact setup. The current artifact is still leakage-aware, but the wording should match the saved files.

The saved purge audit for the active 63-day benchmark reports:

| Fold | Train date count | Validation date count | Purged date count | Overlap purge date count | Embargo date count |
|---|---:|---:|---:|---:|---:|
| Fold 1 | 185 | 83 | 67 | 62 | 5 |
| Fold 2 | 268 | 83 | 67 | 62 | 5 |
| Fold 3 | 351 | 83 | 67 | 62 | 5 |
| Fold 4 | 434 | 83 | 67 | 62 | 5 |
| Fold 5 | 517 | 82 | 67 | 62 | 5 |
| Holdout | 599 | 89 | 67 | 62 | 5 |

This validation design is one of the most important parts of the project. It is the difference between a model that is tested on future-like data and a model that accidentally learns from the future.

---

## 8. Evaluation Metrics

Accuracy by itself is not enough for this project. Stock movement labels can be noisy and close to balanced, and a model can get a decent accuracy without ranking probabilities well.

The main metrics are:

| Metric | Why it matters |
|---|---|
| AUC-ROC | Primary metric because it measures ranking quality across probability thresholds |
| F1-score | Balances precision and recall, useful when one class is harder to capture |
| Precision | Measures how often predicted up movements are actually up |
| Recall | Measures how many actual up movements the model catches |
| Log loss | Measures probability calibration and penalizes overconfident wrong predictions |
| Confusion matrix | Shows false positives and false negatives directly |
| Simple cumulative return simulation | Helps interpret predictions in a presentation-friendly way, but is not a production trading strategy |

The cumulative return simulation should be treated carefully. It can show what might happen if the model's signals were turned into a simple rule, but it does not include all trading realities. It should not be described as a deployable strategy, and it should not be the main proof of model quality.

---

## 9. Results / Current Implementation Status and Preliminary Results

Real results exist in the repository, so this section uses those artifacts. I am still not treating the project as fully finished. The full Layer 1 vs. Layer 2 vs. Layer 3 comparison is still being finalized because the repository contains multiple experiment lanes, label definitions, and sentiment sources. The cleanest final sentiment result is the Capital IQ ladder, and the active final benchmark is the 63-trading-day sector-relative quarterly benchmark.

### 9.1 Current Implementation Status

| Component | Status | What was completed | What still needs work |
|---|---|---|---|
| Financial statement feature pipeline | Mostly complete | SEC-derived quarterly ratios, fundamentals integrity rebuild, ratio sanity improvements, effective-date checks | Continue auditing edge cases and keep only point-in-time available fundamentals |
| Market feature pipeline | Mostly complete | Momentum, volatility, volume, drawdown, beta/sector-relative features, event-aware market features | Market layer showed mixed lift; more stability testing is needed |
| SEC filing sentiment | Partially complete | SEC sentiment features and SHAP artifacts exist in event-panel outputs | Needs clearer final integration into one consistent Layer 1/2/3 report table |
| Capital IQ / news-like sentiment | Strongest sentiment path so far | Capital IQ Key Developments scoring, coverage diagnostics, sector-adjusted sentiment ladder, final report artifacts | High-news-count tickers and low-coverage ticker-years need review |
| Validation framework | Strong but wording-sensitive | 5-fold purged expanding validation, 5-date embargo, 63-day label horizon, 2024 holdout, saved fold and purge audits | If the course requires exactly a 6-month holdout and around 63-day test folds, rerun splits to match that exactly |
| Model training | Partially complete | Logistic Regression, Random Forest, XGBoost, HistGradientBoosting in older comparisons, tuning artifacts for some model families | BaggingClassifier, AdaBoost, Lasso, and Ridge should be added to a single final benchmark matrix if required for final numeric comparison |
| Explainability | Partially complete | SHAP summary and importance CSVs exist for several runs; sentiment importance stability was summarized | Add final beeswarm, waterfall, and dependence plots for the exact selected final Layer 3 model |
| Cumulative return simulation | Not finalized | Concept is included as an interpretation tool | No final cumulative return artifact was found, so no return numbers are claimed |
| Final report | In progress | This report consolidates current locked artifacts | Final formatting and any instructor-specific requirements still need review |

### 9.2 Active 63-Day Sector-Relative Quarterly Benchmark

The active benchmark config is `configs/event_panel_v2_quarterly_63d_sector_relative.yaml`. It uses a 63-trading-day sector-relative sign label and a 2024 holdout. The saved benchmark reports:

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | 2024 Holdout AUC | Holdout Log Loss | Holdout F1 | Promotion |
|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.5022 | 0.0571 | 0.4198 | 0.4992 | 0.7330 | 0.5490 | Reference only |
| Random Forest | 0.5093 | 0.0711 | 0.3732 | 0.4487 | 0.7080 | 0.5439 | Reference only |
| XGBoost | 0.5127 | 0.0827 | 0.3522 | 0.5365 | 0.7448 | 0.6018 | Candidate only |

The selected primary model in that artifact is XGBoost, with a 2024 holdout AUC of 0.5365. However, the artifact marks it as `candidate_only` because the reproducibility threshold failed. That means the model is useful evidence, but not a final stable champion.

### 9.3 Capital IQ Layered Sentiment Ladder

The Capital IQ sentiment final report is the cleanest current evidence for the main research question. It compares a core feature set, a core plus market feature set, and two sentiment-added feature sets.

| Rung | Feature set | Closest project layer | Selected model | Mean CV AUC | Worst Fold AUC | 2024 Holdout AUC | Holdout Log Loss | Holdout F1 | Feature count |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| A | Quarterly core | Layer 1 | XGBoost | 0.4878 | 0.3872 | 0.5501 | 0.7289 | 0.5421 | 24 |
| B | Quarterly core + market | Layer 2 | XGBoost | 0.5304 | 0.4083 | 0.5020 | 0.7776 | 0.5357 | 31 |
| C | Core + market + Capital IQ raw sentiment | Layer 3 | Logistic Regression | 0.4927 | 0.4012 | 0.5958 | 0.6939 | 0.4792 | 45 |
| D | Core + market + Capital IQ within-sector adjusted sentiment | Layer 3 | Logistic Regression | 0.4989 | 0.4203 | 0.6038 | 0.7115 | 0.3765 | 53 |

The headline result is the holdout comparison between Layer 2 and the strongest Layer 3 rung:

| Comparison | 2024 Holdout AUC |
|---|---:|
| Core + market selected model | 0.5020 |
| Within-sector adjusted sentiment selected model | 0.6038 |

That is a meaningful holdout lift in the saved artifact. But the CV AUC did not improve in parallel. The within-sector adjusted sentiment rung had mean CV AUC of 0.4989, while the core plus market rung had mean CV AUC of 0.5304. Because of that, the result should be described as promising but fragile.

### 9.4 Apples-to-Apples Model Comparison

The same final report also compares model families across feature sets:

| Feature family | Model | Mean CV AUC | Holdout AUC | Holdout Log Loss | Holdout F1 |
|---|---|---:|---:|---:|---:|
| Core | Logistic Regression | 0.4998 | 0.5180 | 0.7073 | 0.4706 |
| Core | Random Forest | 0.4866 | 0.5180 | 0.6968 | 0.4854 |
| Core | XGBoost | 0.4878 | 0.5501 | 0.7289 | 0.5421 |
| Core + market | Logistic Regression | 0.4941 | 0.5032 | 0.7173 | 0.3958 |
| Core + market | Random Forest | 0.5271 | 0.4775 | 0.7066 | 0.5047 |
| Core + market | XGBoost | 0.5304 | 0.5020 | 0.7776 | 0.5357 |
| Raw sentiment | Logistic Regression | 0.4927 | 0.5958 | 0.6939 | 0.4792 |
| Raw sentiment | Random Forest | 0.5136 | 0.4860 | 0.7001 | 0.4615 |
| Raw sentiment | XGBoost | 0.5163 | 0.5445 | 0.7491 | 0.5660 |
| Within-sector adjusted sentiment | Logistic Regression | 0.4989 | 0.6038 | 0.7115 | 0.3765 |
| Within-sector adjusted sentiment | Random Forest | 0.5052 | 0.5217 | 0.6990 | 0.4211 |
| Within-sector adjusted sentiment | XGBoost | 0.5018 | 0.5698 | 0.7353 | 0.5053 |

One useful pattern is that sentiment improved 2024 holdout AUC for all three model families in the within-sector adjusted sentiment setup compared with core plus market. However, this pattern was weaker in CV. That is exactly why I would not describe the result as final proof.

### 9.5 Stability Diagnostics

The year-by-year pseudo-holdout diagnostics were mixed:

| Year | Model | Core + market AUC | Within-sector sentiment AUC | Sentiment minus control |
|---:|---|---:|---:|---:|
| 2021 | Logistic Regression | 0.4524 | 0.4924 | 0.0400 |
| 2021 | Random Forest | 0.4858 | 0.4731 | -0.0126 |
| 2021 | XGBoost | 0.5310 | 0.4995 | -0.0315 |
| 2022 | Logistic Regression | 0.4548 | 0.4785 | 0.0237 |
| 2022 | Random Forest | 0.4472 | 0.4446 | -0.0026 |
| 2022 | XGBoost | 0.4265 | 0.4251 | -0.0014 |
| 2023 | Logistic Regression | 0.4767 | 0.4517 | -0.0250 |
| 2023 | Random Forest | 0.5380 | 0.5224 | -0.0156 |
| 2023 | XGBoost | 0.5720 | 0.5309 | -0.0411 |
| 2024 | Logistic Regression | 0.4527 | 0.5766 | 0.1239 |
| 2024 | Random Forest | 0.4459 | 0.5441 | 0.0982 |
| 2024 | XGBoost | 0.5553 | 0.5962 | 0.0409 |

The 2024 results look encouraging, but 2023 worsened across all three models. The 2024 bootstrap AUC-delta interval was:

| Iterations | Control model | Sentiment model | Control AUC | Sentiment AUC | Mean delta | 5th percentile delta | 95th percentile delta |
|---:|---|---|---:|---:|---:|---:|---:|
| 1,000 | XGBoost | Logistic Regression | 0.5553 | 0.5766 | 0.0192 | -0.0930 | 0.1307 |

Because the interval crosses zero, the result is not statistically stable enough to support a strong predictive claim.

### 9.6 Older Preliminary Layer Comparison Artifacts

The repository also has older layer comparison files under `outputs/comparison`. These use a different split date and should not replace the active final benchmark, but they are useful preliminary evidence:

| Artifact comparison | Best model | AUC-ROC | F1 | Log loss | Notes |
|---|---|---:|---:|---:|---|
| Layer 1 only | Random Forest | 0.5138 | 0.3892 | 0.6933 | Fundamentals-only baseline |
| Layer 1 + Layer 2 full market | HistGradientBoosting | 0.5047 | 0.5702 | 0.7122 | Market features did not clearly improve AUC |
| Layer 1 + SEC sentiment | HistGradientBoosting | 0.5129 | 0.6291 | 0.6964 | SEC sentiment helped F1 more than AUC |
| Layer 1 + MDA sentiment | Random Forest | 0.5127 | 0.4848 | 0.6924 | MDA sentiment was similar to Layer 1 on AUC |

These should be interpreted cautiously because they are not the same final validation contract as the Capital IQ ladder and the active 63-day quarterly benchmark.

---

## 10. Explainability with SHAP

SHAP is important in this project because it helps answer whether sentiment features actually matter after they are added. If the Layer 3 model improves but SHAP shows that the model mostly depends on market volatility or a data availability flag, then sentiment may not really be the reason for the improvement.

The project uses SHAP-style interpretation in several places, including SHAP summary plots and SHAP importance CSV files. For a complete final version, I would include:

| SHAP output | Purpose |
|---|---|
| Beeswarm plot | Shows which features have high SHAP values and whether high or low feature values push predictions up or down |
| Waterfall plot | Explains one specific prediction from baseline probability to final probability |
| Dependence plot | Shows how one feature's value relates to its SHAP contribution |
| Feature importance comparison | Compares Layer 1, Layer 2, and Layer 3 to see whether sentiment features become important |

In the active 63-day benchmark SHAP importance file, the top features included a mix of earnings growth, volatility, volume, cash ratio, overnight gap, and SEC sentiment probabilities. Some of the top SHAP features were:

| Feature | Mean absolute SHAP |
|---|---:|
| earnings_growth_qoq | 0.2047 |
| vol_ratio_21d_63d | 0.1827 |
| volume_ratio_20d | 0.1338 |
| realized_vol_63d | 0.1296 |
| log_volume | 0.1285 |
| cash_ratio | 0.1284 |
| sec_negative_prob | 0.1128 |
| sec_neutral_prob | 0.0969 |
| sec_positive_change_prev | 0.0873 |

This suggests that sentiment features are not ignored, but they are mixed with market and financial variables. In the Capital IQ final report, `sent_mean_30d` and `news_count_63d` appeared most consistently among tracked sentiment features. That supports a careful interpretation: the model may be picking up both event-text tone and event attention, not pure sentiment alone.

---

## 11. Discussion

The main finding is that sentiment can help in the current artifacts, but the evidence is not stable enough to claim the problem is solved.

The strongest positive result is the Capital IQ within-sector adjusted sentiment rung. It improved 2024 holdout AUC from 0.5020 for core plus market to 0.6038 for core plus market plus adjusted sentiment. That is a meaningful difference in the saved result file. The sentiment layer also improved 2024 holdout AUC across Logistic Regression, Random Forest, and XGBoost in the apples-to-apples comparison.

The main problem is stability. The CV AUC did not improve in the same way as the holdout AUC, and pseudo-holdout years were mixed. The bootstrap AUC-delta interval crossed zero. So the result should be framed as "promising but fragile."

This is still a useful project outcome. A data mining project does not need to prove that a model can beat the market. It needs to show a valid pipeline, a fair evaluation setup, and an honest interpretation of results. The biggest technical lesson is that financial ML is very sensitive to timing. If features are not point-in-time, if rows are joined incorrectly, or if preprocessing uses future data, the model can look good for the wrong reason.

The layered setup also helped with interpretation. It showed that adding market features did not automatically help the holdout result. It also showed that sentiment results depend heavily on the source and construction of the sentiment features. Capital IQ event text appears more useful than sparse short-window news coverage, but the signal still needs more testing.

---

## 12. Limitations

The project has several important limitations:

1. The final universe is narrow. The strongest final package is based on Consumer Staples v1 with 34 tickers.
2. The sentiment lift is fragile. The 2024 holdout result is encouraging, but pseudo-holdout years were mixed.
3. Sentiment coverage is uneven. The 30-day and 63-day windows are usable overall, but some ticker-years still have low coverage.
4. Capital IQ Key Developments are not the same as all financial news. They are event-text items and may have different coverage patterns.
5. Some high-news-count tickers need review. The audit flagged WMT, KO, PEP, and DLTR for suspiciously high counts.
6. The current active artifact uses a 2024 holdout, not an exact 6-month holdout. If the final class requirement is exactly six months, that split should be regenerated.
7. The saved active fold sizes are not exactly around 63 validation dates. The label horizon is 63 trading days, but the validation blocks in the active artifact are around 82-83 dates.
8. Some required algorithms are discussed but not all have final locked metrics under the same validation policy.
9. The cumulative return simulation has not been finalized in the saved artifacts, so no return numbers are claimed.
10. Financial markets are non-stationary. A relationship that works in one year can fail in another year.

---

## 13. Future Work

The next steps are:

1. Rerun the final Layer 1 vs. Layer 2 vs. Layer 3 comparison under one single validation contract.
2. Add BaggingClassifier, AdaBoost, L1-regularized Logistic Regression, and L2-regularized Logistic Regression or Ridge-style baselines to the same final benchmark matrix.
3. If required, regenerate validation splits with exactly a final 6-month holdout and around 63 trading days per validation fold.
4. Add a final cumulative return simulation against an S&P 500 or SPY benchmark, clearly labeled as interpretation only.
5. Build final SHAP beeswarm, waterfall, and dependence plots for the selected Layer 3 model.
6. Investigate high-news-count tickers and low sentiment coverage ticker-years.
7. Test the same protocol on additional sectors before making broader claims.
8. Add stricter saved validation artifacts for every experiment family, including fold maps and purge audits.
9. Explore LightGBM and CatBoost as optional model extensions.
10. Explore SVM with RBF kernel, PCA/UMAP, LSTM/GRU, TabNet, and stacking ensembles as future work only after the baseline pipeline is locked.

I would not start with LSTM, GRU, TabNet, or stacking as the next immediate step. Those are more complex and could hide basic data issues. The priority should be making the final comparison clean and leakage-aware.

---

## 14. Conclusion

This project tested whether sentiment features improve short-term stock movement classification beyond financial statement and market features. The current version establishes a layered pipeline, point-in-time feature alignment, leakage-aware validation, integrity audits, and real model comparison artifacts.

The best evidence for sentiment comes from the Capital IQ Key Developments layer. The within-sector adjusted sentiment model reached a 2024 holdout AUC of 0.6038 compared with 0.5020 for the core plus market benchmark. However, the result was not stable across all validation diagnostics, and the bootstrap interval crossed zero. So the conclusion is not that the model is ready to trade. The conclusion is that sentiment appears promising enough to keep testing, but the evidence is fragile.

The biggest thing I learned is that financial prediction is not only a modeling problem. It is a data timing problem. The model is only meaningful if every feature is aligned to what would have been known at the time. The current project is strongest as a data mining pipeline and validation framework, and the final model results should be presented as preliminary but useful evidence.

---

## 15. References

Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.

Breiman, L. (2001). Random forests. Machine Learning, 45, 5-32.

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.

Lundberg, S. M., and Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

U.S. Securities and Exchange Commission. EDGAR company filings database.

Repository artifacts used:

- `reports/results/final_project_results_summary.md`
- `reports/results/capitaliq_sentiment_final_report.md`
- `outputs/quarterly/modeling/final/capitaliq_sentiment_final_report.md`
- `reports/results/event_panel_v2_quarterly_63d_sector_relative_benchmark.md`
- `reports/results/event_panel_v2_quarterly_63d_sector_relative_benchmark.csv`
- `outputs/quarterly/validation/event_panel_v2_quarterly_63d_sector_relative/fold_summary.csv`
- `outputs/quarterly/validation/event_panel_v2_quarterly_63d_sector_relative/purge_audit.csv`
- `outputs/quarterly/diagnostics/data_layer_integrity/data_layer_integrity_report.md`
- `reports/results/event_panel_v2_quarterly_63d_sector_relative_shap_importance.csv`
- `outputs/comparison/layer_comparison_metrics.json`
- `outputs/comparison/layer3_comparison_metrics.json`
- `outputs/comparison/layer3_mda_comparison_metrics.json`

---

## Appendix A: Under-the-Hood Pipeline Summary

| Pipeline step | What happens |
|---|---|
| 1. Load data | Load SEC filing data, price/volume data, sentiment text, and ticker/date identifiers |
| 2. Clean data | Normalize dates, tickers, numeric columns, and missing values |
| 3. Engineer Layer 1 | Build financial statement ratios and quality scores |
| 4. Engineer Layer 2 | Build momentum, volatility, volume, and sector-relative market features |
| 5. Engineer Layer 3 | Score text with FinBERT and aggregate sentiment features |
| 6. Align by time | Join features to event dates using only available past information |
| 7. Lag features | Lag features at least one trading day where needed |
| 8. Split data | Use chronological expanding-window validation with purging and embargo |
| 9. Train models | Fit models inside each fold using training-only preprocessing |
| 10. Evaluate | Compute AUC-ROC, F1, precision, recall, log loss, confusion matrix, and holdout results |
| 11. Interpret | Use SHAP to inspect which feature layers matter |

## Appendix B: Main Failure Points

| Failure point | Why it matters | Current control |
|---|---|---|
| Look-ahead leakage | Makes the model look better than it really is | Purged chronological validation and training-only preprocessing |
| Sparse sentiment data | Sentiment features may be missing or biased toward covered companies | Coverage diagnostics by window and holdout |
| Bad ticker/date joins | Can attach text or fundamentals to the wrong event | Panel identity and row-count audits |
| Overfitting | Flexible models can memorize noisy patterns | Holdout set, CV folds, model comparison, and bootstrap diagnostics |
| Non-stationary markets | Relationships can change across years | Pseudo-holdout year diagnostics |
| Weak out-of-sample performance | A model may not generalize | 2024 holdout and cautious interpretation |

## Appendix C: Validation Checks

| Check | Current status |
|---|---|
| Chronological splits | Implemented |
| 5-fold expanding-window validation | Implemented |
| 5-day embargo | Implemented in saved artifacts |
| Final holdout | Implemented as 2024 holdout |
| Training-only preprocessing | Implemented in multiple scripts; should be preserved in final matrix |
| Layer 1 vs. Layer 2 vs. Layer 3 comparison | Partially implemented; final unified comparison still being finalized |
| No future data in training | Main design goal; supported by audits and split logic |

