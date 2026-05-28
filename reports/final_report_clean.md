# Sentiment-Augmented Financial Prediction: A Data Mining Approach to Market Signal Extraction

INEG 41403 Data Mining Final Project Report  
Student: [Name]  
Course: INEG 41403 Data Mining  
Date: [Submission Date]

## Abstract

This project studies whether sentiment features from financial text improve short-term stock price movement prediction when they are added to financial statement features and market-based features. The task is a supervised learning classification problem. The model predicts the probability that a stock will move up over a future horizon.

The way I approached this was to build a layered prediction pipeline. Layer 1 uses financial statement ratios from SEC-derived fundamentals. Layer 2 adds market behavior such as momentum, volatility, volume, risk proxies, and relative strength. Layer 3 adds NLP-derived sentiment features from FinBERT scores on financial news, Capital IQ Key Developments, and SEC filing text.

The current version establishes the pipeline design, feature engineering logic, data integrity checks, leakage-aware validation framework, and several locked modeling artifacts. The strongest current sentiment result is from the Capital IQ within-sector adjusted sentiment layer, which reached a 2024 holdout AUC of 0.6038 compared with 0.5020 for the core plus market benchmark. However, the lift was not stable across pseudo-holdout years, and the bootstrap AUC-delta interval crossed zero. The strongest completed contribution is the pipeline design, feature engineering logic, and leakage-aware validation framework rather than a finished trading model.

## 1. Introduction

At first, this looks like a stock prediction problem, but underneath it is really a data alignment and validation problem. A model can look good for the wrong reason if it accidentally sees future information. The biggest risk in this project is leakage.

This project tests whether financial fundamentals, market behavior, and text sentiment each add useful information for predicting future stock movement. The goal is not to build a completed hedge fund model. The goal is to build a serious data mining pipeline and evaluate whether each feature layer adds marginal predictive value.

## 2. Research Question and Objective

The main research question is:

Does incorporating NLP-derived sentiment features into financial statement and market-based models meaningfully improve short-term stock price movement prediction?

The objective is to compare three feature layers:

| Layer | Feature set | Purpose |
|---|---|---|
| Layer 1 | Financial statement ratios from SEC-derived filings | Fundamentals-only baseline |
| Layer 2 | Layer 1 plus market features | Test whether price and volume context improves the baseline |
| Layer 3 | Layer 2 plus FinBERT sentiment features | Test whether sentiment adds marginal information |

The output is a predicted probability of upward movement. The project also compares model metrics, SHAP feature importance, and a simple cumulative-return-style interpretation. The cumulative return view is not a production trading strategy. It is just a way to explain model predictions more clearly.

## 3. Data Sources

The project combines structured financial data, market data, and text-based sentiment data.

| Data type | Main inputs | Role |
|---|---|---|
| SEC-derived fundamentals | Quarterly filing data and financial statement values | Financial ratios and accounting features |
| Price and volume data | Adjusted prices, returns, volume, volatility | Market behavior features |
| SEC filing text | Filing text and MD&A-style text | Filing sentiment features |
| Capital IQ Key Developments | News-like event text and FinBERT scores | Event-text sentiment features |
| Ticker/date identifiers | Ticker, event date, filing date, period end | Join all feature layers correctly |

The active final package focuses on a Consumer Staples v1 universe of 34 tickers. Results should not be generalized beyond that universe without more testing.

## 4. Feature Engineering

The project uses an additive feature design:

clean data -> engineer features -> align by date -> lag features -> split chronologically -> train models -> evaluate on future data -> compare feature layers -> interpret with SHAP

### Layer 1: Financial Statement Features

Layer 1 includes ratios and accounting features such as current ratio, quick ratio, cash ratio, debt to assets, debt to equity, operating margin, net margin, ROA, ROE, asset turnover, revenue growth, earnings growth, CFO to net income, accruals ratio, and free cash flow features.

The fundamentals pipeline also included an integrity rebuild. This reduced ratio sanity flags sharply:

| Universe | Metric | Before | After |
|---|---:|---:|---:|
| Universe V2 | Feature ratio sanity flags | 10,514 | 181 |
| Universe V2 | Panel ratio sanity flags | 8,260 | 158 |
| Universe V2 | Amount relationship flags | 2,374 | 3 |
| Universe V1 | Feature ratio sanity flags | 2,966 | 63 |
| Universe V1 | Panel ratio sanity flags | 2,402 | 53 |
| Universe V1 | Amount relationship flags | 806 | 0 |

### Layer 2: Market Features

Layer 2 adds market behavior before the prediction point. These features include relative returns, realized volatility, volatility ratios, beta to sector, overnight gap, return shocks, drawdown, return z-score, volume ratio, log volume, and abnormal volume.

Market features did not always help. In the Capital IQ ladder, the core plus market model had better mean CV AUC than the core-only model, but lower 2024 holdout AUC:

| Feature set | Selected model | Mean CV AUC | Holdout AUC |
|---|---|---:|---:|
| Quarterly core | XGBoost | 0.4878 | 0.5501 |
| Quarterly core + market | XGBoost | 0.5304 | 0.5020 |

### Layer 3: Sentiment Features

Layer 3 adds sentiment features from FinBERT-scored text. These include sentiment score, positive probability, negative probability, neutral probability, sentiment change, news counts, text chunk counts, and sector-adjusted sentiment features.

Capital IQ Key Developments improved the coverage problem:

| Coverage window | Overall coverage | 2024 holdout coverage |
|---|---:|---:|
| 7 days | 60.11% | 55.47% |
| 30 days | 88.18% | 86.13% |
| 63 days | 93.77% | 92.70% |

## 5. Preprocessing and Temporal Alignment

The main challenge was not just training a model, but making sure the data was lined up correctly. This project helped me understand that a model can look good for the wrong reason if the timing logic is wrong.

The preprocessing plan includes missing value handling, forward-fill price data instead of backfilling, sector median imputation using training data only, winsorizing outliers at the 1st and 99th percentiles, fitting scalers only on training folds, categorical encoding inside CV folds, and lagging features by at least one trading day.

The assignment design calls for RobustScaler. Some older scripts in the repo use StandardScaler, so the final unified benchmark should make that consistent.

The data layer audit reported:

| Audit status | Count |
|---|---:|
| PASS | 98 |
| REVIEW | 2 |
| INFO | 2 |
| LIMITED | 1 |
| FAIL | 0 |

Adding Capital IQ sentiment did not change the panel row count:

| Panel check | Value |
|---|---:|
| Base row count | 1,108 |
| Sentiment-enriched row count | 1,108 |
| Row count difference | 0 |
| Duplicate event rows | 0 |

## 6. Algorithms and Model Design

The project uses supervised binary classification. The required algorithms are:

| Algorithm | Role in this project |
|---|---|
| Logistic Regression | Interpretable linear classification baseline |
| Random Forest | Nonlinear bagged tree model |
| BaggingClassifier | General bagging baseline; should be added to the final matrix if required |
| AdaBoost or Gradient Boosting | Boosting baseline; older scripts include HistGradientBoosting |
| Lasso Regression | L1-style regularized baseline or L1 logistic model for feature selection |
| Ridge Regression | L2-style regularized baseline or L2 logistic model for stable coefficients |
| XGBoost | Strong boosted-tree model used in active benchmark artifacts |

The current locked final artifacts mainly report Logistic Regression, Random Forest, and XGBoost. Older comparisons also include HistGradientBoosting. BaggingClassifier, AdaBoost, Lasso, and Ridge should not have final numbers claimed unless they are run under the same chronological validation policy.

## 7. Validation Strategy

Normal random k-fold cross-validation is wrong for financial time series because it can train on future data and validate on earlier data. That would make the model look better than it really is.

The correct setup is walk-forward or expanding-window validation. TimeSeriesSplit is the closest scikit-learn idea, but the repo uses custom purged expanding-window logic because the panel has multiple rows per date, a forward-return label horizon, and an embargo requirement.

The active final quarterly config uses:

| Setting | Current value |
|---|---|
| Fold style | Purged expanding-window |
| Number of folds | 5 |
| Embargo | 5 trading dates |
| Label horizon | 63 trading days |
| Holdout start | 2024-01-01 |
| Minimum training dates | 252 |
| Final holdout | 2024 holdout in saved artifact |

The prompt design calls for a final 6-month true holdout and around 63 trading days per test fold. The saved active artifact uses a full 2024 holdout and validation blocks around 82-83 dates. If the exact final requirement is six months and 63-day validation blocks, the splits should be regenerated before claiming that exact setup.

## 8. Evaluation Metrics

Accuracy alone is not enough. The main metrics are:

| Metric | Reason |
|---|---|
| AUC-ROC | Primary metric because it measures ranking quality |
| F1-score | Balances precision and recall |
| Precision | Measures how often predicted up movements are right |
| Recall | Measures how many true up movements are caught |
| Log loss | Measures probability quality |
| Confusion matrix | Shows false positives and false negatives |
| Simple cumulative return simulation | Helps explain predictions, but is not a production trading strategy |

## 9. Results and Current Implementation Status

Real results exist in the repository. The project is still not fully done, so the results should be read as current implementation status and preliminary evidence.

| Component | Status | Completed | Still needs work |
|---|---|---|---|
| Financial features | Mostly complete | Ratios, integrity rebuild, effective-date checks | Continue edge-case audits |
| Market features | Mostly complete | Momentum, volatility, volume, relative strength | Mixed lift needs more testing |
| SEC sentiment | Partially complete | SEC sentiment features and SHAP artifacts | Clearer final integration |
| Capital IQ sentiment | Strongest sentiment path | Coverage diagnostics and final sentiment ladder | Review high-news-count tickers |
| Validation | Strong but wording-sensitive | 5 folds, purge, embargo, 2024 holdout | Exact 6-month holdout if required |
| Models | Partially complete | Logistic Regression, Random Forest, XGBoost, HistGradientBoosting | Add Bagging, AdaBoost, Lasso, Ridge to one final matrix |
| SHAP | Partially complete | SHAP summaries and importance files | Final beeswarm, waterfall, dependence plots |
| Cumulative return simulation | Not finalized | Concept defined | No final artifact found |

### Active 63-Day Quarterly Benchmark

| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | 2024 Holdout AUC | Holdout Log Loss | Holdout F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5022 | 0.0571 | 0.4198 | 0.4992 | 0.7330 | 0.5490 |
| Random Forest | 0.5093 | 0.0711 | 0.3732 | 0.4487 | 0.7080 | 0.5439 |
| XGBoost | 0.5127 | 0.0827 | 0.3522 | 0.5365 | 0.7448 | 0.6018 |

XGBoost was selected as the primary candidate model, but the artifact marks it as candidate-only because the reproducibility threshold failed.

### Capital IQ Layered Sentiment Ladder

| Rung | Feature set | Selected model | Mean CV AUC | Worst Fold AUC | 2024 Holdout AUC | Holdout Log Loss | Holdout F1 |
|---|---|---|---:|---:|---:|---:|---:|
| A | Quarterly core | XGBoost | 0.4878 | 0.3872 | 0.5501 | 0.7289 | 0.5421 |
| B | Quarterly core + market | XGBoost | 0.5304 | 0.4083 | 0.5020 | 0.7776 | 0.5357 |
| C | Core + market + raw sentiment | Logistic Regression | 0.4927 | 0.4012 | 0.5958 | 0.6939 | 0.4792 |
| D | Core + market + within-sector adjusted sentiment | Logistic Regression | 0.4989 | 0.4203 | 0.6038 | 0.7115 | 0.3765 |

The strongest Layer 3 result is the within-sector adjusted sentiment model. It improved 2024 holdout AUC from 0.5020 to 0.6038 compared with the core plus market benchmark. However, CV AUC did not improve in parallel, so this should be described as promising but fragile.

### Stability Check

The 2024 sentiment lift was not stable across all pseudo-holdout years:

| Year | Logistic Regression delta | Random Forest delta | XGBoost delta |
|---:|---:|---:|---:|
| 2021 | 0.0400 | -0.0126 | -0.0315 |
| 2022 | 0.0237 | -0.0026 | -0.0014 |
| 2023 | -0.0250 | -0.0156 | -0.0411 |
| 2024 | 0.1239 | 0.0982 | 0.0409 |

The 2024 bootstrap AUC-delta interval was -0.0930 to 0.1307, so it crossed zero.

## 10. Explainability with SHAP

SHAP helps answer whether sentiment features actually matter after they are added. The final report should include a beeswarm plot, waterfall plot, dependence plot, and feature importance comparison across Layer 1, Layer 2, and Layer 3.

In the active 63-day benchmark, top SHAP features included earnings growth, volatility ratio, volume ratio, realized volatility, log volume, cash ratio, and SEC sentiment probabilities. This suggests that sentiment is not ignored, but it is mixed with financial and market signals.

The Capital IQ final report found that sent_mean_30d and news_count_63d appeared most consistently among tracked sentiment features. That suggests the model may be using both event-text sentiment and event attention.

## 11. Discussion

The main finding is that sentiment can help in the current artifacts, but the evidence is not stable enough to claim the project is finished.

The strongest result is the Capital IQ within-sector adjusted sentiment layer. It improved 2024 holdout AUC compared with the core plus market benchmark. But the CV results were weaker, pseudo-holdout years were mixed, and the bootstrap interval crossed zero.

This is still a useful outcome. The project shows that the real data mining problem is building a point-in-time supervised learning pipeline where fundamentals, market data, and sentiment are aligned correctly. The model only matters if the timing logic is right.

## 12. Limitations

The main limitations are:

- The final universe is only Consumer Staples v1 with 34 tickers.
- The sentiment lift is promising but fragile.
- Capital IQ Key Developments are event text, not all financial news.
- Coverage is uneven across some ticker-years.
- High-news-count tickers need review.
- The saved active artifact uses a full 2024 holdout, not exactly a 6-month holdout.
- Some required algorithms do not yet have final locked metrics under the same validation policy.
- The cumulative return simulation is not finalized.
- Markets are non-stationary, so one year of improvement may not generalize.

## 13. Future Work

Future work should:

1. Rerun the final Layer 1 vs. Layer 2 vs. Layer 3 comparison under one validation contract.
2. Add BaggingClassifier, AdaBoost, Lasso-style, and Ridge-style baselines to the same matrix.
3. Regenerate splits if the final requirement is exactly a 6-month holdout and around 63-day test folds.
4. Add a simple cumulative return simulation against an S&P 500 or SPY benchmark.
5. Create final SHAP beeswarm, waterfall, and dependence plots.
6. Investigate high-news-count tickers and low coverage ticker-years.
7. Test the same method on more sectors.
8. Treat LightGBM, CatBoost, SVM with RBF kernel, PCA/UMAP, LSTM/GRU, TabNet, and stacking as optional future extensions.

## 14. Conclusion

This project tested whether sentiment improves stock movement classification beyond financial statement and market features. The current version establishes a layered pipeline, point-in-time feature alignment, leakage-aware validation, integrity audits, and real model comparison artifacts.

The best evidence for sentiment comes from the Capital IQ Key Developments layer. The within-sector adjusted sentiment model reached a 2024 holdout AUC of 0.6038 compared with 0.5020 for the core plus market benchmark. However, the result was not stable across all diagnostics. The conclusion is not that the model is ready to trade. The conclusion is that sentiment appears promising enough to keep testing, but the evidence is still fragile.

The biggest thing I learned is that financial prediction is not only a modeling problem. It is a data timing problem.

## 15. References

Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.

Breiman, L. (2001). Random forests. Machine Learning, 45, 5-32.

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.

Lundberg, S. M., and Lee, S. I. (2017). A unified approach to interpreting model predictions.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.

U.S. Securities and Exchange Commission. EDGAR company filings database.

Repository artifacts used: final project results summary, Capital IQ sentiment final report, active 63-day benchmark, validation fold summaries, purge audit, data layer integrity audit, SHAP importance files, and layer comparison metrics.

