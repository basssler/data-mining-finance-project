# Capital IQ Sentiment Final Report

## Locked Verdict

Capital IQ Key Developments sentiment was tested as an incremental event-text feature layer in the quarterly Consumer Staples panel. The within-sector adjusted sentiment model reached 0.6038 holdout AUC versus 0.5020 for the core + market benchmark. However, the lift was not stable across earlier pseudo-holdout years and the bootstrapped 2024 AUC-delta interval crossed zero. Therefore, the evidence supports Capital IQ event-text sentiment as a promising but fragile feature layer, not as a robust standalone predictor.

This is zero-shot FinBERT + Capital IQ Key Developments feature engineering. Because the v1 universe is Consumer Staples, the adjusted features should be described as within-sector relative sentiment, not broad sector-specific tuning.

## Final Model Ladder

| rung   | title                                                       | selected_model      |   cv_auc_mean |   cv_auc_std |   worst_fold_auc |   holdout_auc |   holdout_log_loss |   holdout_f1 |   feature_count |
|:-------|:------------------------------------------------------------|:--------------------|--------------:|-------------:|-----------------:|--------------:|-------------------:|-------------:|----------------:|
| A      | Quarterly core                                              | xgboost             |        0.4878 |       0.0641 |           0.3872 |        0.5501 |             0.7289 |       0.5421 |              24 |
| B      | Quarterly core + market                                     | xgboost             |        0.5304 |       0.064  |           0.4083 |        0.502  |             0.7776 |       0.5357 |              31 |
| C      | Core + market + Capital IQ raw sentiment                    | logistic_regression |        0.4927 |       0.0648 |           0.4012 |        0.5958 |             0.6939 |       0.4792 |              45 |
| D      | Core + market + Capital IQ within-sector adjusted sentiment | logistic_regression |        0.4989 |       0.0669 |           0.4203 |        0.6038 |             0.7115 |       0.3765 |              53 |

The headline lift is in the 2024 holdout: core + market selected xgboost reached 0.5020 AUC, while the within-sector adjusted Capital IQ sentiment rung reached 0.6038 AUC. CV AUC did not improve in parallel, so the holdout improvement is not enough to claim robust general predictive power.

## Apples-to-Apples Model Comparison

| family                           | model_name          |   cv_auc_mean |   cv_auc_std |   worst_fold_auc |   holdout_auc |   holdout_log_loss |   holdout_f1 |   feature_count |
|:---------------------------------|:--------------------|--------------:|-------------:|-----------------:|--------------:|-------------------:|-------------:|----------------:|
| core                             | logistic_regression |        0.4998 |       0.0548 |           0.4099 |        0.518  |             0.7073 |       0.4706 |              24 |
| core                             | random_forest       |        0.4866 |       0.0698 |           0.3595 |        0.518  |             0.6968 |       0.4854 |              24 |
| core                             | xgboost             |        0.4878 |       0.0641 |           0.3872 |        0.5501 |             0.7289 |       0.5421 |              24 |
| core_plus_market                 | logistic_regression |        0.4941 |       0.0679 |           0.4023 |        0.5032 |             0.7173 |       0.3958 |              31 |
| core_plus_market                 | random_forest       |        0.5271 |       0.0654 |           0.3999 |        0.4775 |             0.7066 |       0.5047 |              31 |
| core_plus_market                 | xgboost             |        0.5304 |       0.064  |           0.4083 |        0.502  |             0.7776 |       0.5357 |              31 |
| raw_sentiment                    | logistic_regression |        0.4927 |       0.0648 |           0.4012 |        0.5958 |             0.6939 |       0.4792 |              45 |
| raw_sentiment                    | random_forest       |        0.5136 |       0.0758 |           0.3768 |        0.486  |             0.7001 |       0.4615 |              45 |
| raw_sentiment                    | xgboost             |        0.5163 |       0.0642 |           0.4044 |        0.5445 |             0.7491 |       0.566  |              45 |
| within_sector_adjusted_sentiment | logistic_regression |        0.4989 |       0.0669 |           0.4203 |        0.6038 |             0.7115 |       0.3765 |              53 |
| within_sector_adjusted_sentiment | random_forest       |        0.5052 |       0.0663 |           0.3802 |        0.5217 |             0.699  |       0.4211 |              53 |
| within_sector_adjusted_sentiment | xgboost             |        0.5018 |       0.0635 |           0.4143 |        0.5698 |             0.7353 |       0.5053 |              53 |

The sentiment rung improved 2024 holdout AUC across logistic regression, random forest, and xgboost, but that pattern was weaker in CV. This supports a promising but fragile interpretation.

## Feature Ablation

| ablation                         | selected_model      |   cv_auc_mean |   cv_auc_std |   worst_fold_auc |   holdout_auc |   holdout_log_loss |   feature_count |
|:---------------------------------|:--------------------|--------------:|-------------:|-----------------:|--------------:|-------------------:|----------------:|
| core_plus_market_only            | xgboost             |        0.5304 |       0.064  |           0.4083 |        0.502  |             0.7776 |              31 |
| sentiment_means_only             | logistic_regression |        0.4905 |       0.0683 |           0.397  |        0.5289 |             0.7081 |              33 |
| news_counts_only                 | logistic_regression |        0.481  |       0.0551 |           0.4172 |        0.5128 |             0.7211 |              37 |
| sentiment_momentum_only          | logistic_regression |        0.4928 |       0.0721 |           0.3983 |        0.4992 |             0.7166 |              32 |
| sector_adjusted_only             | xgboost             |        0.509  |       0.0714 |           0.3839 |        0.5742 |             0.7348 |              35 |
| all_capitaliq_sentiment_features | logistic_regression |        0.4989 |       0.0669 |           0.4203 |        0.6038 |             0.7115 |              53 |

The best ablation used all Capital IQ sentiment features. Sector-adjusted sentiment alone beat news-count-only and sentiment-means-only ablations on 2024 holdout AUC, which points more toward event-text sentiment and within-sector context than pure event-volume coverage. The result is still not stable enough to treat as definitive.

## Year-by-Year Stability

|   year | model_name          |   core_plus_market |   within_sector_adjusted_sentiment |   auc_delta_sentiment_minus_control |
|-------:|:--------------------|-------------------:|-----------------------------------:|------------------------------------:|
|   2021 | logistic_regression |             0.4524 |                             0.4924 |                              0.04   |
|   2021 | random_forest       |             0.4858 |                             0.4731 |                             -0.0126 |
|   2021 | xgboost             |             0.531  |                             0.4995 |                             -0.0315 |
|   2022 | logistic_regression |             0.4548 |                             0.4785 |                              0.0237 |
|   2022 | random_forest       |             0.4472 |                             0.4446 |                             -0.0026 |
|   2022 | xgboost             |             0.4265 |                             0.4251 |                             -0.0014 |
|   2023 | logistic_regression |             0.4767 |                             0.4517 |                             -0.025  |
|   2023 | random_forest       |             0.538  |                             0.5224 |                             -0.0156 |
|   2023 | xgboost             |             0.572  |                             0.5309 |                             -0.0411 |
|   2024 | logistic_regression |             0.4527 |                             0.5766 |                              0.1239 |
|   2024 | random_forest       |             0.4459 |                             0.5441 |                              0.0982 |
|   2024 | xgboost             |             0.5553 |                             0.5962 |                              0.0409 |

The sentiment layer helped all three model families in 2024, but the pseudo-holdout years were mixed: logistic regression improved in 2021 and 2022, while 2023 worsened across all three families. This is the main reason not to tune now.

## 2024 Bootstrap AUC Delta

|   iterations | control_model   | sentiment_model     |   control_auc |   sentiment_auc |   mean_delta |   p05_delta |   p95_delta |
|-------------:|:----------------|:--------------------|--------------:|----------------:|-------------:|------------:|------------:|
|         1000 | xgboost         | logistic_regression |        0.5553 |          0.5766 |       0.0192 |      -0.093 |      0.1307 |

The 2024 AUC-delta interval is [-0.0930, 0.1307], so it crosses zero. The holdout result is encouraging, but not statistically stable enough to support a strong predictive claim.

## Feature Importance Stability

| rung                                       | model_name          | feature               |   top3_count_validation |   top3_count_holdout |   mean_importance_when_top3 |
|:-------------------------------------------|:--------------------|:----------------------|------------------------:|---------------------:|----------------------------:|
| capitaliq_ladder_sector_adjusted_sentiment | logistic_regression | sent_mean_30d         |                       4 |                    1 |                      0.5911 |
| capitaliq_ladder_sector_adjusted_sentiment | logistic_regression | news_count_63d        |                       3 |                    1 |                      0.4875 |
| capitaliq_ladder_raw_sentiment             | logistic_regression | news_count_63d        |                       2 |                    1 |                      0.3696 |
| capitaliq_ladder_sector_adjusted_sentiment | random_forest       | sector_news_count_63d |                       3 |                    0 |                      0.0362 |
| capitaliq_ladder_raw_sentiment             | logistic_regression | sent_mean_63d         |                       1 |                    0 |                      0.4655 |
| capitaliq_ladder_raw_sentiment             | logistic_regression | sent_mean_30d         |                       1 |                    0 |                      0.4859 |
| capitaliq_ladder_raw_sentiment             | xgboost             | news_count_63d        |                       1 |                    0 |                      0.0308 |
| capitaliq_ladder_sector_adjusted_sentiment | random_forest       | news_count_63d        |                       1 |                    0 |                      0.0446 |
| capitaliq_ladder_sector_adjusted_sentiment | logistic_regression | sector_news_count_63d |                       1 |                    0 |                      0.749  |
| capitaliq_ladder_sector_adjusted_sentiment | random_forest       | sent_mean_30d         |                       1 |                    0 |                      0.0363 |

`sent_mean_30d` and `news_count_63d` appeared most consistently among the tracked sentiment features. Momentum features did not appear consistently. The evidence is therefore better framed as Capital IQ event-text sentiment plus event-attention context, not as pure sentiment alone.

## Figures

- `outputs\quarterly\modeling\final\figures\ablation_holdout_auc.png`
- `outputs\quarterly\modeling\final\figures\apples_to_apples_holdout_auc.png`
- `outputs\quarterly\modeling\final\figures\bootstrap_auc_delta_interval.png`
- `outputs\quarterly\modeling\final\figures\cv_vs_holdout_auc.png`
- `outputs\quarterly\modeling\final\figures\feature_stability_top3_counts.png`
- `outputs\quarterly\modeling\final\figures\ladder_holdout_auc.png`
- `outputs\quarterly\modeling\final\figures\year_holdout_auc_delta.png`

## Final Interpretation

The final project conclusion should be conservative: the richer Capital IQ Key Developments layer solved the missing-news coverage problem and produced meaningful 2024 holdout lift, but the lift did not survive all stability diagnostics. The mature conclusion is that Capital IQ event-text sentiment appears informative in some market regimes and deserves future testing, but the current evidence is promising rather than robust.

## Future Work

A small appendix-only sensitivity check could tune logistic regression on the within-sector adjusted/all Capital IQ sentiment rung using CV-only selection over `C` and `class_weight`, then report the 2024 holdout once. That should not replace the locked untuned comparison as the main result.
