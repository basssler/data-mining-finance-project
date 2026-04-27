# Final Project Artifact Manifest

## Created Package Files

| artifact | purpose |
|---|---|
| `reports/results/final_project_results_summary.md` | Full final project summary for submission. |
| `reports/results/final_project_results_one_page.md` | One-page executive version. |
| `reports/results/final_project_artifact_manifest.md` | Manifest of final package inputs and outputs. |
| `reports/results/final_project_presentation_outline.md` | Presentation structure and slide-level message. |

## Locked Input Artifacts

| artifact | status | notes |
|---|---|---|
| `outputs/quarterly/diagnostics/fundamental_rebuild/staged_rebuild_summary.md` | present | Primary fundamentals rebuild summary. |
| `outputs/quarterly/diagnostics/fundamental_rebuild/ratio_amount_before_after_summary.csv` | present | Before/after ratio and amount sanity counts. |
| `outputs/quarterly/diagnostics/fundamental_rebuild/*fundamental_unit_consistency_audit.md` | not found | No separate Markdown file with this name was found under the fundamental rebuild diagnostics directory during packaging. |
| `outputs/quarterly/diagnostics/data_layer_integrity/data_layer_integrity_report.md` | present | Final integrity audit narrative. |
| `outputs/quarterly/diagnostics/data_layer_integrity/data_layer_integrity_summary.csv` | present | Row-level audit check summary. |
| `outputs/quarterly/diagnostics/data_layer_integrity/artifact_manifest.csv` | present | Existing generated manifest for major data artifacts. |
| `outputs/quarterly/modeling/final/capitaliq_sentiment_final_report.md` | present | Locked sentiment final report. |
| `outputs/quarterly/modeling/final/capitaliq_sentiment_final_tables.csv` | present | Locked modeling ladder and diagnostics tables. |
| `outputs/quarterly/modeling/final/figures/ablation_holdout_auc.png` | present | Final figure. |
| `outputs/quarterly/modeling/final/figures/apples_to_apples_holdout_auc.png` | present | Final figure. |
| `outputs/quarterly/modeling/final/figures/bootstrap_auc_delta_interval.png` | present | Final figure. |
| `outputs/quarterly/modeling/final/figures/cv_vs_holdout_auc.png` | present | Final figure. |
| `outputs/quarterly/modeling/final/figures/feature_stability_top3_counts.png` | present | Final figure. |
| `outputs/quarterly/modeling/final/figures/ladder_holdout_auc.png` | present | Final figure. |
| `outputs/quarterly/modeling/final/figures/year_holdout_auc_delta.png` | present | Final figure. |
| `configs/event_panel_v2_quarterly_63d_sector_relative.yaml` | present | Active canonical label/config contract. |
| `outputs/quarterly/labels/label_map_excess_63d.parquet` | present | Active 63-day sector-relative label map. |
| `configs/quarterly/current_benchmark_set.yaml` | present | Active benchmark anchor points to the 63-day sector-relative config. |

## Key Data Artifacts Referenced By Existing Manifest

| artifact | role |
|---|---|
| `data/raw/fundamentals/raw_fundamentals.parquet` | Consumer Staples v1 raw fundamentals. |
| `data/interim/fundamentals/fundamentals_quarterly_clean.parquet` | Consumer Staples v1 cleaned quarterly fundamentals. |
| `data/interim/features/layer1_financial_features.parquet` | Consumer Staples v1 financial features. |
| `outputs/quarterly/panels/quarterly_event_panel_features.parquet` | Base quarterly event panel. |
| `data/processed/capitaliq_keydev_news_prepared.parquet` | Prepared Capital IQ Key Developments rows. |
| `data/processed/news_scores_finbert_capitaliq_keydev.parquet` | Zero-shot FinBERT sentiment scores for Capital IQ Key Developments. |
| `outputs/quarterly/panels/quarterly_event_panel_sector_sentiment_capitaliq.parquet` | Sentiment-enriched quarterly event panel. |

## Final Numbers To Carry Into Slides

| metric | value |
|---|---:|
| Data-layer audit PASS | 98 |
| Data-layer audit REVIEW | 2 |
| Data-layer audit INFO | 2 |
| Data-layer audit LIMITED | 1 |
| Data-layer audit FAIL | 0 |
| Overall 7d sentiment coverage | 60.11% |
| Overall 30d sentiment coverage | 88.18% |
| Overall 63d sentiment coverage | 93.77% |
| 2024 30d sentiment coverage | 86.13% |
| 2024 63d sentiment coverage | 92.70% |
| Core + market selected-model holdout AUC | 0.5020 |
| Within-sector adjusted sentiment selected-model holdout AUC | 0.6038 |
| Bootstrap 2024 AUC-delta p05 | -0.0930 |
| Bootstrap 2024 AUC-delta p95 | 0.1307 |

## Packaging Constraints Observed

- No modeling was rerun.
- No tuning, SHAP, Optuna, label construction, split changes, feature changes, or data rebuilds were run.
- The package summarizes locked artifacts only.
- The final language treats Capital IQ sentiment as a promising but fragile feature layer, not as a robust standalone result.
