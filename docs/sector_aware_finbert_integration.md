# Sector-Aware FinBERT Integration

Cristescu et al. (2025) show that sector-specific fine-tuning improves FinBERT sentiment classification quality, but they also find that sentiment is reactive rather than predictive of next-day returns. Therefore, this project treats sector-aware sentiment as a measurement-quality improvement and empirically tests whether it adds incremental predictive lift in a quarterly stock-movement model.

## Why This Matters

The paper reports a material sentiment-classification improvement from sector-specific FinBERT fine-tuning on U.S. sector-tagged financial news headlines. The project should not convert that result into a prediction claim. The relevant research question here is narrower: whether cleaner and more sector-contextual sentiment measurements add incremental information beyond fundamentals and market features in the quarterly event pipeline.

## Initial Implementation

The first implementation is feature-engineering based, not model fine-tuning. `src/features/sector_sentiment_features.py` builds event-level features from a pre-scored news table containing zero-shot FinBERT probabilities:

- ticker-level rolling sentiment means, volatility, counts, momentum, and optional confidence averages
- sector-level rolling sentiment means and counts
- sector-adjusted sentiment and news-share features
- news coverage and low-coverage flags

The builder uses only news where `news_date < event_date` or the available prediction timestamp. Missing sentiment remains `NaN`; coverage flags are emitted so the existing modeling pipeline can use train-fold-only imputation rather than feature-builder backfills.

## Experiment Comparison

The intended comparison is registered in `configs/quarterly/sector_aware_finbert_experiments.yaml`:

- A. `quarterly_core`
- B. `quarterly_core + market`
- C. `quarterly_core + market + zero-shot FinBERT sentiment`
- D. `quarterly_core + market + sector-adjusted FinBERT sentiment`

This ladder tests incremental lift. It should be run only after a point-in-time external-news score table has been scored and merged onto the quarterly feature panel.

## Leakage Controls

The feature builder applies a strict date rule: news is eligible only if its normalized news date is strictly before the event/prediction date. If exact timestamps are unavailable, this conservative date-only rule avoids same-day event leakage. The builder does not use labels, does not fill missing sentiment from future periods, and preserves the input event row count.

## Diagnostics

Use `write_sector_sentiment_feature_diagnostics` to write `outputs/quarterly/diagnostics/sector_sentiment_feature_diagnostics.md`. The report includes event-row counts, 7/30/63-day coverage, sector coverage, average news counts by sector, sentiment-feature missingness, and sparse-coverage warnings.

## Generate News Scores

The sector-aware panel bridge expects a scored news table at `data/processed/news_scores_finbert.parquet`. Generate it from a point-in-time-safe news file with zero-shot FinBERT:

```powershell
python src/build_news_scores_finbert.py `
  --input-news data/raw/analyst/analyst_ratings_processed.csv `
  --output data/processed/news_scores_finbert.parquet `
  --ticker-col stock `
  --date-col date `
  --text-col title `
  --batch-size 16 `
  --device auto
```

For a smoke test, add `--max-rows 25`. The scorer uses `ProsusAI/finbert`, reads the model label mapping from `model.config.id2label`, and writes `ticker`, `date`, `finbert_pos`, `finbert_neu`, `finbert_neg`, `confidence`, and `finbert_score`. This is feature engineering only; it does not fine-tune FinBERT and does not use labels or future returns.

## Build Script Usage

The reproducible bridge from a scored news table to an enriched quarterly panel is:

```powershell
python src/build_sector_sentiment_panel.py `
  --event-panel outputs/quarterly/panels/quarterly_event_panel_features.parquet `
  --news-scores data/processed/news_scores_finbert.parquet `
  --output-panel outputs/quarterly/panels/quarterly_event_panel_sector_sentiment.parquet `
  --diagnostics-output outputs/quarterly/diagnostics/sector_sentiment_feature_diagnostics.md `
  --config-output outputs/quarterly/diagnostics/sector_sentiment_next_steps.md
```

Use `--ticker-sector-map path/to/ticker_sector_map.csv` if the event panel does not already contain a `sector` column. The news scores file must already be point-in-time safe and contain at least `ticker`, a date column, `finbert_pos`, `finbert_neu`, and `finbert_neg`. The script applies a conservative date-only leakage rule: only news with `news_date < event_date` or `prediction_date` is eligible, and same-day news is excluded unless a future implementation adds reliable timestamp and market-close logic.

The script writes the enriched panel and diagnostics only. It does not fine-tune FinBERT and does not run the modeling benchmark. The sector-aware experiment ladder is documented in `configs/quarterly/sector_aware_finbert_experiments.yaml` and should be compared against the `quarterly_core` and `quarterly_core + market` baselines before interpreting any sentiment feature lift.

## Interpretation

SHAP should be used after the benchmark runs to inspect whether sentiment features matter relative to fundamentals and market features. Weak or null sentiment results are valid evidence that better measured sentiment did not add robust quarterly predictive lift in this dataset. Strong results should still be treated as empirical model evidence, not as proof that FinBERT directly predicts stocks.
