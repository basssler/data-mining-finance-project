# Final Project Results Summary

## Project Objective

This project tests whether event-aligned financial fundamentals, market context, and Capital IQ Key Developments sentiment can improve quarterly filing-event classification for the Consumer Staples v1 universe. The final task is intentionally narrow: assemble evidence from locked artifacts, not rerun modeling, tuning, label construction, split changes, feature changes, or data rebuilds.

## Final Label Contract

The active project contract is the **63-trading-day sector-relative sign label**.

| field | final setting |
|---|---|
| Active config | `configs/event_panel_v2_quarterly_63d_sector_relative.yaml` |
| Active benchmark anchor | `configs/quarterly/current_benchmark_set.yaml` |
| Label variant | `event_v2_63d_sector_relative_sign` |
| Label artifact | `outputs/quarterly/labels/label_map_excess_63d.parquet` |
| Horizon | 63 trading days |
| Target mode | sign |
| Benchmark mode | sector equal weight, excluding self |
| Holdout start | 2024-01-01 |
| Validation policy | 5-fold purged expanding window with 2024 holdout |

The active universe is **Consumer Staples v1, 34 tickers**. Results should not be generalized beyond this universe without additional testing.

## Data Layers

The final package covers three locked data layers:

| layer | role |
|---|---|
| Fundamentals | Quarterly SEC-derived accounting features aligned to filing events. |
| Market | Event-time market context and benchmark-relative return setup. |
| Capital IQ Key Developments sentiment | Event-text layer built from zero-shot FinBERT + Capital IQ Key Developments feature engineering. |

The sentiment features are interpreted as **within-sector relative sentiment**, not as sector-specific tuning.

## Fundamentals Integrity Fix Summary

The staged fundamentals rebuild corrected the main unit-contamination issue before final packaging. It rebuilt raw fundamentals, clean fundamentals, Layer 1 financial features, and event panels for Universe V2 and Consumer Staples v1 without running modeling, SHAP, Optuna, sentiment scoring, or tuning.

| universe | metric | before | after |
|---|---:|---:|---:|
| Universe V2 | feature ratio sanity flags | 10,514 | 181 |
| Universe V2 | panel ratio sanity flags | 8,260 | 158 |
| Universe V2 | amount relationship flags | 2,374 | 3 |
| Universe V1 | feature ratio sanity flags | 2,966 | 63 |
| Universe V1 | panel ratio sanity flags | 2,402 | 53 |
| Universe V1 | amount relationship flags | 806 | 0 |

Effective-date diagnostics reported zero selected facts after cutoff for both Universe V2 and Universe V1. The final Consumer Staples v1 quarterly panel retained 1,108 rows after filtering one duplicate same-quarter KHC original filing.

## Data Integrity and Leakage Controls

The data-layer audit reports:

| status | count |
|---|---:|
| PASS | 98 |
| REVIEW | 2 |
| INFO | 2 |
| LIMITED | 1 |
| FAIL | 0 |

Key controls documented in the audit include panel identity preservation, no row-count change from adding sentiment, duplicate event-row checks, feature-family isolation, label-field preservation, market as-of checks, no future-return-like panel columns, purged expanding-window validation, embargo coverage for the 63-day label horizon, and separate 2024 holdout evaluation.

Remaining review items are suspiciously high news counts for selected tickers and low coverage in some ticker-years. The limited item is missing fold-map/purge-audit/fold-summary artifacts for `sector_aware_finbert_capitaliq_experiments`; the main Capital IQ ladder artifacts have validation checks recorded.

## Capital IQ Sentiment Coverage Summary

Capital IQ Key Developments coverage solved the most severe missing-news issue at longer windows.

| coverage window | overall coverage | 2024 holdout coverage |
|---|---:|---:|
| 7d | 60.11% | 55.47% |
| 30d | 88.18% | 86.13% |
| 63d | 93.77% | 92.70% |

The 30d and 63d windows are strong enough for a feature-layer test, but coverage remains uneven across ticker-years.

## Modeling Ladder Summary

The final Capital IQ sentiment ladder is locked and untuned.

| rung | feature set | selected model | holdout AUC | feature count |
|---|---|---|---:|---:|
| A | Quarterly core | xgboost | 0.5501 | 24 |
| B | Quarterly core + market | xgboost | 0.5020 | 31 |
| C | Core + market + Capital IQ raw sentiment | logistic regression | 0.5958 | 45 |
| D | Core + market + Capital IQ within-sector adjusted sentiment | logistic regression | 0.6038 | 53 |

The headline comparison is between the core + market selected model and the within-sector adjusted sentiment selected model:

| comparison | holdout AUC |
|---|---:|
| Core + market selected model | 0.5020 |
| Within-sector adjusted sentiment selected model | 0.6038 |

The holdout improvement was meaningful, but not stable across pseudo-holdout years. The 2024 bootstrap AUC-delta interval was **-0.0930 to 0.1307**, crossing zero.

## Final Verdict

The final verdict is conservative: Capital IQ Key Developments sentiment is a **promising but fragile feature layer**. The richer event-text layer produced meaningful 2024 holdout lift, and the within-sector adjusted sentiment rung reached 0.6038 holdout AUC versus 0.5020 for the core + market benchmark. However, cross-year pseudo-holdout diagnostics were mixed, CV did not improve in parallel, and the bootstrap interval crossed zero.

The result supports further testing of zero-shot FinBERT + Capital IQ Key Developments feature engineering as event-context enrichment for Consumer Staples v1. It does not support a broad claim of robust generalization.

## Why Tuning Was Not Run

Tuning was not run because the locked result already showed an unstable lift profile. Running Optuna or other hyperparameter searches after seeing the 2024 holdout result would risk overfitting the final interpretation to the holdout period. The appropriate final presentation is the untuned, locked ladder plus stability diagnostics.

## Remaining Limitations

- The universe is Consumer Staples v1 only, with 34 tickers.
- Capital IQ Key Developments are event-text items, not general news coverage.
- Sentiment coverage is strong at 30d and 63d but uneven across ticker-years.
- Some high-news-count tickers need review.
- The 2024 lift was not stable across pseudo-holdout years.
- Bootstrap uncertainty crossed zero.
- Some validation detail is limited where exact fold assignment artifacts were not saved for older experiment families.

## Future Work

- Re-run the same locked protocol on additional sectors before making broader claims.
- Add stricter saved validation artifacts for every experiment family, including fold maps and purge audits.
- Investigate high-news-count ticker behavior and low-coverage ticker-years.
- Test whether Capital IQ event categories, recency decay, and event-type stratification improve stability.
- If tuning is later added, perform CV-only tuning with a predeclared search space and a single final 2024 holdout read.

## Source Artifacts

- `outputs/quarterly/diagnostics/fundamental_rebuild/staged_rebuild_summary.md`
- `outputs/quarterly/diagnostics/fundamental_rebuild/ratio_amount_before_after_summary.csv`
- `outputs/quarterly/diagnostics/data_layer_integrity/data_layer_integrity_report.md`
- `outputs/quarterly/diagnostics/data_layer_integrity/data_layer_integrity_summary.csv`
- `outputs/quarterly/diagnostics/data_layer_integrity/artifact_manifest.csv`
- `outputs/quarterly/modeling/final/capitaliq_sentiment_final_report.md`
- `outputs/quarterly/modeling/final/capitaliq_sentiment_final_tables.csv`
- `outputs/quarterly/modeling/final/figures/`
- `configs/event_panel_v2_quarterly_63d_sector_relative.yaml`
- `outputs/quarterly/labels/label_map_excess_63d.parquet`
- `configs/quarterly/current_benchmark_set.yaml`
