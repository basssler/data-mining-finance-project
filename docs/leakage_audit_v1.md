# Leakage Audit V1

## Final Decision

YES: the current event_v1 panel is temporally valid after two minimal timing fixes applied in this phase.

Pre-fix status was NO. Two blocking issues were found and corrected before closing the audit:
- Layer 1 fundamentals leaked same-day after-close 10-Q/10-K filings into 137 panel rows across 137 unique filing keys.
- SEC sentiment leaked same-day after-close filings into 262 panel rows across 262 unique filings.

## Scope

- Audited the current event_v1 full panel and the upstream timing paths that feed it.
- Kept labels, validation, model family, and training logic unchanged.
- Rebuilt only the artifacts needed to remove leakage and verify the corrected panel.

## Current Panel Facts

- Current `event_v1_full` rows: `68,513` across `34` tickers.
- Date range: `2015-07-31` to `2024-12-31`.
- 2024 holdout starts on `2024-01-02` and ends on `2024-12-31`. The last pre-holdout date is `2023-12-29`.
- Holdout training stops at `2023-12-15` with `9` purged dates before the holdout block.

## Fixes Applied

- `src/panel_builder.py`: fundamentals now align on a tradable `effective_model_date` derived from SEC metadata when available, with a next-tradable-date fallback for unmatched or ambiguous filings.
- `src/sec_sentiment_event_v1.py`: SEC sentiment now uses the same `effective_model_date` discipline instead of plain `filing_date`, again with a conservative next-tradable-date fallback when no exact timing match exists.
- Rebuilt `layer1_modeling_panel.parquet`, `layer3_sec_sentiment_event_v1.parquet`, and all event_v1 panel parquet outputs.
- Existing training reports under `reports/results/` were not regenerated in this phase and still reflect pre-fix panels.

## Sample Audit

- Sample file: `reports\audits\leakage_audit_sample_v1.csv`
- Sample size: `18` rows.
- Every sampled row passed the end-to-end timing check after the fixes.

## After-Close Handling

- SEC filing metadata rows by timing bucket: pre-market `2,069`, market-hours `1,274`, after-close `1,284`, missing-timestamp `0`.
- Structured SEC filing-event features were already correct before this phase: after-close rows are shifted to the next tradable date, pre-market rows are available the same day, and the missing-timestamp path is coded conservatively. The current artifact had zero rows in the missing-timestamp bucket.
- Analyst events: `3,246` after-close, `104,074` before-open, `36,039` market-hours. `2,578` raw source rows had invalid timestamps and were dropped before feature generation.
- Layer 1 fundamentals and SEC sentiment were the only paths that violated the after-close rule pre-fix. Both now respect next-tradable-date exposure.

## Current Boundary Checks

- Layer 1 current pre-effective row count: `0`.
- SEC sentiment current pre-effective row count: `0`.
- Layer 1 fallback rows without exact SEC timing match: `36` rows across `10` filing keys. These are now exposed conservatively on the next tradable date.
- SEC sentiment fallback rows on the current row grid: `45`. These rows also use next-tradable-date exposure.

## Benchmark And Sector Timing

- Label construction is row-date anchored: `label_start_date = row_date`, and `label_end_date` is the fifth future trading date for that ticker.
- `benchmark_forward_return_5d` is a same-date leave-one-out cross-sectional benchmark built only inside the label table. It is not a feature and is not fit or transformed on holdout rows.
- Layer 2 sector controls are also same-date leave-one-out cross-sections, but they use only trailing returns and rolling windows ending on `row_date`.

## Holdout Isolation

- No hyperparameter tuning uses 2024 rows. The best model is selected by mean CV metrics only.
- No preprocessing is fit on holdout data. Feature usability, clipping bounds, imputers, and scalers are fit on train or pre-holdout rows only.
- The 2024 holdout is scored once after the model is refit on pre-holdout data.
- Validation and holdout use the same purge/embargo discipline from `src/validation_event_v1.py`.

## Missingness And Suspicious Features

1. `gross_margin` is `100.00%` missing in the current full panel. This is a data-quality problem, not a temporal leak, but it makes the column unusable.
2. `receivables_turnover`, `inventory_turnover`, `earnings_growth_yoy`, and `revenue_growth_yoy` all exceed `20%` missingness in the current full panel (27.37%, 24.29%, 24.61%, 22.33%).
3. Analyst sentiment summary features are extremely sparse in the analyst variant (`analyst_mean_sentiment_1d` 98.48%, `analyst_mean_sentiment_5d` 93.47%, `analyst_sentiment_std_5d` 93.47% missing). They appear timestamp-safe but operationally fragile.
4. Several grouped 8-K decay features are mostly empty because the categories are rare (`sec_8k_regulatory_legal_decay_3d` 90.87% missing, `sec_8k_financing_securities_decay_3d` 31.13% missing). This is sparsity, not leakage.
5. The remaining timestamp ambiguity is concentrated in unmatched filings. They are no longer a leakage risk because the current code delays them to the next tradable date, but that fallback can be slightly late for any truly pre-market unmatched filing.

## Residual Uncertainty

- The conservative fallback removes leakage risk but may delay a small subset of unmatched filings by one trading day.
- This phase did not rerun model training, so any saved metrics or predictions from prior runs should be treated as pre-fix artifacts.

## Conclusion

The current event_v1 panel is temporally valid after the two timing fixes above. Before those fixes, the panel was not clean enough to justify a redesign decision.
