# Quarterly Final Core Confirmation

## What Changed

- Froze a versioned confirmation config at [configs/quarterly/quarterly_final_core_confirmation_v1.yaml](C:/Users/maxba/Documents/GitHub/data-mining-finance-project/configs/quarterly/quarterly_final_core_confirmation_v1.yaml).
- Kept the rescued `levels_plus_deltas_plus_cross_sectional` quarterly feature stack unchanged.
- Changed only the benchmark label to the provisional `21d_excess_thresholded` map at `+/-1.5%`.
- Kept the same purged walk-forward setup: holdout start `2024-01-01`, `5` folds, `5` embargo days, `252` minimum train dates.

## Exact Benchmark Settings

- Panel: `data/interim/event_panel_v2_quarterly_feature_design.parquet`
- Label: `outputs/quarterly/labels/checkpoint_21d_excess_thresholded.parquet`
- Holdout policy: unchanged, evaluation window begins `2024-01-04`
- Validation policy: unchanged purged walk-forward
- Candidate models: `logistic_regression`, `random_forest`, `xgboost`
- Selected model: `random_forest`

## Final Benchmark Result

- `selected_model=random_forest`
- `cv_auc_mean=0.5126`
- `cv_auc_std=0.0346`
- `worst_fold_auc=0.4652`
- `holdout_auc=0.5291`
- `holdout_row_count=66`
- `dropped_ambiguous_count=303`

Usable feature counts by family:
- `base_or_other=41`
- `legacy_delta=45`
- `level=7`
- `delta=7`
- `delta_refined=6`
- `cross_sectional=7`
- `acceleration=2`
- `stability=1`

Selected-run fold readout:
- Fold AUCs: `0.5664`, `0.4652`, `0.5278`, `0.4884`, `0.5154`
- Worst fold is fold 2 at `0.4652`
- Holdout AUC is above every validation fold except fold 1

## Comparison Vs Prior

Versus prior `21d_excess_sign` label run on the quarterly core:
- Holdout AUC improved from `0.5084` to `0.5291` (`+0.0208`)
- CV AUC mean improved from `0.5002` to `0.5126` (`+0.0125`)
- Worst fold fell from `0.4757` to `0.4652` (`-0.0105`)
- Holdout rows fell from `82` to `66`

Versus the older `63d_sign` default quarterly-core benchmark:
- Holdout AUC improved from `0.4427` to `0.5291` (`+0.0865`)
- CV AUC mean fell from `0.5243` to `0.5126` (`-0.0117`)
- Holdout rows fell from `101` to `66`

Interpretation:
- This is not a cosmetic win versus `21d_excess_sign`; both holdout and CV mean improved.
- The result is still not fully stable because the worst validation fold remains below random.
- The holdout sample is modest, but `66` rows is materially more usable than the quantile-label case and not so thin that the result should be dismissed outright.

## Delta-Lane Readout

Refined deltas that survived into the selected model:
- `qfd_v3_total_debt_to_assets_d1`
- `qfd_v3_net_margin_d1`
- `qfd_v3_cfo_to_net_income_d1`
- `qfd_v3_operating_margin_d1`
- `qfd_v3_accruals_ratio_d1`
- `qfd_v3_leverage_change_qoq_d1`

Refined deltas still too sparse to matter in this quarterly core:
- `qfd_v3_capex_intensity_d1`
- `qfd_v3_free_cash_flow_margin_d1`
- `qfd_v3_free_cash_flow_to_net_income_d1`
- `qfd_v3_gross_margin_d1`
- `qfd_v3_interest_coverage_d1`
- `qfd_v3_shareholder_payout_ratio_d1`

Recommendation:
- The delta lane is good enough for now.
- Do not spend another checkpoint round on broad delta expansion before Phase 8+.
- If this lane is revisited later, it should only be to unlock the still-all-missing accounting concepts cheaply from source data.

## Promotion Decision

1. `21d_excess_thresholded` should become the new official quarterly benchmark label.
2. The promotion should be treated as a benchmarking decision, not proof that the quarterly core is robust.
3. The quarterly core is strong enough to move to Phase 8+, with the explicit caveat that fold stability is still mediocre and should remain under watch in later-phase work.
