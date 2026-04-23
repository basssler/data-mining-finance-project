# Quarterly Checkpoint Sprint Summary

## What Changed

- Added a narrow `qfd_v3` refined-delta family for higher-signal quarterly change features instead of expanding the feature set broadly.
- Switched quarterly cross-sectional grouping to quarter-aware sector buckets so z-scores and percentile ranks are computed within a real quarterly peer set.
- Ran a refreshed quarterly family ablation, a label-family comparison under the unchanged purged walk-forward setup, and a light pruning checkpoint.

## What Improved

- Cross-sectional coverage improved materially: `cross_sectional` family mean missingness moved to `13.22%` from the prior `39.10%`, with per-feature availability now `81.88%` to `92.79%`.
- Cross-sectional usefulness improved after the coverage rescue. In the selected `levels_plus_deltas_plus_cross_sectional` model, `7` cross-sectional features survived as usable versus the prior `3`.
- The refreshed best family stack is now `levels_plus_deltas_plus_cross_sectional` with `xgboost`: `cv_auc_mean=0.5200`, `cv_auc_std=0.0307`, `worst_fold_auc=0.4921`, `holdout_auc=0.5072`, `holdout_row_count=101`.
- The strongest practical label result is `21d_excess_thresholded`: selected model `random_forest`, `cv_auc_mean=0.5126`, `cv_auc_std=0.0346`, `worst_fold_auc=0.4652`, `holdout_auc=0.5291`, `holdout_row_count=66`, `class_1_rate=0.5176`, `dropped_ambiguous_count=303`.

## What Did Not Improve

- The refined delta lane is still sparse overall: `delta_refined` mean missingness is `58.09%`, and only `4` refined-delta features survived in the selected cross-sectional benchmark model.
- The deltas-only family did not become the best standalone benchmark. Its refreshed result was `logistic_regression` with `cv_auc_mean=0.5076`, `cv_auc_std=0.0193`, `worst_fold_auc=0.4769`, `holdout_auc=0.4487`, `holdout_row_count=101`.
- Light pruning did not help. Against the `levels_plus_deltas_plus_cross_sectional` baseline:
  - `unpruned_logistic`: `cv_auc_mean=0.5044`, `cv_auc_std=0.0230`, `worst_fold_auc=0.4695`, `holdout_auc=0.4447`
  - `unpruned_elastic_net`: `cv_auc_mean=0.5125`, `cv_auc_std=0.0163`, `worst_fold_auc=0.4926`, `holdout_auc=0.4399`
  - `correlation_pruned_logistic`: `cv_auc_mean=0.4957`, `cv_auc_std=0.0283`, `worst_fold_auc=0.4477`, `holdout_auc=0.4346`
  - `correlation_pruned_elastic_net`: `cv_auc_mean=0.5057`, `cv_auc_std=0.0160`, `worst_fold_auc=0.4791`, `holdout_auc=0.4254`
- `21d_excess_quantile` posted the top raw holdout AUC (`0.5500`), but it only had `9` holdout rows and dropped `939` rows as ambiguous, so it is too thin to promote as the default benchmark label.

## Recommended Next Move Before Phase 8+

- Keep `levels_plus_deltas_plus_cross_sectional` as the quarterly feature benchmark family.
- Promote `21d_excess_thresholded` to the next benchmark label candidate, not `21d_excess_quantile`, because it improves the label readout without collapsing the evaluation sample.
- Run one more quarterly-core round focused on:
  - replacing the still-100%-missing unlocked accounting inputs only where the raw source concepts can actually be recovered cheaply
  - rerunning the feature-family ablation under `21d_excess_thresholded`
  - testing one small xgboost regularization pass on the cross-sectional family winner rather than broader pruning work
