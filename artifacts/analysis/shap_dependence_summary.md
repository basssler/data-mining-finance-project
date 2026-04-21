# SHAP Dependence Summary

| feature_name | pattern_type | threshold_behavior | stability | candidate_transform | notes |
| --- | --- | --- | --- | --- | --- |
| abs_return_shock_1d | roughly_monotonic | possible_threshold | medium | keep_raw_or_winsorize |  |
| beta_63d_to_sector | roughly_monotonic | possible_threshold | high | keep_raw_or_winsorize |  |
| cash_ratio | tail_driven | possible_threshold | medium | consider_regime_bin_or_tail_clip |  |
| log_volume | nonlinear_or_noisy | possible_threshold | low | consider_nonlinear_transform |  |
| realized_vol_21d | roughly_monotonic | possible_threshold | high | keep_raw_or_winsorize |  |
| rel_return_10d | roughly_monotonic | possible_threshold | medium | keep_raw_or_winsorize |  |
| rel_return_5d | tail_driven | possible_threshold | low | consider_regime_bin_or_tail_clip |  |
| sec_positive_prob | roughly_monotonic | possible_threshold | medium | keep_raw_or_winsorize |  |
| vol_ratio_21d_63d | roughly_monotonic | possible_threshold | high | keep_raw_or_winsorize |  |
| volume_ratio_20d | roughly_monotonic | possible_threshold | high | keep_raw_or_winsorize |  |
