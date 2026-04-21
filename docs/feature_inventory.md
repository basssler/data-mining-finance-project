# WRDS Stock Prediction Feature Inventory

## Layer 1: Fundamentals
- `current_ratio`
- `quick_ratio`
- `debt_to_equity`
- `debt_to_assets`
- `interest_coverage`
- `gross_margin`
- `operating_margin`
- `net_margin`
- `roa`
- `roe`
- `asset_turnover`
- `revenue_growth_yoy`
- `earnings_growth_yoy`
- `accruals_ratio`
- `cfo_to_net_income`

## Layer 2: Market
- `ret_5d`
- `ret_10d`
- `ret_21d`
- `ret_63d`
- `trend_efficiency_21d`
- `sign_flip_count_21d`
- `rolling_abs_return_sum_21d`
- `vol_21d`
- `downside_semivariance_21d`
- `max_drawdown_63d`
- `vol_of_vol_21d`
- `volume_over_20d_avg`
- `dollar_volume_log`
- `obv_21d_change`
- `distance_from_63d_high`
- `distance_from_252d_high`
- `drawdown_from_peak_21d`
- `ret_21d_minus_market`
- `ret_21d_minus_sector`

## Peer-Relative Transforms
- `peer_<feature>_pct`
- `peer_<feature>_rz`

The current raw inputs for peer transforms are:
- `ret_21d`
- `vol_21d`
- `distance_from_63d_high`
- `debt_to_equity`
- `roa`
- `revenue_growth_yoy`

## Planned Next Layers
- Sentiment features from external timestamped text sources.
- Optional macro/regime features from external macro series.
