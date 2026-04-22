from __future__ import annotations

import numpy as np
import pandas as pd

QUARTERLY_DELTA_SOURCE_COLUMNS = [
    "operating_margin",
    "net_margin",
    "roa",
    "asset_turnover",
    "revenue_growth_qoq",
    "earnings_growth_qoq",
    "accruals_ratio",
    "liquidity_profile_score",
    "solvency_profile_score",
    "profitability_profile_score",
    "growth_quality_profile_score",
    "overall_financial_health_score",
]

QUARTERLY_FEATURE_COLUMNS = [
    "qfd_delta_operating_margin",
    "qfd_delta_net_margin",
    "qfd_delta_roa",
    "qfd_delta_asset_turnover",
    "qfd_delta_revenue_growth_qoq",
    "qfd_delta_earnings_growth_qoq",
    "qfd_delta_accruals_ratio",
    "qfd_delta_liquidity_profile_score",
    "qfd_delta_solvency_profile_score",
    "qfd_delta_profitability_profile_score",
    "qfd_delta_growth_quality_profile_score",
    "qfd_delta_overall_financial_health_score",
    "qfd_margin_delta_combo",
    "qfd_growth_delta_combo",
    "qfd_profile_delta_combo",
    "qfd_av_surprise_consistency",
    "qfd_av_surprise_pct_consistency",
    "qfd_av_latest_surprise_vs_trailing_pct",
    "qfd_av_latest_surprise_vs_trailing_pct_abs",
    "qfd_av_latest_surprise_above_trailing_flag",
    "qfd_av_latest_surprise_vs_trailing_pct_bucket",
    "qfd_av_revision_pressure",
    "qfd_av_revision_acceleration",
    "qfd_av_revision_breadth_pressure",
    "qfd_av_estimate_breadth_mean",
    "qfd_av_revision_x_surprise",
    "qfd_av_revision_x_surprise_capped",
    "qfd_av_revision_direction_flag",
    "qfd_av_revision_magnitude_bucket",
    "qfd_av_revision_plus_surprise",
    "qfd_av_revision_surprise_same_sign_flag",
    "qfd_av_surprise_consistency_bucket",
    "qfd_av_trailing_surprise_pct_std_clipped",
    "qfd_av_trailing_surprise_pct_std_bucket",
    "qfd_av_revision_x_growth_quality",
    "qfd_av_surprise_x_financial_health",
    "qfd_log_days_since_last_earnings_release",
    "qfd_log_days_since_prior_event",
    "qfd_log_days_since_prior_same_event_type",
    "qfd_event_gap_ratio",
    "qfd_event_gap_difference",
    "qfd_prior_event_gap_bucket",
    "qfd_prior_same_type_gap_bucket",
    "qfd_short_cycle_flag",
    "qfd_short_same_type_cycle_flag",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = _numeric(numerator)
    denominator = _numeric(denominator)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = denominator.notna() & (denominator != 0) & numerator.notna()
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def _log1p_nonnegative(series: pd.Series) -> pd.Series:
    numeric = _numeric(series)
    return np.log1p(numeric.clip(lower=0))


def _clip_symmetric(series: pd.Series, cap: float) -> pd.Series:
    numeric = _numeric(series)
    return numeric.clip(lower=-float(cap), upper=float(cap))


def _bucket_nonnegative(series: pd.Series, thresholds: list[float]) -> pd.Series:
    numeric = _numeric(series)
    result = pd.Series(np.nan, index=numeric.index, dtype="float64")
    valid = numeric.notna()
    if not valid.any():
        return result
    clipped = numeric.loc[valid].clip(lower=0)
    bucket = pd.Series(0.0, index=clipped.index, dtype="float64")
    for threshold in thresholds:
        bucket = bucket + clipped.ge(float(threshold)).astype("float64")
    result.loc[valid] = bucket
    return result


def _bucket_signed_magnitude(series: pd.Series, thresholds: list[float]) -> pd.Series:
    numeric = _numeric(series)
    result = pd.Series(np.nan, index=numeric.index, dtype="float64")
    valid = numeric.notna()
    if not valid.any():
        return result
    magnitude_bucket = _bucket_nonnegative(numeric.abs(), thresholds)
    result.loc[valid] = magnitude_bucket.loc[valid] * np.sign(numeric.loc[valid]).astype("float64")
    return result


def build_quarterly_feature_design_panel(panel_df: pd.DataFrame) -> pd.DataFrame:
    prepared = panel_df.copy()
    sort_columns = ["ticker", "effective_model_date", "event_date", "source_id"]
    prepared["effective_model_date"] = pd.to_datetime(prepared["effective_model_date"], errors="coerce")
    prepared["event_date"] = pd.to_datetime(prepared["event_date"], errors="coerce")
    prepared = prepared.sort_values(sort_columns).reset_index(drop=True)

    grouped = prepared.groupby("ticker", sort=False)
    for column in QUARTERLY_DELTA_SOURCE_COLUMNS:
        prepared[column] = _numeric(prepared[column])
        previous = grouped[column].shift(1)
        prepared[f"qfd_delta_{column}"] = prepared[column] - previous

    prepared["av_latest_prior_eps_surprise_pct_before_event"] = _numeric(
        prepared["av_latest_prior_eps_surprise_pct_before_event"]
    )
    prepared["av_trailing_4q_eps_surprise_pct_mean"] = _numeric(
        prepared["av_trailing_4q_eps_surprise_pct_mean"]
    )
    prepared["av_trailing_4q_eps_surprise_pct_std"] = _numeric(
        prepared["av_trailing_4q_eps_surprise_pct_std"]
    )
    prepared["av_latest_prior_eps_surprise_before_event"] = _numeric(
        prepared["av_latest_prior_eps_surprise_before_event"]
    )
    prepared["av_trailing_4q_eps_surprise_mean"] = _numeric(prepared["av_trailing_4q_eps_surprise_mean"])
    prepared["av_trailing_4q_eps_surprise_std"] = _numeric(prepared["av_trailing_4q_eps_surprise_std"])
    prepared["av_eps_estimate_revision_30d"] = _numeric(prepared["av_eps_estimate_revision_30d"])
    prepared["av_eps_estimate_revision_90d"] = _numeric(prepared["av_eps_estimate_revision_90d"])
    prepared["av_eps_estimate_analyst_count_before_event"] = _numeric(
        prepared["av_eps_estimate_analyst_count_before_event"]
    )
    prepared["av_revenue_estimate_analyst_count_before_event"] = _numeric(
        prepared["av_revenue_estimate_analyst_count_before_event"]
    )
    prepared["av_days_since_last_earnings_release"] = _numeric(prepared["av_days_since_last_earnings_release"])
    prepared["days_since_prior_event"] = _numeric(prepared["days_since_prior_event"])
    prepared["days_since_prior_same_event_type"] = _numeric(prepared["days_since_prior_same_event_type"])

    prepared["qfd_margin_delta_combo"] = (
        prepared["qfd_delta_operating_margin"] + prepared["qfd_delta_net_margin"]
    )
    prepared["qfd_growth_delta_combo"] = (
        prepared["qfd_delta_revenue_growth_qoq"] + prepared["qfd_delta_earnings_growth_qoq"]
    )
    prepared["qfd_profile_delta_combo"] = (
        prepared["qfd_delta_profitability_profile_score"]
        + prepared["qfd_delta_growth_quality_profile_score"]
        + prepared["qfd_delta_overall_financial_health_score"]
    )

    prepared["qfd_av_surprise_consistency"] = _safe_divide(
        prepared["av_trailing_4q_eps_surprise_mean"],
        1.0 + prepared["av_trailing_4q_eps_surprise_std"].abs(),
    )
    prepared["qfd_av_surprise_pct_consistency"] = _safe_divide(
        prepared["av_trailing_4q_eps_surprise_pct_mean"],
        1.0 + prepared["av_trailing_4q_eps_surprise_pct_std"].abs(),
    )
    prepared["qfd_av_latest_surprise_vs_trailing_pct"] = (
        prepared["av_latest_prior_eps_surprise_pct_before_event"]
        - prepared["av_trailing_4q_eps_surprise_pct_mean"]
    )
    prepared["qfd_av_latest_surprise_vs_trailing_pct_abs"] = prepared[
        "qfd_av_latest_surprise_vs_trailing_pct"
    ].abs()
    latest_surprise_valid = (
        prepared["av_latest_prior_eps_surprise_pct_before_event"].notna()
        & prepared["av_trailing_4q_eps_surprise_pct_mean"].notna()
    )
    prepared["qfd_av_latest_surprise_above_trailing_flag"] = pd.Series(
        np.nan,
        index=prepared.index,
        dtype="float64",
    )
    prepared.loc[latest_surprise_valid, "qfd_av_latest_surprise_above_trailing_flag"] = (
        prepared.loc[latest_surprise_valid, "av_latest_prior_eps_surprise_pct_before_event"]
        >= prepared.loc[latest_surprise_valid, "av_trailing_4q_eps_surprise_pct_mean"]
    ).astype("float64")
    prepared["qfd_av_latest_surprise_vs_trailing_pct_bucket"] = _bucket_signed_magnitude(
        prepared["qfd_av_latest_surprise_vs_trailing_pct"],
        thresholds=[1.0, 3.0, 6.0],
    )
    prepared["qfd_av_revision_pressure"] = _safe_divide(
        prepared["av_eps_estimate_revision_30d"],
        1.0 + prepared["av_eps_estimate_analyst_count_before_event"].abs(),
    )
    prepared["qfd_av_revision_acceleration"] = (
        prepared["av_eps_estimate_revision_30d"] - prepared["av_eps_estimate_revision_90d"]
    )
    prepared["qfd_av_estimate_breadth_mean"] = prepared[
        [
            "av_eps_estimate_analyst_count_before_event",
            "av_revenue_estimate_analyst_count_before_event",
        ]
    ].mean(axis=1)
    prepared["qfd_av_revision_breadth_pressure"] = _safe_divide(
        prepared["av_eps_estimate_revision_30d"],
        1.0 + prepared["qfd_av_estimate_breadth_mean"].abs(),
    )
    prepared["qfd_av_revision_x_surprise"] = (
        prepared["av_eps_estimate_revision_30d"]
        * prepared["av_latest_prior_eps_surprise_pct_before_event"]
    )
    prepared["qfd_av_revision_x_surprise_capped"] = (
        _clip_symmetric(prepared["av_eps_estimate_revision_30d"], cap=5.0)
        * _clip_symmetric(prepared["av_latest_prior_eps_surprise_pct_before_event"], cap=10.0)
    )
    prepared["qfd_av_revision_direction_flag"] = pd.Series(
        np.sign(prepared["av_eps_estimate_revision_30d"]),
        index=prepared.index,
        dtype="float64",
    ).where(prepared["av_eps_estimate_revision_30d"].notna(), np.nan)
    prepared["qfd_av_revision_magnitude_bucket"] = _bucket_signed_magnitude(
        prepared["av_eps_estimate_revision_30d"],
        thresholds=[0.25, 1.0, 3.0],
    )
    prepared["qfd_av_revision_plus_surprise"] = (
        prepared["av_eps_estimate_revision_30d"]
        + prepared["av_latest_prior_eps_surprise_pct_before_event"]
    )
    revision_valid = prepared["av_eps_estimate_revision_30d"].notna()
    surprise_valid = prepared["av_latest_prior_eps_surprise_pct_before_event"].notna()
    joint_valid = revision_valid & surprise_valid
    prepared["qfd_av_revision_surprise_same_sign_flag"] = pd.Series(
        np.nan,
        index=prepared.index,
        dtype="float64",
    )
    prepared.loc[joint_valid, "qfd_av_revision_surprise_same_sign_flag"] = (
        (
            np.sign(prepared.loc[joint_valid, "av_eps_estimate_revision_30d"])
            == np.sign(prepared.loc[joint_valid, "av_latest_prior_eps_surprise_pct_before_event"])
        )
        & prepared.loc[joint_valid, "av_eps_estimate_revision_30d"].ne(0)
        & prepared.loc[joint_valid, "av_latest_prior_eps_surprise_pct_before_event"].ne(0)
    ).astype("float64")
    prepared["qfd_av_surprise_consistency_bucket"] = _bucket_signed_magnitude(
        prepared["qfd_av_surprise_pct_consistency"],
        thresholds=[0.5, 1.5, 3.0],
    )
    prepared["qfd_av_trailing_surprise_pct_std_clipped"] = prepared["av_trailing_4q_eps_surprise_pct_std"].clip(
        lower=0,
        upper=10.0,
    )
    prepared["qfd_av_trailing_surprise_pct_std_bucket"] = _bucket_nonnegative(
        prepared["av_trailing_4q_eps_surprise_pct_std"],
        thresholds=[0.5, 1.5, 3.0],
    )
    prepared["qfd_av_revision_x_growth_quality"] = (
        prepared["av_eps_estimate_revision_30d"] * prepared["growth_quality_profile_score"]
    )
    prepared["qfd_av_surprise_x_financial_health"] = (
        prepared["av_latest_prior_eps_surprise_pct_before_event"]
        * prepared["overall_financial_health_score"]
    )

    prepared["qfd_log_days_since_last_earnings_release"] = _log1p_nonnegative(
        prepared["av_days_since_last_earnings_release"]
    )
    prepared["qfd_log_days_since_prior_event"] = _log1p_nonnegative(prepared["days_since_prior_event"])
    prepared["qfd_log_days_since_prior_same_event_type"] = _log1p_nonnegative(
        prepared["days_since_prior_same_event_type"]
    )
    prepared["qfd_event_gap_ratio"] = _safe_divide(
        prepared["days_since_prior_event"],
        1.0 + prepared["days_since_prior_same_event_type"],
    )
    prepared["qfd_event_gap_difference"] = (
        prepared["days_since_prior_event"] - prepared["days_since_prior_same_event_type"]
    )
    prepared["qfd_prior_event_gap_bucket"] = _bucket_nonnegative(
        prepared["days_since_prior_event"],
        thresholds=[45.0, 90.0, 135.0, 180.0],
    )
    prepared["qfd_prior_same_type_gap_bucket"] = _bucket_nonnegative(
        prepared["days_since_prior_same_event_type"],
        thresholds=[60.0, 120.0, 180.0, 270.0],
    )
    prepared["qfd_short_cycle_flag"] = (
        prepared["days_since_prior_event"].le(75).where(prepared["days_since_prior_event"].notna(), np.nan)
    ).astype("float64")
    prepared["qfd_short_same_type_cycle_flag"] = (
        prepared["days_since_prior_same_event_type"]
        .le(120)
        .where(prepared["days_since_prior_same_event_type"].notna(), np.nan)
    ).astype("float64")

    return prepared
