from __future__ import annotations

import numpy as np
import pandas as pd

from src.universe import get_project_sector_map

MIN_INDUSTRY_CROSS_SECTIONAL_GROUP_SIZE = 3

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

QFD_V2_LEVEL_SOURCE_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roa",
    "debt_to_assets",
    "working_capital_to_total_assets",
    "accruals_ratio",
    "cfo_to_net_income",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
]

QFD_V2_QOQ_CHANGE_SOURCE_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roa",
    "debt_to_assets",
    "working_capital_to_total_assets",
    "accruals_ratio",
    "cfo_to_net_income",
]

QFD_V2_YOY_CHANGE_SOURCE_COLUMNS = [
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "cfo_to_net_income",
]

QFD_V2_ACCELERATION_SOURCE_COLUMNS = [
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "operating_margin",
    "net_margin",
]

QFD_V2_STABILITY_SOURCE_COLUMNS = [
    "operating_margin",
    "net_margin",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "cfo_to_net_income",
]

QFD_V3_DELTA_SOURCE_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "free_cash_flow_margin",
    "interest_coverage",
    "leverage_change_qoq",
    "total_debt_to_assets",
    "capex_intensity",
    "shareholder_payout_ratio",
    "cfo_to_net_income",
    "accruals_ratio",
    "free_cash_flow_to_net_income",
]

QFD_V2_LEVEL_COLUMNS = [f"qfd_v2_{column}_lvl" for column in QFD_V2_LEVEL_SOURCE_COLUMNS]
QFD_V2_DELTA_COLUMNS = [f"qfd_v2_{column}_d1" for column in QFD_V2_QOQ_CHANGE_SOURCE_COLUMNS] + [
    f"qfd_v2_{column}_d4" for column in QFD_V2_YOY_CHANGE_SOURCE_COLUMNS
]
QFD_V2_ACCELERATION_COLUMNS = [f"qfd_v2_{column}_accel" for column in QFD_V2_ACCELERATION_SOURCE_COLUMNS]
QFD_V2_CROSS_SECTIONAL_COLUMNS = [
    "qfd_v2_roa_zsec",
    "qfd_v2_debt_to_assets_zsec",
    "qfd_v2_operating_margin_d1_zsec",
    "qfd_v2_accruals_ratio_zsec",
    "qfd_v2_operating_margin_d1_ranksec",
    "qfd_v2_profitability_profile_score_ranksec",
    "qfd_v2_debt_to_assets_d1_ranksec",
]
QFD_V2_STABILITY_COLUMNS = [f"qfd_v2_{column}_stb4q" for column in QFD_V2_STABILITY_SOURCE_COLUMNS] + [
    "qfd_v2_operating_margin_pos_count_stb4q",
    "qfd_v2_net_margin_pos_count_stb4q",
]

QFD_EVENT_AWARE_PRE_EVENT_MARKET_COLUMNS = [
    "qfd_mkt_pre_event_return_5d",
    "qfd_mkt_pre_event_return_21d",
    "qfd_mkt_pre_event_excess_return_5d_sector",
    "qfd_mkt_pre_event_excess_return_21d_sector",
    "qfd_mkt_pre_event_excess_return_5d_market",
    "qfd_mkt_pre_event_excess_return_21d_market",
    "qfd_mkt_pre_event_volume_z_20d",
    "qfd_mkt_pre_event_vol_5d",
    "qfd_mkt_pre_event_vol_21d",
    "qfd_mkt_pre_event_vol_ratio_5_21",
]

QFD_EVENT_AWARE_FIRST_TRADABLE_MARKET_COLUMNS = [
    "qfd_mkt_first_tradable_gap",
    "qfd_mkt_first_tradable_abnormal_volume",
]

QFD_EVENT_AWARE_MARKET_COLUMNS = (
    QFD_EVENT_AWARE_PRE_EVENT_MARKET_COLUMNS + QFD_EVENT_AWARE_FIRST_TRADABLE_MARKET_COLUMNS
)

QFD_EVENT_SENTIMENT_LEVEL_COLUMNS = [
    "qfd_es_filing_sentiment_score",
    "qfd_es_filing_positive_prob",
    "qfd_es_filing_negative_prob",
    "qfd_es_filing_neutral_prob",
    "qfd_es_filing_sentiment_abs",
]

QFD_EVENT_SENTIMENT_DELTA_COLUMNS = [
    "qfd_es_filing_sentiment_delta_prev_q",
    "qfd_es_filing_positive_prob_delta_prev_q",
    "qfd_es_filing_negative_prob_delta_prev_q",
    "qfd_es_negative_tone_jump",
]

QFD_EVENT_SENTIMENT_SURPRISE_COLUMNS = [
    "qfd_es_sentiment_surprise_vs_90d",
    "qfd_es_negative_surprise_vs_90d",
]

QFD_EVENT_SENTIMENT_DISPERSION_COLUMNS = [
    "qfd_es_sentiment_uncertainty",
    "qfd_es_sentiment_entropy",
    "qfd_es_sentiment_polarity_balance",
]

QFD_EVENT_SENTIMENT_ATTENTION_COLUMNS = [
    "qfd_es_source_count",
    "qfd_es_text_chunk_count",
    "qfd_es_log_text_chunk_count",
]

QFD_EVENT_SENTIMENT_COLUMNS = (
    QFD_EVENT_SENTIMENT_LEVEL_COLUMNS
    + QFD_EVENT_SENTIMENT_DELTA_COLUMNS
    + QFD_EVENT_SENTIMENT_SURPRISE_COLUMNS
    + QFD_EVENT_SENTIMENT_DISPERSION_COLUMNS
    + QFD_EVENT_SENTIMENT_ATTENTION_COLUMNS
)

QFD_V2_FEATURE_COLUMNS = (
    QFD_V2_LEVEL_COLUMNS
    + QFD_V2_DELTA_COLUMNS
    + QFD_V2_ACCELERATION_COLUMNS
    + QFD_V2_CROSS_SECTIONAL_COLUMNS
    + QFD_V2_STABILITY_COLUMNS
)

QFD_V3_DELTA_COLUMNS = [f"qfd_v3_{column}_d1" for column in QFD_V3_DELTA_SOURCE_COLUMNS]

LEGACY_QUARTERLY_FEATURE_COLUMNS = [
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

QUARTERLY_FEATURE_COLUMNS = (
    LEGACY_QUARTERLY_FEATURE_COLUMNS
    + QFD_V2_FEATURE_COLUMNS
    + QFD_V3_DELTA_COLUMNS
    + QFD_EVENT_AWARE_MARKET_COLUMNS
    + QFD_EVENT_SENTIMENT_COLUMNS
)

QUARTERLY_FEATURE_FAMILY_DEFINITIONS = [
    {"feature_family": "legacy_delta", "feature_type": "delta", "version": "qfd_v1", "columns": LEGACY_QUARTERLY_FEATURE_COLUMNS},
    {"feature_family": "level", "feature_type": "level", "version": "qfd_v2", "columns": QFD_V2_LEVEL_COLUMNS},
    {"feature_family": "delta", "feature_type": "delta", "version": "qfd_v2", "columns": QFD_V2_DELTA_COLUMNS},
    {"feature_family": "acceleration", "feature_type": "acceleration", "version": "qfd_v2", "columns": QFD_V2_ACCELERATION_COLUMNS},
    {"feature_family": "cross_sectional", "feature_type": "cross_sectional", "version": "qfd_v2", "columns": QFD_V2_CROSS_SECTIONAL_COLUMNS},
    {"feature_family": "stability", "feature_type": "stability", "version": "qfd_v2", "columns": QFD_V2_STABILITY_COLUMNS},
    {"feature_family": "delta_refined", "feature_type": "delta", "version": "qfd_v3", "columns": QFD_V3_DELTA_COLUMNS},
    {
        "feature_family": "event_aware_market_pre_event",
        "feature_type": "market_event",
        "version": "qfd_mkt_v1",
        "columns": QFD_EVENT_AWARE_PRE_EVENT_MARKET_COLUMNS,
    },
    {
        "feature_family": "event_aware_market_first_tradable",
        "feature_type": "market_event",
        "version": "qfd_mkt_v1",
        "columns": QFD_EVENT_AWARE_FIRST_TRADABLE_MARKET_COLUMNS,
    },
    {
        "feature_family": "event_sentiment_level",
        "feature_type": "sentiment_event",
        "version": "qfd_es_v1",
        "columns": QFD_EVENT_SENTIMENT_LEVEL_COLUMNS,
    },
    {
        "feature_family": "event_sentiment_delta",
        "feature_type": "sentiment_event",
        "version": "qfd_es_v1",
        "columns": QFD_EVENT_SENTIMENT_DELTA_COLUMNS,
    },
    {
        "feature_family": "event_sentiment_surprise",
        "feature_type": "sentiment_event",
        "version": "qfd_es_v1",
        "columns": QFD_EVENT_SENTIMENT_SURPRISE_COLUMNS,
    },
    {
        "feature_family": "event_sentiment_dispersion",
        "feature_type": "sentiment_event",
        "version": "qfd_es_v1",
        "columns": QFD_EVENT_SENTIMENT_DISPERSION_COLUMNS,
    },
    {
        "feature_family": "event_sentiment_attention",
        "feature_type": "sentiment_event",
        "version": "qfd_es_v1",
        "columns": QFD_EVENT_SENTIMENT_ATTENTION_COLUMNS,
    },
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


def _ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = _numeric(df[column])


def _trailing_std(series: pd.Series, window: int) -> pd.Series:
    numeric = _numeric(series)
    return numeric.rolling(window=window, min_periods=window).std(ddof=0)


def _trailing_mean_time_window(values: pd.Series, timestamps: pd.Series, window_days: int) -> pd.Series:
    numeric_values = _numeric(values)
    datetime_index = pd.to_datetime(timestamps, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = numeric_values.notna() & datetime_index.notna()
    if not bool(valid.any()):
        return result

    frame = pd.DataFrame(
        {
            "timestamp": datetime_index.loc[valid].to_numpy(),
            "value": numeric_values.loc[valid].to_numpy(),
            "original_index": values.index[valid],
        }
    ).sort_values("timestamp")
    rolling_mean = (
        frame.set_index("timestamp")["value"].rolling(f"{int(window_days)}D", closed="left").mean()
    )
    frame["rolling_mean"] = rolling_mean.to_numpy()
    result.loc[frame["original_index"]] = frame["rolling_mean"].to_numpy()
    return result


def _group_trailing_mean_time_window(
    df: pd.DataFrame,
    value_column: str,
    timestamp_column: str,
    group_column: str,
    window_days: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, frame in df.groupby(group_column, sort=False):
        result.loc[frame.index] = _trailing_mean_time_window(
            frame[value_column],
            frame[timestamp_column],
            window_days=window_days,
        )
    return result


def _rolling_positive_count(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    numeric = _numeric(series)
    positive = numeric.gt(0).where(numeric.notna(), np.nan)
    if min_periods is None:
        min_periods = window
    return positive.rolling(window=window, min_periods=min_periods).sum()


def _group_zscore(series: pd.Series) -> pd.Series:
    numeric = _numeric(series)
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    mean = numeric.mean()
    return (numeric - mean) / std


def _group_percentile_rank(series: pd.Series) -> pd.Series:
    numeric = _numeric(series)
    return numeric.rank(method="average", pct=True)


def _build_leave_one_out_mean(series: pd.Series, group_keys: pd.Series) -> pd.Series:
    numeric_series = _numeric(series)
    group_sum = numeric_series.groupby(group_keys).transform("sum")
    group_count = numeric_series.groupby(group_keys).transform("count")
    denominator = group_count - 1
    result = pd.Series(np.nan, index=numeric_series.index, dtype="float64")
    valid = denominator > 0
    result.loc[valid] = (group_sum.loc[valid] - numeric_series.loc[valid]) / denominator.loc[valid]
    return result


def build_event_aware_market_feature_group_map() -> pd.DataFrame:
    rows = []
    for column in QFD_EVENT_AWARE_PRE_EVENT_MARKET_COLUMNS:
        rows.append({"feature_name": column, "market_feature_group": "pre_event"})
    for column in QFD_EVENT_AWARE_FIRST_TRADABLE_MARKET_COLUMNS:
        rows.append({"feature_name": column, "market_feature_group": "first_tradable"})
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)


def build_sentiment_group_map() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for feature_name in [
        "sec_sentiment_score",
        "sec_positive_prob",
        "sec_negative_prob",
        "sec_neutral_prob",
        "sec_sentiment_abs",
        "sec_sentiment_change_prev",
        "sec_positive_change_prev",
        "sec_negative_change_prev",
        "sec_chunk_count",
        "sec_log_chunk_count",
    ]:
        rows.append({"feature_name": feature_name, "sentiment_group": "broad_filing_sentiment"})
    for feature_name in QFD_EVENT_SENTIMENT_COLUMNS:
        rows.append({"feature_name": feature_name, "sentiment_group": "event_specific_sentiment"})
        rows.append({"feature_name": feature_name, "sentiment_group": "combined_sentiment_block"})
    for feature_name in [
        "sec_sentiment_score",
        "sec_positive_prob",
        "sec_negative_prob",
        "sec_neutral_prob",
        "sec_sentiment_abs",
        "sec_sentiment_change_prev",
        "sec_positive_change_prev",
        "sec_negative_change_prev",
        "sec_chunk_count",
        "sec_log_chunk_count",
    ]:
        rows.append({"feature_name": feature_name, "sentiment_group": "combined_sentiment_block"})
    return pd.DataFrame(rows).drop_duplicates().sort_values(["sentiment_group", "feature_name"]).reset_index(drop=True)


def build_event_sentiment_coverage_diagnostics(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    feature_col = "qfd_es_filing_sentiment_score"
    if feature_col not in panel_df.columns:
        return pd.DataFrame(
            columns=[
                "scope",
                "scope_value",
                "rows",
                "covered_rows",
                "coverage_pct",
                "mean_source_count",
                "mean_chunk_count",
            ]
        )

    scoped_frames = [("overall", "all", panel_df)]
    if "event_date" in panel_df.columns:
        event_year = pd.to_datetime(panel_df["event_date"], errors="coerce").dt.year
        for year, group in panel_df.assign(_event_year=event_year).groupby("_event_year", dropna=True):
            scoped_frames.append(("event_year", str(int(year)), group.drop(columns="_event_year")))

    for scope, scope_value, frame in scoped_frames:
        covered = frame[feature_col].notna()
        source_count = _numeric(frame.get("qfd_es_source_count", pd.Series(np.nan, index=frame.index)))
        chunk_count = _numeric(frame.get("qfd_es_text_chunk_count", pd.Series(np.nan, index=frame.index)))
        rows.append(
            {
                "scope": scope,
                "scope_value": scope_value,
                "rows": int(len(frame)),
                "covered_rows": int(covered.sum()),
                "coverage_pct": float(covered.mean() * 100.0) if len(frame) else np.nan,
                "mean_source_count": float(source_count.loc[covered].mean()) if bool(covered.any()) else np.nan,
                "mean_chunk_count": float(chunk_count.loc[covered].mean()) if bool(covered.any()) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _prepare_price_feature_table(price_df: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "open", "close", "adj_close", "volume"}
    missing = sorted(required.difference(price_df.columns))
    if missing:
        raise ValueError("price_df is missing required columns: " + ", ".join(missing))

    prepared = price_df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string").str.upper()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").astype("datetime64[ns]")
    for column in ["open", "close", "adj_close", "volume"]:
        prepared[column] = _numeric(prepared[column])
    prepared = prepared.dropna(subset=["ticker", "date", "adj_close"]).copy()
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)

    sector_map = get_project_sector_map()
    prepared["sector"] = prepared["ticker"].map(sector_map).astype("string").fillna("unknown_sector")

    grouped = prepared.groupby("ticker", sort=False)
    prepared["qfd_market_daily_return_1d"] = grouped["adj_close"].pct_change()
    prepared["qfd_mkt_pre_event_return_5d"] = _safe_divide(prepared["adj_close"], grouped["adj_close"].shift(5)) - 1.0
    prepared["qfd_mkt_pre_event_return_21d"] = _safe_divide(prepared["adj_close"], grouped["adj_close"].shift(21)) - 1.0

    sector_key = pd.MultiIndex.from_frame(prepared[["date", "sector"]])
    prepared["qfd_market_sector_return_5d"] = _build_leave_one_out_mean(prepared["qfd_mkt_pre_event_return_5d"], sector_key)
    prepared["qfd_market_sector_return_21d"] = _build_leave_one_out_mean(prepared["qfd_mkt_pre_event_return_21d"], sector_key)
    prepared["qfd_market_return_5d"] = _build_leave_one_out_mean(prepared["qfd_mkt_pre_event_return_5d"], prepared["date"])
    prepared["qfd_market_return_21d"] = _build_leave_one_out_mean(prepared["qfd_mkt_pre_event_return_21d"], prepared["date"])
    prepared["qfd_mkt_pre_event_excess_return_5d_sector"] = (
        prepared["qfd_mkt_pre_event_return_5d"] - prepared["qfd_market_sector_return_5d"]
    )
    prepared["qfd_mkt_pre_event_excess_return_21d_sector"] = (
        prepared["qfd_mkt_pre_event_return_21d"] - prepared["qfd_market_sector_return_21d"]
    )
    prepared["qfd_mkt_pre_event_excess_return_5d_market"] = (
        prepared["qfd_mkt_pre_event_return_5d"] - prepared["qfd_market_return_5d"]
    )
    prepared["qfd_mkt_pre_event_excess_return_21d_market"] = (
        prepared["qfd_mkt_pre_event_return_21d"] - prepared["qfd_market_return_21d"]
    )

    volume_mean_20d = grouped["volume"].rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)
    volume_std_20d = grouped["volume"].rolling(window=20, min_periods=20).std(ddof=0).reset_index(level=0, drop=True)
    prepared["qfd_mkt_pre_event_volume_z_20d"] = _safe_divide(prepared["volume"] - volume_mean_20d, volume_std_20d)
    prepared["qfd_mkt_pre_event_vol_5d"] = (
        grouped["qfd_market_daily_return_1d"].rolling(window=5, min_periods=5).std(ddof=0).reset_index(level=0, drop=True)
    )
    prepared["qfd_mkt_pre_event_vol_21d"] = (
        grouped["qfd_market_daily_return_1d"].rolling(window=21, min_periods=21).std(ddof=0).reset_index(level=0, drop=True)
    )
    prepared["qfd_mkt_pre_event_vol_ratio_5_21"] = _safe_divide(
        prepared["qfd_mkt_pre_event_vol_5d"],
        prepared["qfd_mkt_pre_event_vol_21d"],
    )

    prior_close = grouped["close"].shift(1)
    prior_volume_mean_20d = (
        grouped["volume"].shift(1).rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)
    )
    prepared["qfd_mkt_first_tradable_gap"] = _safe_divide(prepared["open"], prior_close) - 1.0
    prepared["qfd_mkt_first_tradable_abnormal_volume"] = _safe_divide(prepared["volume"], prior_volume_mean_20d)

    keep_columns = [
        "ticker",
        "date",
        *QFD_EVENT_AWARE_MARKET_COLUMNS,
    ]
    return prepared[keep_columns].copy()


def _attach_event_aware_market_features(panel_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    event_market_df = _prepare_price_feature_table(price_df)
    event_market_df = event_market_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    prepared = panel_df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string").str.upper()
    prepared["effective_model_date"] = pd.to_datetime(prepared["effective_model_date"], errors="coerce").astype("datetime64[ns]")
    prepared["event_date"] = pd.to_datetime(prepared["event_date"], errors="coerce").astype("datetime64[ns]")
    timing_bucket = prepared.get("timing_bucket", pd.Series(pd.NA, index=prepared.index, dtype="string")).astype("string")

    prepared["qfd_market_timing_rule"] = np.where(
        timing_bucket.eq("after_close"),
        "after_close_uses_event_date_close",
        "pre_tradable_uses_prior_session_close",
    )
    prepared["qfd_market_pre_event_anchor_date"] = pd.NaT
    after_close_mask = timing_bucket.eq("after_close")
    prepared.loc[after_close_mask, "qfd_market_pre_event_anchor_date"] = prepared.loc[after_close_mask, "event_date"]
    prepared.loc[~after_close_mask, "qfd_market_pre_event_anchor_date"] = prepared.loc[~after_close_mask, "effective_model_date"]

    pre_event_columns = ["ticker", "date"] + QFD_EVENT_AWARE_PRE_EVENT_MARKET_COLUMNS
    pre_event_snapshot = event_market_df[pre_event_columns].rename(columns={"date": "qfd_market_pre_event_date"})

    after_close_panel = prepared.loc[after_close_mask].sort_values(["qfd_market_pre_event_anchor_date", "ticker"]).reset_index()
    if not after_close_panel.empty:
        merged_after_close = pd.merge_asof(
            left=after_close_panel,
            right=pre_event_snapshot.sort_values(["qfd_market_pre_event_date", "ticker"]).reset_index(drop=True),
            left_on="qfd_market_pre_event_anchor_date",
            right_on="qfd_market_pre_event_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=True,
        )
    else:
        merged_after_close = after_close_panel.copy()

    pre_market_panel = prepared.loc[~after_close_mask].sort_values(["qfd_market_pre_event_anchor_date", "ticker"]).reset_index()
    if not pre_market_panel.empty:
        merged_pre_market = pd.merge_asof(
            left=pre_market_panel,
            right=pre_event_snapshot.sort_values(["qfd_market_pre_event_date", "ticker"]).reset_index(drop=True),
            left_on="qfd_market_pre_event_anchor_date",
            right_on="qfd_market_pre_event_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=False,
        )
    else:
        merged_pre_market = pre_market_panel.copy()

    market_enriched = pd.concat([merged_after_close, merged_pre_market], ignore_index=True, sort=False)
    market_enriched = market_enriched.sort_values("index").set_index("index")
    prepared = prepared.join(
        market_enriched[[column for column in ["qfd_market_pre_event_date"] + QFD_EVENT_AWARE_PRE_EVENT_MARKET_COLUMNS if column in market_enriched.columns]],
        how="left",
    )

    first_tradable_snapshot = event_market_df[["ticker", "date"] + QFD_EVENT_AWARE_FIRST_TRADABLE_MARKET_COLUMNS].rename(
        columns={"date": "qfd_market_first_tradable_date"}
    )
    prepared = prepared.merge(
        first_tradable_snapshot,
        left_on=["ticker", "effective_model_date"],
        right_on=["ticker", "qfd_market_first_tradable_date"],
        how="left",
        validate="many_to_one",
    )
    return prepared


def build_feature_family_map() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for definition in QUARTERLY_FEATURE_FAMILY_DEFINITIONS:
        for column in definition["columns"]:
            rows.append(
                {
                    "feature_name": column,
                    "feature_family": str(definition["feature_family"]),
                    "feature_type": str(definition["feature_type"]),
                    "feature_version": str(definition["version"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_version", "feature_family", "feature_name"]).reset_index(drop=True)


def build_feature_family_coverage(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for definition in QUARTERLY_FEATURE_FAMILY_DEFINITIONS:
        columns = [column for column in definition["columns"] if column in panel_df.columns]
        if not columns:
            continue
        missing_pct = panel_df[columns].isna().mean().mul(100.0)
        available_pct = panel_df[columns].notna().mean().mul(100.0)
        rows.append(
            {
                "feature_family": str(definition["feature_family"]),
                "feature_type": str(definition["feature_type"]),
                "feature_version": str(definition["version"]),
                "feature_count": int(len(columns)),
                "mean_missing_pct": float(missing_pct.mean()),
                "max_missing_pct": float(missing_pct.max()),
                "mean_available_pct": float(available_pct.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_version", "feature_family"]).reset_index(drop=True)


def build_cross_sectional_coverage(panel_df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in QFD_V2_CROSS_SECTIONAL_COLUMNS if column in panel_df.columns]
    rows: list[dict[str, float | str]] = []
    for column in columns:
        rows.append(
            {
                "feature_name": column,
                "feature_family": "cross_sectional",
                "feature_version": "qfd_v2",
                "missing_pct": float(panel_df[column].isna().mean() * 100.0),
                "available_pct": float(panel_df[column].notna().mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_pct", "feature_name"]).reset_index(drop=True)


def _resolve_cross_sectional_group_key(prepared: pd.DataFrame) -> pd.Series:
    if "sector" in prepared.columns:
        sector_key = prepared["sector"].astype("string").str.strip()
    else:
        sector_map = get_project_sector_map()
        sector_key = prepared["ticker"].astype("string").str.upper().map(sector_map).astype("string")
    sector_key = sector_key.mask(sector_key.eq("")).fillna("unknown_sector")

    if "industry" in prepared.columns:
        industry_key = prepared["industry"].astype("string").str.strip()
    elif "industry_classification_raw" in prepared.columns:
        industry_key = prepared["industry_classification_raw"].astype("string").str.split(";").str[0].str.strip()
    else:
        industry_key = pd.Series(pd.NA, index=prepared.index, dtype="string")
    industry_key = industry_key.mask(industry_key.eq("")).mask(industry_key.str.lower().isin(["unknown", "unknown_industry"]))

    fiscal_year_source = prepared.get("event_fiscal_year")
    if fiscal_year_source is None:
        fiscal_year_source = prepared.get("fiscal_year")
    if fiscal_year_source is None:
        effective_dates = pd.to_datetime(prepared.get("effective_model_date"), errors="coerce")
        fiscal_year_source = effective_dates.dt.year
    fiscal_year = pd.Series(pd.to_numeric(fiscal_year_source, errors="coerce"), index=prepared.index).astype("Int64").astype("string")

    fiscal_period = prepared.get("event_fiscal_period")
    if fiscal_period is None:
        fiscal_period = prepared.get("fiscal_period")
    if fiscal_period is None:
        effective_dates = pd.to_datetime(prepared.get("effective_model_date"), errors="coerce")
        quarter = effective_dates.dt.quarter.astype("Int64").astype("string")
        fiscal_period = pd.Series("Q" + quarter.fillna("unknown"), index=prepared.index, dtype="string")
    fiscal_period = pd.Series(fiscal_period, index=prepared.index).astype("string")

    quarter_key = ("fy" + fiscal_year.fillna("unknown") + "_" + fiscal_period.fillna("unknown")).astype("string")

    industry_group_key = (quarter_key + "__industry__" + industry_key.fillna("missing")).astype("string")
    industry_group_size = industry_group_key.groupby(industry_group_key, dropna=False).transform("size")
    use_industry = industry_key.notna() & industry_group_size.ge(MIN_INDUSTRY_CROSS_SECTIONAL_GROUP_SIZE)
    use_sector = ~use_industry & sector_key.ne("unknown_sector")

    group_key = pd.Series(pd.NA, index=prepared.index, dtype="string")
    group_key.loc[use_industry] = (
        quarter_key.loc[use_industry] + "__industry__" + industry_key.loc[use_industry]
    ).astype("string")
    group_key.loc[use_sector] = (quarter_key.loc[use_sector] + "__sector__" + sector_key.loc[use_sector]).astype("string")
    group_key = group_key.fillna((quarter_key + "__global").astype("string"))
    return group_key.astype("string")


def build_quarterly_feature_design_panel(panel_df: pd.DataFrame, price_df: pd.DataFrame | None = None) -> pd.DataFrame:
    prepared = panel_df.copy()
    sort_columns = ["ticker", "effective_model_date", "event_date", "source_id"]
    prepared["effective_model_date"] = pd.to_datetime(prepared["effective_model_date"], errors="coerce")
    prepared["event_date"] = pd.to_datetime(prepared["event_date"], errors="coerce")
    prepared = prepared.sort_values(sort_columns).reset_index(drop=True)
    cross_sectional_group_key = _resolve_cross_sectional_group_key(prepared)

    grouped = prepared.groupby("ticker", sort=False)
    _ensure_numeric_columns(
        prepared,
        QUARTERLY_DELTA_SOURCE_COLUMNS
        + QFD_V2_LEVEL_SOURCE_COLUMNS
        + QFD_V3_DELTA_SOURCE_COLUMNS
        + [
            "profitability_profile_score",
            "sec_sentiment_score",
            "sec_positive_prob",
            "sec_negative_prob",
            "sec_neutral_prob",
            "sec_chunk_count",
            "sec_log_chunk_count",
        ],
    )

    for column in QUARTERLY_DELTA_SOURCE_COLUMNS:
        previous = grouped[column].shift(1)
        prepared[f"qfd_delta_{column}"] = prepared[column] - previous

    for column in QFD_V2_LEVEL_SOURCE_COLUMNS:
        prepared[f"qfd_v2_{column}_lvl"] = prepared[column]

    for column in QFD_V2_QOQ_CHANGE_SOURCE_COLUMNS:
        prepared[f"qfd_v2_{column}_d1"] = prepared[column] - grouped[column].shift(1)

    for column in QFD_V3_DELTA_SOURCE_COLUMNS:
        prepared[f"qfd_v3_{column}_d1"] = prepared[column] - grouped[column].shift(1)

    grouped = prepared.groupby("ticker", sort=False)
    for column in QFD_V2_YOY_CHANGE_SOURCE_COLUMNS:
        prepared[f"qfd_v2_{column}_d4"] = prepared[column] - grouped[column].shift(4)

    for column in QFD_V2_ACCELERATION_SOURCE_COLUMNS:
        if column in {"operating_margin", "net_margin"}:
            delta_series = prepared[f"qfd_v2_{column}_d1"]
            prepared[f"qfd_v2_{column}_accel"] = delta_series - grouped[f"qfd_v2_{column}_d1"].shift(1)
        else:
            prepared[f"qfd_v2_{column}_accel"] = prepared[column] - grouped[column].shift(1)

    prepared["qfd_v2_operating_margin_pos_count_stb4q"] = grouped["qfd_v2_operating_margin_d1"].transform(
        lambda series: _rolling_positive_count(series, window=4, min_periods=3)
    )
    prepared["qfd_v2_net_margin_pos_count_stb4q"] = grouped["qfd_v2_net_margin_d1"].transform(
        lambda series: _rolling_positive_count(series, window=4, min_periods=3)
    )

    for column in QFD_V2_STABILITY_SOURCE_COLUMNS:
        prepared[f"qfd_v2_{column}_stb4q"] = grouped[column].transform(lambda series: _trailing_std(series, window=4))

    sector_grouped = prepared.groupby(cross_sectional_group_key, dropna=False, sort=False)
    prepared["qfd_v2_roa_zsec"] = sector_grouped["roa"].transform(_group_zscore)
    prepared["qfd_v2_debt_to_assets_zsec"] = sector_grouped["debt_to_assets"].transform(_group_zscore)
    prepared["qfd_v2_operating_margin_d1_zsec"] = sector_grouped["qfd_v2_operating_margin_d1"].transform(_group_zscore)
    prepared["qfd_v2_accruals_ratio_zsec"] = sector_grouped["accruals_ratio"].transform(_group_zscore)
    prepared["qfd_v2_operating_margin_d1_ranksec"] = sector_grouped["qfd_v2_operating_margin_d1"].transform(
        _group_percentile_rank
    )
    prepared["qfd_v2_profitability_profile_score_ranksec"] = sector_grouped["profitability_profile_score"].transform(
        _group_percentile_rank
    )
    prepared["qfd_v2_debt_to_assets_d1_ranksec"] = sector_grouped["qfd_v2_debt_to_assets_d1"].transform(
        _group_percentile_rank
    )

    _ensure_numeric_columns(
        prepared,
        [
            "av_latest_prior_eps_surprise_pct_before_event",
            "av_trailing_4q_eps_surprise_pct_mean",
            "av_trailing_4q_eps_surprise_pct_std",
            "av_latest_prior_eps_surprise_before_event",
            "av_trailing_4q_eps_surprise_mean",
            "av_trailing_4q_eps_surprise_std",
            "av_eps_estimate_revision_30d",
            "av_eps_estimate_revision_90d",
            "av_eps_estimate_analyst_count_before_event",
            "av_revenue_estimate_analyst_count_before_event",
            "av_days_since_last_earnings_release",
            "days_since_prior_event",
            "days_since_prior_same_event_type",
        ],
    )

    prepared["qfd_margin_delta_combo"] = prepared["qfd_delta_operating_margin"] + prepared["qfd_delta_net_margin"]
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
        prepared["av_latest_prior_eps_surprise_pct_before_event"] - prepared["av_trailing_4q_eps_surprise_pct_mean"]
    )
    prepared["qfd_av_latest_surprise_vs_trailing_pct_abs"] = prepared["qfd_av_latest_surprise_vs_trailing_pct"].abs()
    latest_surprise_valid = (
        prepared["av_latest_prior_eps_surprise_pct_before_event"].notna()
        & prepared["av_trailing_4q_eps_surprise_pct_mean"].notna()
    )
    prepared["qfd_av_latest_surprise_above_trailing_flag"] = pd.Series(np.nan, index=prepared.index, dtype="float64")
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
        prepared["av_eps_estimate_revision_30d"] * prepared["av_latest_prior_eps_surprise_pct_before_event"]
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
        prepared["av_eps_estimate_revision_30d"] + prepared["av_latest_prior_eps_surprise_pct_before_event"]
    )
    revision_valid = prepared["av_eps_estimate_revision_30d"].notna()
    surprise_valid = prepared["av_latest_prior_eps_surprise_pct_before_event"].notna()
    joint_valid = revision_valid & surprise_valid
    prepared["qfd_av_revision_surprise_same_sign_flag"] = pd.Series(np.nan, index=prepared.index, dtype="float64")
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
        prepared["av_latest_prior_eps_surprise_pct_before_event"] * prepared["overall_financial_health_score"]
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
    prepared["qfd_event_gap_difference"] = prepared["days_since_prior_event"] - prepared["days_since_prior_same_event_type"]
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

    prepared["qfd_es_filing_sentiment_score"] = prepared["sec_sentiment_score"]
    prepared["qfd_es_filing_positive_prob"] = prepared["sec_positive_prob"]
    prepared["qfd_es_filing_negative_prob"] = prepared["sec_negative_prob"]
    prepared["qfd_es_filing_neutral_prob"] = prepared["sec_neutral_prob"]
    prepared["qfd_es_filing_sentiment_abs"] = prepared["qfd_es_filing_sentiment_score"].abs()
    prepared["qfd_es_filing_sentiment_delta_prev_q"] = grouped["sec_sentiment_score"].diff()
    prepared["qfd_es_filing_positive_prob_delta_prev_q"] = grouped["sec_positive_prob"].diff()
    prepared["qfd_es_filing_negative_prob_delta_prev_q"] = grouped["sec_negative_prob"].diff()
    prepared["qfd_es_negative_tone_jump"] = prepared["qfd_es_filing_negative_prob_delta_prev_q"]

    sentiment_trailing_mean_90d = _group_trailing_mean_time_window(
        prepared,
        value_column="sec_sentiment_score",
        timestamp_column="effective_model_date",
        group_column="ticker",
        window_days=90,
    )
    negative_trailing_mean_90d = _group_trailing_mean_time_window(
        prepared,
        value_column="sec_negative_prob",
        timestamp_column="effective_model_date",
        group_column="ticker",
        window_days=90,
    )
    prepared["qfd_es_sentiment_surprise_vs_90d"] = prepared["sec_sentiment_score"] - sentiment_trailing_mean_90d
    prepared["qfd_es_negative_surprise_vs_90d"] = prepared["sec_negative_prob"] - negative_trailing_mean_90d

    probability_frame = prepared[["qfd_es_filing_positive_prob", "qfd_es_filing_negative_prob", "qfd_es_filing_neutral_prob"]]
    prepared["qfd_es_sentiment_uncertainty"] = 1.0 - probability_frame.max(axis=1)
    clipped_probabilities = probability_frame.clip(lower=1e-12)
    prepared["qfd_es_sentiment_entropy"] = (
        -(clipped_probabilities * np.log(clipped_probabilities)).sum(axis=1)
    ).where(probability_frame.notna().all(axis=1), np.nan)
    prepared["qfd_es_sentiment_polarity_balance"] = (
        1.0 - (prepared["qfd_es_filing_positive_prob"] - prepared["qfd_es_filing_negative_prob"]).abs()
    ).where(
        prepared["qfd_es_filing_positive_prob"].notna() & prepared["qfd_es_filing_negative_prob"].notna(),
        np.nan,
    )
    prepared["qfd_es_source_count"] = prepared["qfd_es_filing_sentiment_score"].notna().astype("float64")
    prepared["qfd_es_text_chunk_count"] = prepared["sec_chunk_count"]
    prepared["qfd_es_log_text_chunk_count"] = prepared["sec_log_chunk_count"]

    if price_df is not None:
        prepared = _attach_event_aware_market_features(prepared, price_df)
    else:
        for column in QFD_EVENT_AWARE_MARKET_COLUMNS:
            if column not in prepared.columns:
                prepared[column] = np.nan

    return prepared
