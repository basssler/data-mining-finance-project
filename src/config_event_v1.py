"""Shared configuration for the additive event_v1 experiment lane.

This module is the single source of truth for:
- default evaluation policy
- event_v1 dataset and report paths
- candidate feature lists per panel

The locked benchmark path remains untouched. All event_v1 artifacts write to
new, versioned locations only.
"""

from __future__ import annotations

from pathlib import Path

from src.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT

EVENT_V1_NAME = "event_v1"
DEFAULT_PANEL_NAME = "event_v1_layer1"
PANEL_CHOICES = [
    "event_v1_layer1",
    "event_v1_layer1_layer2",
    "event_v1_full",
]

DEFAULT_HORIZON_DAYS = 5
DEFAULT_EMBARGO_DAYS = 5
DEFAULT_N_SPLITS = 5
DEFAULT_MIN_TRAIN_DATES = 756
DEFAULT_HOLDOUT_START = "2024-01-01"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_MISSINGNESS_PCT = 20.0
DEFAULT_CLIP_LOWER_QUANTILE = 0.01
DEFAULT_CLIP_UPPER_QUANTILE = 0.99
DEFAULT_BENCHMARK_MODE = "sector_equal_weight_ex_self"
DEFAULT_SENTIMENT_SOURCE = "full"
DEFAULT_NEUTRAL_BAND_BPS: int | None = None

IDENTIFIER_COLUMNS = ["ticker", "date"]
LABEL_COLUMNS = [
    "forward_return_5d",
    "benchmark_forward_return_5d",
    "excess_forward_return_5d",
    "target_event_v1",
    "within_neutral_band",
]
TARGET_COLUMN = "target_event_v1"

LAYER1_FEATURE_COLUMNS = [
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "working_capital_to_total_assets",
    "debt_to_equity",
    "debt_to_assets",
    "long_term_debt_ratio",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roa",
    "roe",
    "asset_turnover",
    "inventory_turnover",
    "receivables_turnover",
    "revenue_growth_qoq",
    "revenue_growth_yoy",
    "earnings_growth_qoq",
    "earnings_growth_yoy",
    "cfo_to_net_income",
    "accruals_ratio",
]

LAYER1_METADATA_COLUMNS = [
    "ticker",
    "date",
    "cik",
    "filing_date",
    "period_end",
    "fiscal_period",
    "fiscal_year",
    "form_type",
]

LAYER2_V2_FEATURE_COLUMNS = [
    "rel_return_5d",
    "rel_return_10d",
    "rel_return_21d",
    "realized_vol_21d",
    "realized_vol_63d",
    "vol_ratio_21d_63d",
    "beta_63d_to_sector",
    "overnight_gap_1d",
    "abs_return_shock_1d",
    "drawdown_21d",
    "return_zscore_21d",
    "volume_ratio_20d",
    "log_volume",
    "abnormal_volume_flag",
]

LAYER3_EVENT_FEATURE_COLUMNS = [
    "sec_event_score_latest",
    "sec_event_abs_latest",
    "sec_event_days_since_filing",
    "sec_event_decay_30d",
    "sec_event_score_decayed",
    "sec_event_delta_prev",
    "sec_event_abs_delta_prev",
    "sec_event_neg_to_pos_flip",
    "sec_event_pos_to_neg_flip",
    "sec_event_uncertainty",
]

LAYER3_EVENT_INTERACTION_COLUMNS = [
    "sec_event_delta_x_vol21",
    "sec_event_magnitude_x_days_since",
    "sec_event_negative_x_abnormal_volume",
]

PRICE_INPUT_PATH = INTERIM_DATA_DIR / "prices" / "prices_with_labels.parquet"
LAYER1_BASE_PANEL_PATH = PROCESSED_DATA_DIR / "modeling" / "layer1_modeling_panel.parquet"
FULL_SENTIMENT_INPUT_PATH = INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_features.parquet"
MDA_SENTIMENT_INPUT_PATH = INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_mda_features.parquet"

LABEL_OUTPUT_PATH = INTERIM_DATA_DIR / "labels" / "labels_event_v1.parquet"
MARKET_FEATURE_V2_OUTPUT_PATH = INTERIM_DATA_DIR / "features" / "layer2_market_features_v2.parquet"
SENTIMENT_EVENT_V1_OUTPUT_PATH = (
    INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_event_v1.parquet"
)

EVENT_V1_LAYER1_PANEL_PATH = PROCESSED_DATA_DIR / "modeling" / "event_v1_layer1_panel.parquet"
EVENT_V1_LAYER1_LAYER2_PANEL_PATH = (
    PROCESSED_DATA_DIR / "modeling" / "event_v1_layer1_layer2_panel.parquet"
)
EVENT_V1_FULL_PANEL_PATH = PROCESSED_DATA_DIR / "modeling" / "event_v1_full_panel.parquet"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_RESULTS_DIR = REPORTS_DIR / "results"
EVENT_V1_LAYER1_METRICS_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_metrics.json"
EVENT_V1_LAYER1_LAYER2_METRICS_PATH = (
    REPORTS_RESULTS_DIR / "event_v1_layer1_layer2_metrics.json"
)
EVENT_V1_FULL_METRICS_PATH = REPORTS_RESULTS_DIR / "event_v1_full_metrics.json"

EVENT_V1_LAYER1_PREDICTIONS_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_predictions.parquet"
EVENT_V1_LAYER1_LAYER2_PREDICTIONS_PATH = (
    REPORTS_RESULTS_DIR / "event_v1_layer1_layer2_predictions.parquet"
)
EVENT_V1_FULL_PREDICTIONS_PATH = REPORTS_RESULTS_DIR / "event_v1_full_predictions.parquet"
EVENT_V1_SUMMARY_PATH = REPORTS_RESULTS_DIR / "event_v1_summary.md"


def ensure_event_v1_directories() -> None:
    """Create event_v1 output directories if they do not already exist."""
    for folder in [
        LABEL_OUTPUT_PATH.parent,
        MARKET_FEATURE_V2_OUTPUT_PATH.parent,
        SENTIMENT_EVENT_V1_OUTPUT_PATH.parent,
        EVENT_V1_LAYER1_PANEL_PATH.parent,
        REPORTS_RESULTS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def get_panel_path(panel_name: str) -> Path:
    """Return the panel parquet path for the requested event_v1 setup."""
    mapping = {
        "event_v1_layer1": EVENT_V1_LAYER1_PANEL_PATH,
        "event_v1_layer1_layer2": EVENT_V1_LAYER1_LAYER2_PANEL_PATH,
        "event_v1_full": EVENT_V1_FULL_PANEL_PATH,
    }
    try:
        return mapping[panel_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported event_v1 panel: {panel_name}") from exc


def get_metrics_output_path(panel_name: str) -> Path:
    """Return the metrics output path for a panel choice."""
    mapping = {
        "event_v1_layer1": EVENT_V1_LAYER1_METRICS_PATH,
        "event_v1_layer1_layer2": EVENT_V1_LAYER1_LAYER2_METRICS_PATH,
        "event_v1_full": EVENT_V1_FULL_METRICS_PATH,
    }
    try:
        return mapping[panel_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported event_v1 panel: {panel_name}") from exc


def get_predictions_output_path(panel_name: str) -> Path:
    """Return the predictions parquet path for a panel choice."""
    mapping = {
        "event_v1_layer1": EVENT_V1_LAYER1_PREDICTIONS_PATH,
        "event_v1_layer1_layer2": EVENT_V1_LAYER1_LAYER2_PREDICTIONS_PATH,
        "event_v1_full": EVENT_V1_FULL_PREDICTIONS_PATH,
    }
    try:
        return mapping[panel_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported event_v1 panel: {panel_name}") from exc


def get_candidate_feature_columns(panel_name: str) -> list[str]:
    """Return the candidate feature list for a given event_v1 panel."""
    if panel_name == "event_v1_layer1":
        return list(LAYER1_FEATURE_COLUMNS)
    if panel_name == "event_v1_layer1_layer2":
        return list(LAYER1_FEATURE_COLUMNS + LAYER2_V2_FEATURE_COLUMNS)
    if panel_name == "event_v1_full":
        return list(
            LAYER1_FEATURE_COLUMNS
            + LAYER2_V2_FEATURE_COLUMNS
            + LAYER3_EVENT_FEATURE_COLUMNS
            + LAYER3_EVENT_INTERACTION_COLUMNS
        )
    raise ValueError(f"Unsupported event_v1 panel: {panel_name}")


def get_sentiment_input_path(sentiment_source: str) -> Path:
    """Return the filing-level sentiment input path for the selected source."""
    if sentiment_source == "full":
        return FULL_SENTIMENT_INPUT_PATH
    if sentiment_source == "mda":
        return MDA_SENTIMENT_INPUT_PATH
    raise ValueError(f"Unsupported sentiment source: {sentiment_source}")
