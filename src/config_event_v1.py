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

from src.paths import DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR

EVENT_V1_NAME = "event_v1"
DEFAULT_PANEL_NAME = "event_v1_layer1"
PANEL_CHOICES = [
    "event_v1_layer1",
    "event_v1_layer1_analyst",
    "event_v1_layer1_secfilings",
    "event_v1_layer1_sec8kgrouped",
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
    "interest_coverage",
    "total_debt_to_assets",
    "capex_intensity",
    "free_cash_flow",
    "free_cash_flow_margin",
    "free_cash_flow_to_net_income",
    "leverage_change_qoq",
    "shareholder_payout_ratio",
    "revenue_growth_qoq",
    "revenue_growth_yoy",
    "earnings_growth_qoq",
    "earnings_growth_yoy",
    "cfo_to_net_income",
    "accruals_ratio",
    "liquidity_profile_score",
    "solvency_profile_score",
    "profitability_profile_score",
    "growth_quality_profile_score",
    "overall_financial_health_score",
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

ANALYST_EVENT_FEATURE_COLUMNS = [
    "analyst_event_count_1d",
    "analyst_event_count_5d",
    "analyst_upgrade_count_5d",
    "analyst_downgrade_count_5d",
    "analyst_reiterate_count_5d",
    "analyst_pt_up_count_5d",
    "analyst_pt_down_count_5d",
    "analyst_net_revision_score_5d",
    "analyst_mean_sentiment_1d",
    "analyst_mean_sentiment_5d",
    "analyst_sentiment_std_5d",
    "analyst_days_since_event",
]

SEC_FILING_EVENT_FEATURE_COLUMNS = [
    "sec_is_8k_today",
    "sec_is_10q_today",
    "sec_is_10k_today",
    "sec_filing_count_1d",
    "sec_filing_count_5d",
    "sec_after_close_filing_count_1d",
    "sec_pre_market_filing_count_1d",
    "sec_8k_decay_3d",
    "sec_8k_decay_5d",
    "sec_10q_decay_3d",
    "sec_10k_decay_3d",
    "sec_days_since_any_filing",
    "sec_days_since_8k",
    "sec_days_since_10q",
    "sec_days_since_10k",
]

SEC_8K_GROUPED_EVENT_FEATURE_COLUMNS = [
    "sec_8k_earnings_results_today",
    "sec_8k_guidance_outlook_today",
    "sec_8k_leadership_governance_today",
    "sec_8k_financing_securities_today",
    "sec_8k_material_agreement_mna_today",
    "sec_8k_regulatory_legal_today",
    "sec_8k_earnings_results_decay_3d",
    "sec_8k_guidance_outlook_decay_3d",
    "sec_8k_leadership_governance_decay_3d",
    "sec_8k_financing_securities_decay_3d",
    "sec_8k_material_agreement_mna_decay_3d",
    "sec_8k_regulatory_legal_decay_3d",
    "sec_8k_item_count_1d",
    "sec_8k_item_count_5d",
    "sec_days_since_8k_earnings_results",
    "sec_days_since_8k_guidance_outlook",
    "sec_days_since_8k_leadership_governance",
]

PRICE_INPUT_PATH = INTERIM_DATA_DIR / "prices" / "prices_with_labels.parquet"
LAYER1_BASE_PANEL_PATH = PROCESSED_DATA_DIR / "modeling" / "layer1_modeling_panel.parquet"
FULL_SENTIMENT_INPUT_PATH = INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_features.parquet"
MDA_SENTIMENT_INPUT_PATH = INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_mda_features.parquet"
ANALYST_INPUT_PATH = RAW_DATA_DIR / "analyst" / "analyst_ratings_processed.csv"

LABEL_OUTPUT_PATH = INTERIM_DATA_DIR / "labels" / "labels_event_v1.parquet"
MARKET_FEATURE_V2_OUTPUT_PATH = INTERIM_DATA_DIR / "features" / "layer2_market_features_v2.parquet"
SENTIMENT_EVENT_V1_OUTPUT_PATH = (
    INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_event_v1.parquet"
)
ANALYST_EVENT_V1_OUTPUT_PATH = INTERIM_DATA_DIR / "analyst" / "layer3_analyst_event_v1.parquet"
SEC_FILING_EVENTS_V1_OUTPUT_PATH = INTERIM_DATA_DIR / "sec" / "layer3_sec_filing_events_v1.parquet"
SEC_FILING_METADATA_V1_PATH = INTERIM_DATA_DIR / "sec" / "sec_filing_metadata_v1.parquet"
SEC_8K_GROUPED_EVENTS_V1_OUTPUT_PATH = (
    INTERIM_DATA_DIR / "sec" / "layer3_sec_8k_grouped_events_v1.parquet"
)

EVENT_V1_LAYER1_PANEL_PATH = PROCESSED_DATA_DIR / "modeling" / "event_v1_layer1_panel.parquet"
EVENT_V1_LAYER1_ANALYST_PANEL_PATH = (
    DATA_DIR / "modeling" / "event_v1" / "event_v1_layer1_analyst_panel.parquet"
)
EVENT_V1_LAYER1_SECFILINGS_PANEL_PATH = (
    DATA_DIR / "modeling" / "event_v1" / "event_v1_layer1_secfilings_panel.parquet"
)
EVENT_V1_LAYER1_SEC8KGROUPED_PANEL_PATH = (
    DATA_DIR / "modeling" / "event_v1" / "event_v1_layer1_sec8kgrouped_panel.parquet"
)
EVENT_V1_LAYER1_LAYER2_PANEL_PATH = (
    PROCESSED_DATA_DIR / "modeling" / "event_v1_layer1_layer2_panel.parquet"
)
EVENT_V1_FULL_PANEL_PATH = PROCESSED_DATA_DIR / "modeling" / "event_v1_full_panel.parquet"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_RESULTS_DIR = REPORTS_DIR / "results"
EVENT_V1_LAYER1_METRICS_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_metrics.json"
EVENT_V1_LAYER1_ANALYST_JSON_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_analyst.json"
EVENT_V1_LAYER1_ANALYST_MD_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_analyst.md"
EVENT_V1_LAYER1_SECFILINGS_JSON_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_secfilings.json"
EVENT_V1_LAYER1_SECFILINGS_MD_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_secfilings.md"
EVENT_V1_LAYER1_SEC8KGROUPED_JSON_PATH = (
    REPORTS_RESULTS_DIR / "event_v1_layer1_sec8kgrouped.json"
)
EVENT_V1_LAYER1_SEC8KGROUPED_MD_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_sec8kgrouped.md"
EVENT_V1_LAYER1_LAYER2_METRICS_PATH = (
    REPORTS_RESULTS_DIR / "event_v1_layer1_layer2_metrics.json"
)
EVENT_V1_FULL_METRICS_PATH = REPORTS_RESULTS_DIR / "event_v1_full_metrics.json"

EVENT_V1_LAYER1_PREDICTIONS_PATH = REPORTS_RESULTS_DIR / "event_v1_layer1_predictions.parquet"
EVENT_V1_LAYER1_ANALYST_PREDICTIONS_PATH = (
    REPORTS_RESULTS_DIR / "event_v1_layer1_analyst_predictions.parquet"
)
EVENT_V1_LAYER1_SECFILINGS_PREDICTIONS_PATH = (
    REPORTS_RESULTS_DIR / "event_v1_layer1_secfilings_predictions.parquet"
)
EVENT_V1_LAYER1_SEC8KGROUPED_PREDICTIONS_PATH = (
    REPORTS_RESULTS_DIR / "event_v1_layer1_sec8kgrouped_predictions.parquet"
)
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
        ANALYST_EVENT_V1_OUTPUT_PATH.parent,
        SEC_FILING_EVENTS_V1_OUTPUT_PATH.parent,
        SEC_8K_GROUPED_EVENTS_V1_OUTPUT_PATH.parent,
        EVENT_V1_LAYER1_PANEL_PATH.parent,
        EVENT_V1_LAYER1_ANALYST_PANEL_PATH.parent,
        EVENT_V1_LAYER1_SECFILINGS_PANEL_PATH.parent,
        EVENT_V1_LAYER1_SEC8KGROUPED_PANEL_PATH.parent,
        REPORTS_RESULTS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def get_panel_path(panel_name: str) -> Path:
    """Return the panel parquet path for the requested event_v1 setup."""
    mapping = {
        "event_v1_layer1": EVENT_V1_LAYER1_PANEL_PATH,
        "event_v1_layer1_analyst": EVENT_V1_LAYER1_ANALYST_PANEL_PATH,
        "event_v1_layer1_secfilings": EVENT_V1_LAYER1_SECFILINGS_PANEL_PATH,
        "event_v1_layer1_sec8kgrouped": EVENT_V1_LAYER1_SEC8KGROUPED_PANEL_PATH,
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
        "event_v1_layer1_analyst": EVENT_V1_LAYER1_ANALYST_JSON_PATH,
        "event_v1_layer1_secfilings": EVENT_V1_LAYER1_SECFILINGS_JSON_PATH,
        "event_v1_layer1_sec8kgrouped": EVENT_V1_LAYER1_SEC8KGROUPED_JSON_PATH,
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
        "event_v1_layer1_analyst": EVENT_V1_LAYER1_ANALYST_PREDICTIONS_PATH,
        "event_v1_layer1_secfilings": EVENT_V1_LAYER1_SECFILINGS_PREDICTIONS_PATH,
        "event_v1_layer1_sec8kgrouped": EVENT_V1_LAYER1_SEC8KGROUPED_PREDICTIONS_PATH,
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
    if panel_name == "event_v1_layer1_analyst":
        return list(LAYER1_FEATURE_COLUMNS + ANALYST_EVENT_FEATURE_COLUMNS)
    if panel_name == "event_v1_layer1_secfilings":
        return list(LAYER1_FEATURE_COLUMNS + SEC_FILING_EVENT_FEATURE_COLUMNS)
    if panel_name == "event_v1_layer1_sec8kgrouped":
        return list(LAYER1_FEATURE_COLUMNS + SEC_8K_GROUPED_EVENT_FEATURE_COLUMNS)
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


def get_markdown_output_path(panel_name: str) -> Path:
    """Return the Markdown results path for a panel choice."""
    mapping = {
        "event_v1_layer1": REPORTS_RESULTS_DIR / "event_v1_layer1.md",
        "event_v1_layer1_analyst": EVENT_V1_LAYER1_ANALYST_MD_PATH,
        "event_v1_layer1_secfilings": EVENT_V1_LAYER1_SECFILINGS_MD_PATH,
        "event_v1_layer1_sec8kgrouped": EVENT_V1_LAYER1_SEC8KGROUPED_MD_PATH,
        "event_v1_layer1_layer2": REPORTS_RESULTS_DIR / "event_v1_layer1_layer2.md",
        "event_v1_full": REPORTS_RESULTS_DIR / "event_v1_full.md",
    }
    try:
        return mapping[panel_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported event_v1 panel: {panel_name}") from exc


def get_sentiment_input_path(sentiment_source: str) -> Path:
    """Return the filing-level sentiment input path for the selected source."""
    if sentiment_source == "full":
        return FULL_SENTIMENT_INPUT_PATH
    if sentiment_source == "mda":
        return MDA_SENTIMENT_INPUT_PATH
    raise ValueError(f"Unsupported sentiment source: {sentiment_source}")
