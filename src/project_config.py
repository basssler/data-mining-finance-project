"""Shared configuration and filesystem paths for the WRDS stock pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.paths import (
    CONFIGS_DIR,
    DATA_DIR,
    DAILY_CONFIGS_DIR,
    DAILY_DOCS_DIR,
    DOCS_DIR,
    INTERIM_DATA_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    QUARTERLY_CONFIGS_DIR,
    QUARTERLY_DOCS_DIR,
    QUARTERLY_OUTPUTS_CHAMPIONS_DIR,
    QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR,
    QUARTERLY_OUTPUTS_DIR,
    QUARTERLY_OUTPUTS_EXPERIMENTS_DIR,
    QUARTERLY_OUTPUTS_FEATURES_DIR,
    QUARTERLY_OUTPUTS_LABELS_DIR,
    QUARTERLY_OUTPUTS_PANELS_DIR,
    QUARTERLY_OUTPUTS_VALIDATION_DIR,
    RAW_DATA_DIR,
)

CONFIG_PATH = PROJECT_ROOT / "project_config.yaml"
DEFAULT_PROJECT_CONFIG: dict[str, Any] = {
    "prediction_horizons": [63],
    "universe": {
        "name": "event_panel_v2_consumer_staples_34",
        "description": "Canonical quarterly event workflow universe with one row per filing event.",
        "tickers": [
            "WMT",
            "COST",
            "PG",
            "KO",
            "PM",
            "MDLZ",
            "PEP",
            "MO",
            "CL",
            "TGT",
            "MNST",
            "KR",
            "KDP",
            "ADM",
            "SYY",
            "KMB",
            "HSY",
            "DG",
            "CHD",
            "STZ",
            "DLTR",
            "GIS",
            "KHC",
            "TSN",
            "EL",
            "BG",
            "CLX",
            "MKC",
            "SJM",
            "CAG",
            "TAP",
            "HRL",
            "BF-B",
            "CPB",
        ],
    },
    "date_range": {"start": "2015-01-01", "end": "2024-12-31"},
    "frequency": "quarterly_event",
    "objective": "binary_classification",
    "label_definition": {
        "name": "event_v2_63d_sign",
        "benchmark_mode": "sector_equal_weight_ex_self",
        "positive_class_rule": "63_trading_day_excess_return_greater_than_zero",
        "negative_class_rule": "63_trading_day_excess_return_less_than_or_equal_to_zero",
    },
    "primary_id_fields": ["ticker", "event_date", "effective_model_date", "filing_type"],
    "classification_fields": {"sector_field": "gics_sector", "industry_field": "gics_industry"},
    "peer_groups": ["sector", "sector_equal_weight_ex_self"],
    "validation": {
        "train_validation_start": "2015-01-01",
        "train_validation_end": "2023-12-31",
        "holdout_start": "2024-01-01",
        "holdout_end": "2024-12-31",
        "n_splits": 5,
        "gap_days": 5,
        "min_train_days": 252,
        "scheme": "purged_expanding_window",
    },
    "paths": {
        "raw_wrds_dir": "data/raw",
        "interim_wrds_dir": "data/interim",
        "interim_feature_dir": "data/interim/features",
        "processed_dir": "data/processed",
        "reports_dir": "reports/results",
    },
    "daily_workflow": {
        "docs_dir": "docs/daily",
        "configs_dir": "configs/daily",
        "legacy_primary_config": "configs/daily/legacy_event_panel_v2_primary.yaml",
    },
    "quarterly_workflow": {
        "manifest_path": "configs/quarterly/experiments/benchmark_ladder.yaml",
        "docs_dir": "docs/quarterly",
        "outputs_dir": "outputs/quarterly",
        "baseline_config": "configs/event_panel_v2_quarterly.yaml",
        "candidate_config": "configs/event_panel_v2_quarterly_stability_core_additive.yaml",
        "registry_doc": "docs/quarterly/promoted_models.md",
    },
}


def load_project_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML project config when PyYAML is installed, else use built-in defaults."""
    if not config_path.exists():
        return DEFAULT_PROJECT_CONFIG.copy()
    try:
        import yaml  # type: ignore
    except ImportError:
        return DEFAULT_PROJECT_CONFIG.copy()

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Project config must deserialize to a dictionary.")
    return config


PROJECT_CONFIG = load_project_config()

PREDICTION_HORIZONS = [int(value) for value in PROJECT_CONFIG["prediction_horizons"]]
DATE_RANGE = PROJECT_CONFIG["date_range"]
VALIDATION_CONFIG = PROJECT_CONFIG["validation"]
UNIVERSE_CONFIG = PROJECT_CONFIG["universe"]
PATH_CONFIG = PROJECT_CONFIG["paths"]
QUARTERLY_WORKFLOW_CONFIG = PROJECT_CONFIG.get("quarterly_workflow", {})

RAW_WRDS_DIR = PROJECT_ROOT / PATH_CONFIG["raw_wrds_dir"]
INTERIM_WRDS_DIR = PROJECT_ROOT / PATH_CONFIG["interim_wrds_dir"]
INTERIM_FEATURE_DIR = PROJECT_ROOT / PATH_CONFIG["interim_feature_dir"]
PROCESSED_STOCK_DIR = PROJECT_ROOT / PATH_CONFIG["processed_dir"]
REPORTS_RESULTS_DIR = PROJECT_ROOT / PATH_CONFIG["reports_dir"]

WRDS_FUNDAMENTALS_PATH = RAW_WRDS_DIR / "wrds_compustat_fundamentals.parquet"
WRDS_CRSP_DAILY_PATH = RAW_WRDS_DIR / "wrds_crsp_daily.parquet"
WRDS_CCM_LINKS_PATH = RAW_WRDS_DIR / "wrds_ccm_links.parquet"
SECURITY_MASTER_PATH = INTERIM_WRDS_DIR / "security_master.parquet"
LABELED_PANEL_PATH = INTERIM_WRDS_DIR / "labeled_price_panel.parquet"
FUNDAMENTAL_FEATURE_PATH = INTERIM_FEATURE_DIR / "features_fundamental_daily.parquet"
MARKET_FEATURE_PATH = INTERIM_FEATURE_DIR / "features_market.parquet"
PEER_RELATIVE_FEATURE_PATH = INTERIM_FEATURE_DIR / "features_peer_relative.parquet"
MODEL_PANEL_PATH = PROCESSED_STOCK_DIR / "model_panel.parquet"
FEATURE_DICTIONARY_PATH = PROCESSED_STOCK_DIR / "feature_dictionary.json"
MISSINGNESS_REPORT_PATH = REPORTS_RESULTS_DIR / "stock_prediction_missingness.csv"
MARKET_MISSINGNESS_REPORT_PATH = REPORTS_RESULTS_DIR / "stock_prediction_market_missingness.csv"
PEER_RELATIVE_DICTIONARY_PATH = REPORTS_RESULTS_DIR / "stock_prediction_peer_features.json"
SPLIT_REPORT_PATH = REPORTS_RESULTS_DIR / "stock_prediction_splits.json"
DATA_SCHEMA_PATH = DOCS_DIR / "data_schema.md"
FEATURE_INVENTORY_PATH = DOCS_DIR / "feature_inventory.md"
VALIDATION_PLAN_PATH = DOCS_DIR / "validation_plan.md"
QUARTERLY_WORKFLOW_MANIFEST_PATH = PROJECT_ROOT / str(
    QUARTERLY_WORKFLOW_CONFIG.get("manifest_path", "configs/quarterly/experiments/benchmark_ladder.yaml")
)


def ensure_stock_prediction_directories() -> None:
    """Create the directories used by the new stock prediction lane."""
    for folder in [
        RAW_WRDS_DIR,
        INTERIM_WRDS_DIR,
        INTERIM_FEATURE_DIR,
        PROCESSED_STOCK_DIR,
        REPORTS_RESULTS_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        OUTPUTS_DIR,
        DOCS_DIR,
        CONFIGS_DIR,
        DAILY_CONFIGS_DIR,
        DAILY_DOCS_DIR,
        QUARTERLY_CONFIGS_DIR,
        QUARTERLY_DOCS_DIR,
        QUARTERLY_OUTPUTS_DIR,
        QUARTERLY_OUTPUTS_PANELS_DIR,
        QUARTERLY_OUTPUTS_LABELS_DIR,
        QUARTERLY_OUTPUTS_VALIDATION_DIR,
        QUARTERLY_OUTPUTS_FEATURES_DIR,
        QUARTERLY_OUTPUTS_EXPERIMENTS_DIR,
        QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR,
        QUARTERLY_OUTPUTS_CHAMPIONS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def dump_json(data: dict[str, Any] | list[Any], path: Path) -> None:
    """Persist JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
