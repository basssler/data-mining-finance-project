from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.label_comparison_event_v2 import (
    VariantSpec,
    attach_labels_to_event_panel,
    build_daily_label_table,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.train_event_panel_v2 import (
    load_config,
    load_event_panel,
    load_prebuilt_label_map,
    resolve_candidate_features,
    resolve_max_missingness_pct,
)


DEFAULT_BASE_PANEL = Path("outputs") / "quarterly" / "panels" / "quarterly_event_panel_features.parquet"
DEFAULT_ENRICHED_PANEL = (
    Path("outputs") / "quarterly" / "panels" / "quarterly_event_panel_sector_sentiment_capitaliq.parquet"
)
DEFAULT_OUTPUT = Path("outputs") / "quarterly" / "diagnostics" / "capitaliq_sentiment_pre_model_audit.md"
DEFAULT_CONFIGS = [
    Path("configs") / "quarterly" / "capitaliq_ladder_core.yaml",
    Path("configs") / "quarterly" / "capitaliq_ladder_core_plus_market.yaml",
    Path("configs") / "quarterly" / "capitaliq_ladder_raw_sentiment.yaml",
    Path("configs") / "quarterly" / "capitaliq_ladder_sector_adjusted_sentiment.yaml",
]
SENTIMENT_COLUMNS = [
    "sent_mean_7d",
    "sent_mean_30d",
    "sent_mean_63d",
    "sent_vol_30d",
    "sent_vol_63d",
    "news_count_7d",
    "news_count_30d",
    "news_count_63d",
    "sent_momentum_7v30",
    "sent_momentum_30v63",
    "confidence_mean_30d",
    "sector_sent_mean_30d",
    "sector_sent_mean_63d",
    "sector_news_count_30d",
    "sector_news_count_63d",
    "sector_adj_sent_30d",
    "sector_adj_sent_63d",
    "sector_adj_news_share_30d",
    "sector_adj_news_share_63d",
    "has_news_7d",
    "has_news_30d",
    "has_news_63d",
    "low_news_coverage_30d",
    "low_news_coverage_63d",
]
TEXT_OR_PROVENANCE_COLUMNS = {
    "headline",
    "capitaliq_headline",
    "capitaliq_source",
    "source",
    "source_file",
    "row_company",
    "exchange_ticker",
    "capitaliq_type",
    "date",
    "event_date",
    "event_date_raw",
    "prediction_date",
    "filing_date",
    "effective_model_date",
    "tradable_date",
    "feature_snapshot_timestamp",
}
TARGET_COLUMNS = {
    "target",
    "target_sign",
    "forward_return",
    "benchmark_forward_return",
    "excess_forward_return",
    "label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Capital IQ sentiment panel before modeling.")
    parser.add_argument("--base-panel", default=str(DEFAULT_BASE_PANEL))
    parser.add_argument("--enriched-panel", default=str(DEFAULT_ENRICHED_PANEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--configs", nargs="*", default=[str(path) for path in DEFAULT_CONFIGS])
    return parser.parse_args()


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _load_raw_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["ticker"] = frame["ticker"].astype("string")
    return frame


def _load_labels_for_config(config: dict) -> pd.DataFrame:
    label_cfg = config["label"]
    if label_cfg.get("path"):
        return load_prebuilt_label_map(Path(label_cfg["path"]))
    prices_df = normalize_price_data(load_price_data(Path(config["prices"]["path"])))
    return build_daily_label_table(
        prices_df,
        horizon_days=int(label_cfg["horizon_days"]),
        benchmark_mode=str(label_cfg.get("benchmark_mode", "sector_equal_weight_ex_self")),
    )


def _labeled_panel_for_config(panel_path: Path, config: dict) -> pd.DataFrame:
    panel = load_event_panel(panel_path)
    label_df = _load_labels_for_config(config)
    labeled = attach_labels_to_event_panel(panel, label_df)
    variant = VariantSpec(
        variant_name=str(config["label"]["variant_name"]),
        horizon_days=int(config["label"]["horizon_days"]),
        label_mode=str(config["label"]["mode"]),
        threshold=float(config["label"].get("threshold", 0.015)),
    )
    if variant.label_mode == "sign" and "target_sign" in labeled.columns:
        labeled["target"] = labeled["target_sign"]
    return labeled


def _feature_audit_for_config(config_path: Path, enriched_df: pd.DataFrame) -> dict:
    config = load_config(config_path)
    panel = load_event_panel(Path(config["panel"]["path"]))
    candidates = resolve_candidate_features(panel, config)
    explicit_exclusions = set(config.get("feature_exclusions", {}).get("explicit", []))
    post_explicit = [column for column in candidates if column not in explicit_exclusions]
    forbidden_text = sorted(set(post_explicit).intersection(TEXT_OR_PROVENANCE_COLUMNS))
    forbidden_target = sorted(set(post_explicit).intersection(TARGET_COLUMNS))
    missing_declared = [
        column
        for column in config.get("feature_inclusions", {}).get("additional", [])
        if column not in enriched_df.columns
    ]
    sentiment_in_features = [column for column in post_explicit if column in SENTIMENT_COLUMNS]
    pre_holdout = panel.loc[panel["date"] < pd.Timestamp(config["holdout"]["start"])]
    missingness = {column: float(pre_holdout[column].isna().mean() * 100.0) for column in post_explicit}
    max_missingness = resolve_max_missingness_pct(config)
    usable_precheck = [
        column
        for column in post_explicit
        if missingness[column] <= max_missingness and pre_holdout[column].dropna().nunique() > 1
    ]
    return {
        "config_path": str(config_path),
        "panel_path": str(config["panel"]["path"]),
        "models": list(config["models"]),
        "candidate_count": len(candidates),
        "post_explicit_count": len(post_explicit),
        "usable_precheck_count": len(usable_precheck),
        "sentiment_candidate_count": len(sentiment_in_features),
        "sentiment_usable_precheck_count": len([c for c in usable_precheck if c in SENTIMENT_COLUMNS]),
        "forbidden_text": forbidden_text,
        "forbidden_target": forbidden_target,
        "missing_declared": missing_declared,
    }


def run_audit(
    *,
    base_panel_path: Path,
    enriched_panel_path: Path,
    output_path: Path,
    config_paths: list[Path],
) -> bool:
    base_raw = _load_raw_panel(base_panel_path)
    enriched_raw = _load_raw_panel(enriched_panel_path)
    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "Row count matches base quarterly panel",
            len(base_raw) == len(enriched_raw),
            f"base={len(base_raw):,}; enriched={len(enriched_raw):,}",
        )
    )
    id_match = "event_id" in base_raw.columns and "event_id" in enriched_raw.columns and set(base_raw["event_id"]) == set(
        enriched_raw["event_id"]
    )
    checks.append(("Event ID set unchanged", id_match, "event_id sets compared one-to-one"))

    compare_cols = [column for column in ["ticker", "event_date", "effective_model_date", "tradable_date"] if column in base_raw]
    merged = base_raw[["event_id", *compare_cols]].merge(
        enriched_raw[["event_id", *compare_cols]],
        on="event_id",
        suffixes=("_base", "_enriched"),
        validate="one_to_one",
    )
    unchanged_cols = []
    changed_cols = []
    for column in compare_cols:
        left = merged[f"{column}_base"].astype("string").fillna("<NA>")
        right = merged[f"{column}_enriched"].astype("string").fillna("<NA>")
        (unchanged_cols if left.equals(right) else changed_cols).append(column)
    checks.append(
        (
            "Ticker/event-date/prediction-date fields unchanged",
            not changed_cols,
            f"unchanged={unchanged_cols}; changed={changed_cols}",
        )
    )

    duplicate_count = int(enriched_raw.duplicated(subset=["ticker", "event_date"]).sum())
    checks.append(("No duplicate ticker-event rows", duplicate_count == 0, f"duplicates={duplicate_count:,}"))

    missing_sentiment = [column for column in SENTIMENT_COLUMNS if column not in enriched_raw.columns]
    checks.append(("All expected sentiment columns exist", not missing_sentiment, f"missing={missing_sentiment}"))

    coverage_lines = []
    for column in ["has_news_7d", "has_news_30d", "has_news_63d"]:
        coverage = float(pd.to_numeric(enriched_raw[column], errors="coerce").fillna(0).mean() * 100.0)
        coverage_lines.append(f"{column}={coverage:.2f}%")
    holdout_mask = pd.to_datetime(enriched_raw["event_date"], errors="coerce") >= pd.Timestamp("2024-01-01")
    holdout_30d = float(pd.to_numeric(enriched_raw.loc[holdout_mask, "has_news_30d"], errors="coerce").fillna(0).mean() * 100.0)
    holdout_63d = float(pd.to_numeric(enriched_raw.loc[holdout_mask, "has_news_63d"], errors="coerce").fillna(0).mean() * 100.0)
    checks.append(
        (
            "2024 holdout sentiment coverage is nonzero",
            holdout_30d > 0 and holdout_63d > 0,
            f"2024 has_news_30d={holdout_30d:.2f}%; has_news_63d={holdout_63d:.2f}%",
        )
    )

    first_config = load_config(config_paths[0])
    base_labeled = _labeled_panel_for_config(base_panel_path, first_config)
    enriched_labeled = _labeled_panel_for_config(enriched_panel_path, first_config)
    label_compare_cols = [
        column
        for column in ["target", "target_sign", "forward_return", "benchmark_forward_return", "excess_forward_return"]
        if column in base_labeled.columns and column in enriched_labeled.columns
    ]
    label_join = base_labeled[["event_id", *label_compare_cols]].merge(
        enriched_labeled[["event_id", *label_compare_cols]],
        on="event_id",
        suffixes=("_base", "_enriched"),
        validate="one_to_one",
    )
    changed_label_cols = []
    for column in label_compare_cols:
        left = pd.to_numeric(label_join[f"{column}_base"], errors="coerce")
        right = pd.to_numeric(label_join[f"{column}_enriched"], errors="coerce")
        if not left.fillna(-999999999).equals(right.fillna(-999999999)):
            changed_label_cols.append(column)
    checks.append(("Labels unchanged from base panel", not changed_label_cols, f"changed={changed_label_cols}"))

    feature_audits = [_feature_audit_for_config(path, enriched_raw) for path in config_paths]
    for audit in feature_audits:
        checks.append(
            (
                f"No raw text/date/source/provenance model features in {Path(audit['config_path']).name}",
                not audit["forbidden_text"],
                f"forbidden={audit['forbidden_text']}",
            )
        )
        checks.append(
            (
                f"No target/label model features in {Path(audit['config_path']).name}",
                not audit["forbidden_target"],
                f"forbidden={audit['forbidden_target']}",
            )
        )
        checks.append(
            (
                f"All declared features exist in {Path(audit['config_path']).name}",
                not audit["missing_declared"],
                f"missing={audit['missing_declared']}",
            )
        )

    all_passed = all(condition for _, condition, _ in checks)
    lines = [
        "# Capital IQ Sentiment Pre-Model Audit",
        "",
        f"- Base panel: `{base_panel_path.as_posix()}`",
        f"- Enriched panel: `{enriched_panel_path.as_posix()}`",
        f"- Base rows: `{len(base_raw):,}`",
        f"- Enriched rows: `{len(enriched_raw):,}`",
        f"- Sentiment coverage: `{'; '.join(coverage_lines)}`",
        f"- 2024 holdout coverage: `has_news_30d={holdout_30d:.2f}%`; `has_news_63d={holdout_63d:.2f}%`",
        "",
        "## Checks",
    ]
    lines.extend([f"- **{_status(condition)}** {name}: {detail}" for name, condition, detail in checks])
    lines.extend(["", "## Feature Config Summary"])
    for audit in feature_audits:
        lines.extend(
            [
                f"- `{audit['config_path']}`",
                f"  - panel: `{audit['panel_path']}`",
                f"  - models: `{', '.join(audit['models'])}`",
                f"  - candidate features after explicit exclusions: `{audit['post_explicit_count']}`",
                f"  - usable precheck features: `{audit['usable_precheck_count']}`",
                f"  - sentiment candidates after explicit exclusions: `{audit['sentiment_candidate_count']}`",
                f"  - sentiment usable precheck features: `{audit['sentiment_usable_precheck_count']}`",
            ]
        )
    lines.extend(["", f"Overall result: **{_status(all_passed)}**", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit: {output_path}")
    print(f"Overall result: {_status(all_passed)}")
    return all_passed


def main() -> None:
    args = parse_args()
    ok = run_audit(
        base_panel_path=Path(args.base_panel),
        enriched_panel_path=Path(args.enriched_panel),
        output_path=Path(args.output),
        config_paths=[Path(path) for path in args.configs],
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
