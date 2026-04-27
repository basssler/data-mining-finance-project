"""Audit fundamental amount scale consistency across raw, feature, and panel layers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src import fundamentals_clean
from src.paths import INTERIM_DATA_DIR, OUTPUTS_DIR, RAW_DATA_DIR


DEFAULT_RAW_PATH = RAW_DATA_DIR / "fundamentals" / "raw_fundamentals_universe_v2.parquet"
DEFAULT_FEATURE_PATH = INTERIM_DATA_DIR / "features" / "layer1_financial_features_universe_v2.parquet"
DEFAULT_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_universe_v2.parquet"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "quarterly" / "diagnostics" / "fundamental_unit_consistency"

AMOUNT_CONCEPTS = {
    "accounts_receivable",
    "cash_and_cash_equivalents",
    "current_assets",
    "current_liabilities",
    "inventory",
    "long_term_debt",
    "net_income",
    "operating_cash_flow",
    "operating_income",
    "revenue",
    "shareholders_equity",
    "total_assets",
    "total_liabilities",
}

RATIO_LIMITS = {
    "asset_turnover": 5.0,
    "roa": 1.0,
    "roe": 5.0,
    "net_margin": 1.0,
    "operating_margin": 1.0,
    "debt_to_assets": 2.0,
    "current_ratio": 20.0,
}


def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing audit input: {path}")
    return pd.read_parquet(path)


def audit_raw_candidate_selection(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Flag cases where raw duplicate selection keeps a much smaller amount fact."""
    normalized = fundamentals_clean.normalize_raw_data(raw_df)
    normalized = normalized[normalized["concept_name"].isin(AMOUNT_CONCEPTS)].copy()
    if normalized.empty:
        return pd.DataFrame()

    deduped, _ = fundamentals_clean.deduplicate_concept_rows(normalized)
    selected = deduped[
        ["ticker", "period_end", "concept_name", "value", "unit", "filing_date", "form_type", "raw_tag", "source"]
    ].rename(
        columns={
            "value": "selected_value",
            "unit": "selected_unit",
            "filing_date": "selected_filing_date",
            "form_type": "selected_form_type",
            "raw_tag": "selected_raw_tag",
            "source": "selected_source",
        }
    )

    grouped = normalized.assign(abs_value=normalized["value"].abs())
    summary = (
        grouped.sort_values(["ticker", "period_end", "concept_name", "abs_value"], ascending=[True, True, True, False])
        .groupby(["ticker", "period_end", "concept_name"], dropna=False)
        .agg(
            candidate_count=("value", "size"),
            max_abs_candidate_value=("abs_value", "max"),
            max_abs_candidate_signed_value=("value", "first"),
            max_abs_candidate_filing_date=("filing_date", "first"),
            max_abs_candidate_form_type=("form_type", "first"),
            max_abs_candidate_unit=("unit", "first"),
            positive_candidate_count=("value", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
        .reset_index()
    )

    audit = selected.merge(summary, on=["ticker", "period_end", "concept_name"], how="left")
    audit["selected_abs_value"] = audit["selected_value"].abs()
    audit["selected_to_max_abs_ratio"] = audit["selected_abs_value"] / audit["max_abs_candidate_value"].replace(0, np.nan)
    audit["candidate_scale_flag"] = (
        (audit["candidate_count"] > 1)
        & audit["max_abs_candidate_value"].notna()
        & (audit["selected_to_max_abs_ratio"] < 0.25)
    )
    audit["selected_negative_with_positive_candidates_flag"] = (
        (audit["selected_value"] < 0)
        & (audit["positive_candidate_count"] > 0)
        & (audit["candidate_count"] > 1)
    )
    return audit.sort_values(
        ["candidate_scale_flag", "selected_negative_with_positive_candidates_flag", "selected_to_max_abs_ratio"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def audit_ratio_sanity(df: pd.DataFrame, layer_name: str) -> pd.DataFrame:
    """Flag extreme ratio values that usually indicate mixed units or bad fact selection."""
    rows = []
    available_limits = {column: limit for column, limit in RATIO_LIMITS.items() if column in df.columns}
    for column, limit in available_limits.items():
        values = pd.to_numeric(df[column], errors="coerce")
        mask = values.abs() > limit
        for idx in df.index[mask]:
            row = df.loc[idx]
            rows.append(
                {
                    "layer": layer_name,
                    "row_index": int(idx),
                    "ticker": row.get("ticker", pd.NA),
                    "event_date": row.get("event_date", pd.NA),
                    "period_end": row.get("period_end", pd.NA),
                    "filing_date": row.get("filing_date", row.get("fund_snapshot_filing_date", pd.NA)),
                    "feature": column,
                    "value": float(values.loc[idx]),
                    "absolute_limit": limit,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["layer", "row_index", "ticker", "event_date", "period_end", "filing_date", "feature", "value", "absolute_limit"]
        )
    return pd.DataFrame(rows).sort_values(["layer", "feature", "ticker", "period_end"]).reset_index(drop=True)


def audit_amount_relationships(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Flag amount relationships that are difficult to reconcile economically."""
    required = {"ticker", "period_end", "revenue", "total_assets"}
    if not required.issubset(feature_df.columns):
        return pd.DataFrame()

    df = feature_df.copy()
    revenue = pd.to_numeric(df["revenue"], errors="coerce")
    assets = pd.to_numeric(df["total_assets"], errors="coerce")
    mask = (revenue > 1_000_000_000) & (assets > 0) & (assets < revenue * 0.10)
    output = df.loc[mask, ["ticker", "period_end", "filing_date", "revenue", "total_assets"]].copy()
    if output.empty:
        return output
    output["assets_to_revenue"] = output["total_assets"] / output["revenue"]
    return output.sort_values(["assets_to_revenue", "ticker", "period_end"]).reset_index(drop=True)


def build_summary(
    raw_selection_audit: pd.DataFrame,
    feature_ratio_flags: pd.DataFrame,
    panel_ratio_flags: pd.DataFrame,
    amount_relationship_flags: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "check": "raw_candidate_scale_flags",
            "flagged_rows": int(raw_selection_audit.get("candidate_scale_flag", pd.Series(dtype=bool)).sum()),
            "total_rows": len(raw_selection_audit),
        },
        {
            "check": "raw_selected_negative_with_positive_candidates",
            "flagged_rows": int(raw_selection_audit.get("selected_negative_with_positive_candidates_flag", pd.Series(dtype=bool)).sum()),
            "total_rows": len(raw_selection_audit),
        },
        {
            "check": "feature_ratio_sanity_flags",
            "flagged_rows": len(feature_ratio_flags),
            "total_rows": len(feature_ratio_flags),
        },
        {
            "check": "panel_ratio_sanity_flags",
            "flagged_rows": len(panel_ratio_flags),
            "total_rows": len(panel_ratio_flags),
        },
        {
            "check": "feature_amount_relationship_flags",
            "flagged_rows": len(amount_relationship_flags),
            "total_rows": len(amount_relationship_flags),
        },
    ]
    return pd.DataFrame(rows)


def write_markdown_report(
    summary_df: pd.DataFrame,
    output_path: Path,
    raw_path: Path,
    feature_path: Path,
    panel_path: Path,
) -> None:
    flagged = int(summary_df["flagged_rows"].sum())
    lines = [
        "# Fundamental Unit Consistency Audit",
        "",
        "## Inputs",
        f"- Raw fundamentals: `{raw_path}`",
        f"- Layer 1 features: `{feature_path}`",
        f"- Event panel: `{panel_path}`",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Interpretation",
        (
            f"- Total flagged rows across checks: `{flagged}`. "
            "Raw candidate flags mean the selected clean fact is less than 25% of another raw candidate "
            "for the same ticker, period, and concept."
        ),
        "- Ratio flags usually indicate a mixed-unit, segmented-fact, or duration-selection issue upstream.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(raw_path: Path, feature_path: Path, panel_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = _read_parquet_if_exists(raw_path)
    feature_df = _read_parquet_if_exists(feature_path)
    panel_df = _read_parquet_if_exists(panel_path)

    raw_selection_audit = audit_raw_candidate_selection(raw_df)
    feature_ratio_flags = audit_ratio_sanity(feature_df, "layer1_features")
    panel_ratio_flags = audit_ratio_sanity(panel_df, "event_panel")
    amount_relationship_flags = audit_amount_relationships(feature_df)
    summary_df = build_summary(raw_selection_audit, feature_ratio_flags, panel_ratio_flags, amount_relationship_flags)

    paths = {
        "summary": output_dir / "summary.csv",
        "raw_selection": output_dir / "raw_candidate_selection_audit.csv",
        "feature_ratio_flags": output_dir / "feature_ratio_sanity_flags.csv",
        "panel_ratio_flags": output_dir / "panel_ratio_sanity_flags.csv",
        "amount_relationship_flags": output_dir / "feature_amount_relationship_flags.csv",
        "markdown": output_dir / "fundamental_unit_consistency_audit.md",
    }
    summary_df.to_csv(paths["summary"], index=False)
    raw_selection_audit.to_csv(paths["raw_selection"], index=False)
    feature_ratio_flags.to_csv(paths["feature_ratio_flags"], index=False)
    panel_ratio_flags.to_csv(paths["panel_ratio_flags"], index=False)
    amount_relationship_flags.to_csv(paths["amount_relationship_flags"], index=False)
    write_markdown_report(summary_df, paths["markdown"], raw_path, feature_path, panel_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit fundamental amount scale consistency across layers.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--feature-path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_audit(
        raw_path=Path(args.raw_path),
        feature_path=Path(args.feature_path),
        panel_path=Path(args.panel_path),
        output_dir=Path(args.output_dir),
    )
    print("Fundamental unit consistency audit complete")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
