"""Build a non-mutating before/after report for fundamental fact selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src import fundamentals_clean
from src.accounting_concepts import CONCEPT_SPECS, concept_priority_lookup, source_priority
from src.fundamental_fact_selector import select_fundamental_facts
from src.paths import OUTPUTS_DIR, RAW_DATA_DIR


DEFAULT_RAW_PATH = RAW_DATA_DIR / "fundamentals" / "raw_fundamentals_universe_v2.parquet"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "quarterly" / "diagnostics" / "fundamental_before_after"
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "TSLA"]


def legacy_select(df: pd.DataFrame) -> pd.DataFrame:
    facts = df.copy()
    priority_lookup = concept_priority_lookup()
    facts["_preferred_unit_rank"] = facts.apply(
        lambda row: 0
        if (
            CONCEPT_SPECS.get(str(row["concept_name"])) is not None
            and str(row["unit"]) == CONCEPT_SPECS[str(row["concept_name"])].preferred_unit
        )
        else 1,
        axis=1,
    )
    facts["_tag_priority_rank"] = facts.apply(
        lambda row: priority_lookup.get(str(row["concept_name"]), {}).get(str(row["raw_tag"]), 999),
        axis=1,
    )
    facts["_source_priority_rank"] = facts["source"].map(source_priority).fillna(9).astype("int64")
    facts = facts.sort_values(
        ["ticker", "period_end", "concept_name", "_preferred_unit_rank", "_tag_priority_rank", "_source_priority_rank", "filing_date"],
        ascending=[True, True, True, True, True, True, False],
    )
    return facts.drop_duplicates(subset=["ticker", "period_end", "concept_name"], keep="first").copy()


def selection_scale_flags(raw_df: pd.DataFrame, selected_df: pd.DataFrame) -> int:
    instant_like = [
        concept_name
        for concept_name, spec in CONCEPT_SPECS.items()
        if spec.concept_type in {"instant", "shares"}
    ]
    raw_df = raw_df.loc[raw_df["concept_name"].isin(instant_like)].copy()
    selected_df = selected_df.loc[selected_df["concept_name"].isin(instant_like)].copy()
    if raw_df.empty or selected_df.empty:
        return 0
    grouped = raw_df.assign(abs_value=raw_df["value"].abs()).groupby(
        ["ticker", "period_end", "concept_name"],
        dropna=False,
    )["abs_value"].max().reset_index(name="max_abs_candidate_value")
    merged = selected_df[["ticker", "period_end", "concept_name", "value"]].merge(
        grouped,
        on=["ticker", "period_end", "concept_name"],
        how="left",
    )
    ratio = merged["value"].abs() / merged["max_abs_candidate_value"].replace(0, pd.NA)
    return int((ratio < 0.25).sum())


def selected_negative_with_positive_flags(raw_df: pd.DataFrame, selected_df: pd.DataFrame) -> int:
    positives = raw_df.groupby(["ticker", "period_end", "concept_name"], dropna=False)["value"].apply(
        lambda s: bool((pd.to_numeric(s, errors="coerce") > 0).any())
    ).reset_index(name="has_positive_candidate")
    merged = selected_df[["ticker", "period_end", "concept_name", "value"]].merge(
        positives,
        on=["ticker", "period_end", "concept_name"],
        how="left",
    )
    return int(((merged["value"] < 0) & merged["has_positive_candidate"]).sum())


def aapl_snapshot(selected_df: pd.DataFrame, label: str) -> pd.DataFrame:
    subset = selected_df.loc[
        (selected_df["ticker"] == "AAPL")
        & (selected_df["concept_name"].isin(["revenue", "net_income", "total_assets"]))
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    latest_period = subset["period_end"].max()
    wide = subset.loc[subset["period_end"] == latest_period].pivot_table(
        index=["ticker", "period_end"],
        columns="concept_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide["selection_version"] = label
    if {"revenue", "total_assets"}.issubset(wide.columns):
        wide["asset_turnover"] = wide["revenue"] / wide["total_assets"]
    if {"net_income", "total_assets"}.issubset(wide.columns):
        wide["roa"] = wide["net_income"] / wide["total_assets"]
    if {"net_income", "revenue"}.issubset(wide.columns):
        wide["net_margin"] = wide["net_income"] / wide["revenue"]
    return wide


def aapl_ratio_history(selected_df: pd.DataFrame, label: str) -> pd.DataFrame:
    subset = selected_df.loc[
        (selected_df["ticker"] == "AAPL")
        & (selected_df["concept_name"].isin(["revenue", "net_income", "total_assets"]))
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot_table(
        index=["ticker", "period_end"],
        columns="concept_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide["selection_version"] = label
    if {"revenue", "total_assets"}.issubset(wide.columns):
        wide["asset_turnover"] = wide["revenue"] / wide["total_assets"]
    if {"net_income", "total_assets"}.issubset(wide.columns):
        wide["roa"] = wide["net_income"] / wide["total_assets"]
    if {"net_income", "revenue"}.issubset(wide.columns):
        wide["net_margin"] = wide["net_income"] / wide["revenue"]
    return wide.sort_values(["period_end"]).reset_index(drop=True)


def run_report(raw_path: Path, output_dir: Path, tickers: list[str] | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = fundamentals_clean.normalize_raw_data(pd.read_parquet(raw_path))
    if tickers:
        normalized = normalized.loc[normalized["ticker"].isin(tickers)].copy()
    legacy = legacy_select(normalized)
    new_result = select_fundamental_facts(normalized, include_diagnostics=False)
    new = new_result.selected_facts

    changed = legacy[["ticker", "period_end", "concept_name", "value"]].rename(columns={"value": "legacy_value"}).merge(
        new[["ticker", "period_end", "concept_name", "value", "selection_reason", "derivation_reason"]].rename(columns={"value": "new_value"}),
        on=["ticker", "period_end", "concept_name"],
        how="outer",
    )
    changed["value_changed"] = changed["legacy_value"] != changed["new_value"]

    direct_count = int((new.get("selection_reason", pd.Series(dtype="string")).astype("string") == "reported_quarterly").sum())
    ytd_count = int((new.get("derivation_reason", pd.Series(dtype="string")).astype("string") == "derived_from_ytd_difference").sum())
    summary = pd.DataFrame(
        [
            {"metric": "raw_candidate_scale_flags", "before": selection_scale_flags(normalized, legacy), "after": selection_scale_flags(normalized, new)},
            {
                "metric": "selected_negative_with_positive_candidate_flags",
                "before": selected_negative_with_positive_flags(normalized, legacy),
                "after": selected_negative_with_positive_flags(normalized, new),
            },
            {"metric": "selected_facts_changed", "before": 0, "after": int(changed["value_changed"].fillna(True).sum())},
            {"metric": "direct_reported_quarterly_facts", "before": "not_tracked", "after": direct_count},
            {"metric": "ytd_derived_quarterly_facts", "before": "not_tracked", "after": ytd_count},
            {"metric": "feature_ratio_sanity_flags", "before": "see existing audit", "after": "pending staged feature rebuild"},
            {"metric": "panel_ratio_sanity_flags", "before": "see existing audit", "after": "pending staged panel rebuild"},
        ]
    )
    aapl = pd.concat([aapl_snapshot(legacy, "legacy"), aapl_snapshot(new, "new_selector")], ignore_index=True)
    aapl_history = pd.concat([aapl_ratio_history(legacy, "legacy"), aapl_ratio_history(new, "new_selector")], ignore_index=True)

    paths = {
        "summary": output_dir / "before_after_summary.csv",
        "changed": output_dir / "selected_fact_changes.csv",
        "aapl": output_dir / "aapl_before_after.csv",
        "aapl_history": output_dir / "aapl_ratio_history_before_after.csv",
        "diagnostics": output_dir / "new_selector_rejected_candidate_diagnostics.csv",
        "markdown": output_dir / "before_after_summary.md",
    }
    summary.to_csv(paths["summary"], index=False)
    changed.loc[changed["value_changed"].fillna(True)].to_csv(paths["changed"], index=False)
    aapl.to_csv(paths["aapl"], index=False)
    aapl_history.to_csv(paths["aapl_history"], index=False)
    diagnostic_sample_keys = set(tuple(row) for row in changed.loc[changed["value_changed"].fillna(True)].head(50)[["ticker", "period_end", "concept_name"]].to_numpy())
    diagnostic_sample = normalized.loc[
        normalized.apply(lambda row: (row["ticker"], row["period_end"], row["concept_name"]) in diagnostic_sample_keys, axis=1)
    ].copy()
    select_fundamental_facts(diagnostic_sample, include_diagnostics=True).diagnostics.to_csv(paths["diagnostics"], index=False)
    paths["markdown"].write_text(
        "\n".join(
            [
                "# Fundamentals Before/After Report",
                "",
                f"- Raw input: `{raw_path}`",
                f"- Ticker filter: `{','.join(tickers) if tickers else 'ALL'}`",
                "- This report compares legacy in-memory fact selection to the new selector without rebuilding artifacts.",
                "- Flow concepts are compared on selected facts, but candidate-scale flags are counted only for instant/share concepts so valid quarterly or YTD-derived flows are not treated as scale errors.",
                "- Feature and panel after-state metrics remain pending until the staged rebuild is run.",
                "",
                summary.to_markdown(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build before/after fundamental selector diagnostics.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker filter. Use ALL to run the full raw universe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = None if str(args.tickers).upper() == "ALL" else [ticker.strip().upper() for ticker in str(args.tickers).split(",") if ticker.strip()]
    paths = run_report(Path(args.raw_path), Path(args.output_dir), tickers=tickers)
    print("Fundamentals before/after report complete")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
