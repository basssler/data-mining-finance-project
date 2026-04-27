"""Validate the point-in-time fact selector on a small golden ticker sample."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src import fundamentals_clean
from src.fundamental_fact_selector import select_fundamental_facts
from src.paths import OUTPUTS_DIR, RAW_DATA_DIR


DEFAULT_RAW_PATH = RAW_DATA_DIR / "fundamentals" / "golden_raw_fundamentals.parquet"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "quarterly" / "diagnostics" / "golden_fundamentals_validation"
REQUIRED_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "TSLA"]
CORE_CONCEPTS = ["revenue", "net_income", "total_assets", "total_liabilities", "shareholders_equity", "shares_outstanding"]
REQUIRED_METADATA = ["raw_tag", "accession_number", "start_date", "end_date", "frame", "fact_duration_days"]


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _metadata_complete(df: pd.DataFrame) -> bool:
    return all(column in df.columns and df[column].notna().any() for column in REQUIRED_METADATA)


def _choose_sample_periods(raw_df: pd.DataFrame) -> pd.DataFrame:
    available = raw_df.loc[raw_df["concept_name"].isin(CORE_CONCEPTS)].copy()
    grouped = (
        available.groupby(["ticker", "period_end"], dropna=False)
        .agg(
            concept_count=("concept_name", "nunique"),
            latest_filing_date=("filing_date", "max"),
            has_negative_net_income=("value", lambda s: False),
        )
        .reset_index()
    )
    net_income = available.loc[available["concept_name"] == "net_income", ["ticker", "period_end", "value"]].copy()
    negative_keys = set(
        tuple(row)
        for row in net_income.loc[pd.to_numeric(net_income["value"], errors="coerce") < 0, ["ticker", "period_end"]].to_numpy()
    )
    grouped["has_negative_net_income"] = grouped.apply(
        lambda row: (row["ticker"], row["period_end"]) in negative_keys,
        axis=1,
    )

    rows = []
    for ticker in REQUIRED_TICKERS:
        ticker_rows = grouped.loc[(grouped["ticker"] == ticker) & (grouped["concept_count"] >= 3)].sort_values(
            ["concept_count", "period_end"],
            ascending=[False, False],
        )
        if not ticker_rows.empty:
            rows.append(ticker_rows.iloc[0])
        else:
            rows.append(
                pd.Series(
                    {
                        "ticker": ticker,
                        "period_end": pd.NaT,
                        "concept_count": 0,
                        "latest_filing_date": pd.NaT,
                        "has_negative_net_income": False,
                    }
                )
            )

    negative_rows = grouped.loc[
        (~grouped["ticker"].isin(REQUIRED_TICKERS))
        & grouped["has_negative_net_income"]
        & (grouped["concept_count"] >= 3)
    ].sort_values(["concept_count", "period_end"], ascending=[False, False])
    if not negative_rows.empty:
        rows.append(negative_rows.iloc[0])

    return pd.DataFrame(rows).drop_duplicates(subset=["ticker", "period_end"]).reset_index(drop=True)


def _validate_selected(selected_df: pd.DataFrame, metadata_complete: bool) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame()
    wide = selected_df.pivot_table(
        index=["ticker", "period_end"],
        columns="concept_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    rows = []
    for _, row in wide.iterrows():
        revenue = row.get("revenue", np.nan)
        net_income = row.get("net_income", np.nan)
        assets = row.get("total_assets", np.nan)
        liabilities = row.get("total_liabilities", np.nan)
        equity = row.get("shareholders_equity", np.nan)
        shares = row.get("shares_outstanding", np.nan)
        net_margin = _safe_ratio(net_income, revenue)
        asset_turnover = _safe_ratio(revenue, assets)
        roa = _safe_ratio(net_income, assets)
        roe = _safe_ratio(net_income, equity)
        identity_gap = np.nan
        if pd.notna(assets) and pd.notna(liabilities) and pd.notna(equity) and assets:
            identity_gap = abs(float(assets) - (float(liabilities) + float(equity))) / abs(float(assets))

        checks = {
            "revenue_plausible": pd.notna(revenue) and abs(float(revenue)) > 1_000_000,
            "net_income_plausible_relative_to_revenue": pd.isna(net_margin) or abs(net_margin) <= 1.0,
            "assets_plausible": pd.notna(assets) and float(assets) > 1_000_000,
            "accounting_identity_plausible": pd.isna(identity_gap) or identity_gap <= 0.35,
            "asset_turnover_plausible": pd.isna(asset_turnover) or abs(asset_turnover) <= 5.0,
            "roa_plausible": pd.isna(roa) or abs(roa) <= 1.0,
            "roe_plausible": pd.isna(roe) or abs(roe) <= 5.0,
        }
        status = "PASS" if metadata_complete and all(checks.values()) else ("MISSING_METADATA" if not metadata_complete else "FAIL")
        rows.append(
            {
                "ticker": row["ticker"],
                "period_end": row["period_end"],
                "metadata_complete": metadata_complete,
                "validation_status": status,
                "revenue": revenue,
                "net_income": net_income,
                "total_assets": assets,
                "total_liabilities": liabilities,
                "shareholders_equity": equity,
                "shares_outstanding": shares,
                "net_margin": net_margin,
                "asset_turnover": asset_turnover,
                "roa": roa,
                "roe": roe,
                "identity_gap_pct_assets": identity_gap,
                **checks,
            }
        )
    return pd.DataFrame(rows)


def _apply_golden_effective_dates(sample_raw: pd.DataFrame) -> pd.DataFrame:
    """Use first available filing date per ticker-period as the golden cutoff.

    The production event panel aligns filing availability onto trading dates.
    The golden raw sample has daily filing dates only, so this conservative
    cutoff prevents amended/restated facts filed after the first available
    period filing from winning validation.
    """
    working = sample_raw.copy()
    cutoff = (
        working.groupby(["ticker", "period_end"], dropna=False)["filing_date"]
        .min()
        .reset_index(name="golden_effective_model_date")
    )
    working = working.merge(cutoff, on=["ticker", "period_end"], how="left")
    working["effective_model_date"] = working["golden_effective_model_date"]
    return working.drop(columns=["golden_effective_model_date"])


def _selection_trace(selected: pd.DataFrame) -> pd.DataFrame:
    trace_columns = [
        "ticker",
        "period_end",
        "concept_name",
        "value",
        "unit",
        "raw_tag",
        "frame",
        "fact_duration_days",
        "filing_date",
        "effective_model_date",
        "accession_number",
        "selection_reason",
        "derivation_reason",
        "source_fact_value",
        "prior_ytd_value",
        "source_fact_accession_number",
        "prior_ytd_accession_number",
    ]
    for column in trace_columns:
        if column not in selected.columns:
            selected[column] = pd.NA
    return selected[trace_columns].sort_values(["ticker", "period_end", "concept_name"]).reset_index(drop=True)


def _concept_reason_wide(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    reason = trace.pivot_table(
        index=["ticker", "period_end"],
        columns="concept_name",
        values="selection_reason",
        aggfunc="first",
    ).add_suffix("_selection_reason")
    derivation = trace.pivot_table(
        index=["ticker", "period_end"],
        columns="concept_name",
        values="derivation_reason",
        aggfunc="first",
    ).add_suffix("_derivation_reason")
    return pd.concat([reason, derivation], axis=1).reset_index()


def run_validation(raw_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_parquet(raw_path)
    metadata_complete = _metadata_complete(raw_df)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        normalized = fundamentals_clean.normalize_raw_data(raw_df)
    sample = _choose_sample_periods(normalized)
    sample_keys = set(tuple(row) for row in sample[["ticker", "period_end"]].to_numpy())
    sample_raw = normalized.loc[
        normalized.apply(lambda row: (row["ticker"], row["period_end"]) in sample_keys, axis=1)
    ].copy()
    sample_raw = _apply_golden_effective_dates(sample_raw)
    selection_result = select_fundamental_facts(sample_raw)
    selected = selection_result.selected_facts
    diagnostics = selection_result.diagnostics
    trace = _selection_trace(selected)
    validation = _validate_selected(selected.loc[selected["concept_name"].isin(CORE_CONCEPTS)], metadata_complete)
    reason_wide = _concept_reason_wide(trace)
    if not validation.empty and not reason_wide.empty:
        validation = validation.merge(reason_wide, on=["ticker", "period_end"], how="left")
    if not validation.empty and not trace.empty:
        leakage_check = (
            trace.assign(
                selected_after_effective_model_date=(
                    pd.to_datetime(trace["filing_date"], errors="coerce")
                    > pd.to_datetime(trace["effective_model_date"], errors="coerce")
                )
            )
            .groupby(["ticker", "period_end"], dropna=False)["selected_after_effective_model_date"]
            .any()
            .reset_index()
        )
        validation = validation.merge(leakage_check, on=["ticker", "period_end"], how="left")
        validation["no_post_effective_date_facts_selected"] = ~validation[
            "selected_after_effective_model_date"
        ].fillna(False)
        validation.loc[
            ~validation["no_post_effective_date_facts_selected"],
            "validation_status",
        ] = "FAIL"
    selected_keys = set(tuple(row) for row in validation[["ticker", "period_end"]].to_numpy()) if not validation.empty else set()
    missing_rows = []
    for _, row in sample.iterrows():
        key = (row["ticker"], row["period_end"])
        if pd.isna(row["period_end"]) or key not in selected_keys:
            missing_rows.append(
                {
                    "ticker": row["ticker"],
                    "period_end": row["period_end"],
                    "metadata_complete": metadata_complete,
                    "validation_status": "MISSING_TICKER" if pd.isna(row["period_end"]) else "FAIL",
                }
            )
    if missing_rows:
        validation = pd.concat([validation, pd.DataFrame(missing_rows)], ignore_index=True)

    full_trace = _selection_trace(select_fundamental_facts(_apply_golden_effective_dates(normalized)).selected_facts)
    ytd_examples = full_trace.loc[full_trace["derivation_reason"] == "derived_from_ytd_difference"].copy()

    paths = {
        "sample": output_dir / "golden_sample_periods.csv",
        "selected": output_dir / "golden_selected_facts.csv",
        "trace": output_dir / "golden_selection_trace.csv",
        "ytd_derivations": output_dir / "golden_ytd_derivation_examples.csv",
        "diagnostics": output_dir / "golden_rejected_candidate_diagnostics.csv",
        "validation": output_dir / "golden_validation.csv",
        "markdown": output_dir / "golden_validation_summary.md",
    }
    sample.to_csv(paths["sample"], index=False)
    selected.to_csv(paths["selected"], index=False)
    trace.to_csv(paths["trace"], index=False)
    ytd_examples.to_csv(paths["ytd_derivations"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    validation.to_csv(paths["validation"], index=False)
    status_counts = validation["validation_status"].value_counts(dropna=False).to_dict() if not validation.empty else {}
    markdown = [
        "# Golden Fundamentals Validation",
        "",
        f"- Raw input: `{raw_path}`",
        f"- Metadata complete: `{metadata_complete}`",
        f"- Status counts: `{status_counts}`",
        f"- YTD-derived quarterly facts found in refreshed golden raw history: `{len(ytd_examples)}`",
        "",
        "The golden selector cutoff uses the first available filing date for each ticker-period in this raw-only validation. Production event panels still use the trading-date-aligned `effective_model_date`.",
    ]
    paths["markdown"].write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden fundamentals selector validation.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_validation(Path(args.raw_path), Path(args.output_dir))
    print("Golden fundamentals validation complete")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
