"""Point-in-time-safe SEC/XBRL fundamental fact selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.accounting_concepts import CONCEPT_SPECS, concept_priority_lookup, source_priority


QUARTERLY_DURATION_RANGE = (60, 120)
YTD_Q2_DURATION_RANGE = (121, 220)
YTD_Q3_DURATION_RANGE = (221, 320)
ANNUAL_DURATION_RANGE = (330, 400)


@dataclass(frozen=True)
class FactSelectionResult:
    selected_facts: pd.DataFrame
    diagnostics: pd.DataFrame


def fiscal_period_number(value: object) -> int | None:
    text = str(value).upper().strip()
    if text.startswith("Q") and text[1:].isdigit():
        number = int(text[1:])
        if 1 <= number <= 4:
            return number
    return None


def normalize_selector_input(df: pd.DataFrame) -> pd.DataFrame:
    facts = df.copy()
    priority_lookup = concept_priority_lookup()
    optional_columns = {
        "accession_number": pd.NA,
        "start_date": pd.NaT,
        "end_date": pd.NaT,
        "frame": pd.NA,
        "fact_duration_days": pd.NA,
        "effective_model_date": pd.NaT,
        "selection_reason": pd.NA,
        "derivation_reason": pd.NA,
    }
    for column, default_value in optional_columns.items():
        if column not in facts.columns:
            facts[column] = default_value

    for column in ["ticker", "cik", "concept_name", "raw_tag", "unit", "form_type", "frame", "source", "accession_number"]:
        if column in facts.columns:
            facts[column] = facts[column].astype("string")

    for column in ["filing_date", "period_end", "start_date", "end_date", "effective_model_date"]:
        facts[column] = pd.to_datetime(facts[column], errors="coerce")

    facts["value"] = pd.to_numeric(facts["value"], errors="coerce")
    facts["fiscal_year"] = pd.to_numeric(facts["fiscal_year"], errors="coerce").astype("Int64")
    facts["fact_duration_days"] = pd.to_numeric(facts["fact_duration_days"], errors="coerce")
    missing_duration = facts["fact_duration_days"].isna() & facts["start_date"].notna() & facts["end_date"].notna()
    facts.loc[missing_duration, "fact_duration_days"] = (
        facts.loc[missing_duration, "end_date"] - facts.loc[missing_duration, "start_date"]
    ).dt.days + 1
    facts["_fiscal_period_number"] = facts["fiscal_period"].map(fiscal_period_number)
    facts["_tag_rank"] = facts.apply(
        lambda row: priority_lookup.get(str(row["concept_name"]), {}).get(str(row["raw_tag"]), 999),
        axis=1,
    )
    facts["_source_rank"] = facts["source"].map(source_priority).fillna(9).astype("int64")
    facts["_has_frame_rank"] = facts["frame"].isna().astype("int64")
    return facts


def _duration_between(series: pd.Series, lower: int, upper: int) -> pd.Series:
    return series.notna() & series.between(lower, upper, inclusive="both")


def _is_quarterly_flow(facts: pd.DataFrame) -> pd.Series:
    duration = facts["fact_duration_days"]
    frame = facts["frame"].astype("string")
    return _duration_between(duration, *QUARTERLY_DURATION_RANGE) | (
        frame.notna() & frame.str.match(r"^CY\d{4}Q[1-4]$", na=False)
    )


def _is_ytd_flow(facts: pd.DataFrame) -> pd.Series:
    period_number = facts["_fiscal_period_number"]
    duration = facts["fact_duration_days"]
    return (
        ((period_number == 2) & _duration_between(duration, *YTD_Q2_DURATION_RANGE))
        | ((period_number == 3) & _duration_between(duration, *YTD_Q3_DURATION_RANGE))
    )


def _is_annual_flow(facts: pd.DataFrame) -> pd.Series:
    duration = facts["fact_duration_days"]
    return _duration_between(duration, *ANNUAL_DURATION_RANGE) | facts["fiscal_period"].astype("string").str.upper().eq("FY")


def _is_instant_fact(facts: pd.DataFrame) -> pd.Series:
    frame = facts["frame"].astype("string")
    duration = facts["fact_duration_days"]
    return frame.str.match(r"^CY\d{4}Q[1-4]I$", na=False) | duration.isna() | (duration <= 2)


def _eligible_for_cutoff(facts: pd.DataFrame, cutoff: pd.Timestamp | None) -> pd.DataFrame:
    if cutoff is None or pd.isna(cutoff):
        return facts.copy()
    return facts.loc[facts["filing_date"].notna() & (facts["filing_date"] <= cutoff)].copy()


def _rank_candidates(facts: pd.DataFrame, concept_name: str, period_kind: str) -> pd.DataFrame:
    spec = CONCEPT_SPECS.get(concept_name)
    ranked = facts.copy()
    preferred_unit = spec.preferred_unit if spec else None
    ranked["_unit_rank"] = (ranked["unit"].astype("string") != preferred_unit).astype("int64")
    if period_kind == "quarterly_flow":
        ranked["_period_rank"] = (~_is_quarterly_flow(ranked)).astype("int64")
        ranked["_value_rank"] = 0.0
    elif period_kind == "annual_flow":
        ranked["_period_rank"] = (~_is_annual_flow(ranked)).astype("int64")
        ranked["_value_rank"] = 0.0
    elif period_kind == "instant":
        ranked["_period_rank"] = (~_is_instant_fact(ranked)).astype("int64")
        ranked["_value_rank"] = -ranked["value"].abs().fillna(-np.inf)
    else:
        ranked["_period_rank"] = 0
        ranked["_value_rank"] = 0.0

    ranked["_filing_sort"] = ranked["filing_date"].fillna(pd.Timestamp.min)
    return ranked.sort_values(
        [
            "_unit_rank",
            "_period_rank",
            "_has_frame_rank",
            "_tag_rank",
            "_source_rank",
            "_value_rank",
            "_filing_sort",
        ],
        ascending=[True, True, True, True, True, True, False],
    )


def _period_kind_for_group(group: pd.DataFrame, concept_type: str) -> str:
    if concept_type in {"instant", "shares"}:
        return "instant"
    has_quarterly_period = group["_fiscal_period_number"].notna().any()
    has_quarterly_form = group["form_type"].astype("string").str.upper().str.startswith("10-Q", na=False).any()
    if has_quarterly_period or has_quarterly_form:
        return "quarterly_flow"
    if group["form_type"].astype("string").str.upper().str.startswith("10-K", na=False).any() or group["fiscal_period"].astype("string").str.upper().eq("FY").any():
        return "annual_flow"
    return "quarterly_flow"


def _select_direct(group: pd.DataFrame, concept_name: str, period_kind: str) -> pd.Series | None:
    ranked = _rank_candidates(group, concept_name, period_kind)
    if ranked.empty:
        return None
    if period_kind == "quarterly_flow":
        direct = ranked.loc[_is_quarterly_flow(ranked)]
        if direct.empty and ranked["fact_duration_days"].isna().all() and ranked["frame"].isna().all():
            direct = ranked
        elif direct.empty:
            return None
        return direct.iloc[0].copy()
    if period_kind == "annual_flow":
        annual = ranked.loc[_is_annual_flow(ranked)]
        if annual.empty and ranked["fact_duration_days"].isna().all() and ranked["frame"].isna().all():
            annual = ranked
        elif annual.empty:
            return None
        return annual.iloc[0].copy()
    if period_kind == "instant":
        instant = ranked.loc[_is_instant_fact(ranked)]
        if instant.empty:
            return None
        return instant.iloc[0].copy()
    return ranked.iloc[0].copy()


def _build_ytd_lookup(all_facts: pd.DataFrame) -> dict[tuple, pd.Index]:
    ytd_like = all_facts.loc[_is_quarterly_flow(all_facts) | _is_ytd_flow(all_facts)].copy()
    lookup: dict[tuple, pd.Index] = {}
    for key, group in ytd_like.groupby(
        ["ticker", "cik", "concept_name", "fiscal_year", "_fiscal_period_number", "unit", "raw_tag"],
        dropna=False,
        sort=False,
    ):
        lookup[key] = group.index
    return lookup


def _compatible_prior_ytd(
    all_facts: pd.DataFrame,
    ytd_lookup: dict[tuple, pd.Index],
    current: pd.Series,
    cutoff: pd.Timestamp | None,
) -> pd.DataFrame:
    period_number = fiscal_period_number(current.get("fiscal_period"))
    if period_number is None or period_number <= 1:
        return pd.DataFrame(columns=all_facts.columns)

    key = (
        current["ticker"],
        current["cik"],
        current["concept_name"],
        current["fiscal_year"],
        period_number - 1,
        current["unit"],
        current["raw_tag"],
    )
    prior_index = ytd_lookup.get(key)
    if prior_index is None:
        return pd.DataFrame(columns=all_facts.columns)
    prior = all_facts.loc[prior_index].copy()
    prior = _eligible_for_cutoff(prior, cutoff)
    return prior.loc[_is_quarterly_flow(prior) | _is_ytd_flow(prior)].copy()


def _derive_from_ytd(
    group: pd.DataFrame,
    all_facts: pd.DataFrame,
    ytd_lookup: dict[tuple, pd.Index],
    concept_name: str,
    cutoff: pd.Timestamp | None,
) -> pd.Series | None:
    ytd_candidates = group.loc[_is_ytd_flow(group)].copy()
    if ytd_candidates.empty:
        return None
    current_ranked = _rank_candidates(ytd_candidates, concept_name, "quarterly_flow")
    for _, current in current_ranked.iterrows():
        prior_candidates = _compatible_prior_ytd(all_facts, ytd_lookup, current, cutoff)
        if prior_candidates.empty:
            continue
        prior = _rank_candidates(prior_candidates, concept_name, "quarterly_flow").iloc[0]
        derived = current.copy()
        derived["value"] = float(current["value"]) - float(prior["value"])
        derived["selection_reason"] = "derived_from_ytd_difference"
        derived["derivation_reason"] = "derived_from_ytd_difference"
        derived["source_fact_accession_number"] = current.get("accession_number", pd.NA)
        derived["prior_ytd_accession_number"] = prior.get("accession_number", pd.NA)
        derived["source_fact_value"] = current["value"]
        derived["prior_ytd_value"] = prior["value"]
        return derived
    return None


def _rejection_reason(row: pd.Series, selected: pd.Series, cutoff: pd.Timestamp | None, period_kind: str, concept_name: str) -> str:
    spec = CONCEPT_SPECS.get(concept_name)
    reasons = []
    if cutoff is not None and pd.notna(cutoff) and pd.notna(row.get("filing_date")) and row["filing_date"] > cutoff:
        reasons.append("filed_after_effective_model_date")
    if spec and str(row.get("unit")) != spec.preferred_unit:
        reasons.append("non_preferred_unit")
    if period_kind == "quarterly_flow" and not bool(_is_quarterly_flow(pd.DataFrame([row])).iloc[0]):
        reasons.append("not_direct_quarterly_duration")
    if period_kind == "annual_flow" and not bool(_is_annual_flow(pd.DataFrame([row])).iloc[0]):
        reasons.append("not_annual_duration")
    if period_kind == "instant" and not bool(_is_instant_fact(pd.DataFrame([row])).iloc[0]):
        reasons.append("not_instant_fact")
    if pd.isna(row.get("frame")) and pd.notna(selected.get("frame")):
        reasons.append("less_consolidated_frame_metadata")
    if str(row.get("raw_tag")) != str(selected.get("raw_tag")):
        reasons.append("lower_tag_priority")
    return "|".join(reasons) if reasons else "lower_ranked_candidate"


def _build_diagnostic(group: pd.DataFrame, selected: pd.Series, cutoff: pd.Timestamp | None, period_kind: str) -> dict:
    rejected = group.loc[group.index != selected.name].copy() if selected.name in group.index else group.copy()
    rejected_values = rejected["value"].dropna().astype(float).head(5).tolist()
    rejected_reasons = [
        _rejection_reason(row, selected, cutoff, period_kind, str(selected["concept_name"]))
        for _, row in rejected.head(10).iterrows()
    ]
    return {
        "ticker": selected.get("ticker"),
        "concept_name": selected.get("concept_name"),
        "period_end": selected.get("period_end"),
        "effective_model_date": cutoff,
        "selected_value": selected.get("value"),
        "selected_unit": selected.get("unit"),
        "selected_raw_tag": selected.get("raw_tag"),
        "selected_frame": selected.get("frame"),
        "selected_duration_days": selected.get("fact_duration_days"),
        "selected_filing_date": selected.get("filing_date"),
        "selection_reason": selected.get("selection_reason"),
        "derivation_reason": selected.get("derivation_reason"),
        "rejected_candidate_count": len(rejected),
        "largest_rejected_value": rejected["value"].abs().max() if not rejected.empty else np.nan,
        "rejected_value_examples": "|".join(str(value) for value in rejected_values),
        "rejection_reasons": "|".join(dict.fromkeys(rejected_reasons)),
    }


def select_fundamental_facts(df: pd.DataFrame, *, include_diagnostics: bool = True) -> FactSelectionResult:
    facts = normalize_selector_input(df)
    ytd_lookup = _build_ytd_lookup(facts)
    selected_rows = []
    diagnostic_rows = []
    group_keys = ["ticker", "cik", "period_end", "concept_name"]

    for _, group in facts.groupby(group_keys, dropna=False, sort=True):
        concept_name = str(group["concept_name"].iloc[0])
        spec = CONCEPT_SPECS.get(concept_name)
        concept_type = spec.concept_type if spec else "flow"
        cutoff = group["effective_model_date"].dropna().min() if group["effective_model_date"].notna().any() else None
        eligible = _eligible_for_cutoff(group, cutoff)
        if eligible.empty:
            continue

        period_kind = _period_kind_for_group(eligible, concept_type)
        selected = _select_direct(eligible, concept_name, period_kind)
        if selected is not None:
            if pd.isna(selected.get("selection_reason")):
                selected["selection_reason"] = "reported_quarterly" if period_kind == "quarterly_flow" else f"reported_{period_kind}"
            if pd.isna(selected.get("derivation_reason")):
                selected["derivation_reason"] = "not_derived"
        elif period_kind == "quarterly_flow":
            selected = _derive_from_ytd(eligible, facts, ytd_lookup, concept_name, cutoff)
        if selected is None:
            continue

        selected_rows.append(selected)
        if include_diagnostics:
            diagnostic_rows.append(_build_diagnostic(eligible, selected, cutoff, period_kind))

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else pd.DataFrame(columns=facts.columns)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    return FactSelectionResult(selected_facts=selected_df, diagnostics=diagnostics_df)
