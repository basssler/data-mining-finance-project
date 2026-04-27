import unittest
import warnings

import pandas as pd

from src.fundamentals_clean import (
    attach_effective_model_dates_to_raw,
    build_effective_model_date_diagnostics,
    clean_fundamentals_from_raw,
    normalize_raw_data,
)


def raw_fact(value: float, filing_date: str, accession_number: str, effective_model_date: str | None = None) -> dict:
    return {
        "ticker": "AAA",
        "cik": "0001",
        "filing_date": pd.Timestamp(filing_date),
        "period_end": pd.Timestamp("2024-03-31"),
        "fiscal_period": "Q1",
        "fiscal_year": 2024,
        "form_type": "10-Q",
        "concept_name": "revenue",
        "value": value,
        "unit": "USD",
        "raw_tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "accession_number": accession_number,
        "start_date": pd.Timestamp("2024-01-01"),
        "end_date": pd.Timestamp("2024-03-31"),
        "frame": "CY2024Q1",
        "fact_duration_days": 91,
        "source": "sec_companyfacts",
        "effective_model_date": pd.Timestamp(effective_model_date) if effective_model_date else pd.NaT,
    }


class FundamentalsCleanTimingTests(unittest.TestCase):
    def test_clean_path_excludes_facts_after_effective_model_date(self) -> None:
        raw_df = pd.DataFrame(
            [
                raw_fact(100.0, "2024-05-01", "early"),
                raw_fact(999.0, "2024-05-10", "late"),
            ]
        )
        timing_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "cik": ["0001", "0001"],
                "accession_number": ["early", "late"],
                "filing_date": pd.to_datetime(["2024-05-01", "2024-05-10"]),
                "form_type": ["10-Q", "10-Q"],
                "period_end": pd.to_datetime(["2024-03-31", "2024-03-31"]),
                "fiscal_year": [2024, 2024],
                "fiscal_period": ["Q1", "Q1"],
                "effective_model_date": pd.to_datetime(["2024-05-02", "2024-05-11"]),
                "timing_source": ["test", "test"],
                "timing_assumption": ["event_panel_effective_model_date", "event_panel_effective_model_date"],
            }
        )

        selected, _selection_diagnostics, timing_diagnostics = clean_fundamentals_from_raw(raw_df, timing_df)

        self.assertEqual(float(selected.iloc[0]["value"]), 100.0)
        self.assertEqual(int(timing_diagnostics.iloc[0]["selected_facts_after_effective_model_date"]), 0)

    def test_attach_effective_model_dates_to_raw_uses_accession_number(self) -> None:
        raw_df = normalize_raw_data(pd.DataFrame([raw_fact(100.0, "2024-05-01", "early")]))
        timing_df = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "accession_number": ["early"],
                "filing_date": pd.to_datetime(["2024-05-01"]),
                "form_type": ["10-Q"],
                "effective_model_date": pd.to_datetime(["2024-05-02"]),
                "timing_source": ["test"],
                "timing_assumption": ["event_panel_effective_model_date"],
            }
        )

        attached = attach_effective_model_dates_to_raw(raw_df, timing_df)

        self.assertEqual(pd.Timestamp(attached.iloc[0]["effective_model_date"]), pd.Timestamp("2024-05-02"))

    def test_missing_effective_model_date_warns_and_uses_conservative_cutoff(self) -> None:
        raw_df = pd.DataFrame([raw_fact(100.0, "2024-05-01", "early", effective_model_date=None)])

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            selected, _selection_diagnostics, timing_diagnostics = clean_fundamentals_from_raw(raw_df)

        self.assertTrue(any("no timing table was supplied" in str(item.message).lower() for item in captured))
        self.assertEqual(float(selected.iloc[0]["value"]), 100.0)
        self.assertEqual(int(timing_diagnostics.iloc[0]["selected_facts_after_effective_model_date"]), 0)

    def test_effective_model_date_diagnostics_counts_post_cutoff_candidates(self) -> None:
        candidates = pd.DataFrame(
            [
                raw_fact(100.0, "2024-05-01", "early", "2024-05-02"),
                raw_fact(999.0, "2024-05-10", "late", "2024-05-02"),
            ]
        )
        selected = candidates.iloc[[0]].copy()

        diagnostics = build_effective_model_date_diagnostics(candidates, selected)

        self.assertEqual(int(diagnostics.iloc[0]["candidate_facts_excluded_after_effective_model_date"]), 1)
        self.assertEqual(int(diagnostics.iloc[0]["selected_facts_after_effective_model_date"]), 0)


if __name__ == "__main__":
    unittest.main()
