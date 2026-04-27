import math
import unittest

import pandas as pd

from src.fundamental_fact_selector import select_fundamental_facts


def fact(
    *,
    ticker="AAA",
    concept_name="revenue",
    value=100.0,
    unit="USD",
    raw_tag=None,
    filing_date="2024-05-01",
    period_end="2024-03-31",
    start_date="2024-01-01",
    end_date="2024-03-31",
    frame="CY2024Q1",
    fiscal_period="Q1",
    fiscal_year=2024,
    form_type="10-Q",
    effective_model_date="2024-05-02",
    accession_number="a1",
):
    if raw_tag is None:
        raw_tag = "Assets" if concept_name == "total_assets" else "RevenueFromContractWithCustomerExcludingAssessedTax"
    return {
        "ticker": ticker,
        "cik": "0001",
        "concept_name": concept_name,
        "raw_tag": raw_tag,
        "value": value,
        "unit": unit,
        "filing_date": pd.Timestamp(filing_date),
        "accession_number": accession_number,
        "form_type": form_type,
        "period_end": pd.Timestamp(period_end),
        "start_date": pd.Timestamp(start_date) if start_date is not None else pd.NaT,
        "end_date": pd.Timestamp(end_date) if end_date is not None else pd.NaT,
        "frame": frame,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "fact_duration_days": (
            (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
            if start_date is not None and end_date is not None
            else pd.NA
        ),
        "source": "sec_companyfacts",
        "effective_model_date": pd.Timestamp(effective_model_date),
    }


class FundamentalFactSelectorTests(unittest.TestCase):
    def selected_value(self, rows, concept_name="revenue") -> float:
        result = select_fundamental_facts(pd.DataFrame(rows))
        selected = result.selected_facts.loc[result.selected_facts["concept_name"] == concept_name].iloc[0]
        return float(selected["value"])

    def test_quarterly_revenue_prefers_framed_quarterly_over_segmented_no_frame(self) -> None:
        rows = [
            fact(value=1000.0, frame="CY2024Q1", accession_number="good"),
            fact(value=50.0, frame=None, accession_number="segment"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        selected = result.selected_facts.iloc[0]

        self.assertEqual(float(selected["value"]), 1000.0)
        self.assertEqual(str(selected["accession_number"]), "good")

    def test_q2_ytd_revenue_converts_to_quarterly_by_subtracting_q1_ytd(self) -> None:
        rows = [
            fact(value=100.0, fiscal_period="Q1", period_end="2024-03-31", start_date="2024-01-01", end_date="2024-03-31", frame="CY2024Q1"),
            fact(value=260.0, fiscal_period="Q2", period_end="2024-06-30", start_date="2024-01-01", end_date="2024-06-30", frame=None, accession_number="q2-ytd"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        q2 = result.selected_facts.loc[result.selected_facts["period_end"] == pd.Timestamp("2024-06-30")].iloc[0]

        self.assertEqual(float(q2["value"]), 160.0)
        self.assertEqual(str(q2["derivation_reason"]), "derived_from_ytd_difference")

    def test_q3_ytd_revenue_converts_to_quarterly_by_subtracting_q2_ytd(self) -> None:
        rows = [
            fact(value=260.0, fiscal_period="Q2", period_end="2024-06-30", start_date="2024-01-01", end_date="2024-06-30", frame=None, accession_number="q2-ytd"),
            fact(value=450.0, fiscal_period="Q3", period_end="2024-09-30", start_date="2024-01-01", end_date="2024-09-30", frame=None, accession_number="q3-ytd"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        q3 = result.selected_facts.loc[result.selected_facts["period_end"] == pd.Timestamp("2024-09-30")].iloc[0]

        self.assertEqual(float(q3["value"]), 190.0)
        self.assertEqual(str(q3["source_fact_accession_number"]), "q3-ytd")
        self.assertEqual(str(q3["prior_ytd_accession_number"]), "q2-ytd")

    def test_ytd_conversion_rejected_when_prior_ytd_missing(self) -> None:
        rows = [
            fact(value=260.0, fiscal_period="Q2", period_end="2024-06-30", start_date="2024-01-01", end_date="2024-06-30", frame=None, accession_number="q2-ytd"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))

        self.assertTrue(result.selected_facts.empty)

    def test_balance_sheet_assets_prefer_instant_frame(self) -> None:
        rows = [
            fact(concept_name="total_assets", value=1000.0, frame="CY2024Q1I", start_date=None, accession_number="instant"),
            fact(concept_name="total_assets", value=80.0, frame=None, accession_number="segment"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        selected = result.selected_facts.iloc[0]

        self.assertEqual(float(selected["value"]), 1000.0)
        self.assertEqual(str(selected["accession_number"]), "instant")

    def test_annual_flow_facts_prefer_annual_duration_for_10k(self) -> None:
        rows = [
            fact(value=1000.0, form_type="10-K", fiscal_period="FY", period_end="2024-12-31", start_date="2024-01-01", end_date="2024-12-31", frame="CY2024", accession_number="annual"),
            fact(value=200.0, form_type="10-K", fiscal_period="FY", period_end="2024-12-31", start_date="2024-10-01", end_date="2024-12-31", frame=None, accession_number="quarter"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        selected = result.selected_facts.iloc[0]

        self.assertEqual(float(selected["value"]), 1000.0)
        self.assertEqual(str(selected["accession_number"]), "annual")

    def test_latest_filing_loses_to_better_frame_and_duration(self) -> None:
        rows = [
            fact(value=1000.0, filing_date="2024-05-01", frame="CY2024Q1", accession_number="good"),
            fact(value=20.0, filing_date="2024-05-02", frame=None, accession_number="latest-segment"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        selected = result.selected_facts.iloc[0]

        self.assertEqual(float(selected["value"]), 1000.0)
        self.assertEqual(str(selected["accession_number"]), "good")

    def test_facts_after_effective_model_date_are_ineligible(self) -> None:
        rows = [
            fact(value=100.0, filing_date="2024-05-01", effective_model_date="2024-05-02", accession_number="available"),
            fact(value=999.0, filing_date="2024-05-03", effective_model_date="2024-05-02", accession_number="future"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        selected = result.selected_facts.iloc[0]

        self.assertEqual(float(selected["value"]), 100.0)
        self.assertEqual(str(selected["accession_number"]), "available")

    def test_tiny_segmented_assets_rejected_when_consolidated_assets_exist(self) -> None:
        rows = [
            fact(concept_name="total_assets", value=250_000_000_000.0, frame="CY2024Q1I", start_date=None, accession_number="consolidated"),
            fact(concept_name="total_assets", value=155_000_000.0, frame=None, accession_number="tiny-segment"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        selected = result.selected_facts.iloc[0]

        self.assertEqual(float(selected["value"]), 250_000_000_000.0)
        self.assertEqual(str(selected["accession_number"]), "consolidated")

    def test_aapl_style_selected_values_produce_plausible_asset_turnover(self) -> None:
        rows = [
            fact(concept_name="revenue", value=58_010_000_000.0, frame="CY2024Q1", accession_number="rev"),
            fact(concept_name="total_assets", value=261_194_000_000.0, frame="CY2024Q1I", start_date=None, accession_number="assets"),
            fact(concept_name="total_assets", value=155_000_000.0, frame=None, start_date=None, accession_number="tiny-assets"),
        ]

        result = select_fundamental_facts(pd.DataFrame(rows))
        wide = result.selected_facts.pivot(index=["ticker", "period_end"], columns="concept_name", values="value").reset_index()
        asset_turnover = float(wide.iloc[0]["revenue"]) / float(wide.iloc[0]["total_assets"])

        self.assertTrue(math.isfinite(asset_turnover))
        self.assertLess(asset_turnover, 1.0)


if __name__ == "__main__":
    unittest.main()
