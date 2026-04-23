import math
import shutil
import unittest
from pathlib import Path

import pandas as pd

from src.feature_engineering import (
    build_unlocked_feature_feasibility_report,
    engineer_features,
    normalize_input_data,
    save_unlocked_feature_feasibility_report,
)
from src.fundamentals_clean import (
    build_coverage_diagnostics,
    deduplicate_concept_rows,
    normalize_raw_data,
    save_coverage_diagnostics,
)


class AccountingExpansionTests(unittest.TestCase):
    def test_deduplicate_concept_rows_prefers_priority_tag_and_unit(self) -> None:
        raw_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "cik": ["0001", "0001", "0001"],
                "filing_date": pd.to_datetime(["2024-05-01", "2024-05-03", "2024-05-02"]),
                "period_end": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-03-31"]),
                "fiscal_period": ["Q1", "Q1", "Q1"],
                "fiscal_year": [2024, 2024, 2024],
                "form_type": ["10-Q", "10-Q", "10-Q"],
                "concept_name": ["revenue", "revenue", "revenue"],
                "value": [100.0, 90.0, 95.0],
                "unit": ["USD", "USDm", "USD"],
                "raw_tag": ["Revenue", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
                "source": ["sec_companyfacts", "sec_companyfacts", "edgartools_companyfacts"],
            }
        )

        normalized = normalize_raw_data(raw_df)
        deduped, removed_count = deduplicate_concept_rows(normalized)

        self.assertEqual(removed_count, 2)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(str(deduped.iloc[0]["raw_tag"]), "SalesRevenueNet")
        self.assertTrue(math.isclose(float(deduped.iloc[0]["value"]), 95.0, abs_tol=1e-12))

    def test_build_coverage_diagnostics_includes_missingness(self) -> None:
        concept_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB"],
                "concept_name": ["revenue", "gross_profit", "revenue"],
            }
        )
        clean_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB"],
                "period_end": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31"]),
                "revenue": [100.0, 110.0, 90.0],
                "gross_profit": [40.0, None, None],
            }
        )

        overall_df, by_ticker_df, by_year_df = build_coverage_diagnostics(concept_df, clean_df)

        gross_profit_row = overall_df.loc[overall_df["concept_name"] == "gross_profit"].iloc[0]
        self.assertTrue(math.isclose(float(gross_profit_row["coverage_pct"]), 33.33333333333333, rel_tol=1e-9))
        self.assertTrue(math.isclose(float(gross_profit_row["missing_pct_clean_table"]), 66.66666666666666, rel_tol=1e-9))
        self.assertTrue(((by_ticker_df["concept_name"] == "revenue") & (by_ticker_df["ticker"] == "AAA")).any())
        self.assertTrue(((by_year_df["concept_name"] == "revenue") & (by_year_df["period_year"] == 2024)).any())

    def test_engineer_features_builds_new_accounting_metrics(self) -> None:
        clean_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "cik": ["0001", "0001"],
                "filing_date": pd.to_datetime(["2024-04-30", "2024-07-30"]),
                "period_end": pd.to_datetime(["2024-03-31", "2024-06-30"]),
                "fiscal_period": ["Q1", "Q2"],
                "fiscal_year": [2024, 2024],
                "form_type": ["10-Q", "10-Q"],
                "revenue": [100.0, 120.0],
                "gross_profit": [40.0, 54.0],
                "cogs": [60.0, 66.0],
                "net_income": [10.0, 12.0],
                "total_assets": [200.0, 220.0],
                "total_liabilities": [120.0, 130.0],
                "current_assets": [90.0, 100.0],
                "current_liabilities": [45.0, 50.0],
                "cash_and_cash_equivalents": [20.0, 24.0],
                "shareholders_equity": [80.0, 90.0],
                "operating_income": [15.0, 18.0],
                "ebit": [16.0, 19.0],
                "operating_cash_flow": [14.0, 18.0],
                "long_term_debt": [50.0, 54.0],
                "short_term_debt": [10.0, 12.0],
                "inventory": [25.0, 28.0],
                "accounts_receivable": [18.0, 20.0],
                "interest_expense": [2.0, 2.0],
                "capex": [4.0, 5.0],
                "sga": [12.0, 13.0],
                "r_and_d": [6.0, 7.0],
                "share_repurchases": [3.0, 4.0],
                "dividends_paid": [1.0, 1.5],
            }
        )

        engineered = engineer_features(normalize_input_data(clean_df))

        self.assertTrue(math.isclose(float(engineered.loc[1, "gross_margin"]), 0.45, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(engineered.loc[1, "inventory_turnover"]), 66.0 / 26.5, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(engineered.loc[1, "interest_coverage"]), 9.5, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(engineered.loc[1, "free_cash_flow"]), 13.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(engineered.loc[1, "capex_intensity"]), 5.0 / 120.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(engineered.loc[1, "shareholder_payout_ratio"]), 5.5 / 18.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(engineered.loc[1, "leverage_change_qoq"]), (66.0 / 220.0) - (60.0 / 200.0), abs_tol=1e-12))

    def test_save_diagnostics_reports_writes_expected_files(self) -> None:
        concept_df = pd.DataFrame({"ticker": ["AAA"], "concept_name": ["revenue"]})
        clean_df = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "period_end": pd.to_datetime(["2024-03-31"]),
                "revenue": [100.0],
            }
        )
        feature_df = pd.DataFrame({"gross_margin": [0.4], "free_cash_flow": [10.0]})

        diagnostics_dir = Path("outputs") / "test_tmp" / "accounting_expansion_case"
        if diagnostics_dir.exists():
            shutil.rmtree(diagnostics_dir)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        try:
            coverage_paths = save_coverage_diagnostics(concept_df, clean_df, output_dir=diagnostics_dir)
            feasibility_path = save_unlocked_feature_feasibility_report(feature_df, output_dir=diagnostics_dir)

            self.assertTrue(coverage_paths["overall"].exists())
            self.assertTrue(coverage_paths["by_ticker"].exists())
            self.assertTrue(coverage_paths["by_year"].exists())
            self.assertTrue(coverage_paths["concept_map"].exists())
            self.assertTrue(feasibility_path.exists())
        finally:
            shutil.rmtree(diagnostics_dir, ignore_errors=True)

    def test_build_unlocked_feature_feasibility_report_marks_coverage(self) -> None:
        feature_df = pd.DataFrame(
            {
                "gross_margin": [0.4, 0.3, None, 0.2],
                "interest_coverage": [None, None, None, None],
                "free_cash_flow": [10.0, 12.0, 8.0, None],
            }
        )

        report_df = build_unlocked_feature_feasibility_report(feature_df)

        gross_margin_row = report_df.loc[report_df["feature_name"] == "gross_margin"].iloc[0]
        interest_coverage_row = report_df.loc[report_df["feature_name"] == "interest_coverage"].iloc[0]
        self.assertEqual(bool(gross_margin_row["materially_feasible"]), True)
        self.assertEqual(bool(interest_coverage_row["materially_feasible"]), False)


if __name__ == "__main__":
    unittest.main()
