import unittest

import pandas as pd

from src.audit.fundamental_unit_consistency_audit import (
    audit_amount_relationships,
    audit_ratio_sanity,
    audit_raw_candidate_selection,
)


class FundamentalUnitConsistencyAuditTests(unittest.TestCase):
    def test_raw_candidate_audit_does_not_flag_when_consolidated_duplicate_wins(self) -> None:
        raw_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "cik": ["0001", "0001"],
                "filing_date": pd.to_datetime(["2024-05-01", "2024-05-03"]),
                "period_end": pd.to_datetime(["2024-03-31", "2024-03-31"]),
                "fiscal_period": ["Q1", "Q1"],
                "fiscal_year": [2024, 2024],
                "form_type": ["10-Q", "10-Q"],
                "concept_name": ["total_assets", "total_assets"],
                "value": [1_000_000_000.0, 100_000_000.0],
                "unit": ["USD", "USD"],
                "raw_tag": ["Assets", "Assets"],
                "source": ["sec_companyfacts", "sec_companyfacts"],
            }
        )

        audit_df = audit_raw_candidate_selection(raw_df)

        self.assertEqual(len(audit_df), 1)
        self.assertFalse(bool(audit_df.iloc[0]["candidate_scale_flag"]))
        self.assertAlmostEqual(float(audit_df.iloc[0]["selected_to_max_abs_ratio"]), 1.0)

    def test_ratio_sanity_flags_extreme_feature_values(self) -> None:
        feature_df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "period_end": pd.to_datetime(["2024-03-31", "2024-03-31"]),
                "filing_date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
                "asset_turnover": [12.0, 0.8],
                "net_margin": [0.2, 0.1],
            }
        )

        flags = audit_ratio_sanity(feature_df, "test_layer")

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags.iloc[0]["ticker"], "AAA")
        self.assertEqual(flags.iloc[0]["feature"], "asset_turnover")

    def test_amount_relationship_flags_assets_too_small_for_revenue(self) -> None:
        feature_df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "period_end": pd.to_datetime(["2024-03-31", "2024-03-31"]),
                "filing_date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
                "revenue": [10_000_000_000.0, 100_000_000.0],
                "total_assets": [100_000_000.0, 90_000_000.0],
            }
        )

        flags = audit_amount_relationships(feature_df)

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags.iloc[0]["ticker"], "AAA")


if __name__ == "__main__":
    unittest.main()
