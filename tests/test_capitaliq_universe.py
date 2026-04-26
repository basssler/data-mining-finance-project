import unittest
from pathlib import Path

import pandas as pd

from src.build_quarterly_event_panel import merge_capitaliq_metadata
from src.capitaliq_universe import (
    REQUIRED_COLUMNS,
    build_cik_resolution_audit,
    load_capitaliq_universe,
    normalize_ticker,
    save_capitaliq_universe,
)
from src.quarterly_feature_design import QUARTERLY_FEATURE_COLUMNS


class CapitalIqUniverseTests(unittest.TestCase):
    def _tempdir(self) -> Path:
        root = Path.cwd() / "data" / "interim" / "capitaliq_test_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _write_sample_csv(self, path: Path) -> None:
        pd.DataFrame(
            {
                "Company Name": ["Alpha Corp", "Beta Inc", "Gamma Ltd", "Broken Co"],
                "Exchange:Ticker": ["nyse: aaa ", "NasdaqGS:BBB", "NYSE:CCC", "NYSE:DDD"],
                "Ticker": [" aaa ", "bbb", "ccc", "ddd"],
                "Company Type": ["Public Company"] * 4,
                "Primary Exchange": ["NYSE", "Nasdaq Global Select", "NYSE", "NYSE"],
                "Secondary Listings": ["-", "-", "-", "-"],
                "Geographic Locations": ["United States of America (Primary)"] * 4,
                "Market Cap USDmm Latest": ["1,234.5", "2500", "", "99"],
                "Security Tickers": ["NYSE:AAA", "NasdaqGS:BBB", "NYSE:CCC", "NYSE:DDD"],
                "Primary Sector": ["Technology", "Health Care", "Industrials", ""],
                "Industry Classifications": [
                    "Technology (Primary); Software (Primary)",
                    "Health Care (Primary); Biotechnology",
                    "Industrials (Primary); Machinery",
                    "",
                ],
                "Headquarters Country/Region": ["United States"] * 4,
                "CIK": ["123", "1; 0000000002", "", "not-a-cik"],
                "Equity Currency": ["US Dollar"] * 4,
            }
        ).to_csv(path, index=False)

    def test_loader_normalizes_columns_tickers_market_cap_and_ciks(self) -> None:
        tmpdir = self._tempdir()
        csv_path = tmpdir / "capitaliq.csv"
        self._write_sample_csv(csv_path)

        universe = load_capitaliq_universe(csv_path, sec_ticker_to_cik={"BBB": "2"})

        self.assertEqual(list(universe.columns), REQUIRED_COLUMNS)
        self.assertEqual(universe.loc[0, "ticker"], "AAA")
        self.assertEqual(universe.loc[0, "exchange_ticker"], "nyse:AAA")
        self.assertTrue(abs(float(universe.loc[0, "market_cap_usdmm_latest"]) - 1234.5) < 1e-12)
        self.assertEqual(universe.loc[0, "cik_resolved"], "0000000123")
        self.assertEqual(universe.loc[0, "cik_resolution_status"], "resolved_single")
        self.assertEqual(universe.loc[1, "cik_candidates"], "0000000001;0000000002")
        self.assertEqual(universe.loc[1, "cik_resolved"], "0000000002")
        self.assertEqual(universe.loc[1, "cik_resolution_status"], "resolved_sec_match")
        self.assertEqual(universe.loc[2, "cik_resolution_status"], "missing_cik")
        self.assertEqual(universe.loc[3, "cik_resolution_status"], "invalid_cik")

    def test_ticker_normalization(self) -> None:
        self.assertEqual(normalize_ticker(" nvda "), "NVDA")
        self.assertEqual(normalize_ticker("brk b"), "BRKB")

    def test_ambiguous_cik_is_audited(self) -> None:
        tmpdir = self._tempdir()
        csv_path = tmpdir / "capitaliq.csv"
        self._write_sample_csv(csv_path)
        universe = load_capitaliq_universe(csv_path)
        audit = build_cik_resolution_audit(universe)

        beta = audit.loc[audit["ticker"] == "BBB"].iloc[0]
        self.assertEqual(beta["cik_resolution_status"], "ambiguous_multiple")
        self.assertEqual(beta["cik_resolved"], "0000000001")

    def test_save_capitaliq_universe_writes_parquet_and_diagnostics(self) -> None:
        tmp = self._tempdir()
        csv_path = tmp / "capitaliq.csv"
        output_path = tmp / "company_universe.parquet"
        audit_path = tmp / "capitaliq_universe_audit.csv"
        sector_path = tmp / "capitaliq_sector_coverage.csv"
        cik_path = tmp / "capitaliq_cik_resolution_audit.csv"
        self._write_sample_csv(csv_path)

        save_capitaliq_universe(
            csv_path=csv_path,
            output_path=output_path,
            universe_audit_path=audit_path,
            sector_coverage_path=sector_path,
            cik_resolution_audit_path=cik_path,
        )

        self.assertTrue(output_path.exists())
        self.assertTrue(audit_path.exists())
        self.assertTrue(sector_path.exists())
        self.assertTrue(cik_path.exists())
        saved = pd.read_parquet(output_path)
        self.assertEqual(len(saved), 4)

    def test_sector_industry_metadata_merges_into_quarterly_panel(self) -> None:
        panel = pd.DataFrame(
            {
                "ticker": pd.Series(["aaa", "ZZZ"], dtype="string"),
                "cik": pd.Series(["0000000123", "0000000002"], dtype="string"),
                "company_name": pd.Series(["Old Alpha", "Old Zed"], dtype="string"),
                "sector": pd.Series(["old", pd.NA], dtype="string"),
                "industry": pd.Series(["unknown", "unknown"], dtype="string"),
            }
        )
        metadata = pd.DataFrame(
            {
                "ticker": pd.Series(["AAA", "BBB"], dtype="string"),
                "cik_resolved": pd.Series(["0000000123", "0000000002"], dtype="string"),
                "company_name": pd.Series(["Alpha Corp", "Beta Inc"], dtype="string"),
                "sector": pd.Series(["Technology", "Health Care"], dtype="string"),
                "industry": pd.Series(["Software", "Biotechnology"], dtype="string"),
                "primary_exchange": pd.Series(["NYSE", "Nasdaq"], dtype="string"),
            }
        )

        enriched = merge_capitaliq_metadata(panel, metadata_df=metadata)

        self.assertEqual(enriched.loc[0, "company_name"], "Alpha Corp")
        self.assertEqual(enriched.loc[0, "sector"], "Technology")
        self.assertEqual(enriched.loc[0, "industry"], "Software")
        self.assertEqual(enriched.loc[1, "company_name"], "Beta Inc")
        self.assertEqual(enriched.loc[1, "sector"], "Health Care")
        self.assertEqual(enriched.loc[1, "industry"], "Biotechnology")

    def test_missing_capitaliq_metadata_is_graceful_fallback(self) -> None:
        panel = pd.DataFrame({"ticker": ["AAA"], "sector": ["Staples"]})

        enriched = merge_capitaliq_metadata(panel, metadata_df=pd.DataFrame())

        pd.testing.assert_frame_equal(enriched, panel)

    def test_latest_market_cap_is_not_a_model_feature(self) -> None:
        self.assertNotIn("market_cap_usdmm_latest", QUARTERLY_FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
