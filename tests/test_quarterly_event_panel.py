import unittest

import pandas as pd

from src.build_quarterly_event_panel import (
    MODELING_METADATA_COLUMNS,
    _derive_timing_fields,
    _stable_event_id,
    build_base_panel,
    build_row_count_audit,
    build_timing_audit,
    build_timing_confidence_counts,
    build_timing_samples,
    enrich_feature_panel,
    promote_raw_event_master,
)


class QuarterlyEventPanelTests(unittest.TestCase):
    def test_stable_event_id_is_rerun_stable_for_same_identity(self) -> None:
        event_id_a = _stable_event_id("AAA", "10-Q", "acc1", "2024-05-01")
        event_id_b = _stable_event_id("AAA", "10-Q", "acc1", "2024-05-01")
        event_id_c = _stable_event_id("AAA", "10-Q", "acc2", "2024-05-01")

        self.assertEqual(event_id_a, event_id_b)
        self.assertNotEqual(event_id_a, event_id_c)

    def test_derive_timing_fields_uses_conservative_market_hours_rule(self) -> None:
        timing_df = _derive_timing_fields(
            filing_date=pd.Series(pd.to_datetime(["2024-05-01"] * 4)),
            filing_timestamp_local=pd.Series(
                pd.to_datetime(
                    [
                        "2024-05-01 08:00:00",
                        "2024-05-01 12:00:00",
                        "2024-05-01 16:30:00",
                        None,
                    ]
                )
            ),
            timing_bucket=pd.Series(["pre_market", "market_hours", "after_close", None], dtype="string"),
        )

        self.assertEqual(timing_df.loc[0, "release_session"], "pre_market")
        self.assertEqual(timing_df.loc[0, "timing_confidence"], "exact")
        self.assertEqual(timing_df.loc[0, "availability_base_date"], pd.Timestamp("2024-05-01"))
        self.assertEqual(timing_df.loc[1, "release_session"], "market_hours")
        self.assertEqual(timing_df.loc[1, "availability_base_date"], pd.Timestamp("2024-05-02"))
        self.assertEqual(timing_df.loc[2, "release_session"], "after_close")
        self.assertEqual(timing_df.loc[2, "availability_base_date"], pd.Timestamp("2024-05-02"))
        self.assertEqual(timing_df.loc[3, "timing_confidence"], "inferred_date_only")
        self.assertEqual(timing_df.loc[3, "availability_base_date"], pd.Timestamp("2024-05-02"))

    def test_promote_raw_event_master_prefers_original_and_latest_amendment_fallback(self) -> None:
        raw_master_df = pd.DataFrame(
            {
                "event_id": ["orig", "amend", "amend_only_old", "amend_only_new", "dup_old", "dup_new"],
                "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
                "cik": ["1", "1", "2", "2", "3", "3"],
                "event_type": ["10-Q", "10-Q/A", "10-K/A", "10-K/A", "10-Q", "10-Q"],
                "base_event_type": ["10-Q", "10-Q", "10-K", "10-K", "10-Q", "10-Q"],
                "fiscal_year": pd.Series([2024, 2024, 2023, 2023, 2024, 2024], dtype="Int64"),
                "fiscal_period": pd.Series(["Q1", "Q1", "FY", "FY", "Q2", "Q2"], dtype="string"),
                "period_end": pd.to_datetime(["2024-03-31", "2024-03-31", "2023-12-31", "2023-12-31", "2024-06-30", "2024-06-30"]),
                "event_timestamp_raw": pd.to_datetime(
                    [
                        "2024-05-01 08:00:00",
                        "2024-05-10 08:00:00",
                        "2024-02-20 08:00:00",
                        "2024-03-01 08:00:00",
                        "2024-08-01 08:00:00",
                        "2024-08-01 09:00:00",
                    ]
                ),
                "event_timezone": pd.Series(["America/New_York"] * 6, dtype="string"),
                "event_date_raw": pd.to_datetime(["2024-05-01", "2024-05-10", "2024-02-20", "2024-03-01", "2024-08-01", "2024-08-01"]),
                "release_session": pd.Series(["pre_market"] * 6, dtype="string"),
                "timing_confidence": pd.Series(["exact"] * 6, dtype="string"),
                "tradable_timestamp": pd.to_datetime(
                    [
                        "2024-05-01 09:30:00",
                        "2024-05-10 09:30:00",
                        "2024-02-20 09:30:00",
                        "2024-03-01 09:30:00",
                        "2024-08-01 09:30:00",
                        "2024-08-01 09:30:00",
                    ]
                ),
                "tradable_date": pd.to_datetime(["2024-05-01", "2024-05-10", "2024-02-20", "2024-03-01", "2024-08-01", "2024-08-01"]),
                "feature_snapshot_timestamp": pd.to_datetime(["2024-05-01 09:30:00"] * 6),
                "source_file_id": pd.Series(["s1", "s2", "s3", "s4", "s5", "s5"], dtype="string"),
                "feature_version": pd.Series(["v1"] * 6, dtype="string"),
                "label_version": pd.Series(["label"] * 6, dtype="string"),
                "validation_group": pd.Series(["y2024_q2", "y2024_q2", "y2024_q1", "y2024_q1", "y2024_q3", "y2024_q3"], dtype="string"),
                "sector": pd.Series(["Staples"] * 6, dtype="string"),
                "industry": pd.Series(["unknown"] * 6, dtype="string"),
                "company_name": pd.Series(["A", "A", "B", "B", "C", "C"], dtype="string"),
                "filing_timestamp_utc": pd.to_datetime(["2024-05-01 13:00:00+00:00"] * 6, utc=True),
                "effective_model_date": pd.to_datetime(["2024-05-01", "2024-05-10", "2024-02-20", "2024-03-01", "2024-08-01", "2024-08-01"]),
                "timing_bucket": pd.Series(["pre_market"] * 6, dtype="string"),
                "is_amendment": [False, True, True, True, False, False],
                "promotion_status": pd.Series(["raw"] * 6, dtype="string"),
                "promotion_reason": pd.Series(["unfiltered"] * 6, dtype="string"),
            }
        )

        filtered_df, duplicate_audit_df = promote_raw_event_master(raw_master_df)

        self.assertEqual(set(filtered_df["event_id"]), {"orig", "amend_only_new", "dup_new"})
        self.assertIn("amendment_with_original_present", duplicate_audit_df["drop_reason"].tolist())
        self.assertIn("older_amendment_same_quarter", duplicate_audit_df["drop_reason"].tolist())
        self.assertIn("duplicate_accession_older_row", duplicate_audit_df["drop_reason"].tolist())

    def test_enrich_feature_panel_adds_quarterly_metadata_columns(self) -> None:
        filtered_master_df = pd.DataFrame(
            {
                "event_id": ["evt1"],
                "ticker": pd.Series(["AAA"], dtype="string"),
                "event_type": pd.Series(["10-Q"], dtype="string"),
                "source_file_id": pd.Series(["acc1"], dtype="string"),
                "event_timestamp_raw": pd.to_datetime(["2024-05-01 08:00:00"]),
                "event_timezone": pd.Series(["America/New_York"], dtype="string"),
                "event_date_raw": pd.to_datetime(["2024-05-01"]),
                "release_session": pd.Series(["pre_market"], dtype="string"),
                "timing_confidence": pd.Series(["exact"], dtype="string"),
                "tradable_timestamp": pd.to_datetime(["2024-05-01 09:30:00"]),
                "tradable_date": pd.to_datetime(["2024-05-01"]),
                "feature_snapshot_timestamp": pd.to_datetime(["2024-05-01 09:30:00"]),
                "feature_version": pd.Series(["quarterly_event_panel_v1"], dtype="string"),
                "label_version": pd.Series(["event_v2_63d_sign"], dtype="string"),
                "validation_group": pd.Series(["y2024_q2"], dtype="string"),
                "sector": pd.Series(["Consumer Staples"], dtype="string"),
                "industry": pd.Series(["unknown"], dtype="string"),
                "promotion_reason": pd.Series(["kept_original_filing_for_quarter"], dtype="string"),
            }
        )
        panel_df = pd.DataFrame(
            {
                "ticker": pd.Series(["AAA"], dtype="string"),
                "cik": pd.Series(["1"], dtype="string"),
                "company_name": pd.Series(["AAA Inc"], dtype="string"),
                "event_type": pd.Series(["10-Q"], dtype="string"),
                "event_date": pd.to_datetime(["2024-05-01"]),
                "event_timestamp": pd.to_datetime(["2024-05-01 08:00:00"]),
                "filing_timestamp_utc": pd.to_datetime(["2024-05-01 13:00:00+00:00"], utc=True),
                "effective_model_date": pd.to_datetime(["2024-05-01"]),
                "timing_bucket": pd.Series(["pre_market"], dtype="string"),
                "source_id": pd.Series(["acc1"], dtype="string"),
                "event_period_end": pd.to_datetime(["2024-03-31"]),
                "event_fiscal_period": pd.Series(["Q1"], dtype="string"),
                "event_fiscal_year": pd.Series([2024], dtype="Int64"),
                "current_filing_fundamentals_available": [True],
            }
        )

        feature_panel_df = enrich_feature_panel(panel_df, filtered_master_df)
        base_panel_df = build_base_panel(feature_panel_df)
        row_count_audit_df = build_row_count_audit(filtered_master_df, filtered_master_df, base_panel_df, feature_panel_df)

        self.assertTrue(set(MODELING_METADATA_COLUMNS).issubset(feature_panel_df.columns))
        self.assertEqual(feature_panel_df.loc[0, "event_id"], "evt1")
        self.assertEqual(feature_panel_df.loc[0, "fiscal_period"], "Q1")
        self.assertEqual(int(feature_panel_df.loc[0, "fiscal_year"]), 2024)
        self.assertEqual(len(base_panel_df.columns), len(MODELING_METADATA_COLUMNS) + 1)
        self.assertEqual(row_count_audit_df["row_count"].tolist(), [1, 1, 1, 1])

    def test_timing_audit_and_samples_cover_sessions_and_confidence(self) -> None:
        filtered_master_df = pd.DataFrame(
            {
                "event_id": ["a", "b", "c", "d"],
                "ticker": pd.Series(["AAA", "BBB", "CCC", "DDD"], dtype="string"),
                "event_type": pd.Series(["10-Q", "10-Q", "10-K", "10-K"], dtype="string"),
                "event_date_raw": pd.to_datetime(["2024-05-01", "2024-05-01", "2024-05-01", "2024-05-01"]),
                "event_timestamp_raw": pd.to_datetime(
                    ["2024-05-01 08:00:00", "2024-05-01 12:00:00", "2024-05-01 17:00:00", None]
                ),
                "event_timezone": pd.Series(["America/New_York"] * 4, dtype="string"),
                "release_session": pd.Series(["pre_market", "market_hours", "after_close", "unknown"], dtype="string"),
                "timing_confidence": pd.Series(["exact", "exact", "exact", "inferred_date_only"], dtype="string"),
                "tradable_date": pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-02", "2024-05-02"]),
                "tradable_timestamp": pd.to_datetime(
                    ["2024-05-01 09:30:00", "2024-05-02 09:30:00", "2024-05-02 09:30:00", "2024-05-02 09:30:00"]
                ),
                "source_file_id": pd.Series(["s1", "s2", "s3", "s4"], dtype="string"),
                "promotion_reason": pd.Series(["keep"] * 4, dtype="string"),
            }
        )

        timing_audit_df = build_timing_audit(filtered_master_df, filtered_master_df)
        timing_confidence_counts_df = build_timing_confidence_counts(filtered_master_df, filtered_master_df)
        timing_sample_df = build_timing_samples(filtered_master_df, samples_per_session=1)

        self.assertEqual(set(timing_sample_df["release_session"]), {"pre_market", "market_hours", "after_close", "unknown"})
        self.assertEqual(
            timing_audit_df.loc[
                (timing_audit_df["artifact_name"] == "quarterly_event_master")
                & (timing_audit_df["audit_metric"] == "date_only_conservative_events"),
                "row_count",
            ].iloc[0],
            1,
        )
        self.assertEqual(
            timing_confidence_counts_df.loc[
                (timing_confidence_counts_df["artifact_name"] == "quarterly_event_master")
                & (timing_confidence_counts_df["release_session"] == "unknown"),
                "timing_confidence",
            ].iloc[0],
            "inferred_date_only",
        )
        after_close_row = timing_sample_df.loc[timing_sample_df["release_session"] == "after_close"].iloc[0]
        self.assertGreater(after_close_row["tradable_date"], after_close_row["event_date_raw"])
        market_hours_row = timing_sample_df.loc[timing_sample_df["release_session"] == "market_hours"].iloc[0]
        self.assertGreater(market_hours_row["tradable_date"], market_hours_row["event_date_raw"])
        pre_market_row = timing_sample_df.loc[timing_sample_df["release_session"] == "pre_market"].iloc[0]
        self.assertEqual(pre_market_row["tradable_date"], pre_market_row["event_date_raw"])


if __name__ == "__main__":
    unittest.main()
