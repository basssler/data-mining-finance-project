import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.label_comparison_event_v2 import (
    build_daily_label_table,
    resolve_max_missingness_pct,
    select_usable_features,
)
from src.validation_event_v1 import make_event_v1_splits


class BenchmarkCoreCleanupTests(unittest.TestCase):
    def test_build_daily_label_table_computes_non_5d_forward_returns_from_prices(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=64)
        prices_df = pd.DataFrame(
            {
                "ticker": ["ADM"] * len(dates) + ["CAG"] * len(dates),
                "date": list(dates) * 2,
                "adj_close": list(range(100, 164)) + list(range(200, 264)),
                "forward_return_5d": [999.0] * (len(dates) * 2),
            }
        )

        label_df = build_daily_label_table(
            prices_df,
            horizon_days=63,
            benchmark_mode="universe_equal_weight_ex_self",
        )
        first_day = label_df.loc[label_df["date"] == dates[0]].set_index("ticker")

        self.assertTrue(math.isclose(float(first_day.loc["ADM", "forward_return"]), 63.0 / 100.0, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(first_day.loc["CAG", "forward_return"]), 63.0 / 200.0, rel_tol=0.0, abs_tol=1e-12))
        self.assertNotEqual(float(first_day.loc["ADM", "forward_return"]), 999.0)

    def test_build_daily_label_table_uses_sector_grouping_when_requested(self) -> None:
        prices_df = pd.DataFrame(
            {
                "ticker": ["ADM", "CAG", "AMZN", "F"],
                "date": pd.to_datetime(["2024-01-02"] * 4),
                "adj_close": [100.0, 100.0, 100.0, 100.0],
                "forward_return_5d": [0.10, 0.00, 0.20, 0.30],
            }
        )

        sector_df = build_daily_label_table(
            prices_df,
            horizon_days=5,
            benchmark_mode="sector_equal_weight_ex_self",
        ).set_index("ticker")
        universe_df = build_daily_label_table(
            prices_df,
            horizon_days=5,
            benchmark_mode="universe_equal_weight_ex_self",
        ).set_index("ticker")

        self.assertTrue(math.isclose(float(sector_df.loc["ADM", "benchmark_forward_return"]), 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(sector_df.loc["AMZN", "benchmark_forward_return"]), 0.30, abs_tol=1e-12))
        self.assertTrue(
            math.isclose(
                float(universe_df.loc["ADM", "benchmark_forward_return"]),
                (0.0 + 0.2 + 0.3) / 3.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                float(universe_df.loc["AMZN", "benchmark_forward_return"]),
                (0.1 + 0.0 + 0.3) / 3.0,
                abs_tol=1e-12,
            )
        )

    def test_build_daily_label_table_builds_63d_sector_relative_sign_labels(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=64)
        prices_df = pd.DataFrame(
            {
                "ticker": ["ADM"] * len(dates) + ["CAG"] * len(dates) + ["AMZN"] * len(dates),
                "date": list(dates) * 3,
                "adj_close": [100.0] + [100.0] * 62 + [120.0] + [100.0] + [100.0] * 62 + [105.0] + [100.0] + [100.0] * 62 + [130.0],
                "forward_return_5d": [-1.0] * (len(dates) * 3),
            }
        )

        label_df = build_daily_label_table(
            prices_df,
            horizon_days=63,
            benchmark_mode="sector_equal_weight_ex_self",
        ).loc[lambda df: df["date"] == dates[0]].set_index("ticker")

        self.assertTrue(math.isclose(float(label_df.loc["ADM", "forward_return"]), 0.20, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(label_df.loc["CAG", "benchmark_forward_return"]), 0.20, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(label_df.loc["ADM", "benchmark_forward_return"]), 0.05, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(label_df.loc["ADM", "excess_forward_return"]), 0.15, abs_tol=1e-12))
        self.assertEqual(int(label_df.loc["ADM", "target_sign"]), 1)
        self.assertEqual(int(label_df.loc["CAG", "target_sign"]), 0)
        self.assertTrue(pd.isna(label_df.loc["AMZN", "benchmark_forward_return"]))
        self.assertTrue(pd.isna(label_df.loc["AMZN", "target_sign"]))

    def test_build_daily_label_table_sector_mode_fails_for_missing_sector_mapping(self) -> None:
        prices_df = pd.DataFrame(
            {
                "ticker": ["UNKNOWN"],
                "date": pd.to_datetime(["2024-01-02"]),
                "adj_close": [100.0],
                "forward_return_5d": [0.05],
            }
        )

        with self.assertRaisesRegex(ValueError, "Missing sectors for: UNKNOWN"):
            build_daily_label_table(
                prices_df,
                horizon_days=5,
                benchmark_mode="sector_equal_weight_ex_self",
            )

    def test_select_usable_features_respects_configured_missingness_threshold(self) -> None:
        train_df = pd.DataFrame(
            {
                "feature_keep_if_30": [1.0, None, 3.0, 4.0],
                "feature_always_keep": [1.0, 2.0, 3.0, 4.0],
            }
        )
        candidate_columns = ["feature_keep_if_30", "feature_always_keep"]

        default_usable, _, default_dropped_missing, _ = select_usable_features(train_df, candidate_columns)
        configured_usable, _, configured_dropped_missing, _ = select_usable_features(
            train_df,
            candidate_columns,
            max_missingness_pct=30.0,
        )

        self.assertEqual(resolve_max_missingness_pct(), 20.0)
        self.assertEqual(resolve_max_missingness_pct({"max_missingness_pct": 30}), 30.0)
        self.assertEqual(default_usable, ["feature_always_keep"])
        self.assertEqual(default_dropped_missing, ["feature_keep_if_30"])
        self.assertEqual(configured_usable, ["feature_keep_if_30", "feature_always_keep"])
        self.assertEqual(configured_dropped_missing, [])

    def test_make_event_v1_splits_uses_67_date_purge_span_for_quarterly_lane(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=420)
        split_df = pd.DataFrame(
            {
                "date": dates,
                "row_id": np.arange(len(dates)),
            }
        )

        split_payload = make_event_v1_splits(
            df=split_df,
            date_col="date",
            horizon_days=63,
            embargo_days=5,
            holdout_start=str(dates[360].date()),
            min_train_dates=252,
            n_splits=2,
        )

        first_fold = split_payload["folds"][0]
        holdout = split_payload["holdout"]
        expected_purge_count = 63 + 5 - 1

        self.assertEqual(first_fold["purged_date_count"], expected_purge_count)
        self.assertEqual(holdout["purged_date_count"], expected_purge_count)
        self.assertEqual(first_fold["train_date_count"], 252 - expected_purge_count)
        self.assertEqual(first_fold["date_metadata"]["train_end_date"], str(dates[184].date()))
        self.assertEqual(first_fold["date_metadata"]["purge_start_date"], str(dates[185].date()))
        self.assertEqual(first_fold["date_metadata"]["purge_end_date"], str(dates[251].date()))
        self.assertEqual(first_fold["date_metadata"]["validation_start_date"], str(dates[252].date()))
        self.assertEqual(holdout["date_metadata"]["purge_start_date"], str(dates[293].date()))
        self.assertEqual(holdout["date_metadata"]["purge_end_date"], str(dates[359].date()))
        self.assertEqual(holdout["date_metadata"]["validation_start_date"], str(dates[360].date()))

    def test_quarterly_benchmark_markdown_does_not_contain_5_day_label_text(self) -> None:
        markdown_path = Path("reports/results/event_panel_v2_quarterly_benchmark.md")
        markdown = markdown_path.read_text(encoding="utf-8")

        self.assertIn("63-trading-day excess return sign", markdown)
        self.assertNotIn("5-day", markdown)
        self.assertNotIn("5 trading-day", markdown)


if __name__ == "__main__":
    unittest.main()
