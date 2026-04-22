import math
import unittest

import pandas as pd

from src.label_comparison_event_v2 import (
    build_daily_label_table,
    resolve_max_missingness_pct,
    select_usable_features,
)


class BenchmarkCoreCleanupTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
