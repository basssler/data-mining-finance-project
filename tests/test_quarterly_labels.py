import unittest

import pandas as pd

from src.build_quarterly_labels import (
    build_fold_distribution_summary,
    build_quantile_label_map,
    build_thresholded_label_map,
    summarize_label_distribution,
)
from src.train_event_panel_v2 import build_default_label_description


class QuarterlyLabelTests(unittest.TestCase):
    def test_thresholded_label_map_counts_ambiguous_rows(self) -> None:
        base_df = pd.DataFrame(
            {
                "event_id": ["a", "b", "c"],
                "ticker": ["AAA", "BBB", "CCC"],
                "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
                "validation_group": ["y2024_q1", "y2024_q1", "y2024_q1"],
                "forward_return": [0.03, 0.001, -0.04],
                "benchmark_forward_return": [0.01, 0.01, 0.01],
                "excess_forward_return": [0.02, -0.009, -0.05],
            }
        )

        label_df = build_thresholded_label_map(base_df, horizon_days=21, threshold=0.015)
        summary = summarize_label_distribution(label_df)

        self.assertEqual(summary["label_available_count"], 2)
        self.assertEqual(summary["dropped_ambiguous_count"], 1)
        self.assertEqual(summary["class_1_count"], 1)
        self.assertEqual(summary["class_0_count"], 1)

    def test_quantile_label_map_drops_middle_bucket_and_small_groups(self) -> None:
        base_df = pd.DataFrame(
            {
                "event_id": ["a", "b", "c", "d", "e"],
                "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "date": pd.to_datetime(
                    ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
                ),
                "validation_group": ["y2024_q1"] * 5,
                "forward_return": [0.0] * 5,
                "benchmark_forward_return": [0.0] * 5,
                "excess_forward_return": [-0.03, 0.00, 0.04, -0.01, 0.02],
            }
        )

        label_df = build_quantile_label_map(base_df, horizon_days=21)

        first_day = label_df.loc[label_df["date"] == pd.Timestamp("2024-01-02")]
        second_day = label_df.loc[label_df["date"] == pd.Timestamp("2024-01-03")]

        self.assertEqual(int(first_day["label_available"].sum()), 2)
        self.assertEqual(int(first_day["dropped_ambiguous"].sum()), 1)
        self.assertEqual(int(second_day["label_available"].sum()), 0)
        self.assertEqual(int(second_day["dropped_ambiguous"].sum()), 2)

    def test_fold_distribution_summary_returns_fold_keys(self) -> None:
        rows = []
        dates = pd.date_range("2023-01-02", periods=12, freq="B")
        for date in dates:
            for idx, ticker in enumerate(["AAA", "BBB", "CCC"], start=1):
                excess = float(idx) / 100.0
                rows.append(
                    {
                        "event_id": f"{ticker}-{date.date()}",
                        "ticker": ticker,
                        "date": date,
                        "validation_group": "test",
                        "forward_return": excess,
                        "benchmark_forward_return": 0.0,
                        "excess_forward_return": excess,
                        "target": 1 if idx == 3 else 0,
                        "label_available": True,
                        "dropped_ambiguous": False,
                    }
                )
        label_df = pd.DataFrame(rows)

        by_fold = build_fold_distribution_summary(
            label_df=label_df,
            horizon_days=1,
            holdout_start="2023-01-16",
            n_splits=2,
            embargo_days=1,
            min_train_dates=3,
        )

        self.assertIn("fold_1_validation", by_fold)
        self.assertIn("fold_2_validation", by_fold)
        self.assertIn("holdout_eval", by_fold)

    def test_thresholded_label_description_is_human_readable(self) -> None:
        description = build_default_label_description(
            {
                "horizon_days": 21,
                "mode": "thresholded",
                "threshold": 0.015,
            }
        )
        self.assertEqual(description, "21-trading-day thresholded excess return (+/-1.5%)")


if __name__ == "__main__":
    unittest.main()
