import unittest

from src.train_event_panel_v2 import resolve_report_metadata


class TrainEventPanelV2MetadataTests(unittest.TestCase):
    def test_resolve_report_metadata_prefers_explicit_config_metadata(self) -> None:
        config = {
            "panel": {"name": "event_panel_v2_quarterly"},
            "label": {
                "variant_name": "event_v2_63d_sign",
                "horizon_days": 63,
                "mode": "sign",
                "benchmark_mode": "sector_equal_weight_ex_self",
            },
            "metadata": {
                "report_title": "Event Panel V2 Quarterly Benchmark",
                "label_description": "63-trading-day excess return sign",
            },
        }

        metadata = resolve_report_metadata(config)

        self.assertEqual(metadata["panel_name"], "event_panel_v2_quarterly")
        self.assertEqual(metadata["panel_display_name"], "event_panel_v2_quarterly")
        self.assertEqual(metadata["report_title"], "Event Panel V2 Quarterly Benchmark")
        self.assertEqual(metadata["label_description"], "63-trading-day excess return sign")
        self.assertEqual(
            metadata["setup_note"],
            "This report is the new post-fix anchor to use before universe expansion.",
        )
        self.assertIn("cleaner anchor", metadata["interpretation_note"])

    def test_resolve_report_metadata_builds_default_label_description(self) -> None:
        config = {
            "panel": {"name": "event_panel_v2_primary"},
            "label": {
                "variant_name": "event_v2_5d_sign",
                "horizon_days": 5,
                "mode": "sign",
                "benchmark_mode": "sector_equal_weight_ex_self",
            },
        }

        metadata = resolve_report_metadata(config)

        self.assertEqual(metadata["report_title"], "event_panel_v2_primary Benchmark")
        self.assertEqual(metadata["label_description"], "5-trading-day excess return sign")

    def test_resolve_report_metadata_supports_custom_positioning_notes(self) -> None:
        config = {
            "panel": {"name": "event_panel_v2_quarterly_alpha_vantage"},
            "label": {
                "variant_name": "event_v2_63d_sign",
                "horizon_days": 63,
                "mode": "sign",
                "benchmark_mode": "sector_equal_weight_ex_self",
            },
            "metadata": {
                "setup_note": "Parallel experiment only.",
                "interpretation_note": "Improves the quarterly holdout but is not the default.",
            },
        }

        metadata = resolve_report_metadata(config)

        self.assertEqual(metadata["setup_note"], "Parallel experiment only.")
        self.assertEqual(
            metadata["interpretation_note"],
            "Improves the quarterly holdout but is not the default.",
        )


if __name__ == "__main__":
    unittest.main()
