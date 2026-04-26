import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.quarterly_stability import build_family_view_summary, pick_best_stable_row
from src.train_event_panel_v2 import choose_best_model_with_stability, filter_models_by_holdout_gate


class QuarterlyStabilityTests(unittest.TestCase):
    def test_choose_best_model_with_stability_prefers_worst_fold_then_dispersion(self) -> None:
        best_model = choose_best_model_with_stability(
            [
                {
                    "model_name": "xgboost",
                    "worst_fold_auc": 0.44,
                    "cv_auc_std": 0.06,
                    "cv_auc_mean": 0.54,
                    "cv_log_loss_mean": 0.77,
                    "holdout_auc": 0.50,
                },
                {
                    "model_name": "random_forest",
                    "worst_fold_auc": 0.46,
                    "cv_auc_std": 0.04,
                    "cv_auc_mean": 0.52,
                    "cv_log_loss_mean": 0.79,
                    "holdout_auc": 0.49,
                },
            ]
        )

        self.assertEqual(best_model, "random_forest")

    def test_choose_best_model_with_stability_rejects_material_holdout_lag(self) -> None:
        records = [
            {
                "model_name": "logistic_regression",
                "worst_fold_auc": 0.56,
                "cv_auc_std": 0.01,
                "cv_auc_mean": 0.5726,
                "cv_log_loss_mean": 0.70,
                "holdout_auc": 0.4135,
            },
            {
                "model_name": "random_forest",
                "worst_fold_auc": 0.46,
                "cv_auc_std": 0.04,
                "cv_auc_mean": 0.5162,
                "cv_log_loss_mean": 0.79,
                "holdout_auc": 0.5367,
            },
            {
                "model_name": "catboost",
                "worst_fold_auc": 0.45,
                "cv_auc_std": 0.04,
                "cv_auc_mean": 0.5598,
                "cv_log_loss_mean": 0.78,
                "holdout_auc": 0.5348,
            },
        ]

        eligible = filter_models_by_holdout_gate(records, max_holdout_auc_lag=0.02)
        best_model = choose_best_model_with_stability(records, max_holdout_auc_lag=0.02)

        self.assertNotIn("logistic_regression", [record["model_name"] for record in eligible])
        self.assertEqual(best_model, "random_forest")

    def test_pick_best_stable_row_prefers_fold_survival_then_concentration(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "experiment_family": "quarterly_core",
                    "interaction_style": "additive",
                    "model_name": "xgboost",
                    "holdout_auc": 0.49,
                    "worst_fold_auc": 0.43,
                    "cv_auc_std": 0.06,
                    "dominant_feature_top3_folds": 4,
                    "cv_auc_mean": 0.53,
                    "cv_log_loss_mean": 0.77,
                },
                {
                    "experiment_family": "quarterly_core",
                    "interaction_style": "bucketed",
                    "model_name": "random_forest",
                    "holdout_auc": 0.49,
                    "worst_fold_auc": 0.46,
                    "cv_auc_std": 0.04,
                    "dominant_feature_top3_folds": 2,
                    "cv_auc_mean": 0.51,
                    "cv_log_loss_mean": 0.79,
                },
                {
                    "experiment_family": "quarterly_core",
                    "interaction_style": "raw_components",
                    "model_name": "logistic_regression",
                    "holdout_auc": 0.48,
                    "worst_fold_auc": 0.47,
                    "cv_auc_std": 0.03,
                    "dominant_feature_top3_folds": 1,
                    "cv_auc_mean": 0.5,
                    "cv_log_loss_mean": 0.78,
                },
            ]
        )

        best = pick_best_stable_row(df)

        self.assertEqual(str(best["interaction_style"]), "bucketed")
        self.assertEqual(str(best["model_name"]), "random_forest")

    def test_build_family_view_summary_returns_three_views_per_family(self) -> None:
        matrix_df = pd.DataFrame(
            [
                {
                    "experiment_family": "quarterly_core",
                    "interaction_style": "additive",
                    "design_note": "core_additive",
                    "model_name": "xgboost",
                    "holdout_auc": 0.48,
                    "cv_auc_mean": 0.53,
                    "cv_log_loss_mean": 0.78,
                    "cv_auc_std": 0.05,
                    "worst_fold_auc": 0.44,
                    "dominant_feature_name": "feature_a",
                    "dominant_feature_top3_folds": 4,
                    "holdout_row_count": 101,
                    "is_selected_primary_model": True,
                },
                {
                    "experiment_family": "quarterly_core",
                    "interaction_style": "bucketed",
                    "design_note": "core_bucketed",
                    "model_name": "random_forest",
                    "holdout_auc": 0.49,
                    "cv_auc_mean": 0.51,
                    "cv_log_loss_mean": 0.8,
                    "cv_auc_std": 0.03,
                    "worst_fold_auc": 0.46,
                    "dominant_feature_name": "feature_b",
                    "dominant_feature_top3_folds": 2,
                    "holdout_row_count": 101,
                    "is_selected_primary_model": False,
                },
            ]
        )

        summary_df = build_family_view_summary(matrix_df)

        self.assertEqual(summary_df["view_name"].tolist(), ["trainer_selected", "best_holdout", "best_stable"])
        self.assertEqual(str(summary_df.iloc[1]["interaction_style"]), "bucketed")
        self.assertEqual(str(summary_df.iloc[2]["interaction_style"]), "bucketed")

    def test_stability_configs_have_distinct_outputs_and_quarterly_metadata(self) -> None:
        config_paths = sorted(Path("configs").glob("event_panel_v2_quarterly_stability_*.yaml"))
        self.assertEqual(len(config_paths), 8)

        outputs = []
        for config_path in config_paths:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            metadata = config["metadata"]
            outputs.append(config["outputs"]["csv"])
            self.assertIn("Quarterly Stability", metadata["report_title"])
            self.assertIn(metadata["experiment_family"], {"quarterly_core", "quarterly_plus_sentiment"})
            self.assertIn(
                metadata["interaction_style"],
                {"raw_components", "additive", "capped", "bucketed"},
            )
            self.assertEqual(config["promotion"]["strategy"], "stability_aware")
            self.assertTrue(str(config["outputs"]["validation_dir"]).startswith("outputs/quarterly/validation/"))

        self.assertEqual(len(outputs), len(set(outputs)))


if __name__ == "__main__":
    unittest.main()
