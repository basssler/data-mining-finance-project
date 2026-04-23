import unittest
from pathlib import Path

import yaml

from src.label_comparison_event_v2 import build_hist_gradient_boosting_pipeline
from src.train_event_panel_v2 import resolve_model_params, resolve_tuning_spec


class ModelUpgradeTuningTests(unittest.TestCase):
    def test_resolve_tuning_spec_reads_new_contract(self) -> None:
        config = {
            "tuning": {
                "enabled": True,
                "n_trials": 12,
                "objective": {
                    "metric": "mean_cv_auc",
                    "stability_penalty": 0.25,
                    "concentration_penalty": 0.05,
                },
                "models": {"xgboost": {"n_trials": 20}},
                "reproducibility": {
                    "seeds": [11, 22],
                    "max_holdout_auc_std": 0.01,
                    "max_cv_auc_std": 0.02,
                },
            }
        }

        spec = resolve_tuning_spec(config)

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["n_trials"], 12)
        self.assertEqual(spec["objective_metric"], "mean_cv_auc")
        self.assertEqual(spec["models"]["xgboost"]["n_trials"], 20)
        self.assertEqual(spec["reproducibility_seeds"], [11, 22])
        self.assertAlmostEqual(spec["max_holdout_auc_std"], 0.01)
        self.assertAlmostEqual(spec["max_cv_auc_std"], 0.02)

    def test_resolve_model_params_strips_non_estimator_flags(self) -> None:
        config = {
            "xgboost": {
                "prefer_gpu_if_clean": True,
                "fallback_to_cpu": True,
                "note": "ignore me",
                "max_depth": 5,
            }
        }

        params = resolve_model_params(config, "xgboost", override_params={"learning_rate": 0.1}, seed_override=7)

        self.assertEqual(params["max_depth"], 5)
        self.assertEqual(params["learning_rate"], 0.1)
        self.assertEqual(params["random_state"], 7)
        self.assertNotIn("prefer_gpu_if_clean", params)
        self.assertNotIn("fallback_to_cpu", params)
        self.assertNotIn("note", params)

    def test_hist_gradient_boosting_model_is_supported(self) -> None:
        pipeline = build_hist_gradient_boosting_pipeline()

        self.assertEqual(pipeline.named_steps["model"].__class__.__name__, "HistGradientBoostingClassifier")

    def test_quarterly_tuned_upgrade_config_declares_new_model_ladder(self) -> None:
        config_path = Path("configs/quarterly/quarterly_tuned_model_upgrade_v1.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["models"],
            [
                "logistic_regression",
                "random_forest",
                "hist_gradient_boosting",
                "xgboost",
                "lightgbm",
                "catboost",
            ],
        )
        self.assertTrue(config["tuning"]["enabled"])
        self.assertEqual(
            config["promotion"]["reference_results_csv"],
            "reports/results/quarterly_phase9_event_specific_sentiment_champion_benchmark.csv",
        )
        self.assertIn("concentration_csv", config["outputs"])
        self.assertIn("tuning_dir", config["outputs"])


if __name__ == "__main__":
    unittest.main()
