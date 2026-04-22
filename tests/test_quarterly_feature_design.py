import math
import unittest

import pandas as pd

from src.quarterly_feature_design import build_quarterly_feature_design_panel


class QuarterlyFeatureDesignTests(unittest.TestCase):
    def test_build_quarterly_feature_design_panel_uses_prior_filing_only(self) -> None:
        panel_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "effective_model_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "event_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "source_id": ["a1", "a2"],
                "operating_margin": [0.10, 0.16],
                "net_margin": [0.05, 0.09],
                "roa": [0.04, 0.06],
                "asset_turnover": [1.1, 1.2],
                "revenue_growth_qoq": [0.02, 0.08],
                "earnings_growth_qoq": [0.01, 0.07],
                "accruals_ratio": [0.03, 0.01],
                "liquidity_profile_score": [0.5, 0.7],
                "solvency_profile_score": [0.4, 0.6],
                "profitability_profile_score": [0.3, 0.8],
                "growth_quality_profile_score": [0.2, 0.9],
                "overall_financial_health_score": [0.35, 0.75],
                "av_latest_prior_eps_surprise_before_event": [0.2, 0.5],
                "av_latest_prior_eps_surprise_pct_before_event": [2.0, 5.0],
                "av_trailing_4q_eps_surprise_mean": [0.1, 0.3],
                "av_trailing_4q_eps_surprise_std": [0.2, 0.4],
                "av_trailing_4q_eps_surprise_pct_mean": [1.5, 2.5],
                "av_trailing_4q_eps_surprise_pct_std": [0.5, 1.0],
                "av_eps_estimate_revision_30d": [1.0, 4.0],
                "av_eps_estimate_revision_90d": [2.0, 3.0],
                "av_eps_estimate_analyst_count_before_event": [10.0, 20.0],
                "av_revenue_estimate_analyst_count_before_event": [12.0, 18.0],
                "av_days_since_last_earnings_release": [20.0, 15.0],
                "days_since_prior_event": [90.0, 91.0],
                "days_since_prior_same_event_type": [90.0, 92.0],
            }
        )

        enriched = build_quarterly_feature_design_panel(panel_df)

        self.assertTrue(pd.isna(enriched.loc[0, "qfd_delta_operating_margin"]))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_delta_operating_margin"]), 0.06, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_growth_delta_combo"]), 0.12, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_av_revision_acceleration"]), 1.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_av_estimate_breadth_mean"]), 19.0, abs_tol=1e-12))

    def test_build_quarterly_feature_design_panel_builds_non_leaky_event_history(self) -> None:
        panel_df = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "effective_model_date": pd.to_datetime(["2024-04-10"]),
                "event_date": pd.to_datetime(["2024-04-10"]),
                "source_id": ["a1"],
                "operating_margin": [0.10],
                "net_margin": [0.05],
                "roa": [0.04],
                "asset_turnover": [1.1],
                "revenue_growth_qoq": [0.02],
                "earnings_growth_qoq": [0.01],
                "accruals_ratio": [0.03],
                "liquidity_profile_score": [0.5],
                "solvency_profile_score": [0.4],
                "profitability_profile_score": [0.3],
                "growth_quality_profile_score": [0.2],
                "overall_financial_health_score": [0.35],
                "av_latest_prior_eps_surprise_before_event": [0.2],
                "av_latest_prior_eps_surprise_pct_before_event": [2.0],
                "av_trailing_4q_eps_surprise_mean": [0.1],
                "av_trailing_4q_eps_surprise_std": [0.2],
                "av_trailing_4q_eps_surprise_pct_mean": [1.5],
                "av_trailing_4q_eps_surprise_pct_std": [0.5],
                "av_eps_estimate_revision_30d": [1.0],
                "av_eps_estimate_revision_90d": [2.0],
                "av_eps_estimate_analyst_count_before_event": [10.0],
                "av_revenue_estimate_analyst_count_before_event": [12.0],
                "av_days_since_last_earnings_release": [20.0],
                "days_since_prior_event": [60.0],
                "days_since_prior_same_event_type": [121.0],
            }
        )

        enriched = build_quarterly_feature_design_panel(panel_df)

        self.assertTrue(
            math.isclose(float(enriched.loc[0, "qfd_log_days_since_prior_event"]), math.log1p(60.0), abs_tol=1e-12)
        )
        self.assertTrue(math.isclose(float(enriched.loc[0, "qfd_event_gap_difference"]), -61.0, abs_tol=1e-12))
        self.assertEqual(float(enriched.loc[0, "qfd_short_cycle_flag"]), 1.0)
        self.assertEqual(float(enriched.loc[0, "qfd_short_same_type_cycle_flag"]), 0.0)

    def test_build_quarterly_feature_design_panel_builds_stability_variants(self) -> None:
        panel_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "effective_model_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "event_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "source_id": ["a1", "a2"],
                "operating_margin": [0.10, 0.16],
                "net_margin": [0.05, 0.09],
                "roa": [0.04, 0.06],
                "asset_turnover": [1.1, 1.2],
                "revenue_growth_qoq": [0.02, 0.08],
                "earnings_growth_qoq": [0.01, 0.07],
                "accruals_ratio": [0.03, 0.01],
                "liquidity_profile_score": [0.5, 0.7],
                "solvency_profile_score": [0.4, 0.6],
                "profitability_profile_score": [0.3, 0.8],
                "growth_quality_profile_score": [0.2, 0.9],
                "overall_financial_health_score": [0.35, 0.75],
                "av_latest_prior_eps_surprise_before_event": [0.2, 0.5],
                "av_latest_prior_eps_surprise_pct_before_event": [2.0, 5.0],
                "av_trailing_4q_eps_surprise_mean": [0.1, 0.3],
                "av_trailing_4q_eps_surprise_std": [0.2, 0.4],
                "av_trailing_4q_eps_surprise_pct_mean": [1.5, 2.5],
                "av_trailing_4q_eps_surprise_pct_std": [0.5, 1.0],
                "av_eps_estimate_revision_30d": [1.0, 4.0],
                "av_eps_estimate_revision_90d": [2.0, 3.0],
                "av_eps_estimate_analyst_count_before_event": [10.0, 20.0],
                "av_revenue_estimate_analyst_count_before_event": [12.0, 18.0],
                "av_days_since_last_earnings_release": [20.0, 15.0],
                "days_since_prior_event": [90.0, 91.0],
                "days_since_prior_same_event_type": [90.0, 92.0],
            }
        )

        enriched = build_quarterly_feature_design_panel(panel_df)

        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_av_revision_x_surprise_capped"]), 20.0, abs_tol=1e-12))
        self.assertEqual(float(enriched.loc[1, "qfd_av_revision_magnitude_bucket"]), 3.0)
        self.assertTrue(
            math.isclose(float(enriched.loc[1, "qfd_av_revision_plus_surprise"]), 9.0, abs_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(float(enriched.loc[1, "qfd_av_latest_surprise_vs_trailing_pct_abs"]), 2.5, abs_tol=1e-12)
        )
        self.assertEqual(float(enriched.loc[1, "qfd_av_latest_surprise_above_trailing_flag"]), 1.0)
        self.assertEqual(float(enriched.loc[1, "qfd_av_latest_surprise_vs_trailing_pct_bucket"]), 1.0)
        self.assertEqual(float(enriched.loc[1, "qfd_av_trailing_surprise_pct_std_bucket"]), 1.0)
        self.assertEqual(float(enriched.loc[1, "qfd_prior_event_gap_bucket"]), 2.0)
        self.assertEqual(float(enriched.loc[1, "qfd_prior_same_type_gap_bucket"]), 1.0)
        self.assertEqual(float(enriched.loc[1, "qfd_av_revision_surprise_same_sign_flag"]), 1.0)


if __name__ == "__main__":
    unittest.main()
