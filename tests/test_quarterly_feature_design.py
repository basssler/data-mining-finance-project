import math
import unittest

import pandas as pd

from src.quarterly_feature_design import (
    build_event_sentiment_coverage_diagnostics,
    build_feature_family_coverage,
    build_feature_family_map,
    build_quarterly_feature_design_panel,
    build_sentiment_group_map,
)


class QuarterlyFeatureDesignTests(unittest.TestCase):
    def test_build_quarterly_feature_design_panel_uses_prior_filing_only(self) -> None:
        panel_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "effective_model_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "event_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "source_id": ["a1", "a2"],
                "sector": pd.Series(["Staples", "Staples"], dtype="string"),
                "operating_margin": [0.10, 0.16],
                "net_margin": [0.05, 0.09],
                "roa": [0.04, 0.06],
                "gross_margin": [0.30, 0.36],
                "debt_to_assets": [0.40, 0.37],
                "working_capital_to_total_assets": [0.12, 0.15],
                "cfo_to_net_income": [0.90, 1.10],
                "revenue_growth_yoy": [0.08, 0.13],
                "earnings_growth_yoy": [0.05, 0.12],
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
                "sec_sentiment_score": [0.10, 0.25],
                "sec_positive_prob": [0.60, 0.70],
                "sec_negative_prob": [0.20, 0.10],
                "sec_neutral_prob": [0.20, 0.20],
                "sec_chunk_count": [8.0, 11.0],
                "sec_log_chunk_count": [math.log1p(8.0), math.log1p(11.0)],
            }
        )

        enriched = build_quarterly_feature_design_panel(panel_df)

        self.assertTrue(pd.isna(enriched.loc[0, "qfd_delta_operating_margin"]))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_delta_operating_margin"]), 0.06, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_growth_delta_combo"]), 0.12, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_av_revision_acceleration"]), 1.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_av_estimate_breadth_mean"]), 19.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_v2_operating_margin_d1"]), 0.06, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_v2_debt_to_assets_d1"]), -0.03, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_v2_revenue_growth_yoy_accel"]), 0.05, abs_tol=1e-12))
        self.assertTrue(
            math.isclose(float(enriched.loc[1, "qfd_es_filing_sentiment_delta_prev_q"]), 0.15, abs_tol=1e-12)
        )
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_es_negative_tone_jump"]), -0.10, abs_tol=1e-12))
        self.assertEqual(float(enriched.loc[1, "qfd_es_source_count"]), 1.0)
        self.assertTrue(math.isclose(float(enriched.loc[1, "qfd_es_text_chunk_count"]), 11.0, abs_tol=1e-12))

    def test_build_quarterly_feature_design_panel_builds_non_leaky_event_history(self) -> None:
        panel_df = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "effective_model_date": pd.to_datetime(["2024-04-10"]),
                "event_date": pd.to_datetime(["2024-04-10"]),
                "source_id": ["a1"],
                "sector": pd.Series(["Staples"], dtype="string"),
                "operating_margin": [0.10],
                "net_margin": [0.05],
                "roa": [0.04],
                "gross_margin": [0.30],
                "debt_to_assets": [0.40],
                "working_capital_to_total_assets": [0.12],
                "cfo_to_net_income": [0.90],
                "revenue_growth_yoy": [0.08],
                "earnings_growth_yoy": [0.05],
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
                "sec_sentiment_score": [0.10],
                "sec_positive_prob": [0.55],
                "sec_negative_prob": [0.25],
                "sec_neutral_prob": [0.20],
                "sec_chunk_count": [7.0],
                "sec_log_chunk_count": [math.log1p(7.0)],
            }
        )

        enriched = build_quarterly_feature_design_panel(panel_df)

        self.assertTrue(
            math.isclose(float(enriched.loc[0, "qfd_log_days_since_prior_event"]), math.log1p(60.0), abs_tol=1e-12)
        )
        self.assertTrue(math.isclose(float(enriched.loc[0, "qfd_event_gap_difference"]), -61.0, abs_tol=1e-12))
        self.assertEqual(float(enriched.loc[0, "qfd_short_cycle_flag"]), 1.0)
        self.assertEqual(float(enriched.loc[0, "qfd_short_same_type_cycle_flag"]), 0.0)
        self.assertTrue(
            math.isclose(float(enriched.loc[0, "qfd_es_sentiment_uncertainty"]), 0.45, abs_tol=1e-12)
        )

    def test_build_quarterly_feature_design_panel_builds_stability_variants(self) -> None:
        panel_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "effective_model_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "event_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "source_id": ["a1", "a2"],
                "sector": pd.Series(["Staples", "Staples"], dtype="string"),
                "operating_margin": [0.10, 0.16],
                "net_margin": [0.05, 0.09],
                "roa": [0.04, 0.06],
                "gross_margin": [0.30, 0.36],
                "debt_to_assets": [0.40, 0.37],
                "working_capital_to_total_assets": [0.12, 0.15],
                "cfo_to_net_income": [0.90, 1.10],
                "revenue_growth_yoy": [0.08, 0.13],
                "earnings_growth_yoy": [0.05, 0.12],
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
                "sec_sentiment_score": [0.10, 0.25],
                "sec_positive_prob": [0.60, 0.70],
                "sec_negative_prob": [0.20, 0.10],
                "sec_neutral_prob": [0.20, 0.20],
                "sec_chunk_count": [8.0, 11.0],
                "sec_log_chunk_count": [math.log1p(8.0), math.log1p(11.0)],
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
        self.assertTrue(
            math.isclose(float(enriched.loc[1, "qfd_es_sentiment_polarity_balance"]), 0.40, abs_tol=1e-12)
        )

    def test_build_quarterly_feature_design_panel_builds_cross_sectional_and_stability_variants(self) -> None:
        panel_df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
                "effective_model_date": pd.to_datetime(
                    [
                        "2024-01-10",
                        "2024-01-10",
                        "2024-04-10",
                        "2024-04-10",
                        "2024-07-10",
                        "2024-07-10",
                        "2024-10-10",
                        "2024-10-10",
                    ]
                ),
                "event_date": pd.to_datetime(
                    [
                        "2024-01-10",
                        "2024-01-10",
                        "2024-04-10",
                        "2024-04-10",
                        "2024-07-10",
                        "2024-07-10",
                        "2024-10-10",
                        "2024-10-10",
                    ]
                ),
                "source_id": ["a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4"],
                "sector": pd.Series(["Staples"] * 8, dtype="string"),
                "gross_margin": [0.30, 0.20, 0.32, 0.19, 0.35, 0.18, 0.39, 0.17],
                "operating_margin": [0.10, 0.06, 0.12, 0.05, 0.15, 0.04, 0.19, 0.03],
                "net_margin": [0.05, 0.03, 0.06, 0.02, 0.08, 0.01, 0.10, 0.00],
                "roa": [0.04, 0.02, 0.05, 0.02, 0.06, 0.01, 0.08, 0.01],
                "debt_to_assets": [0.40, 0.70, 0.37, 0.72, 0.35, 0.75, 0.32, 0.77],
                "working_capital_to_total_assets": [0.12, 0.05, 0.13, 0.04, 0.14, 0.04, 0.16, 0.03],
                "cfo_to_net_income": [0.90, 0.70, 0.95, 0.68, 1.00, 0.66, 1.10, 0.65],
                "revenue_growth_yoy": [0.08, 0.03, 0.09, 0.02, 0.11, 0.01, 0.14, 0.00],
                "earnings_growth_yoy": [0.05, 0.02, 0.07, 0.01, 0.10, 0.00, 0.14, -0.01],
                "asset_turnover": [1.1, 0.9, 1.15, 0.88, 1.2, 0.85, 1.25, 0.82],
                "revenue_growth_qoq": [0.02, 0.01, 0.03, 0.00, 0.05, -0.01, 0.06, -0.02],
                "earnings_growth_qoq": [0.01, 0.00, 0.03, -0.01, 0.05, -0.02, 0.06, -0.03],
                "accruals_ratio": [0.03, 0.06, 0.02, 0.07, 0.02, 0.08, 0.01, 0.09],
                "liquidity_profile_score": [0.5, 0.3, 0.55, 0.28, 0.60, 0.27, 0.70, 0.25],
                "solvency_profile_score": [0.4, 0.2, 0.45, 0.19, 0.50, 0.18, 0.60, 0.17],
                "profitability_profile_score": [0.3, 0.1, 0.40, 0.09, 0.55, 0.08, 0.70, 0.07],
                "growth_quality_profile_score": [0.2, 0.1, 0.25, 0.08, 0.35, 0.07, 0.45, 0.06],
                "overall_financial_health_score": [0.35, 0.15, 0.40, 0.14, 0.50, 0.13, 0.60, 0.12],
                "av_latest_prior_eps_surprise_before_event": [0.2] * 8,
                "av_latest_prior_eps_surprise_pct_before_event": [2.0] * 8,
                "av_trailing_4q_eps_surprise_mean": [0.1] * 8,
                "av_trailing_4q_eps_surprise_std": [0.2] * 8,
                "av_trailing_4q_eps_surprise_pct_mean": [1.5] * 8,
                "av_trailing_4q_eps_surprise_pct_std": [0.5] * 8,
                "av_eps_estimate_revision_30d": [1.0] * 8,
                "av_eps_estimate_revision_90d": [0.5] * 8,
                "av_eps_estimate_analyst_count_before_event": [10.0] * 8,
                "av_revenue_estimate_analyst_count_before_event": [12.0] * 8,
                "av_days_since_last_earnings_release": [20.0] * 8,
                "days_since_prior_event": [90.0] * 8,
                "days_since_prior_same_event_type": [92.0] * 8,
                "sec_sentiment_score": [0.10, -0.10, 0.15, -0.05, 0.20, -0.02, 0.24, 0.00],
                "sec_positive_prob": [0.60, 0.30, 0.65, 0.35, 0.70, 0.40, 0.72, 0.42],
                "sec_negative_prob": [0.20, 0.50, 0.15, 0.45, 0.10, 0.40, 0.08, 0.38],
                "sec_neutral_prob": [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
                "sec_chunk_count": [8.0, 6.0, 9.0, 7.0, 10.0, 7.0, 12.0, 8.0],
                "sec_log_chunk_count": [
                    math.log1p(8.0),
                    math.log1p(6.0),
                    math.log1p(9.0),
                    math.log1p(7.0),
                    math.log1p(10.0),
                    math.log1p(7.0),
                    math.log1p(12.0),
                    math.log1p(8.0),
                ],
            }
        )

        enriched = build_quarterly_feature_design_panel(panel_df)
        aaa_last = enriched.loc[enriched["source_id"] == "a4"].iloc[0]
        bbb_last = enriched.loc[enriched["source_id"] == "b4"].iloc[0]

        self.assertGreater(float(aaa_last["qfd_v2_roa_zsec"]), 0.0)
        self.assertLess(float(bbb_last["qfd_v2_roa_zsec"]), 0.0)
        self.assertEqual(float(aaa_last["qfd_v2_operating_margin_d1_ranksec"]), 1.0)
        self.assertEqual(float(bbb_last["qfd_v2_operating_margin_d1_ranksec"]), 0.5)
        self.assertTrue(0.03 < float(aaa_last["qfd_v2_operating_margin_stb4q"]) < 0.04)
        self.assertEqual(float(aaa_last["qfd_v2_operating_margin_pos_count_stb4q"]), 3.0)

    def test_build_quarterly_feature_design_metadata_helpers(self) -> None:
        family_map = build_feature_family_map()
        self.assertIn("qfd_v2_roa_zsec", family_map["feature_name"].tolist())
        self.assertIn("cross_sectional", family_map["feature_family"].tolist())
        self.assertIn("qfd_mkt_pre_event_return_5d", family_map["feature_name"].tolist())
        self.assertIn("event_aware_market_pre_event", family_map["feature_family"].tolist())
        self.assertIn("qfd_es_filing_sentiment_score", family_map["feature_name"].tolist())
        self.assertIn("event_sentiment_level", family_map["feature_family"].tolist())

        coverage = build_feature_family_coverage(
            pd.DataFrame(
                {
                    "qfd_v2_operating_margin_lvl": [0.1, None],
                    "qfd_v2_operating_margin_d1": [None, 0.02],
                    "qfd_v2_roa_zsec": [0.5, -0.5],
                    "qfd_mkt_pre_event_return_5d": [0.01, 0.02],
                    "qfd_es_filing_sentiment_score": [0.1, None],
                }
            )
        )
        self.assertIn("feature_family", coverage.columns)
        self.assertIn("mean_missing_pct", coverage.columns)
        self.assertIn("event_sentiment_level", coverage["feature_family"].tolist())

        sentiment_group_map = build_sentiment_group_map()
        self.assertIn("event_specific_sentiment", sentiment_group_map["sentiment_group"].tolist())
        self.assertIn("combined_sentiment_block", sentiment_group_map["sentiment_group"].tolist())

        sentiment_coverage = build_event_sentiment_coverage_diagnostics(
            pd.DataFrame(
                {
                    "event_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                    "qfd_es_filing_sentiment_score": [0.1, None],
                    "qfd_es_source_count": [1.0, 0.0],
                    "qfd_es_text_chunk_count": [8.0, None],
                }
            )
        )
        self.assertIn("coverage_pct", sentiment_coverage.columns)
        overall_row = sentiment_coverage.loc[sentiment_coverage["scope"] == "overall"].iloc[0]
        self.assertTrue(math.isclose(float(overall_row["coverage_pct"]), 50.0, abs_tol=1e-12))

    def test_build_quarterly_feature_design_panel_adds_event_aware_market_features(self) -> None:
        price_df = pd.DataFrame(
            {
                "ticker": ["AAA"] * 27 + ["BBB"] * 27,
                "date": list(pd.date_range("2024-01-02", periods=27, freq="B")) * 2,
                "open": [100 + idx for idx in range(27)] + [200 + idx for idx in range(27)],
                "close": [100 + idx for idx in range(27)] + [200 + idx for idx in range(27)],
                "adj_close": [100 + idx for idx in range(27)] + [200 + idx for idx in range(27)],
                "volume": [1000 + idx * 10 for idx in range(27)] + [1200 + idx * 10 for idx in range(27)],
            }
        )
        panel_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "effective_model_date": pd.to_datetime(["2024-02-06", "2024-02-07"]),
                "event_date": pd.to_datetime(["2024-02-06", "2024-02-06"]),
                "source_id": ["evt_after_close", "evt_pre_market"],
                "timing_bucket": pd.Series(["after_close", "pre_market"], dtype="string"),
                "sector": pd.Series(["Staples", "Staples"], dtype="string"),
                "operating_margin": [0.10, 0.16],
                "net_margin": [0.05, 0.09],
                "roa": [0.04, 0.06],
                "gross_margin": [0.30, 0.36],
                "debt_to_assets": [0.40, 0.37],
                "working_capital_to_total_assets": [0.12, 0.15],
                "cfo_to_net_income": [0.90, 1.10],
                "revenue_growth_yoy": [0.08, 0.13],
                "earnings_growth_yoy": [0.05, 0.12],
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
                "sec_sentiment_score": [0.10, 0.25],
                "sec_positive_prob": [0.60, 0.70],
                "sec_negative_prob": [0.20, 0.10],
                "sec_neutral_prob": [0.20, 0.20],
                "sec_chunk_count": [8.0, 11.0],
                "sec_log_chunk_count": [math.log1p(8.0), math.log1p(11.0)],
            }
        )

        enriched = build_quarterly_feature_design_panel(panel_df, price_df=price_df)
        after_close_row = enriched.loc[enriched["source_id"] == "evt_after_close"].iloc[0]
        pre_market_row = enriched.loc[enriched["source_id"] == "evt_pre_market"].iloc[0]

        self.assertEqual(after_close_row["qfd_market_pre_event_date"], pd.Timestamp("2024-02-06"))
        self.assertEqual(pre_market_row["qfd_market_pre_event_date"], pd.Timestamp("2024-02-06"))
        self.assertEqual(after_close_row["qfd_market_first_tradable_date"], pd.Timestamp("2024-02-06"))
        self.assertEqual(pre_market_row["qfd_market_first_tradable_date"], pd.Timestamp("2024-02-07"))
        self.assertTrue(math.isclose(float(after_close_row["qfd_mkt_pre_event_return_5d"]), 5.0 / 120.0, rel_tol=0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(pre_market_row["qfd_mkt_first_tradable_gap"]), 1.0 / 125.0, rel_tol=0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(pre_market_row["qfd_mkt_first_tradable_abnormal_volume"]), 1260.0 / 1155.0, rel_tol=0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
