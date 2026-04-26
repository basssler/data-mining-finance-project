import math
import unittest

import pandas as pd

from src.features.sector_sentiment_features import build_sector_sentiment_features


class SectorSentimentFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ticker_sector_df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "sector": ["Tech", "Tech", "Health"],
            }
        )

    def test_rolling_ticker_sentiment_uses_only_past_news(self) -> None:
        news_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-09"]),
                "finbert_pos": [0.8, 0.4, 0.9],
                "finbert_neu": [0.1, 0.4, 0.05],
                "finbert_neg": [0.1, 0.2, 0.05],
            }
        )
        event_df = pd.DataFrame({"ticker": ["AAA"], "event_date": pd.to_datetime(["2024-01-10"]), "sector": ["Tech"]})

        features = build_sector_sentiment_features(news_df, self.ticker_sector_df, event_df)

        self.assertTrue(math.isclose(float(features.loc[0, "sent_mean_7d"]), 0.525, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(features.loc[0, "sent_mean_30d"]), 0.5833333333333334, abs_tol=1e-12))
        self.assertEqual(float(features.loc[0, "news_count_7d"]), 2.0)
        self.assertEqual(float(features.loc[0, "has_news_7d"]), 1.0)

    def test_news_on_or_after_event_date_is_excluded(self) -> None:
        news_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-09", "2024-01-10", "2024-01-11"]),
                "finbert_pos": [0.9, 0.1, 0.1],
                "finbert_neu": [0.05, 0.8, 0.8],
                "finbert_neg": [0.05, 0.1, 0.1],
            }
        )
        event_df = pd.DataFrame({"ticker": ["AAA"], "event_date": pd.to_datetime(["2024-01-10"]), "sector": ["Tech"]})

        features = build_sector_sentiment_features(news_df, self.ticker_sector_df, event_df)

        self.assertEqual(float(features.loc[0, "news_count_30d"]), 1.0)
        self.assertTrue(math.isclose(float(features.loc[0, "sent_mean_30d"]), 0.85, abs_tol=1e-12))

    def test_sector_mean_and_adjusted_sentiment_are_correct(self) -> None:
        news_df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "finbert_pos": [0.8, 0.3],
                "finbert_neu": [0.1, 0.4],
                "finbert_neg": [0.1, 0.3],
            }
        )
        event_df = pd.DataFrame({"ticker": ["AAA"], "event_date": pd.to_datetime(["2024-01-10"]), "sector": ["Tech"]})

        features = build_sector_sentiment_features(news_df, self.ticker_sector_df, event_df)

        self.assertTrue(math.isclose(float(features.loc[0, "sector_sent_mean_30d"]), 0.35, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(features.loc[0, "sector_adj_sent_30d"]), 0.35, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(features.loc[0, "sector_adj_news_share_30d"]), 0.5, abs_tol=1e-12))

    def test_missing_news_produces_nan_sentiment_and_zero_flags(self) -> None:
        news_df = pd.DataFrame(
            {
                "ticker": ["BBB"],
                "date": pd.to_datetime(["2024-01-01"]),
                "finbert_pos": [0.3],
                "finbert_neu": [0.4],
                "finbert_neg": [0.3],
            }
        )
        event_df = pd.DataFrame({"ticker": ["CCC"], "event_date": pd.to_datetime(["2024-01-10"]), "sector": ["Health"]})

        features = build_sector_sentiment_features(news_df, self.ticker_sector_df, event_df)

        self.assertTrue(pd.isna(features.loc[0, "sent_mean_30d"]))
        self.assertEqual(float(features.loc[0, "has_news_30d"]), 0.0)
        self.assertEqual(float(features.loc[0, "low_news_coverage_30d"]), 1.0)

    def test_no_duplicate_ticker_event_rows_are_created(self) -> None:
        news_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "finbert_pos": [0.8, 0.7, 0.3],
                "finbert_neu": [0.1, 0.2, 0.4],
                "finbert_neg": [0.1, 0.1, 0.3],
            }
        )
        event_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "event_date": pd.to_datetime(["2024-01-10", "2024-04-10"]),
                "sector": ["Tech", "Tech"],
            }
        )

        features = build_sector_sentiment_features(news_df, self.ticker_sector_df, event_df)

        self.assertEqual(len(features), 2)
        self.assertEqual(features[["ticker", "event_date"]].drop_duplicates().shape[0], 2)

    def test_empty_sector_news_is_handled_gracefully(self) -> None:
        news_df = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": pd.to_datetime(["2024-01-01"]),
                "finbert_pos": [0.8],
                "finbert_neu": [0.1],
                "finbert_neg": [0.1],
            }
        )
        event_df = pd.DataFrame({"ticker": ["CCC"], "event_date": pd.to_datetime(["2024-01-10"]), "sector": ["Health"]})

        features = build_sector_sentiment_features(news_df, self.ticker_sector_df, event_df)

        self.assertTrue(pd.isna(features.loc[0, "sector_sent_mean_30d"]))
        self.assertEqual(float(features.loc[0, "sector_news_count_30d"]), 0.0)
        self.assertTrue(pd.isna(features.loc[0, "sector_adj_news_share_30d"]))

    def test_confidence_mean_is_built_when_available(self) -> None:
        news_df = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "finbert_pos": [0.8, 0.7],
                "finbert_neu": [0.1, 0.2],
                "finbert_neg": [0.1, 0.1],
                "confidence": [0.9, 0.7],
            }
        )
        event_df = pd.DataFrame({"ticker": ["AAA"], "event_date": pd.to_datetime(["2024-01-10"]), "sector": ["Tech"]})

        features = build_sector_sentiment_features(news_df, self.ticker_sector_df, event_df)

        self.assertTrue(math.isclose(float(features.loc[0, "confidence_mean_30d"]), 0.8, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
