import argparse
import math
import unittest
from pathlib import Path

import pandas as pd

from src.build_sector_sentiment_panel import build_sector_sentiment_panel


class BuildSectorSentimentPanelTests(unittest.TestCase):
    def _test_dir(self) -> Path:
        path = Path.cwd() / "outputs" / "test_build_sector_sentiment_panel" / self._testMethodName
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _args(
        self,
        *,
        event_panel: Path,
        news_scores: Path,
        output_panel: Path,
        diagnostics_output: Path,
        ticker_sector_map: str = "",
        fail_on_empty: bool = False,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            event_panel=str(event_panel),
            news_scores=str(news_scores),
            output_panel=str(output_panel),
            diagnostics_output=str(diagnostics_output),
            ticker_sector_map=ticker_sector_map,
            event_date_col="",
            ticker_col="ticker",
            sector_col="sector",
            windows=[7, 30, 63],
            config_output="",
            fail_on_empty=fail_on_empty,
        )

    def test_script_reads_files_and_writes_enriched_panel(self) -> None:
        tmp = self._test_dir()
        event_path = tmp / "events.csv"
        news_path = tmp / "news.csv"
        output_path = tmp / "enriched.csv"
        diagnostics_path = tmp / "diagnostics.md"
        pd.DataFrame(
            {
                "ticker": ["aaa", "BBB"],
                "event_date": ["2024-01-10", "2024-01-10"],
                "sector": ["Tech", "Tech"],
                "target": [1, 0],
            }
        ).to_csv(event_path, index=False)
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB"],
                "date": ["2024-01-05", "2024-01-10", "2024-01-03"],
                "finbert_pos": [0.8, 0.1, 0.4],
                "finbert_neu": [0.1, 0.8, 0.2],
                "finbert_neg": [0.1, 0.1, 0.4],
            }
        ).to_csv(news_path, index=False)

        enriched = build_sector_sentiment_panel(
            self._args(
                event_panel=event_path,
                news_scores=news_path,
                output_panel=output_path,
                diagnostics_output=diagnostics_path,
            )
        )

        self.assertTrue(output_path.exists())
        self.assertTrue(diagnostics_path.exists())
        self.assertEqual(len(enriched), 2)
        output = pd.read_csv(output_path)
        self.assertEqual(len(output), 2)
        self.assertTrue(math.isclose(float(output.loc[0, "sent_mean_30d"]), 0.7, abs_tol=1e-12))
        self.assertEqual(float(output.loc[0, "news_count_30d"]), 1.0)
        self.assertEqual(float(output.loc[0, "has_news_30d"]), 1.0)

    def test_same_day_news_is_excluded_and_past_news_is_included(self) -> None:
        tmp = self._test_dir()
        event_path = tmp / "events.csv"
        news_path = tmp / "news.csv"
        output_path = tmp / "enriched.csv"
        diagnostics_path = tmp / "diagnostics.md"
        pd.DataFrame({"ticker": ["AAA"], "event_date": ["2024-01-10"], "sector": ["Tech"]}).to_csv(
            event_path, index=False
        )
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "date": ["2024-01-09", "2024-01-10"],
                "finbert_pos": [0.9, 0.1],
                "finbert_neu": [0.05, 0.8],
                "finbert_neg": [0.05, 0.1],
            }
        ).to_csv(news_path, index=False)

        enriched = build_sector_sentiment_panel(
            self._args(
                event_panel=event_path,
                news_scores=news_path,
                output_panel=output_path,
                diagnostics_output=diagnostics_path,
            )
        )

        self.assertEqual(float(enriched.loc[0, "news_count_30d"]), 1.0)
        self.assertTrue(math.isclose(float(enriched.loc[0, "sent_mean_30d"]), 0.85, abs_tol=1e-12))

    def test_missing_news_creates_flags_without_crashing(self) -> None:
        tmp = self._test_dir()
        event_path = tmp / "events.csv"
        news_path = tmp / "news.csv"
        output_path = tmp / "enriched.csv"
        diagnostics_path = tmp / "diagnostics.md"
        pd.DataFrame({"ticker": ["AAA"], "event_date": ["2024-01-10"], "sector": ["Tech"]}).to_csv(
            event_path, index=False
        )
        pd.DataFrame(
            columns=["ticker", "date", "finbert_pos", "finbert_neu", "finbert_neg"]
        ).to_csv(news_path, index=False)

        enriched = build_sector_sentiment_panel(
            self._args(
                event_panel=event_path,
                news_scores=news_path,
                output_panel=output_path,
                diagnostics_output=diagnostics_path,
            )
        )

        self.assertTrue(pd.isna(enriched.loc[0, "sent_mean_30d"]))
        self.assertEqual(float(enriched.loc[0, "has_news_30d"]), 0.0)
        self.assertEqual(float(enriched.loc[0, "low_news_coverage_30d"]), 1.0)

    def test_missing_sector_map_fails_unless_sector_exists_in_event_panel(self) -> None:
        tmp = self._test_dir()
        event_path = tmp / "events.csv"
        news_path = tmp / "news.csv"
        output_path = tmp / "enriched.csv"
        diagnostics_path = tmp / "diagnostics.md"
        pd.DataFrame({"ticker": ["AAA"], "event_date": ["2024-01-10"]}).to_csv(event_path, index=False)
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": ["2024-01-09"],
                "finbert_pos": [0.9],
                "finbert_neu": [0.05],
                "finbert_neg": [0.05],
            }
        ).to_csv(news_path, index=False)

        with self.assertRaisesRegex(ValueError, "Ticker-sector mapping is required"):
            build_sector_sentiment_panel(
                self._args(
                    event_panel=event_path,
                    news_scores=news_path,
                    output_panel=output_path,
                    diagnostics_output=diagnostics_path,
                )
            )

    def test_external_sector_map_allows_event_panel_without_sector(self) -> None:
        tmp = self._test_dir()
        event_path = tmp / "events.csv"
        news_path = tmp / "news.csv"
        sector_path = tmp / "sector_map.csv"
        output_path = tmp / "enriched.csv"
        diagnostics_path = tmp / "diagnostics.md"
        pd.DataFrame({"ticker": ["AAA"], "event_date": ["2024-01-10"]}).to_csv(event_path, index=False)
        pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]}).to_csv(sector_path, index=False)
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": ["2024-01-09"],
                "finbert_pos": [0.9],
                "finbert_neu": [0.05],
                "finbert_neg": [0.05],
            }
        ).to_csv(news_path, index=False)

        enriched = build_sector_sentiment_panel(
            self._args(
                event_panel=event_path,
                news_scores=news_path,
                output_panel=output_path,
                diagnostics_output=diagnostics_path,
                ticker_sector_map=str(sector_path),
            )
        )

        self.assertEqual(len(enriched), 1)
        self.assertEqual(str(enriched.loc[0, "sector"]), "Tech")


if __name__ == "__main__":
    unittest.main()
