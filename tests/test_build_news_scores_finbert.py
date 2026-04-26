import argparse
import math
import unittest
from pathlib import Path

import pandas as pd

from src.build_news_scores_finbert import build_news_scores_finbert, normalize_label_mapping


class BuildNewsScoresFinbertTests(unittest.TestCase):
    def _test_dir(self) -> Path:
        path = Path.cwd() / "outputs" / "test_build_news_scores_finbert" / self._testMethodName
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _args(
        self,
        *,
        input_news: Path,
        output: Path,
        text_col: str = "",
        ticker_col: str = "",
        date_col: str = "",
        source_col: str = "",
        max_rows: int = 0,
        min_text_length: int = 5,
        fail_on_empty: bool = False,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            input_news=str(input_news),
            output=str(output),
            text_col=text_col,
            ticker_col=ticker_col,
            date_col=date_col,
            source_col=source_col,
            batch_size=16,
            max_rows=max_rows,
            device="cpu",
            min_text_length=min_text_length,
            fail_on_empty=fail_on_empty,
        )

    @staticmethod
    def _fake_scorer(texts: list[str]) -> pd.DataFrame:
        rows = []
        for index, _ in enumerate(texts):
            pos = 0.60 + index * 0.01
            neu = 0.30
            neg = 0.10 - index * 0.01
            rows.append(
                {
                    "finbert_pos": pos,
                    "finbert_neu": neu,
                    "finbert_neg": neg,
                    "confidence": max(pos, neu, neg),
                    "finbert_score": pos - neg,
                }
            )
        return pd.DataFrame(rows)

    def test_input_csv_creates_expected_output_schema(self) -> None:
        tmp = self._test_dir()
        input_path = tmp / "news.csv"
        output_path = tmp / "scores.csv"
        pd.DataFrame(
            {
                "stock": ["aapl"],
                "date": ["2024-01-02 10:30:00-05:00"],
                "title": ["Apple shares rise after earnings"],
            }
        ).to_csv(input_path, index=False)

        output = build_news_scores_finbert(
            self._args(input_news=input_path, output=output_path),
            scorer=self._fake_scorer,
        )

        self.assertTrue(output_path.exists())
        self.assertEqual(
            ["ticker", "date", "finbert_pos", "finbert_neu", "finbert_neg", "confidence", "finbert_score"],
            output.columns[:7].tolist(),
        )
        self.assertIn("headline", output.columns)
        self.assertIn("text_id", output.columns)
        self.assertEqual(str(output.loc[0, "ticker"]), "AAPL")
        self.assertTrue(math.isclose(float(output.loc[0, "finbert_score"]), 0.50, abs_tol=1e-12))

    def test_missing_required_columns_fails_clearly(self) -> None:
        tmp = self._test_dir()
        input_path = tmp / "news.csv"
        output_path = tmp / "scores.csv"
        pd.DataFrame({"ticker": ["AAPL"], "date": ["2024-01-02"]}).to_csv(input_path, index=False)

        with self.assertRaisesRegex(ValueError, "No text column found"):
            build_news_scores_finbert(
                self._args(input_news=input_path, output=output_path),
                scorer=self._fake_scorer,
            )

    def test_date_parsing_and_ticker_normalization_work(self) -> None:
        tmp = self._test_dir()
        input_path = tmp / "news.csv"
        output_path = tmp / "scores.csv"
        pd.DataFrame(
            {
                "symbol": [" msft "],
                "published_date": ["2024-01-02T15:00:00-05:00"],
                "headline": ["Microsoft announces a new product"],
            }
        ).to_csv(input_path, index=False)

        output = build_news_scores_finbert(
            self._args(input_news=input_path, output=output_path),
            scorer=self._fake_scorer,
        )

        self.assertEqual(str(output.loc[0, "ticker"]), "MSFT")
        self.assertFalse(pd.isna(output.loc[0, "date"]))

    def test_label_mapping_handles_positive_neutral_negative_labels(self) -> None:
        mapping = normalize_label_mapping({2: "Positive", 0: "negative", 1: "Neutral"})

        self.assertEqual(mapping["positive"], 2)
        self.assertEqual(mapping["neutral"], 1)
        self.assertEqual(mapping["negative"], 0)

    def test_empty_usable_input_writes_empty_unless_fail_on_empty(self) -> None:
        tmp = self._test_dir()
        input_path = tmp / "news.csv"
        output_path = tmp / "scores.csv"
        pd.DataFrame({"ticker": ["AAPL"], "date": ["not-a-date"], "headline": ["Valid text"]}).to_csv(
            input_path, index=False
        )

        output = build_news_scores_finbert(
            self._args(input_news=input_path, output=output_path),
            scorer=self._fake_scorer,
        )

        self.assertTrue(output.empty)
        self.assertTrue(output_path.exists())

        with self.assertRaisesRegex(ValueError, "No usable news rows"):
            build_news_scores_finbert(
                self._args(input_news=input_path, output=tmp / "fail.csv", fail_on_empty=True),
                scorer=self._fake_scorer,
            )

    def test_max_rows_limits_rows(self) -> None:
        tmp = self._test_dir()
        input_path = tmp / "news.csv"
        output_path = tmp / "scores.csv"
        pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "text": ["First valid headline", "Second valid headline", "Third valid headline"],
            }
        ).to_csv(input_path, index=False)

        output = build_news_scores_finbert(
            self._args(input_news=input_path, output=output_path, max_rows=2),
            scorer=self._fake_scorer,
        )

        self.assertEqual(len(output), 2)
        self.assertEqual(output["ticker"].tolist(), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
