from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.features.sector_sentiment_features import (
    SENTIMENT_FEATURE_COLUMNS,
    build_sector_sentiment_features,
    write_sector_sentiment_feature_diagnostics,
)
from src.paths import QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR, QUARTERLY_OUTPUTS_PANELS_DIR

DEFAULT_EVENT_PANEL_PATH = QUARTERLY_OUTPUTS_PANELS_DIR / "quarterly_event_panel_features.parquet"
DEFAULT_NEWS_SCORES_PATH = Path("data") / "processed" / "news_scores_finbert.parquet"
DEFAULT_OUTPUT_PANEL_PATH = QUARTERLY_OUTPUTS_PANELS_DIR / "quarterly_event_panel_sector_sentiment.parquet"
DEFAULT_DIAGNOSTICS_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "sector_sentiment_feature_diagnostics.md"
DEFAULT_CONFIG_OUTPUT_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "sector_sentiment_next_steps.md"
EVENT_DATE_CANDIDATES = ("prediction_date", "event_date", "filing_date", "report_date")
NEWS_DATE_CANDIDATES = ("date", "trading_date", "news_date", "published_at")
REQUIRED_NEWS_COLUMNS = ("finbert_pos", "finbert_neu", "finbert_neg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich a quarterly event panel with sector-aware zero-shot FinBERT news sentiment features."
    )
    parser.add_argument("--event-panel", default=str(DEFAULT_EVENT_PANEL_PATH))
    parser.add_argument("--news-scores", default=str(DEFAULT_NEWS_SCORES_PATH))
    parser.add_argument("--output-panel", default=str(DEFAULT_OUTPUT_PANEL_PATH))
    parser.add_argument("--diagnostics-output", default=str(DEFAULT_DIAGNOSTICS_PATH))
    parser.add_argument("--ticker-sector-map", default="")
    parser.add_argument("--event-date-col", default="")
    parser.add_argument("--ticker-col", default="ticker")
    parser.add_argument("--sector-col", default="sector")
    parser.add_argument("--windows", nargs="+", type=int, default=[7, 30, 63])
    parser.add_argument("--config-output", default="")
    parser.add_argument("--fail-on-empty", action="store_true")
    return parser.parse_args()


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file was not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension for {path}; expected .csv or .parquet")


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output extension for {path}; expected .csv or .parquet")


def _normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.strip()


def resolve_event_date_col(event_df: pd.DataFrame, explicit_event_date_col: str = "") -> str:
    if explicit_event_date_col:
        if explicit_event_date_col not in event_df.columns:
            raise ValueError(f"--event-date-col was provided but is not in event panel: {explicit_event_date_col}")
        return explicit_event_date_col
    for column in EVENT_DATE_CANDIDATES:
        if column in event_df.columns:
            return column
    raise ValueError(
        "No usable event date column found. Provide --event-date-col or include one of: "
        + ", ".join(EVENT_DATE_CANDIDATES)
    )


def resolve_news_date_col(news_df: pd.DataFrame) -> str:
    for column in NEWS_DATE_CANDIDATES:
        if column in news_df.columns:
            return column
    raise ValueError("No usable news date column found; expected one of: " + ", ".join(NEWS_DATE_CANDIDATES))


def validate_inputs(
    event_df: pd.DataFrame,
    news_df: pd.DataFrame,
    *,
    ticker_col: str,
    event_date_col: str,
    news_date_col: str,
) -> None:
    if ticker_col not in event_df.columns:
        raise ValueError(f"event panel is missing ticker column: {ticker_col}")
    if ticker_col not in news_df.columns:
        raise ValueError(f"news scores are missing ticker column: {ticker_col}")
    missing_news_columns = [column for column in REQUIRED_NEWS_COLUMNS if column not in news_df.columns]
    if missing_news_columns:
        raise ValueError("news scores are missing required FinBERT columns: " + ", ".join(missing_news_columns))
    if event_date_col not in event_df.columns:
        raise ValueError(f"event panel is missing event date column: {event_date_col}")
    if news_date_col not in news_df.columns:
        raise ValueError(f"news scores are missing news date column: {news_date_col}")

    event_dates = pd.to_datetime(event_df[event_date_col], errors="coerce")
    if event_dates.notna().sum() == 0:
        raise ValueError(f"event date column has no parseable dates: {event_date_col}")
    if news_df.empty:
        return
    news_dates = pd.to_datetime(news_df[news_date_col], errors="coerce")
    if news_dates.notna().sum() == 0:
        raise ValueError(f"news date column has no parseable dates: {news_date_col}")


def load_or_derive_ticker_sector_map(
    *,
    event_df: pd.DataFrame,
    ticker_sector_map_path: Path | None,
    ticker_col: str,
    sector_col: str,
) -> pd.DataFrame:
    if ticker_sector_map_path is not None:
        ticker_sector_df = _read_frame(ticker_sector_map_path)
        missing = [column for column in (ticker_col, sector_col) if column not in ticker_sector_df.columns]
        if missing:
            raise ValueError("ticker-sector map is missing required columns: " + ", ".join(missing))
    elif sector_col in event_df.columns:
        ticker_sector_df = event_df[[ticker_col, sector_col]].copy()
    else:
        raise ValueError(
            "Ticker-sector mapping is required. Provide --ticker-sector-map or include "
            f"{sector_col!r} in the event panel."
        )

    ticker_sector_df = ticker_sector_df[[ticker_col, sector_col]].copy()
    ticker_sector_df[ticker_col] = _normalize_ticker(ticker_sector_df[ticker_col])
    ticker_sector_df[sector_col] = ticker_sector_df[sector_col].astype("string").str.strip()
    ticker_sector_df = ticker_sector_df.dropna(subset=[ticker_col, sector_col], how="any")
    ticker_sector_df = ticker_sector_df.drop_duplicates(subset=[ticker_col], keep="first").reset_index(drop=True)
    if ticker_sector_df.empty:
        raise ValueError("Ticker-sector mapping is empty after normalization.")
    return ticker_sector_df


def _write_next_steps(path: Path, output_panel: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = "\n".join(
        [
            "# Sector-Aware FinBERT Next Steps",
            "",
            "The enriched panel has been built. Keep interpretation cautious: sector-aware FinBERT is a sentiment measurement feature, not proof of predictability.",
            "",
            "Experiment registry:",
            "",
            "- `configs/quarterly/sector_aware_finbert_experiments.yaml`",
            "",
            "Comparison ladder:",
            "",
            "- A. `quarterly_core`",
            "- B. `quarterly_core + market`",
            "- C. `quarterly_core + market + zero-shot FinBERT sentiment`",
            "- D. `quarterly_core + market + sector-adjusted FinBERT sentiment`",
            "",
            "Manual config step:",
            "",
            f"- Point the sector-aware candidate config panel path at `{output_panel.as_posix()}` before running `src/train_event_panel_v2.py`.",
            "",
        ]
    )
    path.write_text(markdown, encoding="utf-8")


def build_sector_sentiment_panel(args: argparse.Namespace) -> pd.DataFrame:
    event_panel_path = Path(args.event_panel)
    news_scores_path = Path(args.news_scores)
    output_panel_path = Path(args.output_panel)
    diagnostics_output_path = Path(args.diagnostics_output)
    ticker_sector_map_path = Path(args.ticker_sector_map) if args.ticker_sector_map else None
    config_output_path = Path(args.config_output) if args.config_output else None

    event_df = _read_frame(event_panel_path)
    news_df = _read_frame(news_scores_path)
    event_date_col = resolve_event_date_col(event_df, args.event_date_col)
    news_date_col = resolve_news_date_col(news_df)
    validate_inputs(
        event_df,
        news_df,
        ticker_col=args.ticker_col,
        event_date_col=event_date_col,
        news_date_col=news_date_col,
    )
    if args.fail_on_empty and news_df.empty:
        raise ValueError("news scores file is empty and --fail-on-empty was provided")

    ticker_sector_df = load_or_derive_ticker_sector_map(
        event_df=event_df,
        ticker_sector_map_path=ticker_sector_map_path,
        ticker_col=args.ticker_col,
        sector_col=args.sector_col,
    )

    event_tickers = set(_normalize_ticker(event_df[args.ticker_col]).dropna())
    mapped_tickers = set(_normalize_ticker(ticker_sector_df[args.ticker_col]).dropna())
    missing_sector_tickers = sorted(event_tickers.difference(mapped_tickers))
    if missing_sector_tickers:
        print(
            "Warning: some event tickers lack sector mapping: "
            + ", ".join(missing_sector_tickers[:10])
            + (" ..." if len(missing_sector_tickers) > 10 else "")
        )
    if "confidence" not in news_df.columns:
        print("Warning: confidence column is missing; confidence_mean_30d will remain NaN.")

    working_event_df = event_df.copy()
    sector_lookup = ticker_sector_df[[args.ticker_col, args.sector_col]].copy()
    if args.sector_col not in working_event_df.columns:
        working_event_df[args.ticker_col] = _normalize_ticker(working_event_df[args.ticker_col])
        working_event_df = working_event_df.merge(sector_lookup, on=args.ticker_col, how="left")
    else:
        mapped_sector = working_event_df[[args.ticker_col]].copy()
        mapped_sector[args.ticker_col] = _normalize_ticker(mapped_sector[args.ticker_col])
        mapped_sector = mapped_sector.merge(sector_lookup, on=args.ticker_col, how="left")[args.sector_col]
        working_event_df[args.sector_col] = working_event_df[args.sector_col].fillna(mapped_sector)
    had_prediction_date = "prediction_date" in working_event_df.columns
    working_event_df["prediction_date"] = pd.to_datetime(working_event_df[event_date_col], errors="coerce")

    # Conservative leakage rule: exact publication timestamps and market-close
    # timing are often unavailable, so the feature builder admits only
    # news_date < prediction_date and excludes same-day news.
    enriched = build_sector_sentiment_features(
        news_scores_df=news_df,
        ticker_sector_df=ticker_sector_df,
        event_df=working_event_df,
        windows=tuple(args.windows),
        date_col=news_date_col,
        ticker_col=args.ticker_col,
        sector_col=args.sector_col,
    )
    if not had_prediction_date and "prediction_date" not in event_df.columns:
        enriched = enriched.drop(columns=["prediction_date"], errors="ignore")
    if len(enriched) != len(event_df):
        raise ValueError(f"row count changed after enrichment: input={len(event_df)}, output={len(enriched)}")

    duplicate_keys = [args.ticker_col, event_date_col]
    input_duplicates = event_df.duplicated(subset=duplicate_keys).sum()
    output_duplicates = enriched.duplicated(subset=duplicate_keys).sum()
    if input_duplicates == 0 and output_duplicates > 0:
        raise ValueError("duplicate ticker-event rows were introduced unexpectedly")
    if input_duplicates > 0:
        print(f"Warning: input event panel already has {int(input_duplicates)} duplicate ticker-event key rows.")

    coverage_30d = pd.to_numeric(enriched.get("has_news_30d", pd.Series(dtype="float64")), errors="coerce").fillna(0)
    covered_30d = int(coverage_30d.sum())
    if len(enriched) and coverage_30d.mean() < 0.20:
        print("Warning: 30-day news coverage is sparse; treat downstream lift estimates cautiously.")
    if len(enriched) and covered_30d < len(enriched):
        print(f"Warning: {len(enriched) - covered_30d:,} event rows have no eligible prior 30-day news.")

    _write_frame(enriched, output_panel_path)
    write_sector_sentiment_feature_diagnostics(
        enriched,
        diagnostics_output_path,
        news_scores_df=news_df,
        date_col=news_date_col,
        sector_col=args.sector_col,
    )
    if config_output_path is not None:
        _write_next_steps(config_output_path, output_panel_path)

    print("Sector-aware FinBERT panel build complete.")
    print(f"Event panel:        {event_panel_path}")
    print(f"News scores:        {news_scores_path}")
    print(f"Output panel:       {output_panel_path}")
    print(f"Diagnostics:        {diagnostics_output_path}")
    print(f"Rows:               {len(enriched):,}")
    print(f"30d news coverage:  {covered_30d:,} / {len(enriched):,}")
    print("Experiment registry: configs/quarterly/sector_aware_finbert_experiments.yaml")
    return enriched


def main() -> None:
    build_sector_sentiment_panel(parse_args())


if __name__ == "__main__":
    main()
