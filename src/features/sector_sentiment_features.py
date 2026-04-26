from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_WINDOWS = (7, 30, 63)
SENTIMENT_FEATURE_COLUMNS = [
    "sent_mean_7d",
    "sent_mean_30d",
    "sent_mean_63d",
    "sent_vol_30d",
    "sent_vol_63d",
    "news_count_7d",
    "news_count_30d",
    "news_count_63d",
    "sent_momentum_7v30",
    "sent_momentum_30v63",
    "confidence_mean_30d",
    "sector_sent_mean_30d",
    "sector_sent_mean_63d",
    "sector_news_count_30d",
    "sector_news_count_63d",
    "sector_adj_sent_30d",
    "sector_adj_sent_63d",
    "sector_adj_news_share_30d",
    "sector_adj_news_share_63d",
    "has_news_7d",
    "has_news_30d",
    "has_news_63d",
    "low_news_coverage_30d",
    "low_news_coverage_63d",
]


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str], role: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"{role} column not found; expected one of: {', '.join(candidates)}")


def _normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.strip()


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.mean())


def _safe_std(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.std(ddof=0))


def _feature_template() -> dict[str, float]:
    return {column: np.nan for column in SENTIMENT_FEATURE_COLUMNS}


def _prepare_news(
    news_scores_df: pd.DataFrame,
    ticker_sector_df: pd.DataFrame,
    *,
    date_col: str,
    ticker_col: str,
    sector_col: str,
) -> pd.DataFrame:
    required = {ticker_col, "finbert_pos", "finbert_neg"}
    missing = sorted(required.difference(news_scores_df.columns))
    if missing:
        raise ValueError("news_scores_df is missing required columns: " + ", ".join(missing))

    actual_date_col = date_col if date_col in news_scores_df.columns else _first_existing_column(
        news_scores_df,
        ("trading_date", "news_date", "published_at", "date"),
        "news date",
    )
    news = news_scores_df.copy()
    news[ticker_col] = _normalize_ticker(news[ticker_col])
    news["_news_date"] = pd.to_datetime(news[actual_date_col], errors="coerce").dt.normalize()
    news["finbert_pos"] = pd.to_numeric(news["finbert_pos"], errors="coerce")
    news["finbert_neg"] = pd.to_numeric(news["finbert_neg"], errors="coerce")
    news["finbert_score"] = news["finbert_pos"] - news["finbert_neg"]
    if "confidence" in news.columns:
        news["confidence"] = pd.to_numeric(news["confidence"], errors="coerce")

    sector_lookup = ticker_sector_df.copy()
    if not sector_lookup.empty:
        sector_lookup[ticker_col] = _normalize_ticker(sector_lookup[ticker_col])
        sector_lookup[sector_col] = sector_lookup[sector_col].astype("string").str.strip()
        sector_lookup = sector_lookup[[ticker_col, sector_col]].drop_duplicates(subset=[ticker_col], keep="first")
        news = news.merge(sector_lookup, on=ticker_col, how="left", suffixes=("", "_lookup"))
        lookup_col = f"{sector_col}_lookup"
        if lookup_col in news.columns:
            news[sector_col] = news[sector_col].fillna(news[lookup_col])
            news = news.drop(columns=[lookup_col])
    elif sector_col not in news.columns:
        news[sector_col] = pd.NA

    news[sector_col] = news[sector_col].astype("string").str.strip()
    return (
        news.dropna(subset=[ticker_col, "_news_date", "finbert_score"])
        .sort_values([ticker_col, "_news_date"])
        .reset_index(drop=True)
    )


def _prepare_events(event_df: pd.DataFrame, *, ticker_col: str, sector_col: str) -> tuple[pd.DataFrame, str]:
    event_date_col = _first_existing_column(
        event_df,
        ("prediction_date", "feature_snapshot_timestamp", "tradable_date", "event_date", "filing_date"),
        "event/prediction date",
    )
    events = event_df.copy()
    events["_event_row_id"] = np.arange(len(events), dtype="int64")
    events[ticker_col] = _normalize_ticker(events[ticker_col])
    events["_event_date"] = pd.to_datetime(events[event_date_col], errors="coerce").dt.normalize()
    if sector_col not in events.columns:
        events[sector_col] = pd.NA
    events[sector_col] = events[sector_col].astype("string").str.strip()
    return events, event_date_col


def _window_values(frame: pd.DataFrame, event_date: pd.Timestamp, window: int) -> pd.DataFrame:
    start_date = event_date - pd.Timedelta(days=int(window))
    return frame[(frame["_news_date"] >= start_date) & (frame["_news_date"] < event_date)]


def _build_row_features(
    *,
    ticker_news: pd.DataFrame,
    sector_news: pd.DataFrame,
    event_date: pd.Timestamp,
    windows: tuple[int, ...],
    has_confidence: bool,
) -> dict[str, float]:
    features = _feature_template()
    if pd.isna(event_date):
        return features

    ticker_windows = {window: _window_values(ticker_news, event_date, window) for window in windows}
    for window, frame in ticker_windows.items():
        if f"sent_mean_{window}d" in features:
            features[f"sent_mean_{window}d"] = _safe_mean(frame["finbert_score"])
        if f"news_count_{window}d" in features:
            features[f"news_count_{window}d"] = float(len(frame))
        if f"has_news_{window}d" in features:
            features[f"has_news_{window}d"] = float(len(frame) > 0)

    for window in (30, 63):
        frame = ticker_windows.get(window, pd.DataFrame())
        if f"sent_vol_{window}d" in features:
            features[f"sent_vol_{window}d"] = _safe_std(frame["finbert_score"]) if not frame.empty else np.nan
        if f"low_news_coverage_{window}d" in features:
            features[f"low_news_coverage_{window}d"] = float(len(frame) < 3)

    if has_confidence:
        frame_30d = ticker_windows.get(30, pd.DataFrame())
        features["confidence_mean_30d"] = _safe_mean(frame_30d["confidence"]) if not frame_30d.empty else np.nan

    features["sent_momentum_7v30"] = features["sent_mean_7d"] - features["sent_mean_30d"]
    features["sent_momentum_30v63"] = features["sent_mean_30d"] - features["sent_mean_63d"]

    for window in (30, 63):
        sector_window = _window_values(sector_news, event_date, window)
        features[f"sector_sent_mean_{window}d"] = _safe_mean(sector_window["finbert_score"])
        features[f"sector_news_count_{window}d"] = float(len(sector_window))
        features[f"sector_adj_sent_{window}d"] = features[f"sent_mean_{window}d"] - features[f"sector_sent_mean_{window}d"]
        sector_count = features[f"sector_news_count_{window}d"]
        if pd.notna(sector_count) and sector_count > 0:
            features[f"sector_adj_news_share_{window}d"] = features[f"news_count_{window}d"] / sector_count

    return features


def build_sector_sentiment_features(
    news_scores_df: pd.DataFrame,
    ticker_sector_df: pd.DataFrame,
    event_df: pd.DataFrame,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    date_col: str = "date",
    ticker_col: str = "ticker",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Build event-level, sector-aware FinBERT sentiment features.

    News is eligible only when ``news_date < event/prediction date``. The
    builder intentionally leaves missing sentiment as NaN and emits coverage
    flags so downstream train-only imputation can decide how to handle gaps.
    """
    windows = tuple(sorted({int(window) for window in windows}))
    if ticker_col not in event_df.columns:
        raise ValueError(f"event_df is missing required column: {ticker_col}")
    if ticker_col not in ticker_sector_df.columns or sector_col not in ticker_sector_df.columns:
        raise ValueError(f"ticker_sector_df must contain {ticker_col} and {sector_col}")

    events, _ = _prepare_events(event_df, ticker_col=ticker_col, sector_col=sector_col)
    news = _prepare_news(
        news_scores_df,
        ticker_sector_df,
        date_col=date_col,
        ticker_col=ticker_col,
        sector_col=sector_col,
    )
    has_confidence = "confidence" in news.columns

    ticker_groups = {ticker: frame for ticker, frame in news.groupby(ticker_col, sort=False)}
    sector_groups = {sector: frame for sector, frame in news.groupby(sector_col, dropna=False, sort=False)}
    feature_rows: list[dict[str, float]] = []

    for _, event in events.iterrows():
        ticker = event[ticker_col]
        sector = event[sector_col]
        ticker_news = ticker_groups.get(ticker, news.iloc[0:0])
        sector_news = sector_groups.get(sector, news.iloc[0:0])
        feature_rows.append(
            _build_row_features(
                ticker_news=ticker_news,
                sector_news=sector_news,
                event_date=event["_event_date"],
                windows=windows,
                has_confidence=has_confidence,
            )
        )

    feature_df = pd.DataFrame(feature_rows, index=events.index)
    output = pd.concat([events.drop(columns=["_event_row_id", "_event_date"]), feature_df], axis=1)
    if len(output) != len(event_df):
        raise ValueError("sector sentiment feature build changed event row count")
    if output.index.duplicated().any():
        raise ValueError("sector sentiment feature build produced duplicate event rows")
    return output.reset_index(drop=True)


def write_sector_sentiment_feature_diagnostics(
    feature_df: pd.DataFrame,
    output_path: str | Path,
    *,
    news_scores_df: pd.DataFrame | None = None,
    date_col: str = "date",
    sector_col: str = "sector",
    sparse_coverage_threshold: float = 0.20,
) -> None:
    """Write a compact markdown coverage and missingness report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = len(feature_df)
    coverage_lines = []
    for window in (7, 30, 63):
        flag = f"has_news_{window}d"
        covered = int(pd.to_numeric(feature_df.get(flag, pd.Series(dtype="float64")), errors="coerce").fillna(0).sum())
        pct = (covered / rows * 100.0) if rows else np.nan
        coverage_lines.append(f"- {window}d news coverage: {covered:,} / {rows:,} ({pct:.2f}%)")

    sector_lines = []
    if sector_col in feature_df.columns and rows:
        sector_summary = (
            feature_df.groupby(sector_col, dropna=False)
            .agg(
                event_rows=(sector_col, "size"),
                has_30d_news=("has_news_30d", "mean"),
                avg_news_count_30d=("news_count_30d", "mean"),
                avg_news_count_63d=("news_count_63d", "mean"),
            )
            .reset_index()
        )
        for _, row in sector_summary.iterrows():
            sector_lines.append(
                f"- {row[sector_col]}: rows={int(row['event_rows'])}, "
                f"30d_coverage={float(row['has_30d_news']) * 100.0:.2f}%, "
                f"avg_30d_count={float(row['avg_news_count_30d']):.2f}, "
                f"avg_63d_count={float(row['avg_news_count_63d']):.2f}"
            )

    feature_missingness = feature_df[[c for c in SENTIMENT_FEATURE_COLUMNS if c in feature_df.columns]].isna().mean()
    missingness_lines = [f"- {feature}: {pct * 100.0:.2f}%" for feature, pct in feature_missingness.items()]
    news_date_range = "not available from feature frame"
    if news_scores_df is not None and not news_scores_df.empty:
        actual_date_col = date_col if date_col in news_scores_df.columns else None
        if actual_date_col is None:
            for candidate in ("trading_date", "news_date", "published_at", "date"):
                if candidate in news_scores_df.columns:
                    actual_date_col = candidate
                    break
        if actual_date_col is not None:
            news_dates = pd.to_datetime(news_scores_df[actual_date_col], errors="coerce")
            if news_dates.notna().any():
                news_date_range = f"{news_dates.min().date()} to {news_dates.max().date()}"
    coverage_30d = pd.to_numeric(feature_df.get("has_news_30d", pd.Series(dtype="float64")), errors="coerce")
    sparse_warning = ""
    if rows and coverage_30d.fillna(0).mean() < sparse_coverage_threshold:
        sparse_warning = "\n\nWarning: 30-day news coverage is sparse; treat sentiment lift estimates as provisional."

    markdown = "\n".join(
        [
            "# Sector Sentiment Feature Diagnostics",
            "",
            f"- Event rows: {rows:,}",
            f"- Date range of news used: {news_date_range}",
            *coverage_lines,
            "",
            "## Coverage By Sector",
            *(sector_lines or ["- No sector coverage available."]),
            "",
            "## Feature Missingness",
            *(missingness_lines or ["- No sentiment feature columns found."]),
            "",
            "## Interpretation Note",
            (
                "Cristescu et al. (2025) show that sector-specific fine-tuning improves FinBERT sentiment "
                "classification quality, but they also find that sentiment is reactive rather than predictive "
                "of next-day returns. Therefore, this project treats sector-aware sentiment as a "
                "measurement-quality improvement and tests incremental lift cautiously."
            ),
            sparse_warning,
            "",
        ]
    )
    output_path.write_text(markdown, encoding="utf-8")
