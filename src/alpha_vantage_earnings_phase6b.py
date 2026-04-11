"""Alpha Vantage earnings-family ingest and Phase 6B feature engineering."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from src.paths import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.universe import get_layer1_tickers

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_ENV_VAR = "ALPHAVANTAGE_API_KEYS"
ALPHA_VANTAGE_SOURCE_NAME = "alpha_vantage"
ALPHA_VANTAGE_END_POINTS = ("EARNINGS_ESTIMATES", "EARNINGS")
ALPHA_VANTAGE_TIMEZONE = "America/New_York"

RAW_CACHE_DIR = RAW_DATA_DIR / "alpha_vantage"
MANIFEST_PATH = INTERIM_DATA_DIR / "alpha_vantage" / "alpha_vantage_manifest.json"
NORMALIZED_ESTIMATES_PATH = (
    INTERIM_DATA_DIR / "alpha_vantage" / "alpha_vantage_earnings_estimates.parquet"
)
NORMALIZED_EARNINGS_PATH = INTERIM_DATA_DIR / "alpha_vantage" / "alpha_vantage_earnings.parquet"
FEATURE_BLOCK_PATH = (
    INTERIM_DATA_DIR / "features" / "alpha_vantage_earnings_features_phase6b.parquet"
)
MERGED_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_phase6b_alpha_vantage.parquet"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs() -> None:
    for path in [
        MANIFEST_PATH,
        NORMALIZED_ESTIMATES_PATH,
        NORMALIZED_EARNINGS_PATH,
        FEATURE_BLOCK_PATH,
        MERGED_PANEL_PATH,
    ]:
        ensure_parent_dir(path)
    for endpoint in ALPHA_VANTAGE_END_POINTS:
        (RAW_CACHE_DIR / endpoint.lower()).mkdir(parents=True, exist_ok=True)


def provider_symbol_for_ticker(ticker: str) -> str:
    overrides = {
        "BF-B": "BF.B",
    }
    return overrides.get(str(ticker).upper(), str(ticker).upper())


def parse_api_keys(
    env_var: str = ALPHA_VANTAGE_ENV_VAR,
    explicit_keys: list[str] | None = None,
) -> list[str]:
    raw_values: list[str] = []
    if explicit_keys:
        raw_values.extend(explicit_keys)
    env_value = os.environ.get(env_var, "")
    if env_value:
        raw_values.extend(env_value.split(","))
    cleaned = [value.strip() for value in raw_values if value and value.strip()]
    deduped = list(dict.fromkeys(cleaned))
    if not deduped:
        raise ValueError(
            f"No Alpha Vantage API keys were found. Set {env_var} or pass explicit keys."
        )
    return deduped


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}****{key[-4:]}"


@dataclass
class KeyState:
    key: str
    cooldown_until: datetime | None = None
    last_used_at: datetime | None = None
    failed: bool = False
    total_requests: int = 0
    rate_limit_hits: int = 0
    failures: int = 0

    @property
    def masked(self) -> str:
        return mask_key(self.key)

    def available(self, now: datetime) -> bool:
        if self.failed:
            return False
        if self.cooldown_until is None:
            return True
        return now >= self.cooldown_until


@dataclass
class AlphaVantageKeyManager:
    keys: list[str]
    cooldown_seconds: int = 75
    minimum_reuse_seconds: int = 18
    states: list[KeyState] = field(init=False)
    next_index: int = 0

    def __post_init__(self) -> None:
        self.states = [KeyState(key=key) for key in self.keys]

    def get_next_key(self) -> str | None:
        now = datetime.now(UTC)
        for offset in range(len(self.states)):
            index = (self.next_index + offset) % len(self.states)
            state = self.states[index]
            if state.available(now) and (
                state.last_used_at is None
                or (now - state.last_used_at).total_seconds() >= self.minimum_reuse_seconds
            ):
                self.next_index = (index + 1) % len(self.states)
                state.total_requests += 1
                state.last_used_at = now
                return state.key
        return None

    def next_available_wait_seconds(self) -> int:
        now = datetime.now(UTC)
        waits = []
        for state in self.states:
            if state.failed:
                continue
            cooldown_wait = 0
            if state.cooldown_until is not None:
                cooldown_wait = max(0, math.ceil((state.cooldown_until - now).total_seconds()))
            reuse_wait = 0
            if state.last_used_at is not None:
                reuse_wait = max(
                    0,
                    math.ceil(
                        self.minimum_reuse_seconds - (now - state.last_used_at).total_seconds()
                    ),
                )
            waits.append(max(cooldown_wait, reuse_wait))
        return max(0, min(waits)) if waits else 0

    def mark_key_rate_limited(self, key: str, cooldown_seconds: int | None = None) -> None:
        for state in self.states:
            if state.key == key:
                state.rate_limit_hits += 1
                state.cooldown_until = datetime.now(UTC) + timedelta(
                    seconds=int(cooldown_seconds or self.cooldown_seconds)
                )
                return

    def mark_key_failed(self, key: str) -> None:
        for state in self.states:
            if state.key == key:
                state.failures += 1
                state.failed = True
                return

    def masked_key_summary(self) -> list[dict[str, Any]]:
        rows = []
        for state in self.states:
            rows.append(
                {
                    "key": state.masked,
                    "failed": state.failed,
                    "total_requests": state.total_requests,
                    "rate_limit_hits": state.rate_limit_hits,
                    "failures": state.failures,
                    "cooldown_until": state.cooldown_until.isoformat()
                    if state.cooldown_until is not None
                    else None,
                }
            )
        return rows


class AlphaVantageExhaustedError(RuntimeError):
    """Raised when all API keys are temporarily exhausted."""


class AlphaVantageResponseError(RuntimeError):
    """Raised when the API returns a hard failure."""


def is_rate_limit_payload(payload: dict[str, Any]) -> bool:
    message = " ".join(
        str(payload.get(key, "")) for key in ("Information", "Note", "information", "note")
    ).lower()
    return "thank you for using alpha vantage" in message or "rate limit" in message


def extract_error_message(payload: dict[str, Any]) -> str | None:
    for key in ("Error Message", "error", "Error", "Information", "Note", "message"):
        if key in payload and payload[key]:
            return str(payload[key])
    return None


@dataclass
class AlphaVantageClient:
    key_manager: AlphaVantageKeyManager
    request_timeout_seconds: int = 30
    min_request_spacing_seconds: float = 1.2
    max_attempts_per_request: int = 6
    last_request_started_at: float | None = None

    def _throttle(self) -> None:
        if self.last_request_started_at is None:
            return
        elapsed = time.monotonic() - self.last_request_started_at
        remaining = self.min_request_spacing_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def fetch_json(self, endpoint: str, symbol: str) -> tuple[dict[str, Any], str]:
        last_error: str | None = None
        for _ in range(self.max_attempts_per_request):
            key = self.key_manager.get_next_key()
            if key is None:
                wait_seconds = self.key_manager.next_available_wait_seconds()
                raise AlphaVantageExhaustedError(
                    f"All Alpha Vantage keys are cooling down. Retry after ~{wait_seconds} seconds."
                )
            self._throttle()
            params = urlencode({"function": endpoint, "symbol": symbol, "apikey": key})
            url = f"{ALPHA_VANTAGE_BASE_URL}?{params}"
            self.last_request_started_at = time.monotonic()
            try:
                with urllib.request.urlopen(url, timeout=self.request_timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.reason}"
                if exc.code in {429, 500, 502, 503, 504}:
                    self.key_manager.mark_key_rate_limited(key)
                    continue
                self.key_manager.mark_key_failed(key)
                raise AlphaVantageResponseError(last_error) from exc
            except urllib.error.URLError as exc:
                last_error = str(exc.reason)
                self.key_manager.mark_key_rate_limited(key, cooldown_seconds=45)
                continue

            if not isinstance(payload, dict):
                self.key_manager.mark_key_failed(key)
                raise AlphaVantageResponseError("Alpha Vantage returned a non-dict payload.")

            if is_rate_limit_payload(payload):
                self.key_manager.mark_key_rate_limited(key)
                last_error = extract_error_message(payload) or "Rate limit payload"
                continue

            error_message = extract_error_message(payload)
            if error_message and "symbol" not in payload:
                self.key_manager.mark_key_failed(key)
                raise AlphaVantageResponseError(error_message)

            return payload, key

        raise AlphaVantageExhaustedError(last_error or "Alpha Vantage request attempts were exhausted.")


def raw_cache_path(endpoint: str, ticker: str) -> Path:
    return RAW_CACHE_DIR / endpoint.lower() / f"{ticker.upper()}.json"


def build_manifest(tickers: list[str], endpoints: tuple[str, ...] = ALPHA_VANTAGE_END_POINTS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        provider_symbol = provider_symbol_for_ticker(ticker)
        for endpoint in endpoints:
            cache_path = raw_cache_path(endpoint, ticker)
            rows.append(
                {
                    "ticker": str(ticker).upper(),
                    "provider_symbol": provider_symbol,
                    "endpoint": endpoint,
                    "status": "pending",
                    "fetched_at": None,
                    "cache_path": str(cache_path),
                    "error_message": None,
                }
            )
    return rows


def load_manifest(path: Path = MANIFEST_PATH, tickers: list[str] | None = None) -> pd.DataFrame:
    if path.exists():
        manifest_rows = json.loads(path.read_text(encoding="utf-8"))
        manifest_df = pd.DataFrame(manifest_rows)
    else:
        manifest_df = pd.DataFrame(build_manifest(tickers or get_layer1_tickers()))
    if manifest_df.empty:
        raise ValueError("Alpha Vantage manifest is empty.")
    manifest_df["ticker"] = manifest_df["ticker"].astype("string")
    manifest_df["provider_symbol"] = manifest_df["provider_symbol"].astype("string")
    manifest_df["endpoint"] = manifest_df["endpoint"].astype("string")
    manifest_df["status"] = manifest_df["status"].fillna("pending").astype("string")
    manifest_df["cache_path"] = manifest_df["cache_path"].astype("string")
    return manifest_df.sort_values(["ticker", "endpoint"]).reset_index(drop=True)


def save_manifest(manifest_df: pd.DataFrame, path: Path = MANIFEST_PATH) -> None:
    ensure_parent_dir(path)
    payload = manifest_df.to_dict(orient="records")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_manifest_row(
    manifest_df: pd.DataFrame,
    ticker: str,
    endpoint: str,
    *,
    status: str,
    fetched_at: str | None = None,
    error_message: str | None = None,
) -> pd.DataFrame:
    mask = (manifest_df["ticker"] == ticker) & (manifest_df["endpoint"] == endpoint)
    if not bool(mask.any()):
        raise KeyError(f"Manifest row not found for {ticker} / {endpoint}")
    manifest_df.loc[mask, "status"] = status
    if fetched_at is not None:
        manifest_df.loc[mask, "fetched_at"] = fetched_at
    if error_message is not None or status != "complete":
        manifest_df.loc[mask, "error_message"] = error_message
    if status == "complete":
        manifest_df.loc[mask, "error_message"] = None
    return manifest_df


def fetch_raw_payloads(
    tickers: list[str],
    *,
    mode: str = "backfill",
    explicit_keys: list[str] | None = None,
    force_refresh: bool = False,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[pd.DataFrame, AlphaVantageKeyManager]:
    if mode not in {"backfill", "refresh"}:
        raise ValueError(f"Unsupported mode: {mode}")
    ensure_output_dirs()
    manifest_df = load_manifest(manifest_path, tickers=tickers)
    key_manager = AlphaVantageKeyManager(parse_api_keys(explicit_keys=explicit_keys))
    client = AlphaVantageClient(key_manager=key_manager)

    for row in manifest_df.sort_values(["ticker", "endpoint"]).itertuples(index=False):
        ticker = str(row.ticker)
        endpoint = str(row.endpoint)
        cache_path = Path(str(row.cache_path))
        cache_exists = cache_path.exists()
        should_fetch = force_refresh or mode == "refresh" or not cache_exists
        if not should_fetch:
            manifest_df = update_manifest_row(
                manifest_df,
                ticker,
                endpoint,
                status="complete",
                fetched_at=str(row.fetched_at) if row.fetched_at else None,
            )
            continue

        try:
            payload, _ = client.fetch_json(endpoint=endpoint, symbol=str(row.provider_symbol))
        except AlphaVantageExhaustedError:
            save_manifest(manifest_df, manifest_path)
            raise
        except Exception as exc:
            manifest_df = update_manifest_row(
                manifest_df,
                ticker,
                endpoint,
                status="failed",
                error_message=str(exc),
            )
            save_manifest(manifest_df, manifest_path)
            continue

        ensure_parent_dir(cache_path)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifest_df = update_manifest_row(
            manifest_df,
            ticker,
            endpoint,
            status="complete",
            fetched_at=datetime.now(UTC).isoformat(),
        )
        save_manifest(manifest_df, manifest_path)

    save_manifest(manifest_df, manifest_path)
    return manifest_df, key_manager


def _to_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_raw_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_estimates_from_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    estimates_manifest = manifest_df.loc[
        (manifest_df["endpoint"] == "EARNINGS_ESTIMATES") & (manifest_df["status"] == "complete")
    ].copy()
    for item in estimates_manifest.itertuples(index=False):
        cache_path = Path(str(item.cache_path))
        if not cache_path.exists():
            continue
        payload = _load_raw_json(cache_path)
        fetched_at = str(item.fetched_at) if item.fetched_at else None
        for estimate in payload.get("estimates", []):
            rows.append(
                {
                    "ticker": str(item.ticker),
                    "provider_symbol": str(item.provider_symbol),
                    "estimate_date": pd.to_datetime(estimate.get("date"), errors="coerce"),
                    "horizon": estimate.get("horizon"),
                    "eps_estimate_average": _to_float(estimate.get("eps_estimate_average")),
                    "eps_estimate_high": _to_float(estimate.get("eps_estimate_high")),
                    "eps_estimate_low": _to_float(estimate.get("eps_estimate_low")),
                    "eps_estimate_analyst_count": _to_float(
                        estimate.get("eps_estimate_analyst_count")
                    ),
                    "eps_estimate_average_7_days_ago": _to_float(
                        estimate.get("eps_estimate_average_7_days_ago")
                    ),
                    "eps_estimate_average_30_days_ago": _to_float(
                        estimate.get("eps_estimate_average_30_days_ago")
                    ),
                    "eps_estimate_average_60_days_ago": _to_float(
                        estimate.get("eps_estimate_average_60_days_ago")
                    ),
                    "eps_estimate_average_90_days_ago": _to_float(
                        estimate.get("eps_estimate_average_90_days_ago")
                    ),
                    "eps_estimate_revision_up_trailing_7_days": _to_float(
                        estimate.get("eps_estimate_revision_up_trailing_7_days")
                    ),
                    "eps_estimate_revision_down_trailing_7_days": _to_float(
                        estimate.get("eps_estimate_revision_down_trailing_7_days")
                    ),
                    "eps_estimate_revision_up_trailing_30_days": _to_float(
                        estimate.get("eps_estimate_revision_up_trailing_30_days")
                    ),
                    "eps_estimate_revision_down_trailing_30_days": _to_float(
                        estimate.get("eps_estimate_revision_down_trailing_30_days")
                    ),
                    "revenue_estimate_average": _to_float(estimate.get("revenue_estimate_average")),
                    "revenue_estimate_high": _to_float(estimate.get("revenue_estimate_high")),
                    "revenue_estimate_low": _to_float(estimate.get("revenue_estimate_low")),
                    "revenue_estimate_analyst_count": _to_float(
                        estimate.get("revenue_estimate_analyst_count")
                    ),
                    "source": ALPHA_VANTAGE_SOURCE_NAME,
                    "endpoint": "EARNINGS_ESTIMATES",
                    "fetched_at": pd.to_datetime(fetched_at, errors="coerce", utc=True),
                    "raw_cache_path": str(cache_path),
                }
            )
    estimates_df = pd.DataFrame(rows)
    if estimates_df.empty:
        return estimates_df
    estimates_df["ticker"] = estimates_df["ticker"].astype("string")
    estimates_df["provider_symbol"] = estimates_df["provider_symbol"].astype("string")
    estimates_df["horizon"] = estimates_df["horizon"].astype("string")
    return estimates_df.sort_values(["ticker", "estimate_date", "horizon"]).reset_index(drop=True)


def normalize_earnings_from_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    earnings_manifest = manifest_df.loc[
        (manifest_df["endpoint"] == "EARNINGS") & (manifest_df["status"] == "complete")
    ].copy()
    for item in earnings_manifest.itertuples(index=False):
        cache_path = Path(str(item.cache_path))
        if not cache_path.exists():
            continue
        payload = _load_raw_json(cache_path)
        fetched_at = str(item.fetched_at) if item.fetched_at else None
        for earning in payload.get("quarterlyEarnings", []):
            rows.append(
                {
                    "ticker": str(item.ticker),
                    "provider_symbol": str(item.provider_symbol),
                    "earnings_scope": "quarterly",
                    "fiscal_date_ending": pd.to_datetime(
                        earning.get("fiscalDateEnding"), errors="coerce"
                    ),
                    "reported_date": pd.to_datetime(earning.get("reportedDate"), errors="coerce"),
                    "reported_eps": _to_float(earning.get("reportedEPS")),
                    "estimated_eps": _to_float(earning.get("estimatedEPS")),
                    "surprise": _to_float(earning.get("surprise")),
                    "surprise_percentage": _to_float(earning.get("surprisePercentage")),
                    "report_time": str(earning.get("reportTime") or "").strip().lower() or None,
                    "source": ALPHA_VANTAGE_SOURCE_NAME,
                    "endpoint": "EARNINGS",
                    "fetched_at": pd.to_datetime(fetched_at, errors="coerce", utc=True),
                    "raw_cache_path": str(cache_path),
                }
            )
        for earning in payload.get("annualEarnings", []):
            rows.append(
                {
                    "ticker": str(item.ticker),
                    "provider_symbol": str(item.provider_symbol),
                    "earnings_scope": "annual",
                    "fiscal_date_ending": pd.to_datetime(
                        earning.get("fiscalDateEnding"), errors="coerce"
                    ),
                    "reported_date": pd.NaT,
                    "reported_eps": _to_float(earning.get("reportedEPS")),
                    "estimated_eps": None,
                    "surprise": None,
                    "surprise_percentage": None,
                    "report_time": None,
                    "source": ALPHA_VANTAGE_SOURCE_NAME,
                    "endpoint": "EARNINGS",
                    "fetched_at": pd.to_datetime(fetched_at, errors="coerce", utc=True),
                    "raw_cache_path": str(cache_path),
                }
            )
    earnings_df = pd.DataFrame(rows)
    if earnings_df.empty:
        return earnings_df
    earnings_df["ticker"] = earnings_df["ticker"].astype("string")
    earnings_df["provider_symbol"] = earnings_df["provider_symbol"].astype("string")
    earnings_df["earnings_scope"] = earnings_df["earnings_scope"].astype("string")
    earnings_df["report_time"] = earnings_df["report_time"].astype("string")
    return earnings_df.sort_values(["ticker", "earnings_scope", "fiscal_date_ending"]).reset_index(drop=True)


def save_normalized_tables(
    estimates_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
    *,
    estimates_path: Path = NORMALIZED_ESTIMATES_PATH,
    earnings_path: Path = NORMALIZED_EARNINGS_PATH,
) -> None:
    ensure_parent_dir(estimates_path)
    ensure_parent_dir(earnings_path)
    estimates_df.to_parquet(estimates_path, index=False)
    earnings_df.to_parquet(earnings_path, index=False)


def _localize_timestamp(
    date_value: pd.Timestamp | datetime | str | None,
    hour: int,
    minute: int,
) -> pd.Timestamp | pd.NaT:
    if date_value is None or pd.isna(date_value):
        return pd.NaT
    ts = pd.Timestamp(date_value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(ALPHA_VANTAGE_TIMEZONE)
        return ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return ts.tz_localize(ALPHA_VANTAGE_TIMEZONE).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )


def _next_trading_date(trading_dates: list[pd.Timestamp], base_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    if pd.isna(base_date):
        return pd.NaT
    index = np.searchsorted(np.array(trading_dates, dtype="datetime64[ns]"), base_date.to_datetime64(), side="left")
    if index >= len(trading_dates):
        return pd.NaT
    return trading_dates[index]


def _prev_trading_date(trading_dates: list[pd.Timestamp], base_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    if pd.isna(base_date):
        return pd.NaT
    index = np.searchsorted(np.array(trading_dates, dtype="datetime64[ns]"), base_date.to_datetime64(), side="left") - 1
    if index < 0:
        return pd.NaT
    return trading_dates[index]


def build_trading_calendar(prices_df: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    normalized = prices_df[["ticker", "date"]].drop_duplicates().copy()
    normalized["ticker"] = normalized["ticker"].astype("string")
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").astype("datetime64[ns]")
    trading_calendar: dict[str, list[pd.Timestamp]] = {}
    for ticker, group in normalized.groupby("ticker"):
        trading_calendar[str(ticker)] = list(group["date"].sort_values().to_list())
    return trading_calendar


def attach_earnings_availability(
    earnings_df: pd.DataFrame,
    trading_calendar: dict[str, list[pd.Timestamp]],
) -> pd.DataFrame:
    quarterly = earnings_df.loc[earnings_df["earnings_scope"] == "quarterly"].copy()
    if quarterly.empty:
        return quarterly

    availability_timestamps = []
    effective_dates = []
    timing_buckets = []
    for row in quarterly.itertuples(index=False):
        reported_date = pd.to_datetime(row.reported_date, errors="coerce")
        trading_dates = trading_calendar.get(str(row.ticker), [])
        report_time = str(row.report_time or "").strip().lower()

        if report_time in {"pre-market", "before open", "before-open"}:
            availability_ts = _localize_timestamp(reported_date, 8, 0)
            effective_date = _next_trading_date(trading_dates, reported_date)
            timing_bucket = "pre_market"
        elif report_time in {"post-market", "after close", "after-close"}:
            availability_ts = _localize_timestamp(reported_date, 16, 1)
            next_date = _next_trading_date(
                trading_dates,
                (reported_date + pd.Timedelta(days=1)) if pd.notna(reported_date) else reported_date,
            )
            effective_date = next_date
            timing_bucket = "after_close"
        elif report_time in {"during market", "during-market", "market-hours", "market hours"}:
            availability_ts = _localize_timestamp(reported_date, 12, 0)
            effective_date = _next_trading_date(trading_dates, reported_date)
            timing_bucket = "market_hours"
        else:
            next_date = _next_trading_date(
                trading_dates,
                (reported_date + pd.Timedelta(days=1)) if pd.notna(reported_date) else reported_date,
            )
            availability_ts = _localize_timestamp(next_date, 9, 30) if pd.notna(next_date) else pd.NaT
            effective_date = next_date
            timing_bucket = "missing_time_conservative"

        availability_timestamps.append(availability_ts)
        effective_dates.append(effective_date)
        timing_buckets.append(timing_bucket)

    quarterly["availability_timestamp"] = availability_timestamps
    quarterly["availability_effective_date"] = pd.to_datetime(effective_dates, errors="coerce")
    quarterly["availability_timing_bucket"] = timing_buckets
    quarterly["eps_surprise_beat"] = np.where(
        quarterly["surprise"].notna(),
        (quarterly["surprise"] > 0).astype(float),
        np.nan,
    )
    quarterly = quarterly.sort_values(["ticker", "availability_timestamp", "fiscal_date_ending"]).reset_index(
        drop=True
    )
    return quarterly


def build_safe_estimate_history(
    estimates_df: pd.DataFrame,
    quarterly_earnings_df: pd.DataFrame,
) -> pd.DataFrame:
    if estimates_df.empty or quarterly_earnings_df.empty:
        return pd.DataFrame()

    quarterly_estimates = estimates_df.loc[estimates_df["horizon"] == "fiscal quarter"].copy()
    if quarterly_estimates.empty:
        return pd.DataFrame()

    safe_df = quarterly_estimates.merge(
        quarterly_earnings_df[
            [
                "ticker",
                "fiscal_date_ending",
                "reported_date",
                "report_time",
                "availability_timestamp",
                "availability_effective_date",
            ]
        ].rename(columns={"fiscal_date_ending": "estimate_date"}),
        on=["ticker", "estimate_date"],
        how="inner",
        validate="many_to_one",
    )
    safe_df["av_eps_estimate_revision_30d"] = (
        safe_df["eps_estimate_average"] - safe_df["eps_estimate_average_30_days_ago"]
    )
    safe_df["av_eps_estimate_revision_90d"] = (
        safe_df["eps_estimate_average"] - safe_df["eps_estimate_average_90_days_ago"]
    )
    safe_df = safe_df.sort_values(["ticker", "availability_timestamp", "estimate_date"]).reset_index(drop=True)
    return safe_df


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=2).std(ddof=0)


def _build_event_cutoff_timestamps(
    panel_df: pd.DataFrame,
    trading_calendar: dict[str, list[pd.Timestamp]],
) -> pd.DataFrame:
    panel = panel_df.copy()
    panel["event_timestamp"] = pd.to_datetime(panel["event_timestamp"], errors="coerce", utc=False)
    panel["effective_model_date"] = pd.to_datetime(panel["effective_model_date"], errors="coerce").astype(
        "datetime64[ns]"
    )

    cutoff_values = []
    cutoff_modes = []
    for row in panel.itertuples(index=False):
        event_timestamp = row.event_timestamp
        if pd.notna(event_timestamp):
            if getattr(event_timestamp, "tzinfo", None) is None:
                event_timestamp = event_timestamp.tz_localize(ALPHA_VANTAGE_TIMEZONE)
            else:
                event_timestamp = event_timestamp.tz_convert(ALPHA_VANTAGE_TIMEZONE)
            cutoff_values.append(event_timestamp)
            cutoff_modes.append("event_timestamp")
            continue

        trading_dates = trading_calendar.get(str(row.ticker), [])
        previous_trading_date = _prev_trading_date(trading_dates, pd.Timestamp(row.effective_model_date))
        if pd.isna(previous_trading_date):
            cutoff_values.append(pd.NaT)
            cutoff_modes.append("missing_timestamp_no_prev_trading_day")
            continue
        cutoff_values.append(_localize_timestamp(previous_trading_date, 16, 0))
        cutoff_modes.append("missing_timestamp_prev_close")

    panel["phase6b_event_cutoff_timestamp"] = cutoff_values
    panel["phase6b_event_cutoff_mode"] = cutoff_modes
    return panel


def _build_quarterly_history_features(quarterly_earnings_df: pd.DataFrame) -> pd.DataFrame:
    history = quarterly_earnings_df.copy()
    if history.empty:
        for column in [
            "availability_timestamp",
            "fiscal_date_ending",
            "surprise",
            "surprise_percentage",
            "eps_surprise_beat",
            "av_trailing_4q_eps_surprise_mean",
            "av_trailing_4q_eps_surprise_std",
            "av_trailing_4q_eps_surprise_pct_mean",
            "av_trailing_4q_eps_surprise_pct_std",
            "av_trailing_4q_eps_beat_rate",
        ]:
            if column not in history.columns:
                history[column] = pd.Series(dtype="float64")
        return history
    history = history.sort_values(["ticker", "availability_timestamp", "fiscal_date_ending"]).reset_index(drop=True)
    grouped = history.groupby("ticker", group_keys=False)
    history["av_trailing_4q_eps_surprise_mean"] = grouped["surprise"].apply(
        lambda series: series.rolling(window=4, min_periods=1).mean()
    )
    history["av_trailing_4q_eps_surprise_std"] = grouped["surprise"].apply(
        lambda series: _rolling_std(series, window=4)
    )
    history["av_trailing_4q_eps_surprise_pct_mean"] = grouped["surprise_percentage"].apply(
        lambda series: series.rolling(window=4, min_periods=1).mean()
    )
    history["av_trailing_4q_eps_surprise_pct_std"] = grouped["surprise_percentage"].apply(
        lambda series: _rolling_std(series, window=4)
    )
    history["av_trailing_4q_eps_beat_rate"] = grouped["eps_surprise_beat"].apply(
        lambda series: series.rolling(window=4, min_periods=1).mean()
    )
    return history


def build_event_level_feature_block(
    panel_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    estimates_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trading_calendar = build_trading_calendar(prices_df)
    panel = _build_event_cutoff_timestamps(panel_df, trading_calendar)
    quarterly_earnings = attach_earnings_availability(earnings_df, trading_calendar)
    safe_estimates = build_safe_estimate_history(estimates_df, quarterly_earnings)
    quarterly_history = _build_quarterly_history_features(quarterly_earnings)

    if quarterly_history.empty:
        feature_block = panel[
            ["ticker", "event_date", "source_id", "phase6b_event_cutoff_timestamp", "phase6b_event_cutoff_mode"]
        ].copy()
        for column in [
            "av_last_earnings_reported_date",
            "av_last_earnings_fiscal_date",
            "av_last_earnings_availability_timestamp",
            "av_last_earnings_timing_bucket",
            "av_last_estimate_fiscal_date",
            "av_latest_prior_eps_surprise_before_event",
            "av_latest_prior_eps_surprise_pct_before_event",
            "av_trailing_4q_eps_surprise_mean",
            "av_trailing_4q_eps_surprise_std",
            "av_trailing_4q_eps_surprise_pct_mean",
            "av_trailing_4q_eps_surprise_pct_std",
            "av_trailing_4q_eps_beat_rate",
            "av_days_since_last_earnings_release",
            "av_latest_quarterly_eps_estimate_before_event",
            "av_latest_quarterly_revenue_estimate_before_event",
            "av_eps_estimate_analyst_count_before_event",
            "av_revenue_estimate_analyst_count_before_event",
            "av_eps_estimate_revision_30d",
            "av_eps_estimate_revision_90d",
        ]:
            feature_block[column] = np.nan
        feature_block["av_coverage_any"] = False
        diagnostics = {
            "panel_rows": int(len(panel_df)),
            "matched_rows": 0,
            "unmatched_rows": int(len(panel_df)),
            "coverage_by_ticker": feature_block.groupby("ticker")["av_coverage_any"].mean().mul(100).to_dict(),
            "coverage_by_year": (
                feature_block.assign(event_year=pd.to_datetime(feature_block["event_date"]).dt.year)
                .groupby("event_year")["av_coverage_any"]
                .mean()
                .mul(100)
                .to_dict()
            ),
            "feature_missingness_pct": {
                column: 100.0
                for column in [
                    "av_latest_prior_eps_surprise_before_event",
                    "av_latest_prior_eps_surprise_pct_before_event",
                    "av_trailing_4q_eps_surprise_mean",
                    "av_trailing_4q_eps_surprise_std",
                    "av_trailing_4q_eps_surprise_pct_mean",
                    "av_trailing_4q_eps_surprise_pct_std",
                    "av_trailing_4q_eps_beat_rate",
                    "av_days_since_last_earnings_release",
                    "av_latest_quarterly_eps_estimate_before_event",
                    "av_latest_quarterly_revenue_estimate_before_event",
                    "av_eps_estimate_analyst_count_before_event",
                    "av_revenue_estimate_analyst_count_before_event",
                    "av_eps_estimate_revision_30d",
                    "av_eps_estimate_revision_90d",
                ]
            },
            "estimates_feature_policy": (
                "Only quarterly estimate rows linked to an already-reported quarter were promoted into model features. "
                "Future untimestamped estimate snapshots were normalized but not used."
            ),
        }
        return feature_block.sort_values(["ticker", "event_date", "source_id"]).reset_index(drop=True), diagnostics

    earnings_feature_source = quarterly_history[
        [
            "ticker",
            "availability_timestamp",
            "reported_date",
            "fiscal_date_ending",
            "surprise",
            "surprise_percentage",
            "estimated_eps",
            "reported_eps",
            "availability_timing_bucket",
            "av_trailing_4q_eps_surprise_mean",
            "av_trailing_4q_eps_surprise_std",
            "av_trailing_4q_eps_surprise_pct_mean",
            "av_trailing_4q_eps_surprise_pct_std",
            "av_trailing_4q_eps_beat_rate",
        ]
    ].sort_values(["ticker", "availability_timestamp"])

    sortable_panel = panel.loc[panel["phase6b_event_cutoff_timestamp"].notna()].copy()
    unsortable_panel = panel.loc[panel["phase6b_event_cutoff_timestamp"].isna()].copy()
    sortable_panel = sortable_panel.sort_values(["ticker", "phase6b_event_cutoff_timestamp"]).reset_index(drop=True)

    merged = _merge_asof_per_ticker(
        sortable_panel,
        earnings_feature_source,
        left_on="phase6b_event_cutoff_timestamp",
        right_on="availability_timestamp",
    )

    if not safe_estimates.empty:
        estimate_feature_source = safe_estimates[
            [
                "ticker",
                "availability_timestamp",
                "estimate_date",
                "eps_estimate_average",
                "revenue_estimate_average",
                "eps_estimate_analyst_count",
                "revenue_estimate_analyst_count",
                "av_eps_estimate_revision_30d",
                "av_eps_estimate_revision_90d",
            ]
        ].sort_values(["ticker", "availability_timestamp"])
        merged = _merge_asof_per_ticker(
            merged.sort_values(["ticker", "phase6b_event_cutoff_timestamp"]).reset_index(drop=True),
            estimate_feature_source,
            left_on="phase6b_event_cutoff_timestamp",
            right_on="availability_timestamp",
            suffixes=("", "_estimate"),
        )
    else:
        merged["estimate_date"] = pd.NaT
        merged["eps_estimate_average"] = np.nan
        merged["revenue_estimate_average"] = np.nan
        merged["eps_estimate_analyst_count"] = np.nan
        merged["revenue_estimate_analyst_count"] = np.nan
        merged["av_eps_estimate_revision_30d"] = np.nan
        merged["av_eps_estimate_revision_90d"] = np.nan

    unsortable_panel = _append_missing_av_feature_columns(unsortable_panel)
    merged = pd.concat([merged, unsortable_panel], ignore_index=True, sort=False)

    merged["av_latest_prior_eps_surprise_before_event"] = merged["surprise"]
    merged["av_latest_prior_eps_surprise_pct_before_event"] = merged["surprise_percentage"]
    merged["av_days_since_last_earnings_release"] = (
        pd.to_datetime(merged["effective_model_date"], errors="coerce")
        - pd.to_datetime(merged["reported_date"], errors="coerce")
    ).dt.days
    merged["av_latest_quarterly_eps_estimate_before_event"] = merged["eps_estimate_average"]
    merged["av_latest_quarterly_revenue_estimate_before_event"] = merged["revenue_estimate_average"]
    merged["av_eps_estimate_analyst_count_before_event"] = merged["eps_estimate_analyst_count"]
    merged["av_revenue_estimate_analyst_count_before_event"] = merged[
        "revenue_estimate_analyst_count"
    ]

    feature_columns = [
        "ticker",
        "event_date",
        "source_id",
        "phase6b_event_cutoff_timestamp",
        "phase6b_event_cutoff_mode",
        "reported_date",
        "fiscal_date_ending",
        "availability_timestamp",
        "availability_timing_bucket",
        "estimate_date",
        "av_latest_prior_eps_surprise_before_event",
        "av_latest_prior_eps_surprise_pct_before_event",
        "av_trailing_4q_eps_surprise_mean",
        "av_trailing_4q_eps_surprise_std",
        "av_trailing_4q_eps_surprise_pct_mean",
        "av_trailing_4q_eps_surprise_pct_std",
        "av_trailing_4q_eps_beat_rate",
        "av_days_since_last_earnings_release",
        "av_latest_quarterly_eps_estimate_before_event",
        "av_latest_quarterly_revenue_estimate_before_event",
        "av_eps_estimate_analyst_count_before_event",
        "av_revenue_estimate_analyst_count_before_event",
        "av_eps_estimate_revision_30d",
        "av_eps_estimate_revision_90d",
    ]
    feature_block = merged[feature_columns].copy()
    feature_block = feature_block.rename(
        columns={
            "reported_date": "av_last_earnings_reported_date",
            "fiscal_date_ending": "av_last_earnings_fiscal_date",
            "availability_timestamp": "av_last_earnings_availability_timestamp",
            "availability_timing_bucket": "av_last_earnings_timing_bucket",
            "estimate_date": "av_last_estimate_fiscal_date",
        }
    )
    feature_block["av_coverage_any"] = feature_block[
        [
            "av_latest_prior_eps_surprise_before_event",
            "av_latest_quarterly_eps_estimate_before_event",
        ]
    ].notna().any(axis=1)

    matched = feature_block.loc[feature_block["av_last_earnings_availability_timestamp"].notna()].copy()
    if not matched.empty:
        comparison = matched["av_last_earnings_availability_timestamp"] < matched["phase6b_event_cutoff_timestamp"]
        if not bool(comparison.all()):
            bad_count = int((~comparison).sum())
            raise ValueError(
                f"Phase 6B leakage assertion failed: {bad_count} matched Alpha Vantage rows are not strictly before the event cutoff."
            )

    diagnostics = {
        "panel_rows": int(len(panel_df)),
        "matched_rows": int(feature_block["av_coverage_any"].sum()),
        "unmatched_rows": int((~feature_block["av_coverage_any"]).sum()),
        "coverage_by_ticker": (
            feature_block.groupby("ticker")["av_coverage_any"].mean().sort_index().mul(100).round(2).to_dict()
        ),
        "coverage_by_year": (
            feature_block.assign(event_year=pd.to_datetime(feature_block["event_date"]).dt.year)
            .groupby("event_year")["av_coverage_any"]
            .mean()
            .sort_index()
            .mul(100)
            .round(2)
            .to_dict()
        ),
        "feature_missingness_pct": (
            feature_block[
                [
                    "av_latest_prior_eps_surprise_before_event",
                    "av_latest_prior_eps_surprise_pct_before_event",
                    "av_trailing_4q_eps_surprise_mean",
                    "av_trailing_4q_eps_surprise_std",
                    "av_trailing_4q_eps_surprise_pct_mean",
                    "av_trailing_4q_eps_surprise_pct_std",
                    "av_trailing_4q_eps_beat_rate",
                    "av_days_since_last_earnings_release",
                    "av_latest_quarterly_eps_estimate_before_event",
                    "av_latest_quarterly_revenue_estimate_before_event",
                    "av_eps_estimate_analyst_count_before_event",
                    "av_revenue_estimate_analyst_count_before_event",
                    "av_eps_estimate_revision_30d",
                    "av_eps_estimate_revision_90d",
                ]
            ]
            .isna()
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
            .to_dict()
        ),
        "estimates_feature_policy": (
            "Only quarterly estimate rows linked to an already-reported quarter were promoted into model features. "
            "Future untimestamped estimate snapshots were normalized but not used."
        ),
    }
    return feature_block.sort_values(["ticker", "event_date", "source_id"]).reset_index(drop=True), diagnostics


def merge_feature_block_onto_panel(
    panel_df: pd.DataFrame,
    feature_block_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = panel_df.merge(
        feature_block_df,
        on=["ticker", "event_date", "source_id"],
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(["effective_model_date", "ticker", "source_id"]).reset_index(drop=True)


def _append_missing_av_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "reported_date",
        "fiscal_date_ending",
        "availability_timestamp",
        "availability_timing_bucket",
        "estimate_date",
        "surprise",
        "surprise_percentage",
        "eps_estimate_average",
        "revenue_estimate_average",
        "eps_estimate_analyst_count",
        "revenue_estimate_analyst_count",
        "av_eps_estimate_revision_30d",
        "av_eps_estimate_revision_90d",
        "av_trailing_4q_eps_surprise_mean",
        "av_trailing_4q_eps_surprise_std",
        "av_trailing_4q_eps_surprise_pct_mean",
        "av_trailing_4q_eps_surprise_pct_std",
        "av_trailing_4q_eps_beat_rate",
    ]:
        if column not in df.columns:
            df[column] = np.nan
    return df


def _merge_asof_per_ticker(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str = "ticker",
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    if left_df.empty:
        return left_df.copy()
    if right_df.empty:
        return left_df.copy()

    merged_groups = []
    right_grouped = {str(key): group.copy() for key, group in right_df.groupby(by)}
    for ticker, left_group in left_df.groupby(by, sort=False):
        left_group = left_group.sort_values(left_on).reset_index(drop=True)
        right_group = right_grouped.get(str(ticker))
        if right_group is None or right_group.empty:
            merged_groups.append(left_group)
            continue
        right_group = right_group.sort_values(right_on).reset_index(drop=True)
        merged_groups.append(
            pd.merge_asof(
                left=left_group,
                right=right_group,
                left_on=left_on,
                right_on=right_on,
                direction="backward",
                allow_exact_matches=False,
                suffixes=suffixes,
            )
        )
    return pd.concat(merged_groups, ignore_index=True, sort=False)
