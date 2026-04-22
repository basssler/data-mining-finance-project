"""Helpers for enforcing the canonical event_panel_v2 schema contract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.paths import INTERIM_DATA_DIR

CANONICAL_BASE_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2.parquet"


def load_parquet_columns(path: Path) -> list[str]:
    """Load one parquet artifact and return its columns in storage order."""
    if not path.exists():
        raise FileNotFoundError(f"Panel contract file was not found: {path}")
    return list(pd.read_parquet(path).columns)


def load_canonical_base_columns(base_panel_path: Path = CANONICAL_BASE_PANEL_PATH) -> list[str]:
    """Return the canonical event_panel_v2 base columns from the source-of-truth artifact."""
    return load_parquet_columns(base_panel_path)


def assert_matches_canonical_base_contract(
    panel_df: pd.DataFrame,
    panel_name: str,
    *,
    base_panel_path: Path = CANONICAL_BASE_PANEL_PATH,
    allowed_extra_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Assert that a panel contains the canonical base columns plus only allowed additive prefixes."""
    canonical_columns = load_canonical_base_columns(base_panel_path)
    actual_columns = list(panel_df.columns)

    missing_columns = [column for column in canonical_columns if column not in actual_columns]
    unexpected_columns = [
        column
        for column in actual_columns
        if column not in canonical_columns
        and not any(column.startswith(prefix) for prefix in allowed_extra_prefixes)
    ]

    if missing_columns or unexpected_columns:
        details: list[str] = []
        if missing_columns:
            details.append("missing columns: " + ", ".join(missing_columns))
        if unexpected_columns:
            details.append("unexpected columns: " + ", ".join(unexpected_columns))
        raise ValueError(f"{panel_name} failed canonical base schema validation; " + "; ".join(details))

    return canonical_columns


def order_columns_with_canonical_base_first(
    panel_df: pd.DataFrame,
    *,
    canonical_columns: list[str] | None = None,
    base_panel_path: Path = CANONICAL_BASE_PANEL_PATH,
) -> pd.DataFrame:
    """Place canonical base columns first and preserve the current order of additive columns."""
    if canonical_columns is None:
        canonical_columns = load_canonical_base_columns(base_panel_path)

    ordered_columns = [column for column in canonical_columns if column in panel_df.columns]
    ordered_columns += [column for column in panel_df.columns if column not in ordered_columns]
    return panel_df[ordered_columns].copy()
