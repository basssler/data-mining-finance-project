"""Engineer Layer 1 financial-statement-only features from clean fundamentals.

This script reads the cleaned quarterly fundamentals table, creates
accounting-based ratios and growth features, and saves the result as an
interim feature dataset for the Layer 1 baseline model.

Input:
    data/interim/fundamentals/fundamentals_quarterly_clean.parquet

Output:
    data/interim/features/layer1_financial_features.parquet

Notes:
    - This script only creates features that can be computed from financial
      statements alone.
    - It does not create market-based or valuation ratios.
    - Unsafe divisions return NaN instead of raising errors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import INTERIM_DATA_DIR

REQUIRED_METADATA_COLUMNS = [
    "ticker",
    "cik",
    "filing_date",
    "period_end",
    "fiscal_period",
    "fiscal_year",
    "form_type",
]

FEATURE_COLUMNS = [
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "working_capital_to_total_assets",
    "debt_to_equity",
    "debt_to_assets",
    "long_term_debt_ratio",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roa",
    "roe",
    "asset_turnover",
    "inventory_turnover",
    "receivables_turnover",
    "revenue_growth_qoq",
    "revenue_growth_yoy",
    "earnings_growth_qoq",
    "earnings_growth_yoy",
    "cfo_to_net_income",
    "accruals_ratio",
]

FEATURE_CATEGORY_NOTES: Dict[str, str] = {
    "Liquidity": (
        "Liquidity ratios measure short-term financial flexibility and the "
        "ability to cover near-term obligations."
    ),
    "Leverage / Solvency": (
        "Leverage ratios describe how much the company relies on debt or "
        "liabilities relative to its asset base or equity."
    ),
    "Profitability": (
        "Profitability ratios show how efficiently revenue is converted into "
        "operating profit, net income, and returns on capital."
    ),
    "Efficiency": (
        "Efficiency ratios capture how effectively the firm uses assets, "
        "inventory, and receivables to support revenue."
    ),
    "Growth": (
        "Growth features track quarter-over-quarter and year-over-year changes "
        "in revenue and earnings."
    ),
    "Cash Flow Quality": (
        "Cash flow quality metrics compare accounting earnings with operating "
        "cash generation to flag earnings backed by cash versus accruals."
    ),
}


def get_input_path() -> Path:
    """Return the cleaned fundamentals parquet path."""
    return INTERIM_DATA_DIR / "fundamentals" / "fundamentals_quarterly_clean.parquet"


def get_output_path() -> Path:
    """Return the feature parquet path and create its folder."""
    output_dir = INTERIM_DATA_DIR / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer1_financial_features.parquet"


def load_clean_fundamentals(input_path: Path) -> pd.DataFrame:
    """Load the clean fundamentals table and validate required metadata columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Clean fundamentals file was not found: {input_path}")

    df = pd.read_parquet(input_path)

    missing_columns = [column for column in REQUIRED_METADATA_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Clean fundamentals file is missing required metadata columns: "
            + ", ".join(missing_columns)
        )

    return df.copy()


def normalize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and sort rows before lag-based calculations."""
    cleaned = df.copy()

    cleaned["ticker"] = cleaned["ticker"].astype("string")
    cleaned["cik"] = cleaned["cik"].astype("string")
    cleaned["fiscal_period"] = cleaned["fiscal_period"].astype("string")
    cleaned["form_type"] = cleaned["form_type"].astype("string")
    cleaned["filing_date"] = pd.to_datetime(cleaned["filing_date"], errors="coerce")
    cleaned["period_end"] = pd.to_datetime(cleaned["period_end"], errors="coerce")
    cleaned["fiscal_year"] = pd.to_numeric(cleaned["fiscal_year"], errors="coerce").astype("Int64")

    for column in cleaned.columns:
        if column in REQUIRED_METADATA_COLUMNS:
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.sort_values(by=["ticker", "period_end"]).reset_index(drop=True)
    return cleaned


def get_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Return a numeric series or a NaN-filled fallback if the column is absent."""
    if column_name in df.columns:
        return pd.to_numeric(df[column_name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series safely and return NaN when the division is unsafe."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    valid_denominator = denominator.notna() & np.isfinite(denominator) & (denominator != 0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid_denominator] = numerator.loc[valid_denominator] / denominator.loc[valid_denominator]
    return result


def compute_average_balance(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Average the current and prior-quarter balance within each ticker."""
    current_values = get_series(df, column_name)
    prior_values = current_values.groupby(df["ticker"]).shift(1)
    return (current_values + prior_values) / 2.0


def compute_growth(series: pd.Series, ticker_series: pd.Series, periods: int) -> pd.Series:
    """Compute percentage growth using ticker-specific lagged values."""
    prior_values = series.groupby(ticker_series).shift(periods)
    return safe_divide(series - prior_values, prior_values)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Layer 1 financial statement features."""
    features_df = df.copy()

    revenue = get_series(features_df, "revenue")
    net_income = get_series(features_df, "net_income")
    total_assets = get_series(features_df, "total_assets")
    total_liabilities = get_series(features_df, "total_liabilities")
    current_assets = get_series(features_df, "current_assets")
    current_liabilities = get_series(features_df, "current_liabilities")
    cash_and_cash_equivalents = get_series(features_df, "cash_and_cash_equivalents")
    shareholders_equity = get_series(features_df, "shareholders_equity")
    operating_income = get_series(features_df, "operating_income")
    operating_cash_flow = get_series(features_df, "operating_cash_flow")
    long_term_debt = get_series(features_df, "long_term_debt")
    inventory = get_series(features_df, "inventory")
    accounts_receivable = get_series(features_df, "accounts_receivable")

    average_total_assets = compute_average_balance(features_df, "total_assets")
    average_shareholders_equity = compute_average_balance(features_df, "shareholders_equity")
    average_inventory = compute_average_balance(features_df, "inventory")
    average_accounts_receivable = compute_average_balance(features_df, "accounts_receivable")

    # Liquidity ratios
    # current_ratio = current_assets / current_liabilities
    features_df["current_ratio"] = safe_divide(current_assets, current_liabilities)

    # quick_ratio = (current_assets - inventory) / current_liabilities
    features_df["quick_ratio"] = safe_divide(current_assets - inventory, current_liabilities)

    # cash_ratio = cash_and_cash_equivalents / current_liabilities
    features_df["cash_ratio"] = safe_divide(cash_and_cash_equivalents, current_liabilities)

    # working_capital_to_total_assets = (current_assets - current_liabilities) / total_assets
    features_df["working_capital_to_total_assets"] = safe_divide(
        current_assets - current_liabilities,
        total_assets,
    )

    # Leverage / solvency ratios
    # debt_to_equity = total_liabilities / shareholders_equity
    features_df["debt_to_equity"] = safe_divide(total_liabilities, shareholders_equity)

    # debt_to_assets = total_liabilities / total_assets
    features_df["debt_to_assets"] = safe_divide(total_liabilities, total_assets)

    # long_term_debt_ratio = long_term_debt / total_assets
    features_df["long_term_debt_ratio"] = safe_divide(long_term_debt, total_assets)

    # Profitability ratios
    # gross_margin stays NaN unless a usable gross profit column already exists.
    gross_profit = get_series(features_df, "gross_profit")
    features_df["gross_margin"] = safe_divide(gross_profit, revenue)

    # operating_margin = operating_income / revenue
    features_df["operating_margin"] = safe_divide(operating_income, revenue)

    # net_margin = net_income / revenue
    features_df["net_margin"] = safe_divide(net_income, revenue)

    # roa = net_income / average_total_assets
    features_df["roa"] = safe_divide(net_income, average_total_assets)

    # roe = net_income / average_shareholders_equity
    features_df["roe"] = safe_divide(net_income, average_shareholders_equity)

    # Efficiency ratios
    # asset_turnover = revenue / average_total_assets
    features_df["asset_turnover"] = safe_divide(revenue, average_total_assets)

    # inventory_turnover = revenue / average_inventory
    features_df["inventory_turnover"] = safe_divide(revenue, average_inventory)

    # receivables_turnover = revenue / average_accounts_receivable
    features_df["receivables_turnover"] = safe_divide(revenue, average_accounts_receivable)

    # Growth features
    # revenue_growth_qoq = (revenue_t - revenue_t-1) / revenue_t-1
    features_df["revenue_growth_qoq"] = compute_growth(revenue, features_df["ticker"], periods=1)

    # revenue_growth_yoy = (revenue_t - revenue_t-4) / revenue_t-4
    features_df["revenue_growth_yoy"] = compute_growth(revenue, features_df["ticker"], periods=4)

    # earnings_growth_qoq = (net_income_t - net_income_t-1) / net_income_t-1
    features_df["earnings_growth_qoq"] = compute_growth(net_income, features_df["ticker"], periods=1)

    # earnings_growth_yoy = (net_income_t - net_income_t-4) / net_income_t-4
    features_df["earnings_growth_yoy"] = compute_growth(net_income, features_df["ticker"], periods=4)

    # Cash flow quality
    # cfo_to_net_income = operating_cash_flow / net_income
    features_df["cfo_to_net_income"] = safe_divide(operating_cash_flow, net_income)

    # accruals_ratio = (net_income - operating_cash_flow) / average_total_assets
    features_df["accruals_ratio"] = safe_divide(
        net_income - operating_cash_flow,
        average_total_assets,
    )

    return features_df


def calculate_missing_percentages(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Return missing-value percentages for selected columns."""
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return pd.Series(dtype="float64")

    return df[available_columns].isna().mean().mul(100).sort_index()


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print a short feature engineering summary to the console."""
    print("\nFeature Engineering Summary")
    print("-" * 60)
    print(f"Number of rows: {len(df):,}")
    print(f"Number of tickers: {df['ticker'].nunique():,}")
    print(f"Number of engineered feature columns: {len(FEATURE_COLUMNS):,}")

    missing_percentages = calculate_missing_percentages(df, FEATURE_COLUMNS)

    print("\nMissingness for engineered features")
    print("-" * 60)
    for column_name, percentage in missing_percentages.items():
        print(f"{column_name:<30} {percentage:>8.2f}%")


def print_feature_category_notes() -> None:
    """Print plain-English descriptions of the feature groups."""
    print("\nFeature Categories")
    print("-" * 60)
    for category_name, description in FEATURE_CATEGORY_NOTES.items():
        print(f"{category_name}: {description}")

    print("\nSparse Feature Note")
    print("-" * 60)
    print(
        "gross_margin, inventory_turnover, and receivables_turnover may be sparse "
        "because they depend on concepts that are not always available or cleanly "
        "reported in EDGAR-derived fundamentals."
    )


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save the engineered feature table to parquet."""
    df.to_parquet(output_path, index=False)


def main() -> None:
    """Run the Layer 1 feature engineering pipeline."""
    input_path = get_input_path()
    output_path = get_output_path()

    print(f"Loading clean fundamentals from: {input_path}")
    clean_df = load_clean_fundamentals(input_path)

    print("Normalizing input data...")
    normalized_df = normalize_input_data(clean_df)

    print("Engineering Layer 1 financial features...")
    feature_df = engineer_features(normalized_df)

    print(f"Saving feature dataset to: {output_path}")
    save_features(feature_df, output_path)

    print_feature_summary(feature_df)
    print_feature_category_notes()

    print(f"\nSaved {len(feature_df):,} rows with Layer 1 features.")


if __name__ == "__main__":
    main()
