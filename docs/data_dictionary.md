# Data Dictionary

## Overview
This file documents all raw, intermediate, and modeling variables used in the project.

---

## Dataset: prices_raw.csv
### Source
Yahoo Finance via yfinance

### Grain
One row per stock per trading day

| Column Name | Type | Description | Example | Notes |
|---|---|---|---|---|
| ticker | string | Stock ticker symbol | AAPL | Identifier |
| date | date | Trading date | 2024-03-15 | Primary time field |
| open | float | Opening price | 172.44 | Raw market field |
| high | float | Daily high price | 174.10 | Raw market field |
| low | float | Daily low price | 171.88 | Raw market field |
| close | float | Closing price | 173.55 | Raw market field |
| adj_close | float | Adjusted closing price | 173.20 | Use for returns |
| volume | int | Trading volume | 53482900 | Raw market field |

---

## Dataset: fundamentals_raw.csv
### Source
SEC EDGAR / edgartools

### Grain
One row per stock per filing period

| Column Name | Type | Description | Example | Notes |
|---|---|---|---|---|
| ticker | string | Stock ticker symbol | MSFT | Identifier |
| filing_date | date | Filing release date | 2024-02-01 | Use carefully for alignment |
| period_end | date | Fiscal period end date | 2023-12-31 | Not same as filing date |
| total_assets | float | Total assets | 352583000000 | Raw statement field |
| total_liabilities | float | Total liabilities | 205753000000 | Raw statement field |
| current_assets | float | Current assets | ... | Raw statement field |
| current_liabilities | float | Current liabilities | ... | Raw statement field |
| net_income | float | Net income | ... | Raw statement field |
| revenue | float | Revenue | ... | Raw statement field |
| operating_income | float | Operating income | ... | Raw statement field |

---

## Dataset: features_layer1.csv
### Source
Engineered from fundamentals_raw.csv

### Grain
One row per stock-date after aligning fundamentals to daily data

| Column Name | Type | Description | Formula / Rule | Notes |
|---|---|---|---|---|
| current_ratio | float | Liquidity ratio | current_assets / current_liabilities | Layer 1 feature |
| debt_to_equity | float | Solvency ratio | total_liabilities / shareholder_equity | Layer 1 feature |
| net_margin | float | Profitability ratio | net_income / revenue | Layer 1 feature |
| roa | float | Return on assets | net_income / total_assets | Layer 1 feature |
| roe | float | Return on equity | net_income / shareholder_equity | Layer 1 feature |
| revenue_growth_yoy | float | Revenue growth | (rev_t - rev_t-4q) / rev_t-4q | Layer 1 feature |
| liquidity_profile_score | float | Composite short-term health score | Average of fixed-threshold liquidity indicators | Layer 1 feature |
| solvency_profile_score | float | Composite long-term health score | Average of fixed-threshold leverage indicators | Layer 1 feature |
| profitability_profile_score | float | Composite profitability score | Average of fixed-threshold margin and return indicators | Layer 1 feature |
| growth_quality_profile_score | float | Composite growth quality score | Average of growth and cash-quality indicators | Layer 1 feature |
| overall_financial_health_score | float | Overall financial health score | Average of the four profile scores | Layer 1 feature |

---

## Dataset: features_layer2.csv
### Source
Engineered from price data

| Column Name | Type | Description | Formula / Rule | Notes |
|---|---|---|---|---|
| return_5d | float | 5-day backward return | close_t / close_t-5 - 1 | Feature |
| return_21d | float | 21-day backward return | close_t / close_t-21 - 1 | Feature |
| vol_21d | float | 21-day rolling volatility | std(daily returns, 21d) | Feature |
| volume_ratio_20d | float | Relative volume | volume / avg(volume, 20d) | Feature |

---

## Dataset: features_layer3.csv
### Source
Financial news / SEC text sentiment

| Column Name | Type | Description | Formula / Rule | Notes |
|---|---|---|---|---|
| sentiment_score | float | Net sentiment | prob_pos - prob_neg | Layer 3 feature |
| sentiment_momentum_5d | float | Change in sentiment | rolling mean delta | Feature |
| news_volume | int | Number of articles in window | count of articles | Feature |
| mda_sentiment_delta | float | QoQ change in MD&A sentiment | current - prior | Feature |

---

## Final Modeling Table
### Dataset
modeling_table.csv

### Grain
One row per stock-date

| Column Name | Type | Role | Description |
|---|---|---|---|
| ticker | string | ID | Stock symbol |
| date | date | ID | Observation date |
| [feature columns] | numeric | Feature | Model input |
| forward_return_5d | float | Derived target helper | Forward return over next 5 trading days |
| label | int | Target | 1 if forward_return_5d > 0 else 0 |

---

## Notes on Alignment
- Fundamentals must only be used after they become publicly available
- No future information should leak into features
- Daily rows are aligned to trading dates
- Forward return is the target, not a feature
