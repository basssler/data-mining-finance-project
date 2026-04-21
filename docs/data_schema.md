# WRDS Stock Prediction Data Schema

## Canonical Row
One row in the modeling pipeline represents one `(permno, date)` stock-date observation using only information known on or before that date.

## Tables
`data/raw/wrds/wrds_compustat_fundamentals.parquet`
- Grain: one `gvkey` and fiscal quarter.
- Primary keys: `gvkey`, `datadate`.
- Important dates:
  - `datadate`: fiscal period end.
  - `rdq`: reported date from Compustat when available.
  - `availability_date`: pipeline-defined usable date, equal to `rdq` or `datadate + 60 days`.

`data/raw/wrds/wrds_crsp_daily.parquet`
- Grain: one `permno` and trading date.
- Primary keys: `permno`, `date`.
- Important fields: `ret`, `dlret`, `price`, `volume`, `shares_outstanding`, `market_cap`.

`data/raw/wrds/wrds_ccm_links.parquet`
- Grain: one CCM link record.
- Primary keys: `gvkey`, `permno`, `linkdt`, `linkenddt`.
- Join rule: valid only when the security date falls inside `[linkdt, linkenddt]`.

`data/interim/wrds/security_master.parquet`
- Grain: one valid `(permno, date)` mapping.
- Primary keys: `permno`, `date`.
- Join rule:
  - Start from CRSP daily rows.
  - Join to CCM on `permno`.
  - Keep only rows with `date` inside the CCM link validity window.
  - Resolve multiple links by retaining the preferred primary link and flagging ambiguity.

`data/interim/wrds/labeled_price_panel.parquet`
- Grain: one `(permno, date)` observation.
- Primary keys: `permno`, `date`.
- Required core columns:
  - `date`
  - `permno`
  - `gvkey`
  - `ticker`
  - `sic`
  - `gics_sector`
  - `gics_industry`
  - `market_cap`
  - `close`
  - `volume`
  - `ret_1d`
  - `fwd_ret_5d`
  - `fwd_ret_21d`
  - `label_up_5d`
  - `label_up_21d`

`data/interim/features/features_fundamental_daily.parquet`
- Grain: one `(permno, date)` observation.
- Join keys: `permno`, `date`, `gvkey`, `ticker`.
- Date rule: the joined quarter must satisfy `availability_date <= date`.

`data/interim/features/features_market.parquet`
- Grain: one `(permno, date)` observation.
- Join keys: `permno`, `date`, `ticker`.
- Date rule: all rolling windows are trailing-only.

`data/interim/features/features_peer_relative.parquet`
- Grain: one `(permno, date)` observation.
- Join keys: `permno`, `date`, `ticker`.
- Date rule: cross-sectional transforms are computed within each single date only.

`data/processed/stock_prediction/model_panel.parquet`
- Grain: one `(permno, date)` observation.
- Primary keys: `permno`, `date`.
- Merge order: labels -> fundamentals -> market -> peer-relative -> later sentiment/macro layers.

## Join Rules
- Never join on ticker alone.
- Canonical identifier bridge:
  - Compustat uses `gvkey`.
  - CRSP uses `permno`.
  - CCM provides the bridge between them.
- Preserve CCM date validity windows on every bridge from `gvkey` to `permno`.
- Prefer `permno` + `date` as the row-level key after the security master is built.

## Date Conventions
- CRSP market data uses trading date.
- Fundamentals become usable on `availability_date`, not fiscal period end.
- Forward labels use future prices only for the target, never for features.
- Peer-relative statistics use same-date cross-sections only.
