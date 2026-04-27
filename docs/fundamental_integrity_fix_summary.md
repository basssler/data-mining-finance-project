# Fundamental Integrity Fix Summary

## What changed

- Added explicit accounting concept metadata for flow, instant, and share-count concepts.
- Added a deterministic fact selector that ranks candidates by point-in-time eligibility, unit, period type, frame/consolidation signal, canonical tag priority, source priority, and filing recency.
- Added YTD-to-quarter derivation for 10-Q flow concepts when a directly reported quarterly fact is unavailable and compatible prior YTD facts exist.
- Preserved richer SEC/XBRL raw fields for future pulls: accession number, start date, end date, frame, and fact duration.
- Changed future raw fundamentals pulls to prefer SEC Company Facts JSON because it exposes the metadata needed for point-in-time validation.

## What was fixed

- Tiny segmented balance-sheet facts no longer outrank consolidated instant facts when SEC frame/duration metadata is available.
- Later filings no longer automatically win over better unit/frame/duration candidates.
- Facts filed after an event effective model date are excluded when that cutoff is supplied.
- Derived quarterly values are labeled with `derived_from_ytd_difference` and keep source/prior YTD trace fields.

## Current artifact status

- Existing Universe V2 raw fundamentals are metadata-incomplete: they lack frame, start date, end date, accession number, and fact duration fields.
- Existing clean fundamentals, Layer 1 features, and event panels should still be treated as contaminated until the staged rebuild is run from metadata-complete raw fundamentals.
- Golden validation passes on the refreshed metadata-complete golden raw fundamentals sample.
- The selector consumes `effective_model_date` when the input rows include it. The Universe V2 build now creates SEC timing metadata before clean fundamentals and passes it into `fundamentals_clean` before fact selection.
- The pre-rebuild golden check still uses a conservative filing-date cutoff because it intentionally does not build the full event timing table.
- Layer 1 balance-sheet denominator ratios now have an explicit scale convention: generic `asset_turnover`, `roa`, `roe`, and `accruals_ratio` are TTM-normalized; `qtr_` columns are 10-Q period-specific; `annual_` columns are 10-K annual.

## Safe next stage

The golden ticker raw refresh has been isolated to:

```text
data/raw/fundamentals/golden_raw_fundamentals.parquet
```

Rerun validation with:

```powershell
& .\.venv\Scripts\python.exe src\audit\golden_fundamentals_validation.py --raw-path data\raw\fundamentals\golden_raw_fundamentals.parquet
& .\.venv\Scripts\python.exe src\audit\fundamentals_before_after_report.py --raw-path data\raw\fundamentals\golden_raw_fundamentals.parquet --tickers AAPL,MSFT,JPM,XOM,TSLA,KR
```

Only after the golden sample passes should the clean fundamentals, Layer 1 features, and event panels be regenerated in stages.
