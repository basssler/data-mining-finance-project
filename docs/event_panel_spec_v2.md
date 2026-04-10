# Event Panel V2 Spec

## Purpose

Phase 2 rebuilds the main research panel around event rows rather than daily forward-filled rows.
One row in `event_panel_v2` equals one ticker-event.

## Scope

- Included event types: `10-K, 10-Q`
- Excluded for this phase: earnings announcement rows, because the repo does not contain a clean standalone earnings-date source outside the grouped 8-K path. To keep Phase 2 within scope, v2 ships with 10-Q and 10-K events only.
- Existing `event_v1` and daily modeling panels remain unchanged.
- No labels, model family changes, training changes, or validation changes are included in this artifact.

## Row Unit

- `event_date`: raw filing date from the source metadata. In some SEC rows this can differ from the local acceptance-calendar date implied by `event_timestamp`.
- `effective_model_date`: first tradable date when the event is considered available to the model.
- `source_id`: SEC accession number for the filing event.

## Alignment Rules

- After-close filings shift to the next tradable date through `effective_model_date`, measured against the local acceptance timestamp when a timestamp exists.
- Before-open filings are available on the same trading day as the local acceptance timestamp.
- Market-hours filings are available on the same trading day as the local acceptance timestamp.
- Missing timestamps are handled conservatively and shift to the next tradable date.
- `effective_model_date` is re-aligned from the stored SEC `availability_base_date` onto actual ticker trading dates so the event panel uses a consistent tradable calendar.
- Market features are attached strictly from the prior trading day: `market_asof_date < effective_model_date`.
- Fundamentals use the latest valid filing snapshot with `fund_snapshot_effective_model_date <= effective_model_date`.
- Sentiment context is same-filing only and is joined by accession number; no daily forward-filled sentiment layer is carried into v2.
- No grouped 8-K EXP-009 features are included in v2.

## Diagnostics

- Row count: `1,109`
- Ticker count: `34`
- Event date range: `2015-07-31` to `2024-12-19`
- Effective model date range: `2015-07-31` to `2024-12-19`
- Fundamentals feature count: `21`
- Market feature count: `14`
- Filing sentiment feature count: `10`
- Event-context feature count: `2`

### Event Counts By Type

| event_type | rows |
| --- | --- |
| 10-Q | 831 |
| 10-K | 278 |

### Timing Bucket Counts

| timing_bucket | rows |
| --- | --- |
| pre_market | 405 |
| market_hours | 359 |
| after_close | 345 |

### Top Missingness Columns

| column | missing_pct |
| --- | --- |
| gross_margin | 100.0000 |
| receivables_turnover | 27.0514 |
| earnings_growth_yoy | 24.4364 |
| inventory_turnover | 23.9856 |
| revenue_growth_yoy | 22.0920 |
| quick_ratio | 17.9441 |
| earnings_growth_qoq | 14.5176 |
| accruals_ratio | 13.4355 |
| roa | 13.4355 |
| asset_turnover | 13.4355 |
| operating_margin | 13.1650 |
| revenue_growth_qoq | 12.8945 |

### Sample Rows

| ticker | event_type | event_date | event_timestamp | effective_model_date | timing_bucket | source_id | event_period_end | market_asof_date | fund_snapshot_filing_date | fund_snapshot_is_current_event | current_filing_sentiment_available |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADM | 10-K | 2016-02-19 | 2016-02-19 09:51:18-0500 | 2016-02-19 | market_hours | 0000007084-16-000046 |  | 2016-02-18 | 2015-08-04 | False | True |
| MDLZ | 10-K | 2016-02-19 | 2016-02-19 11:41:16-0500 | 2016-02-19 | market_hours | 0001193125-16-469394 |  | 2016-02-18 | 2015-07-31 | False | True |
| MNST | 10-K | 2016-02-29 | 2016-02-29 10:09:11-0500 | 2016-02-29 | market_hours | 0001104659-16-100960 | 2015-06-12 | 2016-02-26 | 2016-02-29 | True | True |
| BF-B | 10-K | 2016-06-16 | 2016-06-15 13:49:15-0400 | 2016-06-15 | market_hours | 0000014693-16-000160 | 2016-06-01 | 2016-06-14 | 2016-06-16 | True | True |
| MDLZ | 10-Q | 2015-07-31 | 2015-07-31 11:28:20-0400 | 2015-07-31 | market_hours | 0001193125-15-272328 | 2015-02-16 | 2015-07-30 | 2015-07-31 | True | True |
| ADM | 10-Q | 2015-08-04 | 2015-08-04 12:31:30-0400 | 2015-08-04 | market_hours | 0000007084-15-000023 | 2015-05-01 | 2015-08-03 | 2015-08-04 | True | True |
| MDLZ | 10-Q | 2015-10-29 | 2015-10-29 11:27:05-0400 | 2015-10-29 | market_hours | 0001193125-15-357422 |  | 2015-10-28 | 2015-07-31 | False | True |
| ADM | 10-Q | 2015-11-03 | 2015-11-03 11:03:34-0500 | 2015-11-03 | market_hours | 0000007084-15-000037 |  | 2015-11-02 | 2015-08-04 | False | True |

## Comparison To Current Daily/Event_V1 Design

The current daily comparison panel `event_v1_layer1_panel` has 68,513 rows; `event_panel_v2` has 1,109 rows, so v2 removes daily forward-filled repetition and stores one observation per information event.
In the daily/event_v1 design, one filing can influence many later rows through daily forward-fill. In v2, the filing itself becomes the observation unit and the latest valid snapshot is attached once at the event boundary.

## Structural Notes

- Exact current-filing fundamentals available: `566` of `1,109` rows
- Exact current-filing sentiment available: `1,109` of `1,109` rows
- Rows where the attached fundamentals snapshot equals the current event: `565` of `1,109` rows
- When exact current-filing fundamentals are unavailable, the panel keeps the row and attaches the latest prior valid fundamentals snapshot instead. That fallback is explicit through `current_filing_fundamentals_available` and `fund_snapshot_is_current_event`.

## Artifact

- Main parquet: `C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\interim\event_panel_v2.parquet`
