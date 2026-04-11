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

- Row count: `4,908`
- Ticker count: `126`
- Event date range: `2014-12-29` to `2024-12-20`
- Effective model date range: `2015-01-02` to `2024-12-23`
- Fundamentals feature count: `21`
- Market feature count: `14`
- Filing sentiment feature count: `10`
- Event-context feature count: `2`

### Event Counts By Type

| event_type | rows |
| --- | --- |
| 10-Q | 3682 |
| 10-K | 1226 |

### Timing Bucket Counts

| timing_bucket | rows |
| --- | --- |
| market_hours | 2086 |
| after_close | 1806 |
| pre_market | 1016 |

### Top Missingness Columns

| column | missing_pct |
| --- | --- |
| gross_margin | 100.0000 |
| sec_positive_prob | 100.0000 |
| sec_sentiment_score | 100.0000 |
| sec_negative_change_prev | 100.0000 |
| sec_positive_change_prev | 100.0000 |
| sec_neutral_prob | 100.0000 |
| sec_negative_prob | 100.0000 |
| sec_chunk_count | 100.0000 |
| sec_log_chunk_count | 100.0000 |
| sec_sentiment_change_prev | 100.0000 |
| sec_sentiment_abs | 100.0000 |
| inventory_turnover | 51.0799 |

### Sample Rows

| ticker | event_type | event_date | event_timestamp | effective_model_date | timing_bucket | source_id | event_period_end | market_asof_date | fund_snapshot_filing_date | fund_snapshot_is_current_event | current_filing_sentiment_available |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADBE | 10-K | 2015-01-20 | 2015-01-20 01:42:30-0500 | 2015-01-20 | pre_market | 0000796343-15-000022 |  | 2015-01-16 |  | False | False |
| META | 10-K | 2015-01-29 | 2015-01-29 11:35:58-0500 | 2015-01-29 | market_hours | 0001326801-15-000006 |  | 2015-01-28 |  | False | False |
| MKC | 10-K | 2015-01-29 | 2015-01-29 15:39:34-0500 | 2015-01-29 | market_hours | 0000063754-15-000013 |  | 2015-01-28 |  | False | False |
| NFLX | 10-K | 2015-01-29 | 2015-01-29 13:55:40-0500 | 2015-01-29 | market_hours | 0001065280-15-000006 |  | 2015-01-28 |  | False | False |
| CAG | 10-Q | 2014-12-29 | 2014-12-29 09:23:51-0500 | 2015-01-02 | pre_market | 0001445305-14-005699 |  |  |  | False | False |
| NKE | 10-Q | 2015-01-07 | 2015-01-07 16:56:03-0500 | 2015-01-08 | after_close | 0000320187-15-000003 |  | 2015-01-07 |  | False | False |
| STZ | 10-Q | 2015-01-08 | 2015-01-08 13:20:44-0500 | 2015-01-08 | market_hours | 0000016918-15-000006 |  | 2015-01-07 |  | False | False |
| MU | 10-Q | 2015-01-09 | 2015-01-09 13:01:10-0500 | 2015-01-09 | market_hours | 0000723125-15-000010 |  | 2015-01-08 |  | False | False |

## Comparison To Current Daily/Event_V1 Design

The current daily comparison panel `event_v1_layer1_panel` has 68,513 rows; `event_panel_v2` has 4,908 rows, so v2 removes daily forward-filled repetition and stores one observation per information event.
In the daily/event_v1 design, one filing can influence many later rows through daily forward-fill. In v2, the filing itself becomes the observation unit and the latest valid snapshot is attached once at the event boundary.

## Structural Notes

- Exact current-filing fundamentals available: `2,341` of `4,908` rows
- Exact current-filing sentiment available: `0` of `4,908` rows
- Rows where the attached fundamentals snapshot equals the current event: `1,130` of `4,908` rows
- When exact current-filing fundamentals are unavailable, the panel keeps the row and attaches the latest prior valid fundamentals snapshot instead. That fallback is explicit through `current_filing_fundamentals_available` and `fund_snapshot_is_current_event`.

## Artifact

- Main parquet: `data\interim\event_panel_v2_universe_v2.parquet`
