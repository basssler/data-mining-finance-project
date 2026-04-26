# Phase 9B External News Coverage Audit

## Scope

- Dataset: `Brianferrell787/financial-news-multisource`
- Mode: `sample`
- Listed parquet files: `4`
- Sampled parquet files: `4`
- Successfully read sampled files: `4`
- Sampled files with access/read errors: `0`
- Sample rows per subset cap: `50,000`
- Capital IQ universe rows: `640`
- Event panel rows: `1,109`
- Event coverage anchor column: `effective_model_date`

## Go/No-Go Readout

- Safe loading design: yes, this audit lists shards and reads Parquet batches; it does not materialize the full dataset.
- Date/timestamp metadata detected in schema: `True`
- Ticker/symbol metadata detected in schema: `True`
- Source metadata detected in schema: `True`
- Best sampled trading-day coverage: `2.8855%`
- Recommendation: **Do not integrate into the current quarterly model.**
- Preferred targeted subsets: `yahoo_finance_felixdrinkall`, `sentarl_combined`, `benzinga_6000stocks`

## Final Recommendation

Do not integrate this dataset into the current quarterly model.

The targeted subset audit confirms that several Hugging Face subsets contain clean ticker, source, and timestamp metadata, especially `benzinga_6000stocks`, `sentarl_combined`, and `yahoo_finance_felixdrinkall`. However, conservative pre-event coverage remains too sparse for model integration. The best trading-day window, `T-10_to_T-1`, covers only 32 of 1,109 events, or 2.89%.

Because coverage is concentrated in limited date ranges and only a small share of quarterly events receive usable pre-event news, this dataset would mostly create missing or zero-valued features. Running FinBERT or training models on this layer is not justified.

Final status: researched and rejected for immediate integration. Revisit later only if the project narrows to a smaller ticker universe, a different date range, or a dedicated news-source prototype.

## Sampled Subset Coverage

| subset                      |   sample_rows | date_min   | date_max   |   rows_with_date_pct |   rows_with_source_pct |   rows_with_ticker_candidate_pct |   rows_with_universe_ticker_match_pct |   unique_universe_tickers_matched |
|:----------------------------|--------------:|:-----------|:-----------|---------------------:|-----------------------:|---------------------------------:|--------------------------------------:|----------------------------------:|
| benzinga_6000stocks         |         50000 | 2009-02-14 | 2015-01-26 |                  100 |                    100 |                           98.808 |                                28.968 |                               376 |
| sentarl_combined            |         50000 | 1997-01-03 | 2004-11-05 |                  100 |                    100 |                           96.46  |                                87.222 |                               438 |
| yahoo_finance_felixdrinkall |         50000 | 2017-01-03 | 2023-03-13 |                  100 |                    100 |                          100     |                                98.052 |                                44 |

## Required Conservative Trading-Day Event Coverage

| window      |   start_offset_trading_days |   end_offset_trading_days |   events |   covered_events |   coverage_pct | note                                                                   |
|:------------|----------------------------:|--------------------------:|---------:|-----------------:|---------------:|:-----------------------------------------------------------------------|
| T-1         |                           1 |                         1 |     1109 |               12 |        1.08206 | Weekday trading-day proxy; excludes event-day and post-event articles. |
| T-3_to_T-1  |                           3 |                         1 |     1109 |               22 |        1.98377 | Weekday trading-day proxy; excludes event-day and post-event articles. |
| T-5_to_T-1  |                           5 |                         1 |     1109 |               26 |        2.34445 | Weekday trading-day proxy; excludes event-day and post-event articles. |
| T-10_to_T-1 |                          10 |                         1 |     1109 |               32 |        2.88548 | Weekday trading-day proxy; excludes event-day and post-event articles. |

## Preliminary Calendar-Day Event Coverage Smoke Test

|   window_days |   events |   covered_events |   coverage_pct | note                                                             |
|--------------:|---------:|-----------------:|---------------:|:-----------------------------------------------------------------|
|             7 |     1109 |               27 |        2.43463 | Sampled lower-bound; excludes event-day and post-event articles. |
|            14 |     1109 |               33 |        2.97565 | Sampled lower-bound; excludes event-day and post-event articles. |
|            30 |     1109 |               45 |        4.05771 | Sampled lower-bound; excludes event-day and post-event articles. |

## Audit Warnings

- This run used `effective_model_date` as the event anchor. Before final integration, verify that this matches the quarterly label timing and `tradable_date` policy.
- Limited-shard sampling can be date-biased because many subsets are stored in chronological shards and sampled shards may cover narrow date slices.
- The calendar-day coverage table is a smoke test only; integration decisions should use the required trading-day windows.
- Treat `fnspid_news` as suspicious until ticker extraction is inspected; it can show a high ticker-match percentage while matching very few unique universe tickers.
- No known suspicious subsets were sampled in this run.

## Schema Columns

`date`, `extra_fields`, `text`

## Top `extra_fields` Keys

| extra_field_key     |   sample_rows_with_key |   sample_rows_non_null | example                                                                                         |
|:--------------------|-----------------------:|-----------------------:|:------------------------------------------------------------------------------------------------|
| stocks              |                 150000 |                 150000 | $INDU;$SPX;COMP;HD;WMT                                                                          |
| text_type           |                 150000 |                 150000 | headline+subhead+abstract                                                                       |
| time_precision      |                 150000 |                 150000 | minute                                                                                          |
| dataset_source      |                 150000 |                 150000 | private:sentarl                                                                                 |
| dataset             |                 150000 |                 150000 | sentarl_20assets                                                                                |
| raw_type            |                 150000 |                 150000 | used_icaif_2021_csv                                                                             |
| tz_hint             |                 150000 |                 150000 | America/New_York                                                                                |
| source              |                 131217 |                 131217 | SentARL                                                                                         |
| url                 |                  68783 |                  68783 | https://www.benzinga.com/content/thestreet-com/109748/rally-may-have-run-its-course             |
| source_domain       |                  50000 |                  50000 | finance.yahoo.com                                                                               |
| news_outlet         |                  50000 |                  50000 | finance.yahoo.com                                                                               |
| authors             |                  50000 |                  50000 | nan                                                                                             |
| mentioned_companies |                  50000 |                  50000 | GOOGL;AMZN;AAPL;NFLX                                                                            |
| related_companies   |                  50000 |                  50000 | IGLD;TDC;IAIC;CTXS;EPAY;OTEX;CRAY;TCX;SMCI;YHOO;WBMD;HMI;DTLK;HPQ;ADSK;MGIC;MSFT;TISA;OMCL;ANSS |
| industries          |                  50000 |                  50000 | 7841;7370;3571;7375                                                                             |

## Leakage Risks

- Use only articles with publication timestamps strictly before the event anchor; this audit excludes event-day and post-event text from coverage windows.
- Rows with date-only metadata are risky for intraday filings because publication ordering cannot be proven.
- Generic market, source popularity, article counts, or future-resolved company metadata can leak if computed with knowledge after the event date.
- Ticker extraction from free text is not conservative enough for modeling; later integration should require explicit ticker/entity metadata or a point-in-time entity map.
- Duplicate/syndicated articles across sources should be collapsed before any later feature work to avoid overweighting wire duplication.

## Output Files

- `outputs\quarterly\diagnostics\external_news_schema.csv`
- `outputs\quarterly\diagnostics\external_news_file_sample.csv`
- `outputs\quarterly\diagnostics\external_news_subset_summary.csv`
- `outputs\quarterly\diagnostics\external_news_extra_fields_summary.csv`
- `outputs\quarterly\diagnostics\external_news_calendar_day_event_coverage.csv`
- `outputs\quarterly\diagnostics\external_news_trading_day_event_coverage.csv`
