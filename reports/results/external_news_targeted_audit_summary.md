# Phase 9B External News Targeted Subset Audit

## Targeted Scope

- Dataset: `Brianferrell787/financial-news-multisource`
- Targeted subsets: `yahoo_finance_felixdrinkall`, `sentarl_combined`, `benzinga_6000stocks`
- Event coverage anchor column: `effective_model_date`

## Final Recommendation

Do not integrate this dataset into the current quarterly model.

The targeted subset audit confirms that several Hugging Face subsets contain clean ticker, source, and timestamp metadata, especially `benzinga_6000stocks`, `sentarl_combined`, and `yahoo_finance_felixdrinkall`. However, conservative pre-event coverage remains too sparse for model integration. The best trading-day window, `T-10_to_T-1`, covers only 32 of 1,109 events, or 2.89%.

Because coverage is concentrated in limited date ranges and only a small share of quarterly events receive usable pre-event news, this dataset would mostly create missing or zero-valued features. Running FinBERT or training models on this layer is not justified.

Final status: researched and rejected for immediate integration. Revisit later only if the project narrows to a smaller ticker universe, a different date range, or a dedicated news-source prototype.

## Targeted Subset Coverage

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

## Warnings

- This run used `effective_model_date` as the event anchor, not necessarily `tradable_date`; verify quarterly label timing before any integration.
- Limited-shard sampling can be date-biased if sampled shards cover narrow chronological slices.
- `fnspid_news` is intentionally excluded from the preferred target list until ticker extraction is inspected.
