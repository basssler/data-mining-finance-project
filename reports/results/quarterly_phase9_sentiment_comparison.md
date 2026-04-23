# Quarterly Phase 9 Sentiment Comparison

## Setup

- Base config: frozen quarterly benchmark anchor `quarterly_core_no_market_anchor_v1`.
- Label contract: `21d_excess_thresholded` via the frozen prebuilt label map.
- Event timing, purged walk-forward validation, and 2024 holdout were kept unchanged.
- Phase 8 market features remained excluded in every setup.

## Selected Models

| Setup | Selected Model | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Rows | Broad Usable | Event-Specific Usable | Combined Usable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| event_specific_sentiment_only | random_forest | 0.5210 | 0.0324 | 0.4789 | 0.5367 | 66 | 0 | 14 | 14 |
| core_no_sentiment | random_forest | 0.5126 | 0.0346 | 0.4652 | 0.5291 | 66 | 0 | 0 | 0 |
| broad_filing_sentiment_only | random_forest | 0.5162 | 0.0390 | 0.4653 | 0.5263 | 66 | 10 | 0 | 10 |
| combined_sentiment_block | random_forest | 0.5015 | 0.0392 | 0.4545 | 0.5179 | 66 | 10 | 14 | 24 |

## Direct Answers

- Does `event_specific_sentiment_only` beat `broad_filing_sentiment_only`? `Yes`
- Does `combined_sentiment_block` beat `core_no_sentiment`? `No`
- Are delta/surprise/dispersion/attention features helping, or just adding noise? `Helping`
- Is Phase 9 strong enough to keep as part of the benchmark stack before Phase 10? `Yes`

## Winner

- Winning setup: `event_specific_sentiment_only` with `random_forest`.
- Winner metrics: CV AUC `0.5210`, CV AUC std `0.0324`, worst fold AUC `0.4789`, holdout AUC `0.5367`.

## Sentiment Survivors In Winner

- `combined_sentiment_block|event_specific_sentiment`: qfd_es_filing_negative_prob, qfd_es_filing_negative_prob_delta_prev_q, qfd_es_filing_neutral_prob, qfd_es_filing_positive_prob, qfd_es_filing_positive_prob_delta_prev_q, qfd_es_filing_sentiment_abs, qfd_es_filing_sentiment_delta_prev_q, qfd_es_filing_sentiment_score, qfd_es_log_text_chunk_count, qfd_es_negative_tone_jump, qfd_es_sentiment_entropy, qfd_es_sentiment_polarity_balance, qfd_es_sentiment_uncertainty, qfd_es_text_chunk_count
