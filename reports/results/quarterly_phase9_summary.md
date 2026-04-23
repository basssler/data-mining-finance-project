# Quarterly Phase 9 Summary

## What Changed

- Rebuilt the quarterly feature-design panel with Phase 9 event-specific sentiment features.
- Re-ran the frozen 21d thresholded quarterly benchmark contract across four sentiment setups only.

## Which Setup Won

- Winner: `event_specific_sentiment_only` using `random_forest`.
- Holdout AUC: `0.5367`.

## Direct Decision

- Event-specific only beats broad filing only: `Yes`.
- Combined sentiment beats no-sentiment core: `No`.
- Keep Phase 9 in the benchmark stack: `Yes`.
- Move on to Phase 10 now: `Yes`.
