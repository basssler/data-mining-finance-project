# Universe V2

## Purpose

Phase 5 expands the locked event-based setup from the original 34-name Consumer Staples universe to a broader large-cap cross-sector universe without changing the observation unit, label horizon, feature families, or model set.

## Source Of Ticker List

- Source artifact: `data/reference/universe_v2_tickers.csv`
- Source type: manually curated internal reference list of liquid U.S. large-cap names already compatible with the repo's existing SEC and Yahoo-based pipelines
- This phase does not introduce a new licensed or external dataset. It only broadens the ticker list used against the same existing data sources and ingestion code paths

## Approximate Count

- Target size: 80 to 150 names
- Current `universe_v2` size: 127 tickers

## Sector Coverage

- Communication Services: 8
- Consumer Discretionary: 10
- Consumer Staples: 34
- Energy: 7
- Financials: 11
- Health Care: 13
- Industrials: 11
- Information Technology: 17
- Materials: 7
- Real Estate: 3
- Utilities: 6

## Selection Logic

- Start from the original 34-name universe so the old benchmark remains directly comparable
- Add long-lived, liquid, well-covered U.S. large-cap names across sectors rather than narrow thematic additions
- Prefer names with broad analyst coverage, stable SEC filing history, and deep daily price history over marginal or recently listed constituents
- Keep the list moderate enough that the engineering burden stays manageable while still increasing cross-sectional breadth substantially

## Exclusions

- No ADR-heavy or non-U.S.-domiciled names were intentionally targeted in this phase
- No recent IPO-heavy or short-history cohorts were intentionally added
- Duplicated share classes were minimized; the legacy `BF-B` ticker remains because it is already part of the baseline universe and existing repo logic supports it
- Extremely new spinoffs and names with obviously short post-2015 public history were avoided

## Why 80 To 150 Names Instead Of The Full S&P 500

- Phase 5 is a scale test, not a final production-universe decision
- A 100-ish name universe is large enough to test whether more cross-sectional breadth helps the locked benchmark
- It keeps SEC ingestion, filing-text pulls, FinBERT scoring, and debugging costs bounded while the event-row design is still being validated
- Jumping directly to the full S&P 500 would add operational risk and runtime cost before the broader-universe decision is earned by evidence

## Operating Rule

- `event_v1`, the old daily panels, and the 34-name locked event panel remain untouched
- `universe_v2` is a parallel research universe for testing whether scale improves the locked event-panel benchmark before any external data sourcing
