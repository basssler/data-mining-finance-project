# Phase 6B Alpha Vantage Earnings Test V1

## Data Source

- External dataset family: Alpha Vantage structured earnings expectations / earnings outcomes.
- Normalized estimates table: `C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\interim\alpha_vantage\alpha_vantage_earnings_estimates.parquet`
- Normalized earnings table: `C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\interim\alpha_vantage\alpha_vantage_earnings.parquet`
- Endpoint scope: `EARNINGS_ESTIMATES` and `EARNINGS` only.

## API Efficiency Design

- Per-symbol, per-endpoint raw JSON cache under `data/raw/alpha_vantage/`.
- Resumable manifest with `pending`, `complete`, and `failed` states stored in `data/interim/alpha_vantage/alpha_vantage_manifest.json`.
- Multi-key round-robin rotation from `ALPHAVANTAGE_API_KEYS` with cooldown handling for rate limits.
- Backfill mode avoids re-pulling any completed symbol-endpoint pair unless refresh is explicitly requested.

## Leakage Controls

- Alpha Vantage earnings-outcome rows are timestamped using `reportedDate` + `reportTime` and must be strictly earlier than the filing-event cutoff.
- Filing events with missing intraday timestamps use previous-trading-day close as the conservative cutoff.
- `EARNINGS_ESTIMATES` rows were normalized in full, but only quarterly estimate rows linked to already-reported quarters were promoted into model features.
- Untimestamped future estimate snapshots were intentionally excluded from modeling because they are not point-in-time safe.

## Feasible Feature Block

- Safe outcome features: latest prior EPS surprise, latest prior EPS surprise percent, trailing 4-quarter EPS surprise mean/std, trailing 4-quarter EPS surprise percent mean/std, trailing 4-quarter EPS beat rate, days since last earnings release.
- Safe estimate-derived features: latest prior quarter EPS estimate, latest prior quarter revenue estimate, EPS/revenue analyst counts, EPS estimate revision vs 30/90 days ago for the most recently reported quarter.
- Not promoted: annual estimate features and revenue revision/surprise features where Alpha Vantage did not expose sufficiently timestamped historical fields.

## Coverage Diagnostics

- Manifest rows: `68`
- Manifest status counts: `{'complete': 68}`
- Full backfill complete: `yes`
- Event-panel rows: `1,109`
- Tickers: `34`
- New Alpha Vantage feature count: `20`
- Coverage by year: `{2015: 100.0, 2016: 91.84, 2017: 96.85, 2018: 96.95, 2019: 96.97, 2020: 96.97, 2021: 96.97, 2022: 96.97, 2023: 96.97, 2024: 97.08}`
- Coverage by ticker: `{'ADM': 100.0, 'BF-B': 0.0, 'BG': 100.0, 'CAG': 100.0, 'CHD': 100.0, 'CL': 100.0, 'CLX': 100.0, 'COST': 100.0, 'CPB': 100.0, 'DG': 100.0, 'DLTR': 100.0, 'EL': 100.0, 'GIS': 100.0, 'HRL': 100.0, 'HSY': 100.0, 'KDP': 100.0, 'KHC': 100.0, 'KMB': 100.0, 'KO': 100.0, 'KR': 100.0, 'MDLZ': 100.0, 'MKC': 100.0, 'MNST': 100.0, 'MO': 100.0, 'PEP': 100.0, 'PG': 100.0, 'PM': 100.0, 'SJM': 100.0, 'STZ': 100.0, 'SYY': 100.0, 'TAP': 100.0, 'TGT': 100.0, 'TSN': 100.0, 'WMT': 100.0}`

## Feature QA

- Missingness by new feature: `{'av_eps_estimate_analyst_count_before_event': 14.07, 'av_eps_estimate_revision_30d': 14.07, 'av_eps_estimate_revision_90d': 14.07, 'av_revenue_estimate_analyst_count_before_event': 14.07, 'av_latest_quarterly_eps_estimate_before_event': 14.07, 'av_latest_quarterly_revenue_estimate_before_event': 14.07, 'av_latest_prior_eps_surprise_pct_before_event': 3.43, 'av_trailing_4q_eps_beat_rate': 3.25, 'av_trailing_4q_eps_surprise_pct_std': 3.25, 'av_trailing_4q_eps_surprise_pct_mean': 3.25, 'av_trailing_4q_eps_surprise_std': 3.25, 'av_trailing_4q_eps_surprise_mean': 3.25, 'av_latest_prior_eps_surprise_before_event': 3.25, 'av_days_since_last_earnings_release': 3.25, 'av_coverage_any': 0.0}`
- Top descriptive target correlations: `{'av_trailing_4q_eps_surprise_pct_std': 0.05023146459389001, 'av_latest_quarterly_eps_estimate_before_event': 0.036278906061228605, 'av_trailing_4q_eps_surprise_pct_mean': 0.03447509966686203, 'av_trailing_4q_eps_surprise_std': 0.02619915653543923, 'av_eps_estimate_revision_30d': 0.02260548676462498, 'av_coverage_any': 0.02084765705344657, 'av_latest_prior_eps_surprise_pct_before_event': 0.015020239204283048, 'av_latest_quarterly_revenue_estimate_before_event': 0.01383401871176752, 'av_eps_estimate_revision_90d': 0.013549708950218377, 'av_revenue_estimate_analyst_count_before_event': -0.01252191119235978}`

## Results

- Baseline selected model: `random_forest` with CV AUC `0.5260` and holdout AUC `0.5237`.
- Alpha Vantage selected model: `xgboost` with CV AUC `0.5233` and holdout AUC `0.5774`.
- The additive benchmark exactly matched baseline because all Alpha Vantage columns were excluded by the existing missingness filter at the current partial-coverage state.

## Recommendation

- Final decision: **FREEZE**
- `PROMOTE` means the additive block improved the locked benchmark cleanly enough to carry forward.
- `FREEZE` means the ingest and merge are kept for reference, but the dataset is not yet strong enough to promote.
- `REJECT` would be reserved for a structurally impractical or leakage-unsafe dataset path.
- Current result: keep the Phase 4 anchor as the primary setup and treat Alpha Vantage earnings data as a tested but not-yet-promoted additive layer.
