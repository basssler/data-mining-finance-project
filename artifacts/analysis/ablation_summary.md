# Ablation Summary

- Selected model held constant: `xgboost`
- Strongest validation setup: `market_only`
- Weakest validation setup: `event_plus_fundamentals`

| Model | Families | Train AUC | Validation AUC | Overfit Gap | Feature Count |
|---|---|---:|---:|---:|---:|
| event_nlp_only | event_nlp | 0.9760 | 0.5283 | 0.4477 | 10 |
| event_plus_fundamentals | event_nlp, fundamentals | 0.9947 | 0.5067 | 0.4880 | 31 |
| event_plus_market | event_nlp, momentum_returns, volatility_risk, liquidity_trading | 0.9994 | 0.5351 | 0.4643 | 24 |
| full_model | event_nlp, momentum_returns, volatility_risk, liquidity_trading, fundamentals, timing_context | 1.0000 | 0.5368 | 0.4632 | 49 |
| fundamentals_only | fundamentals | 0.8390 | 0.5214 | 0.3176 | 21 |
| market_only | momentum_returns, volatility_risk, liquidity_trading | 0.9963 | 0.5430 | 0.4533 | 14 |
| market_plus_fundamentals | momentum_returns, volatility_risk, liquidity_trading, fundamentals | 0.9996 | 0.5412 | 0.4583 | 35 |
