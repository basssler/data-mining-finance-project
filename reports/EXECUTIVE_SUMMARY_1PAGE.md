# Stock Direction Prediction: 1-Minute Executive Summary

## The Question
Can we predict **5-trading-day excess returns** (stock return minus sector return) for 34 large-cap Consumer Staples stocks using fundamentals, market features, and SEC sentiment?

## The Answer
**No, not with meaningful signal.**

## The Numbers
- **Best achieved AUC: 0.5237** (holdout) / 0.5260 (cross-validation)
- **Random guessing AUC: 0.5000**
- **Improvement over random: +2.4% (marginal)**

## What We Tested (4 Phases)

| Phase | What | Result |
|-------|------|--------|
| **4** | Primary benchmark: event-driven panel (1,109 events, 72 features, 34 stocks) | AUC 0.524 ✓ Locked baseline |
| **5** | Universe expansion: can 126 stocks help? | AUC 0.507 ✗ Made it worse |
| **6** | SEC sentiment reproducibility check | AUC 0.524 ✓ Confirmed existing signal |
| **6B** | External earnings data (Alpha Vantage) | Incomplete—waiting on data |

## Why Is 0.52 The Ceiling?

**Structural Limitations:**
1. **5 days is too short** for large-cap stocks (microstructure noise dominates)
2. **34 companies is too small** (insufficient cross-sectional variation)
3. **Fundamentals move quarterly, not daily** (grain mismatch)

**Evidence:**
- Broader universe (126 stocks) made performance **worse**, not better
- Adding external earnings data **dropped features due to sparsity**, not improved signal
- All 3 model types (logistic, RF, XGBoost) converge to **0.52 AUC** regardless of hyperparameters

## The Bottom Line

**This problem is fundamentally data-limited, not method-limited.**

We have the right validation setup, clean features, and proper risk controls. The 0.52 AUC is the true signal ceiling for predicting 5-day excess returns on large-cap liquid stocks.

To unlock signal, we'd need:
- **Longer prediction horizon** (21+ days)
- **Higher-signal events** (earnings surprises, insider trades)
- **Smaller-cap universe** (more mispricing potential)

---

**Project Duration:** 1 month
**Experiments:** 4 phases + 1 incomplete
**Verdict:** Ready for class ✓
