# Stock Direction Prediction: 1-Minute Executive Summary

## The Question
Can we predict **5-trading-day excess returns** (stock return minus sector return) for 34 large-cap Consumer Staples stocks using fundamentals, market features, and SEC sentiment?

## The Answer
**Only modestly, and mainly after adding structure to the fundamentals.**

## The Numbers
- **Best achieved AUC: 0.5388** (holdout) / **0.5415** (cross-validation)
- **Random guessing AUC: 0.5000**
- **Improvement over random: +3.9%**

## What We Tested

| Phase | What | Result |
|-------|------|--------|
| **4** | Enriched primary benchmark: event-driven panel plus financial profile scores | **AUC 0.539** - best current anchor |
| **5** | Universe expansion: can 126 stocks help? | **AUC 0.507** - made it worse |
| **6** | SEC sentiment reproducibility check | Existing signal confirmed |
| **6B** | External earnings data (Alpha Vantage) | Incomplete - waiting on data |

## What Helped

1. **Event-driven sampling** was better than daily forward-fill.
2. **Financial profile scores** helped more than broadening the universe.
3. The best current full-panel model is **XGBoost** on the enriched event panel.

## What The Profile Layer Did

- It grouped firms by **liquidity, solvency, profitability, and growth quality**.
- It improved the main benchmark to **0.5415 CV AUC / 0.5388 holdout AUC**.
- It also revealed that the **financially weak** segment may be more learnable than the full sample.

## Best Segment Finding

- On the `financially_weak` holdout subset, the enriched full-panel model scored **0.5324 AUC**.
- A dedicated `financially_weak` model scored **0.5590 AUC** on that same subset.
- This is promising, but it is still a **secondary result**, not the main project benchmark.

## Why The Edge Is Still Limited

1. **5 days is too short** for large-cap stocks and noise dominates.
2. **34 companies is still a small universe**.
3. **Fundamentals move quarterly, not daily**, so the feature timing does not naturally match the label horizon.

## The Bottom Line

**This problem is still data-limited, not method-limited.**

The project now has a better benchmark and a better story:
- the enriched full-panel model is the main result
- financial profiles improve interpretation
- segment modeling looks most promising for **financially weak firms**

To unlock more signal, the next best moves are:
- **Longer prediction horizon** (21+ days)
- **Higher-signal events** (earnings surprises, insider trades)
- **Smaller-cap universe** or a more targeted event subset

---

**Project Duration:** 1 month  
**Experiments:** 4 major phases plus targeted follow-ups  
**Verdict:** Ready for class
