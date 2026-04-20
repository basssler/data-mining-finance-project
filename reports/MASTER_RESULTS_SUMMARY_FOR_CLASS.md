# Stock Direction Prediction: Complete Results Summary
## Phase 1 Through Phase 6B Plus Financial Profile Follow-Up

**Project Timeline:** ~1 month  
**Final Dataset:** 34 Consumer Staples large-cap stocks (2015-2024)  
**Target:** 5-trading-day excess return sign  
**Primary Event Panel:** 1,109 filing events

---

## Executive Summary

This document summarizes the full experiment sequence and the final class-facing conclusion.

**Key finding:** the project remains difficult, but adding **financial profile scores** created the strongest benchmark so far and gave a more interpretable story than raw fundamentals alone.

- Best current full-panel result: **XGBoost**, CV AUC **0.5415**, holdout AUC **0.5388**
- Best targeted segment result: `financially_weak` segment model, holdout AUC **0.5590** on that subset
- Main interpretation: the signal is still modest, but financial profiling helps organize firms into more meaningful regimes

---

## Phase Progression

### Phase 1-3: Foundation
- Built the SEC fundamentals pipeline, market features, sentiment features, and time-aware validation setup.
- Early results stayed around **0.50-0.51 AUC**.

### Phase 4: Event Panel V2 Baseline
- Switched from daily forward-fill to an event-based panel.
- Original locked benchmark showed a modest gain over the old daily path.
- Earlier anchor: **Random Forest**, CV AUC **0.5260**, holdout AUC **0.5237**.

### Phase 4 Follow-Up: Enriched Financial Profile Benchmark
- Added structured profile scores from accounting ratios:
  - liquidity
  - solvency
  - profitability
  - growth quality
- New benchmark outcome:

| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Selected |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.5254 | 0.7527 | 0.4919 | 0.7492 |  |
| Random Forest | 0.5228 | 0.6987 | 0.4960 | 0.6999 |  |
| XGBoost | **0.5415** | 0.7950 | **0.5388** | 0.7474 | yes |

**Key insight:** the profile layer improved AUC enough to become the best current benchmark, even though the edge is still modest and log loss did not improve uniformly.

### Phase 5: Universe Expansion
- Expanded from 34 names to 126 cross-sector large caps.
- Result: performance worsened, not improved.
- Best expanded-universe holdout AUC stayed around **0.5068**.

**Decision:** do not promote the broader universe.

### Phase 6: SEC Sentiment Reproducibility
- Reconfirmed that SEC sentiment was already embedded in the locked path.
- No new standalone lift.

### Phase 6B: Alpha Vantage External Earnings
- Added an external earnings block.
- Current result is still incomplete because the manifest is not fully backfilled.
- Under current partial coverage, the new Alpha Vantage features were dropped by missingness rules.

**Decision:** keep Phase 6B as incomplete.

---

## Financial Profile Analysis

The profile layer did two jobs:
- improved the full-panel benchmark
- gave an interpretable way to segment firms

### Fine Profile Behavior

| Profile | Rows | Mean Excess Return | Hit Rate |
|---|---:|---:|---:|
| mixed_profile | 453 | -0.0008 | 0.4834 |
| distressed_weak_quality | 354 | 0.0011 | 0.5056 |
| high_growth_fragile | 177 | -0.0016 | 0.4689 |
| short_term_healthy_levered | 51 | 0.0012 | 0.5490 |
| mature_defensive | 14 | 0.0034 | 0.7143 |
| stable_compounder | 8 | 0.0061 | 0.7500 |

### Coarse Segment Behavior

| Segment | Rows | Mean Excess Return | Hit Rate |
|---|---:|---:|---:|
| mixed_other | 499 | 0.0003 | 0.5110 |
| financially_weak | 391 | 0.0005 | 0.4962 |
| growth_fragile | 177 | -0.0016 | 0.4689 |
| financially_strong | 42 | 0.0007 | 0.5714 |

**Interpretation:** the groups are not identical. The weakest descriptive group was `growth_fragile`, while `financially_weak` looked different enough from the rest to justify a targeted modeling check.

---

## Segment Modeling Follow-Up

I ran coarse segment models only where sample size, ticker breadth, date breadth, and holdout coverage were large enough.

### Segment Benchmark Results

| Segment | Model | Mean CV AUC | Holdout AUC | Comment |
|---|---|---:|---:|---|
| financially_weak | XGBoost | **0.5465** | 0.5571 | Most promising segment |
| financially_weak | Random Forest | 0.5272 | **0.5943** | Strong holdout, weaker CV |
| growth_fragile | XGBoost | 0.4316 | 0.5347 | Too unstable to trust yet |
| mixed_other | XGBoost | 0.5632 | 0.4417 | Poor out of sample |
| financially_strong | skipped | n/a | n/a | Too sparse |

### Direct Apples-to-Apples Follow-Up on `financially_weak`

To test whether segmenting actually helps, I compared:
- the **full-panel enriched XGBoost**
- a **financially_weak-only XGBoost**

Both were evaluated on the same `financially_weak` 2024 holdout subset.

| Training Scope | Model | Holdout AUC | Holdout Log Loss |
|---|---|---:|---:|
| full_panel | XGBoost | 0.5324 | 0.7425 |
| financially_weak_only | XGBoost | **0.5590** | 0.7966 |

**Interpretation:** yes, there is evidence that the `financially_weak` subset may benefit from a dedicated model. But the result is still narrow and should be presented as a targeted follow-up, not as the project’s new main benchmark.

---

## What Worked

1. **Event-driven sampling** improved the research design over daily forward-fill.
2. **Financial profile scores** produced the best full-panel benchmark.
3. **Segment analysis** added interpretation and found one promising specialized subgroup.
4. **Validation discipline** prevented the project from overclaiming weak signal.

## What Did Not Work

1. **Universe expansion** added noise rather than stable signal.
2. **SEC sentiment alone** did not create a new incremental win.
3. **External earnings data** cannot yet be judged fairly because coverage is incomplete.
4. **Full segmentation across all groups** is not justified because several groups are too sparse or unstable.

---

## Final Class Conclusion

The honest conclusion is stronger than before, but still disciplined:

- Predicting 5-trading-day excess returns for large-cap consumer staples remains hard.
- The project is still **data-limited**, not just **method-limited**.
- The best improvement came from making the fundamentals more structured and interpretable through **financial profiles**.
- The best current class story is:
  - **main result:** enriched full-panel benchmark at **0.5388 holdout AUC**
  - **insight result:** firms can be grouped into financially meaningful profiles
  - **targeted follow-up:** `financially_weak` firms may support a better specialized model

If continuing beyond class, the most defensible next steps would be:
1. keep the enriched full-panel benchmark as the main anchor
2. continue testing the `financially_weak` segment as a secondary path
3. move to longer horizons or higher-signal event types

---

## File References

- Primary benchmark: `reports/results/event_panel_v2_primary_benchmark.md`
- Financial profile analysis: `reports/results/financial_profile_return_analysis_v1.md`
- Segment benchmark: `reports/results/event_panel_v2_segment_benchmark_v1.md`
- Financially weak follow-up: `reports/results/financially_weak_followup_v1.md`
- Universe expansion: `reports/results/event_panel_v2_universe_v2_benchmark.md`
- SEC sentiment reproducibility: `reports/results/event_panel_v2_sec_sentiment_v1_benchmark.md`
- Alpha Vantage external data: `reports/results/event_panel_v2_phase6b_alpha_vantage_benchmark.md`
