# Legacy Daily Benchmark Registry

This registry freezes the pre-quarterly benchmark so it stays reproducible without competing with the live quarterly path.

| Benchmark | Modeling Unit | Label | Validation | Status | Config | Artifact |
|---|---|---|---|---|---|---|
| `event_panel_v2_primary` | one row = one filing event evaluated on the daily `5`-day label setup | `event_v2_5d_sign` | `5`-fold purged expanding window with `2024` holdout | `legacy_frozen_baseline` | `configs/daily/legacy_event_panel_v2_primary.yaml` | `reports/results/event_panel_v2_primary_benchmark.csv` |

## Notes

- The root-level `configs/event_panel_v2_primary.yaml` file is preserved for compatibility.
- The daily baseline remains useful as a historical reference, but it is not on the live promotion ladder.
