# Benchmark Regeneration Runbook

Use the repo-local `.venv` for every run.

```powershell
.venv\Scripts\Activate.ps1
```

## Regeneration Order

1. Regenerate the live quarterly baseline first.

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly.yaml
```

2. Regenerate the active quarterly candidate and quarterly scaffold diagnostics.

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly_stability_core_additive.yaml
.venv\Scripts\python.exe -m src.quarterly_workflow --write-artifacts
```

3. Regenerate quarterly comparison runs only when the ladder needs to be refreshed.

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly_feature_design_core.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly_feature_design_medium_market.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly_feature_design_sentiment.yaml
```

4. Regenerate legacy or side-lane artifacts only when a historical comparison is needed.

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_primary.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_universe_v2.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_sec_sentiment_v1.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_phase6b_alpha_vantage.yaml
```

## Notes

- Treat `reports/results/event_panel_v2_quarterly_benchmark.csv` as the live baseline comparison anchor.
- Treat `reports/results/event_panel_v2_primary_benchmark.csv` as the frozen daily historical comparator.
- Regenerate the derivative reports after their benchmark CSVs are refreshed; otherwise the narratives can drift from the artifacts.
