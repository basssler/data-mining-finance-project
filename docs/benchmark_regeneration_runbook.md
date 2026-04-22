# Benchmark Regeneration Runbook

Use the repo-local `.venv` for every run.

```powershell
.venv\Scripts\Activate.ps1
```

## Regeneration Order

1. Regenerate the canonical enriched benchmark first.

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_primary.yaml
```

2. Regenerate derivative benchmark CSVs that depend on the canonical benchmark for comparison.

```powershell
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_universe_v2.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_sec_sentiment_v1.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_phase6b_alpha_vantage.yaml
.venv\Scripts\python.exe src\train_event_panel_v2.py --config configs\event_panel_v2_quarterly.yaml
```

3. Regenerate narrative reports from the current benchmark CSVs and manifest state.

```powershell
.venv\Scripts\python.exe src\report_event_panel_v2_universe_v2.py
.venv\Scripts\python.exe src\report_event_panel_v2_sec_sentiment_v1.py
.venv\Scripts\python.exe src\report_event_panel_v2_phase6b_alpha_vantage.py
```

## Notes

- Treat `reports/results/event_panel_v2_primary_benchmark.csv` as the comparison anchor unless a regenerated benchmark clearly replaces it.
- Regenerate the derivative reports after their benchmark CSVs are refreshed; otherwise the narratives can drift from the artifacts.
- For verification-only runs, you can pass alternate markdown output paths to the report scripts to avoid overwriting checked-in generated files outside the intended edit scope.
