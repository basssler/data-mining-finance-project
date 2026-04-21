# Baseline Notes

- Panel: `event_panel_v2`
- Panel path: `data/interim/event_panel_v2.parquet`
- Target variant: `event_v2_5d_sign`
- Holdout start: `2024-01-01`
- Selected model: `xgboost`
- Candidate feature count: `55`
- Usable feature count last fold: `49`
- SHAP plot source: `reports\results\event_panel_v2_primary_shap_summary.png`
- SHAP CSV source: `reports\results\event_panel_v2_primary_shap_importance.csv`
- Preprocessing assumptions: median imputation, train-only clipping, model-specific scaling where applicable.
- Feature timestamp logic: event rows keyed on `effective_model_date`; market features must be prior-day aligned.
