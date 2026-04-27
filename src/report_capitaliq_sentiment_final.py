"""Generate locked final Capital IQ sentiment report artifacts.

This script is intentionally read-only with respect to modeling inputs: it
summarizes existing experiment outputs and does not train or tune models.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELING_DIR = ROOT / "outputs" / "quarterly" / "modeling"
DEFAULT_OUTPUT_DIR = MODELING_DIR / "final"
REPORTS_RESULTS_DIR = ROOT / "reports" / "results"

REQUIRED_INPUTS = {
    "comparison": MODELING_DIR / "capitaliq_sentiment_comparison.csv",
    "apples": MODELING_DIR / "capitaliq_sentiment_apples_to_apples.csv",
    "ablation": MODELING_DIR / "capitaliq_sentiment_ablation.csv",
    "year_holdouts": MODELING_DIR / "capitaliq_sentiment_year_holdouts.md",
    "bootstrap": MODELING_DIR / "capitaliq_sentiment_holdout_bootstrap.md",
    "feature_stability": MODELING_DIR / "capitaliq_sentiment_feature_stability.md",
    "predictions": MODELING_DIR / "capitaliq_sentiment_2024_holdout_predictions.csv",
}

VERDICT = (
    "Capital IQ Key Developments sentiment was tested as an incremental "
    "event-text feature layer in the quarterly Consumer Staples panel. The "
    "within-sector adjusted sentiment model reached 0.6038 holdout AUC versus "
    "0.5020 for the core + market benchmark. However, the lift was not stable "
    "across earlier pseudo-holdout years and the bootstrapped 2024 AUC-delta "
    "interval crossed zero. Therefore, the evidence supports Capital IQ "
    "event-text sentiment as a promising but fragile feature layer, not as a "
    "robust standalone predictor."
)


def require_inputs(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def normalize_cell(value: str) -> object:
    value = value.strip()
    if value.lower() in {"n/a", "nan", ""}:
        return math.nan
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_markdown_tables(path: Path) -> list[pd.DataFrame]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tables: list[pd.DataFrame] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|") or i + 1 >= len(lines):
            i += 1
            continue
        separator = lines[i + 1].strip()
        if not re.fullmatch(r"\|(?:\s*:?-{3,}:?\s*\|)+", separator):
            i += 1
            continue

        block = [line]
        i += 2
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append(lines[i].strip())
            i += 1

        header = [cell.strip() for cell in block[0].strip("|").split("|")]
        rows = []
        for row in block[1:]:
            cells = [normalize_cell(cell) for cell in row.strip("|").split("|")]
            if len(cells) == len(header):
                rows.append(cells)
        tables.append(pd.DataFrame(rows, columns=header))
    return tables


def read_inputs() -> dict[str, pd.DataFrame]:
    comparison = pd.read_csv(REQUIRED_INPUTS["comparison"])
    apples = pd.read_csv(REQUIRED_INPUTS["apples"])
    ablation = pd.read_csv(REQUIRED_INPUTS["ablation"])
    predictions = pd.read_csv(REQUIRED_INPUTS["predictions"])

    year_tables = parse_markdown_tables(REQUIRED_INPUTS["year_holdouts"])
    if len(year_tables) < 2:
        raise ValueError("Expected metrics and delta tables in year holdout markdown.")
    year_metrics, year_delta = year_tables[0], year_tables[1]

    bootstrap_tables = parse_markdown_tables(REQUIRED_INPUTS["bootstrap"])
    if not bootstrap_tables:
        raise ValueError("Expected bootstrap table in bootstrap markdown.")
    bootstrap = bootstrap_tables[0]

    feature_tables = parse_markdown_tables(REQUIRED_INPUTS["feature_stability"])
    if not feature_tables:
        raise ValueError("Expected feature stability table in feature markdown.")
    feature_stability = feature_tables[0]

    return {
        "comparison": comparison,
        "apples_to_apples": apples,
        "ablation": ablation,
        "year_holdout_metrics": year_metrics,
        "year_holdout_delta": year_delta,
        "bootstrap": bootstrap,
        "feature_stability": feature_stability,
        "predictions": predictions,
    }


def add_table_name(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "table", name)
    return out


def write_combined_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    combined = pd.concat(
        [
            add_table_name("untuned_ladder", tables["comparison"]),
            add_table_name("apples_to_apples", tables["apples_to_apples"]),
            add_table_name("ablation", tables["ablation"]),
            add_table_name("year_holdout_metrics", tables["year_holdout_metrics"]),
            add_table_name("year_holdout_delta", tables["year_holdout_delta"]),
            add_table_name("bootstrap", tables["bootstrap"]),
            add_table_name("feature_stability", tables["feature_stability"]),
        ],
        ignore_index=True,
        sort=False,
    )
    path = output_dir / "capitaliq_sentiment_final_tables.csv"
    combined.to_csv(path, index=False)
    return path


def format_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    display = frame.loc[:, cols].copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda value: format_float(value))
    return display.to_markdown(index=False)


def selected_value(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    rows = frame.loc[mask, column]
    if rows.empty:
        raise ValueError(f"No row found for {column}.")
    return float(rows.iloc[0])


def validate_locked_numbers(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    comparison = tables["comparison"]
    ablation = tables["ablation"]
    bootstrap = tables["bootstrap"]

    core_market_auc = selected_value(
        comparison,
        comparison["experiment_family"].eq("quarterly_core_plus_market"),
        "holdout_auc",
    )
    adjusted_auc = selected_value(
        comparison,
        comparison["experiment_family"].eq(
            "quarterly_core_plus_market_plus_capitaliq_sector_adjusted_sentiment"
        ),
        "holdout_auc",
    )
    all_sentiment_auc = selected_value(
        ablation,
        ablation["ablation"].eq("all_capitaliq_sentiment_features"),
        "holdout_auc",
    )
    p05 = float(bootstrap["p05_delta"].iloc[0])
    p95 = float(bootstrap["p95_delta"].iloc[0])

    if abs(core_market_auc - 0.5020) > 0.0001:
        raise ValueError(f"Unexpected core+market holdout AUC: {core_market_auc}")
    if abs(adjusted_auc - 0.6038) > 0.0001:
        raise ValueError(f"Unexpected adjusted sentiment holdout AUC: {adjusted_auc}")
    if abs(all_sentiment_auc - adjusted_auc) > 0.0001:
        raise ValueError("Comparison and ablation adjusted sentiment AUC disagree.")
    if not (p05 < 0 < p95):
        raise ValueError("Expected bootstrap interval to cross zero.")

    return {
        "core_market_auc": core_market_auc,
        "adjusted_auc": adjusted_auc,
        "bootstrap_p05": p05,
        "bootstrap_p95": p95,
    }


def load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")


def plot_ladder(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plt = load_matplotlib()
    comparison = tables["comparison"].copy()
    labels = comparison["title"].str.replace("Capital IQ ", "", regex=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, comparison["holdout_auc"], color=["#4C78A8", "#72B7B2", "#F58518", "#54A24B"])
    ax.axhline(0.5, color="#555555", linewidth=1, linestyle="--")
    ax.set_ylabel("2024 holdout AUC")
    ax.set_title("Capital IQ Sentiment Ladder: 2024 Holdout AUC")
    ax.set_ylim(0.45, max(0.64, comparison["holdout_auc"].max() + 0.03))
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, figures_dir / "ladder_holdout_auc.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(comparison))
    ax.plot(x, comparison["cv_auc_mean"], marker="o", label="CV AUC mean", color="#4C78A8")
    ax.plot(x, comparison["holdout_auc"], marker="o", label="2024 holdout AUC", color="#F58518")
    ax.axhline(0.5, color="#555555", linewidth=1, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("CV AUC vs. 2024 Holdout AUC")
    ax.legend()
    save_fig(fig, figures_dir / "cv_vs_holdout_auc.png")
    plt.close(fig)


def plot_apples(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plt = load_matplotlib()
    apples = tables["apples_to_apples"].copy()
    order = [
        "core",
        "core_plus_market",
        "raw_sentiment",
        "within_sector_adjusted_sentiment",
    ]
    labels = {
        "core": "Core",
        "core_plus_market": "Core + market",
        "raw_sentiment": "Raw sentiment",
        "within_sector_adjusted_sentiment": "Within-sector sentiment",
    }
    pivot = apples.pivot(index="model_name", columns="family", values="holdout_auc").loc[
        ["logistic_regression", "random_forest", "xgboost"], order
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.rename(columns=labels).plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(0.5, color="#555555", linewidth=1, linestyle="--")
    ax.set_ylabel("2024 holdout AUC")
    ax.set_xlabel("")
    ax.set_title("Apples-to-Apples Holdout AUC by Model Family")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")
    save_fig(fig, figures_dir / "apples_to_apples_holdout_auc.png")
    plt.close(fig)


def plot_ablation(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plt = load_matplotlib()
    ablation = tables["ablation"].copy()
    labels = {
        "core_plus_market_only": "Core + market",
        "sentiment_means_only": "Sentiment means",
        "news_counts_only": "News counts",
        "sentiment_momentum_only": "Momentum",
        "sector_adjusted_only": "Sector adjusted",
        "all_capitaliq_sentiment_features": "All sentiment",
    }
    ablation["label"] = ablation["ablation"].map(labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ablation["label"], ablation["holdout_auc"], color="#4C78A8")
    ax.axhline(0.5, color="#555555", linewidth=1, linestyle="--")
    ax.set_ylabel("2024 holdout AUC")
    ax.set_title("Capital IQ Sentiment Ablation")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, figures_dir / "ablation_holdout_auc.png")
    plt.close(fig)


def plot_year_delta(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plt = load_matplotlib()
    delta = tables["year_holdout_delta"].copy()
    delta["year"] = delta["year"].astype(int)
    delta["auc_delta_sentiment_minus_control"] = delta[
        "auc_delta_sentiment_minus_control"
    ].astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in delta.groupby("model_name"):
        ax.plot(
            group["year"],
            group["auc_delta_sentiment_minus_control"],
            marker="o",
            label=model_name,
        )
    ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_ylabel("AUC delta vs. core + market")
    ax.set_title("Year-by-Year Pseudo-Holdout Stability")
    ax.set_xticks(sorted(delta["year"].unique()))
    ax.legend(title="")
    save_fig(fig, figures_dir / "year_holdout_auc_delta.png")
    plt.close(fig)


def plot_bootstrap(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plt = load_matplotlib()
    bootstrap = tables["bootstrap"].iloc[0]
    mean_delta = float(bootstrap["mean_delta"])
    p05 = float(bootstrap["p05_delta"])
    p95 = float(bootstrap["p95_delta"])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar(
        [mean_delta],
        [0],
        xerr=[[mean_delta - p05], [p95 - mean_delta]],
        fmt="o",
        color="#F58518",
        capsize=6,
    )
    ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_yticks([])
    ax.set_xlabel("AUC delta: within-sector sentiment minus core + market")
    ax.set_title("2024 Bootstrap AUC Delta Interval")
    ax.set_xlim(min(-0.12, p05 - 0.02), max(0.15, p95 + 0.02))
    save_fig(fig, figures_dir / "bootstrap_auc_delta_interval.png")
    plt.close(fig)


def plot_feature_stability(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plt = load_matplotlib()
    stability = tables["feature_stability"].copy()
    stability["top3_total"] = (
        pd.to_numeric(stability["top3_count_validation"], errors="coerce").fillna(0)
        + pd.to_numeric(stability["top3_count_holdout"], errors="coerce").fillna(0)
    )
    summary = (
        stability.groupby("feature", as_index=False)["top3_total"]
        .sum()
        .sort_values("top3_total", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(summary["feature"], summary["top3_total"], color="#54A24B")
    ax.set_xlabel("Top-3 appearances across folds/models")
    ax.set_title("Sentiment Feature Stability")
    save_fig(fig, figures_dir / "feature_stability_top3_counts.png")
    plt.close(fig)


def write_figures(tables: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_ladder(tables, figures_dir)
    plot_apples(tables, figures_dir)
    plot_ablation(tables, figures_dir)
    plot_year_delta(tables, figures_dir)
    plot_bootstrap(tables, figures_dir)
    plot_feature_stability(tables, figures_dir)
    return figures_dir


def build_report(tables: dict[str, pd.DataFrame], checks: dict[str, float], figures_dir: Path) -> str:
    comparison = tables["comparison"]
    apples = tables["apples_to_apples"]
    ablation = tables["ablation"]
    year_delta = tables["year_holdout_delta"]
    bootstrap = tables["bootstrap"]
    feature_stability = tables["feature_stability"].copy()

    feature_stability["top3_total"] = (
        pd.to_numeric(feature_stability["top3_count_validation"], errors="coerce").fillna(0)
        + pd.to_numeric(feature_stability["top3_count_holdout"], errors="coerce").fillna(0)
    )
    feature_summary = (
        feature_stability.sort_values("top3_total", ascending=False)
        .head(10)
        .loc[
            :,
            [
                "rung",
                "model_name",
                "feature",
                "top3_count_validation",
                "top3_count_holdout",
                "mean_importance_when_top3",
            ],
        ]
    )

    figure_lines = "\n".join(
        f"- `{path.relative_to(ROOT)}`" for path in sorted(figures_dir.glob("*.png"))
    )

    return f"""# Capital IQ Sentiment Final Report

## Locked Verdict

{VERDICT}

This is zero-shot FinBERT + Capital IQ Key Developments feature engineering. Because the v1 universe is Consumer Staples, the adjusted features should be described as within-sector relative sentiment, not broad sector-specific tuning.

## Final Model Ladder

{markdown_table(comparison, ["rung", "title", "selected_model", "cv_auc_mean", "cv_auc_std", "worst_fold_auc", "holdout_auc", "holdout_log_loss", "holdout_f1", "feature_count"])}

The headline lift is in the 2024 holdout: core + market selected xgboost reached {checks["core_market_auc"]:.4f} AUC, while the within-sector adjusted Capital IQ sentiment rung reached {checks["adjusted_auc"]:.4f} AUC. CV AUC did not improve in parallel, so the holdout improvement is not enough to claim robust general predictive power.

## Apples-to-Apples Model Comparison

{markdown_table(apples, ["family", "model_name", "cv_auc_mean", "cv_auc_std", "worst_fold_auc", "holdout_auc", "holdout_log_loss", "holdout_f1", "feature_count"])}

The sentiment rung improved 2024 holdout AUC across logistic regression, random forest, and xgboost, but that pattern was weaker in CV. This supports a promising but fragile interpretation.

## Feature Ablation

{markdown_table(ablation, ["ablation", "selected_model", "cv_auc_mean", "cv_auc_std", "worst_fold_auc", "holdout_auc", "holdout_log_loss", "feature_count"])}

The best ablation used all Capital IQ sentiment features. Sector-adjusted sentiment alone beat news-count-only and sentiment-means-only ablations on 2024 holdout AUC, which points more toward event-text sentiment and within-sector context than pure event-volume coverage. The result is still not stable enough to treat as definitive.

## Year-by-Year Stability

{markdown_table(year_delta, ["year", "model_name", "core_plus_market", "within_sector_adjusted_sentiment", "auc_delta_sentiment_minus_control"])}

The sentiment layer helped all three model families in 2024, but the pseudo-holdout years were mixed: logistic regression improved in 2021 and 2022, while 2023 worsened across all three families. This is the main reason not to tune now.

## 2024 Bootstrap AUC Delta

{markdown_table(bootstrap, ["iterations", "control_model", "sentiment_model", "control_auc", "sentiment_auc", "mean_delta", "p05_delta", "p95_delta"])}

The 2024 AUC-delta interval is [{checks["bootstrap_p05"]:.4f}, {checks["bootstrap_p95"]:.4f}], so it crosses zero. The holdout result is encouraging, but not statistically stable enough to support a strong predictive claim.

## Feature Importance Stability

{markdown_table(feature_summary, ["rung", "model_name", "feature", "top3_count_validation", "top3_count_holdout", "mean_importance_when_top3"])}

`sent_mean_30d` and `news_count_63d` appeared most consistently among the tracked sentiment features. Momentum features did not appear consistently. The evidence is therefore better framed as Capital IQ event-text sentiment plus event-attention context, not as pure sentiment alone.

## Figures

{figure_lines}

## Final Interpretation

The final project conclusion should be conservative: the richer Capital IQ Key Developments layer solved the missing-news coverage problem and produced meaningful 2024 holdout lift, but the lift did not survive all stability diagnostics. The mature conclusion is that Capital IQ event-text sentiment appears informative in some market regimes and deserves future testing, but the current evidence is promising rather than robust.

## Future Work

A small appendix-only sensitivity check could tune logistic regression on the within-sector adjusted/all Capital IQ sentiment rung using CV-only selection over `C` and `class_weight`, then report the 2024 holdout once. That should not replace the locked untuned comparison as the main result.
"""


def write_report(tables: dict[str, pd.DataFrame], checks: dict[str, float], output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    report = build_report(tables, checks, figures_dir)
    report_path = output_dir / "capitaliq_sentiment_final_report.md"
    report_path.write_text(report, encoding="utf-8")

    REPORTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, REPORTS_RESULTS_DIR / report_path.name)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create final locked Capital IQ sentiment report artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for final report, tables, and figures.",
    )
    args = parser.parse_args()

    require_inputs(REQUIRED_INPUTS)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = read_inputs()
    checks = validate_locked_numbers(tables)
    tables_path = write_combined_tables(tables, output_dir)
    figures_dir = write_figures(tables, output_dir)
    report_path = write_report(tables, checks, output_dir)

    print(f"Wrote {report_path.relative_to(ROOT)}")
    print(f"Wrote {tables_path.relative_to(ROOT)}")
    print(f"Wrote figures under {figures_dir.relative_to(ROOT)}")
    print(f"Copied report to {(REPORTS_RESULTS_DIR / report_path.name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
