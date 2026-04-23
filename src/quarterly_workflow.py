"""Quarterly workflow scaffolding and benchmark ladder reporting."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import (
    QUARTERLY_CONFIGS_DIR,
    QUARTERLY_DOCS_DIR,
    QUARTERLY_OUTPUTS_CHAMPIONS_DIR,
    QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR,
    QUARTERLY_OUTPUTS_EXPERIMENTS_DIR,
    QUARTERLY_OUTPUTS_FEATURES_DIR,
    QUARTERLY_OUTPUTS_LABELS_DIR,
    QUARTERLY_OUTPUTS_PANELS_DIR,
    QUARTERLY_OUTPUTS_VALIDATION_DIR,
)
from src.project_config import QUARTERLY_WORKFLOW_MANIFEST_PATH


@dataclass(frozen=True)
class BenchmarkStep:
    ladder_step: int
    slug: str
    title: str
    objective: str
    config_path: str | None = None
    csv_path: str | None = None
    markdown_path: str | None = None
    notes: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quarterly workflow diagnostics.")
    parser.add_argument("--manifest", default=str(QUARTERLY_WORKFLOW_MANIFEST_PATH))
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Write benchmark ladder diagnostics and quarterly docs scaffold.",
    )
    return parser.parse_args()


def ensure_quarterly_workflow_directories() -> list[Path]:
    """Create the canonical quarterly workflow directories."""
    directories = [
        QUARTERLY_CONFIGS_DIR / "data",
        QUARTERLY_CONFIGS_DIR / "labels",
        QUARTERLY_CONFIGS_DIR / "validation",
        QUARTERLY_CONFIGS_DIR / "models",
        QUARTERLY_CONFIGS_DIR / "experiments",
        QUARTERLY_DOCS_DIR,
        QUARTERLY_OUTPUTS_PANELS_DIR,
        QUARTERLY_OUTPUTS_LABELS_DIR,
        QUARTERLY_OUTPUTS_VALIDATION_DIR,
        QUARTERLY_OUTPUTS_FEATURES_DIR,
        QUARTERLY_OUTPUTS_EXPERIMENTS_DIR,
        QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR,
        QUARTERLY_OUTPUTS_CHAMPIONS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def load_workflow_manifest(manifest_path: Path) -> dict:
    """Read the YAML manifest that defines the benchmark ladder."""
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Quarterly workflow manifest must deserialize to a dictionary.")
    return payload


def load_benchmark_steps(manifest: dict) -> list[BenchmarkStep]:
    """Validate and normalize benchmark ladder steps from the manifest."""
    raw_steps = manifest.get("steps", [])
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Quarterly workflow manifest must define at least one step.")

    steps: list[BenchmarkStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise ValueError("Each quarterly workflow step must be a dictionary.")
        steps.append(
            BenchmarkStep(
                ladder_step=int(raw_step["ladder_step"]),
                slug=str(raw_step["slug"]),
                title=str(raw_step["title"]),
                objective=str(raw_step["objective"]),
                config_path=_optional_str(raw_step.get("config_path")),
                csv_path=_optional_str(raw_step.get("csv_path")),
                markdown_path=_optional_str(raw_step.get("markdown_path")),
                notes=_optional_str(raw_step.get("notes")),
            )
        )
    return steps


def build_benchmark_ladder_frame(steps: list[BenchmarkStep], project_root: Path) -> pd.DataFrame:
    """Summarize quarterly ladder coverage against checked-in configs and outputs."""
    rows: list[dict[str, object]] = []
    for step in steps:
        config_exists = _exists(project_root, step.config_path)
        csv_exists = _exists(project_root, step.csv_path)
        markdown_exists = _exists(project_root, step.markdown_path)
        artifact_count = int(config_exists) + int(csv_exists) + int(markdown_exists)
        rows.append(
            {
                "ladder_step": step.ladder_step,
                "slug": step.slug,
                "title": step.title,
                "objective": step.objective,
                "config_path": step.config_path or "",
                "csv_path": step.csv_path or "",
                "markdown_path": step.markdown_path or "",
                "config_exists": config_exists,
                "csv_exists": csv_exists,
                "markdown_exists": markdown_exists,
                "artifact_count": artifact_count,
                "status": _resolve_status(artifact_count),
                "notes": step.notes or "",
            }
        )
    return pd.DataFrame(rows).sort_values("ladder_step").reset_index(drop=True)


def build_benchmark_log_markdown(manifest: dict, ladder_df: pd.DataFrame) -> str:
    """Render the benchmark ladder into the canonical quarterly benchmark log."""
    workflow_name = str(manifest.get("workflow_name", "Quarterly Benchmark Ladder"))
    updated_note = str(manifest.get("updated_note", "Derived from the checked-in configs and outputs in this repo state."))
    lines = [
        f"# {workflow_name}",
        "",
        updated_note,
        "",
        "| Step | Title | Status | Config | CSV | Markdown |",
        "|---:|---|---|---|---|---|",
    ]
    for _, row in ladder_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["ladder_step"])),
                    str(row["title"]),
                    str(row["status"]),
                    _render_presence(row["config_exists"], row["config_path"]),
                    _render_presence(row["csv_exists"], row["csv_path"]),
                    _render_presence(row["markdown_exists"], row["markdown_path"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Operating Rule",
            "",
            "Advance one major layer at a time: panel, labels, validation, features, model tuning, promotion.",
            "",
            "## Current Gaps",
            "",
        ]
    )
    missing = ladder_df.loc[ladder_df["status"] != "complete", ["ladder_step", "title", "status", "notes"]]
    if missing.empty:
        lines.append("- None. Every ladder rung has a checked-in config and benchmark artifact.")
    else:
        for _, row in missing.iterrows():
            note = f" ({row['notes']})" if str(row["notes"]).strip() else ""
            lines.append(f"- Step {int(row['ladder_step'])}: `{row['title']}` is `{row['status']}`{note}.")
    return "\n".join(lines) + "\n"


def write_quarterly_workflow_artifacts(manifest_path: Path) -> tuple[Path, Path]:
    """Generate benchmark ladder diagnostics and refresh the benchmark log doc."""
    manifest = load_workflow_manifest(manifest_path)
    steps = load_benchmark_steps(manifest)
    ladder_df = build_benchmark_ladder_frame(steps, _resolve_project_root_from_manifest(manifest_path))

    diagnostics_csv_path = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "benchmark_ladder.csv"
    benchmark_log_path = QUARTERLY_DOCS_DIR / "benchmark_log.md"

    diagnostics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_log_path.parent.mkdir(parents=True, exist_ok=True)

    ladder_df.to_csv(diagnostics_csv_path, index=False)
    benchmark_log_path.write_text(build_benchmark_log_markdown(manifest, ladder_df), encoding="utf-8")
    return diagnostics_csv_path, benchmark_log_path


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _exists(project_root: Path, relative_path: str | None) -> bool:
    if not relative_path:
        return False
    return (project_root / relative_path).exists()


def _resolve_status(artifact_count: int) -> str:
    if artifact_count >= 3:
        return "complete"
    if artifact_count > 0:
        return "partial"
    return "planned"


def _render_presence(exists_flag: object, path_value: object) -> str:
    path_text = str(path_value).strip()
    if not path_text:
        return ""
    return f"`{path_text}`" if bool(exists_flag) else f"missing: `{path_text}`"


def _resolve_project_root_from_manifest(manifest_path: Path) -> Path:
    return manifest_path.resolve().parents[3]


def main() -> None:
    args = parse_args()
    ensure_quarterly_workflow_directories()
    manifest_path = Path(args.manifest)
    manifest = load_workflow_manifest(manifest_path)
    steps = load_benchmark_steps(manifest)
    ladder_df = build_benchmark_ladder_frame(steps, _resolve_project_root_from_manifest(manifest_path))
    print(ladder_df[["ladder_step", "title", "status"]].to_string(index=False))
    if args.write_artifacts:
        diagnostics_csv_path, benchmark_log_path = write_quarterly_workflow_artifacts(manifest_path)
        print(f"\nWrote diagnostics CSV to: {diagnostics_csv_path}")
        print(f"Wrote benchmark log to: {benchmark_log_path}")


if __name__ == "__main__":
    main()
