import shutil
import unittest
from pathlib import Path

from src.quarterly_workflow import (
    BenchmarkStep,
    build_benchmark_ladder_frame,
    build_benchmark_log_markdown,
    ensure_quarterly_workflow_directories,
    load_benchmark_steps,
)


class QuarterlyWorkflowTests(unittest.TestCase):
    def test_load_benchmark_steps_parses_required_fields(self) -> None:
        manifest = {
            "steps": [
                {
                    "ladder_step": 2,
                    "slug": "quarterly-baseline",
                    "title": "Quarterly event baseline",
                    "objective": "Run the current quarterly benchmark.",
                    "config_path": "configs/event_panel_v2_quarterly.yaml",
                }
            ]
        }

        steps = load_benchmark_steps(manifest)

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].slug, "quarterly-baseline")
        self.assertEqual(steps[0].config_path, "configs/event_panel_v2_quarterly.yaml")

    def test_build_benchmark_ladder_frame_marks_complete_partial_and_planned(self) -> None:
        temp_dir = Path("outputs") / "test_tmp" / "quarterly_workflow_case"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            root = temp_dir
            (root / "configs").mkdir()
            (root / "reports" / "results").mkdir(parents=True)
            (root / "configs" / "complete.yaml").write_text("x: 1\n", encoding="utf-8")
            (root / "reports" / "results" / "complete.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "reports" / "results" / "complete.md").write_text("# ok\n", encoding="utf-8")
            (root / "configs" / "partial.yaml").write_text("x: 1\n", encoding="utf-8")

            steps = [
                BenchmarkStep(
                    ladder_step=1,
                    slug="complete",
                    title="Complete rung",
                    objective="Has all artifacts.",
                    config_path="configs/complete.yaml",
                    csv_path="reports/results/complete.csv",
                    markdown_path="reports/results/complete.md",
                ),
                BenchmarkStep(
                    ladder_step=2,
                    slug="partial",
                    title="Partial rung",
                    objective="Has only config.",
                    config_path="configs/partial.yaml",
                    csv_path="reports/results/missing.csv",
                    markdown_path="reports/results/missing.md",
                ),
                BenchmarkStep(
                    ladder_step=3,
                    slug="planned",
                    title="Planned rung",
                    objective="Has no artifacts.",
                    config_path="configs/not_there.yaml",
                ),
            ]

            ladder_df = build_benchmark_ladder_frame(steps, root)

            self.assertEqual(ladder_df["status"].tolist(), ["complete", "partial", "planned"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_benchmark_log_markdown_lists_missing_steps(self) -> None:
        ladder_df = build_benchmark_ladder_frame(
            [
                BenchmarkStep(
                    ladder_step=1,
                    slug="baseline",
                    title="Baseline",
                    objective="Historical baseline.",
                    config_path="configs/a.yaml",
                )
            ],
            Path("."),
        )

        markdown = build_benchmark_log_markdown(
            {"workflow_name": "Quarterly Ladder", "updated_note": "Repo-derived."},
            ladder_df,
        )

        self.assertIn("# Quarterly Ladder", markdown)
        self.assertIn("Step 1: `Baseline` is `planned`.", markdown)

    def test_ensure_quarterly_workflow_directories_returns_expected_paths(self) -> None:
        directories = ensure_quarterly_workflow_directories()
        self.assertTrue(any(str(path).endswith("configs\\quarterly\\experiments") for path in directories))
        self.assertTrue(any(str(path).endswith("outputs\\quarterly\\diagnostics") for path in directories))


if __name__ == "__main__":
    unittest.main()
