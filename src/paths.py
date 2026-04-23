"""Common filesystem paths used across the project.

Keeping paths in one file makes the code easier to read and reduces
hard-coded strings in later scripts and notebooks.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"

DAILY_CONFIGS_DIR = CONFIGS_DIR / "daily"
DAILY_DOCS_DIR = DOCS_DIR / "daily"
QUARTERLY_CONFIGS_DIR = CONFIGS_DIR / "quarterly"
QUARTERLY_DOCS_DIR = DOCS_DIR / "quarterly"
QUARTERLY_OUTPUTS_DIR = OUTPUTS_DIR / "quarterly"
QUARTERLY_OUTPUTS_PANELS_DIR = QUARTERLY_OUTPUTS_DIR / "panels"
QUARTERLY_OUTPUTS_LABELS_DIR = QUARTERLY_OUTPUTS_DIR / "labels"
QUARTERLY_OUTPUTS_VALIDATION_DIR = QUARTERLY_OUTPUTS_DIR / "validation"
QUARTERLY_OUTPUTS_FEATURES_DIR = QUARTERLY_OUTPUTS_DIR / "feature_sets"
QUARTERLY_OUTPUTS_EXPERIMENTS_DIR = QUARTERLY_OUTPUTS_DIR / "experiments"
QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR = QUARTERLY_OUTPUTS_DIR / "diagnostics"
QUARTERLY_OUTPUTS_CHAMPIONS_DIR = QUARTERLY_OUTPUTS_DIR / "champions"


def ensure_directories() -> None:
    """Create the main project folders if they do not already exist."""
    for folder in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        OUTPUTS_DIR,
        DOCS_DIR,
        CONFIGS_DIR,
        REPORTS_DIR,
        DAILY_CONFIGS_DIR,
        DAILY_DOCS_DIR,
        QUARTERLY_CONFIGS_DIR,
        QUARTERLY_DOCS_DIR,
        QUARTERLY_OUTPUTS_DIR,
        QUARTERLY_OUTPUTS_PANELS_DIR,
        QUARTERLY_OUTPUTS_LABELS_DIR,
        QUARTERLY_OUTPUTS_VALIDATION_DIR,
        QUARTERLY_OUTPUTS_FEATURES_DIR,
        QUARTERLY_OUTPUTS_EXPERIMENTS_DIR,
        QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR,
        QUARTERLY_OUTPUTS_CHAMPIONS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
