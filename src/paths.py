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


def ensure_directories() -> None:
    """Create the main project folders if they do not already exist."""
    for folder in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        OUTPUTS_DIR,
        DOCS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
