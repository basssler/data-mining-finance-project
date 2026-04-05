# Layer 1 Stock Direction Project

This repository contains the first layer of a supervised learning course project for stock direction prediction.

Layer 1 focuses only on financial statement features pulled from SEC EDGAR data between `2015-01-01` and `2024-12-31`.

## Project Scope

- Problem type: binary classification
- Current layer: financial statement features only
- Primary source: SEC EDGAR, preferably through `edgartools`
- Future target label: `1` if 5-day forward return is greater than 0, else `0`
- Current goal: build a clean and modular data pipeline before modeling

## Folder Overview

- `data/raw/`: original downloaded files or untouched source extracts
- `data/interim/`: partially cleaned or joined datasets
- `data/processed/`: final modeling tables
- `notebooks/`: exploratory notebooks
- `src/`: reusable Python modules for the project
- `outputs/`: charts, metrics, and saved reports
- `docs/`: short notes, project decisions, and documentation

## Suggested Setup on Windows (VS Code)

1. Open this folder in VS Code.
2. Create a virtual environment:

```powershell
python -m venv .venv
```

3. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Example Import Style

When running scripts from the project root, you can import modules like this:

```python
from src.paths import RAW_DATA_DIR
from src.config import START_DATE
```

## Current Status

This is a minimal starter structure. The files in `src/` contain lightweight placeholders and docstrings so each step can be built and tested gradually.
