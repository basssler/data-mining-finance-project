"""Project-wide settings for the Layer 1 fundamentals pipeline.

This file stores simple constants so the rest of the project can
import one source of truth for dates, file names, and default options.
"""

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Layer 1 only: financial statement features from SEC EDGAR.
PROJECT_LAYER = "layer_1_fundamentals"
PRIMARY_DATA_SOURCE = "sec_edgar"

# A small starter universe for testing can be added later.
DEFAULT_TICKERS = []
