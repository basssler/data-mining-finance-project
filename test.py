import pandas as pd
from src.universe import get_layer1_tickers
df = pd.read_parquet("data/interim/fundamentals/fundamentals_quarterly_clean.parquet")
print(sorted(set(get_layer1_tickers()) - set(df["ticker"].dropna().astype(str).unique())))