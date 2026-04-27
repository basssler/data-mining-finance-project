import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def analyze_file(filepath):
    print(f'\nAnalyzing: {filepath}')
    try:
        if filepath.endswith('.xls'):
            dfs = pd.read_html(filepath)
            df = dfs[0]
            header_idx = df[df.iloc[:, 0] == 'Date'].index[0]
            df = df.iloc[header_idx+1:].copy()
            df.columns = dfs[0].iloc[header_idx]
            df = df.dropna(subset=['Date'])
            print(f'Total Rows: {len(df)}')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            print(f'Min Date: {df["Date"].min().date()}')
            print(f'Max Date: {df["Date"].max().date()}')
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            header_idx = next(i for i, line in enumerate(lines) if line.startswith('Date,'))
            df = pd.read_csv(filepath, skiprows=header_idx)
            print(f'Total Rows: {len(df)}')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            print(f'Min Date: {df["Date"].min().date()}')
            print(f'Max Date: {df["Date"].max().date()}')
    except Exception as e:
        print(f'Error: {e}')

analyze_file('C:/Users/maxba/Downloads/Walmart Inc NasdaqGS WMT Key Developments.xls')
analyze_file('C:/Users/maxba/Downloads/Walmart Inc NasdaqGS WMT Key Developments.csv')
