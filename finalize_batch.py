import pandas as pd
import os
import glob
import re
from datetime import datetime

RAW_DIR = r'C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\raw\capital_iq\key_developments'
INVENTORY_PATH = r'C:\Users\maxba\Documents\GitHub\data-mining-finance-project\outputs\quarterly\diagnostics\capital_iq_keydev_export_inventory.csv'
SUMMARY_PATH = r'C:\Users\maxba\Documents\GitHub\data-mining-finance-project\outputs\quarterly\diagnostics\capital_iq_keydev_export_summary.md'

def clean_html_to_df(html_file, source_ticker, year):
    try:
        dfs = pd.read_html(html_file)
        target_df = None
        for df in dfs:
            if df.shape[1] >= 5:
                # Find header row
                header_idx = -1
                for i, row in df.iterrows():
                    row_vals = [str(v) for v in row.values]
                    if 'Date' in row_vals and 'Headline' in row_vals:
                        header_idx = i
                        break
                if header_idx != -1:
                    target_df = df.iloc[header_idx+1:].copy()
                    header_row = df.iloc[header_idx].tolist()
                    col_map = {}
                    for j, val in enumerate(header_row):
                        val_str = str(val).strip()
                        if val_str in ['Date', 'Company', 'Type', 'Headline', 'Other Parties', 'Situation']:
                            col_map[j] = val_str
                    target_df = target_df[list(col_map.keys())]
                    target_df.columns = [col_map[j] for j in target_df.columns]
                    break
        
        if target_df is None:
            return None
            
        # Clean rows
        target_df = target_df.dropna(subset=['Date'])
        date_pattern = re.compile(r'[A-Z][a-z]{2}-\d{1,2}-\d{4}')
        target_df = target_df[target_df['Date'].astype(str).str.contains(date_pattern, na=False)]
        
        # Build Schema
        # source_page_ticker, year, date, row_company, exchange_ticker, type, headline, other_parties, situation, source, direct_parent_match, extraction_method, extraction_timestamp, capital_iq_page_title, notes
        
        final_rows = []
        timestamp = datetime.now().isoformat()
        
        for _, row in target_df.iterrows():
            row_company_raw = str(row.get('Company', ''))
            
            # Extract exchange_ticker e.g. "NasdaqGS:WMT" from "Walmart Inc. (NasdaqGS:WMT)"
            ticker_match = re.search(r'\(([^)]+:[^)]+)\)', row_company_raw)
            exchange_ticker = ticker_match.group(1) if ticker_match else ""
            
            # Clean company name
            row_company = re.sub(r'\s*\([^)]+:[^)]+\)', '', row_company_raw).strip()
            
            # Direct parent match logic
            # Match if the exchange_ticker contains the source_ticker, or if the row_company matches
            # Handle BF-B specifically
            clean_source = source_ticker.replace('-', '.') # BF-B -> BF.B
            is_parent = False
            if exchange_ticker and clean_source in exchange_ticker:
                is_parent = True
            elif clean_source.lower() in row_company.lower():
                is_parent = True
            
            final_rows.append({
                'source_page_ticker': source_ticker,
                'year': year,
                'date': row.get('Date', ''),
                'row_company': row_company,
                'exchange_ticker': exchange_ticker,
                'type': row.get('Type', ''),
                'headline': row.get('Headline', ''),
                'other_parties': row.get('Other Parties', ''),
                'situation': row.get('Situation', ''),
                'source': 'Capital IQ Key Developments',
                'direct_parent_match': is_parent,
                'extraction_method': 'DOM visible table extraction after UI date filter',
                'extraction_timestamp': timestamp,
                'capital_iq_page_title': f"{source_ticker} Key Developments",
                'notes': ''
            })
            
        return pd.DataFrame(final_rows)
    except Exception as e:
        print(f"Error parsing {html_file}: {e}")
        return None

def update_diagnostics(processed_records):
    # Update Inventory
    if os.path.exists(INVENTORY_PATH):
        inv_df = pd.read_csv(INVENTORY_PATH)
    else:
        inv_df = pd.DataFrame()
        
    new_inv_df = pd.concat([inv_df, pd.DataFrame(processed_records)]).drop_duplicates(subset=['ticker', 'year'], keep='last')
    new_inv_df.to_csv(INVENTORY_PATH, index=False)
    
    # Update Summary
    total_tickers = new_inv_df['ticker'].nunique()
    total_files = len(new_inv_df)
    total_rows = new_inv_df['row_count'].sum()
    
    summary_md = f"""# Capital IQ Key Developments Export Summary

- **Total Tickers Attempted**: {total_tickers}
- **Total Files Exported**: {total_files}
- **Total Rows Exported**: {total_rows}

## Date Coverage by Ticker
"""
    for _, row in new_inv_df.sort_values(['ticker', 'year']).iterrows():
        summary_md += f"- **{row['ticker']} {row['year']}**: {row['min_date']} to {row['max_date']} ({row['row_count']} rows)\n"

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write(summary_md)

if __name__ == "__main__":
    # This script will be called after a batch finishes
    html_files = glob.glob(os.path.join(RAW_DIR, '*.html'))
    records = []
    for hf in html_files:
        filename = os.path.basename(hf)
        ticker, year = filename.replace('.html', '').split('_')
        
        df = clean_html_to_df(hf, ticker, year)
        if df is not None:
            csv_path = hf.replace('.html', '.csv')
            df.to_csv(csv_path, index=False)
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            records.append({
                'ticker': ticker,
                'year': year,
                'chunk': 'yearly',
                'file_path': csv_path,
                'row_count': len(df),
                'min_date': df['date'].min().strftime('%Y-%m-%d') if len(df) > 0 else '',
                'max_date': df['date'].max().strftime('%Y-%m-%d') if len(df) > 0 else '',
                'date_column_found': True,
                'headline_column_found': True,
                'type_column_found': True,
                'pagination_detected': False,
                'pages_extracted': 1,
                'capped_warning_found': False,
                'status': 'success' if len(df) > 0 else 'no_rows',
                'notes': ''
            })
            # os.remove(hf) # Optionally remove HTML to save space
            
    update_diagnostics(records)
    print("Batch finalized.")
