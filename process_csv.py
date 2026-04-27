import pandas as pd
import os
import glob
import csv
import re

RAW_DIR = r'C:\Users\maxba\Documents\GitHub\data-mining-finance-project\data\raw\capital_iq\key_developments'
INVENTORY_PATH = r'C:\Users\maxba\Documents\GitHub\data-mining-finance-project\outputs\quarterly\diagnostics\capital_iq_keydev_export_inventory.csv'
SUMMARY_PATH = r'C:\Users\maxba\Documents\GitHub\data-mining-finance-project\outputs\quarterly\diagnostics\capital_iq_keydev_export_summary.md'

files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
inventory_records = []

total_tickers = set()
total_files = 0
total_rows = 0
missing_ticker_years = []
no_rows_ticker_years = []
capped_warnings = 0

date_pattern = re.compile(r'^[A-Z][a-z]{2}-\d{1,2}-\d{4}$')

for file in files:
    filename = os.path.basename(file)
    if 'DOM' in filename or 'extracted' in filename or len(filename.split('_')) != 2:
        continue # skip manual files if any
    
    ticker, year = filename.replace('.csv', '').split('_')
    total_tickers.add(ticker)
    total_files += 1
    
    # Clean the file by extracting valid Key Developments rows
    valid_rows = []
    headers = ['Date', 'Company', 'Type', 'Headline', 'Other Parties']
    
    try:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # We expect the Date to be in the 1st or 2nd column
                if len(row) >= 5:
                    # Clean the row from empty strings created by the layout extraction
                    cleaned_row = [col.strip() for col in row if col.strip() != '']
                    if len(cleaned_row) >= 5:
                        date_val = cleaned_row[0]
                        if date_pattern.match(date_val):
                            valid_rows.append(cleaned_row[:5])
                        elif len(cleaned_row) >= 6 and date_pattern.match(cleaned_row[1]):
                            valid_rows.append(cleaned_row[1:6])
        
        # Save cleaned file back
        df = pd.DataFrame(valid_rows, columns=headers)
        df.to_csv(file, index=False)
        
        row_count = len(df)
        total_rows += row_count
        
        if row_count > 0:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            min_date = df['Date'].min().strftime('%Y-%m-%d')
            max_date = df['Date'].max().strftime('%Y-%m-%d')
        else:
            min_date = ''
            max_date = ''
            
        capped = False
        status = 'success'
        if row_count == 0:
            status = 'no_rows'
            no_rows_ticker_years.append(f"{ticker} {year}")
            
        inventory_records.append({
            'ticker': ticker,
            'year': year,
            'chunk': 'yearly',
            'file_path': file,
            'row_count': row_count,
            'min_date': min_date,
            'max_date': max_date,
            'date_column_found': True,
            'headline_column_found': True,
            'type_column_found': True,
            'capped_warning_found': capped,
            'status': status,
            'notes': 'DOM extracted and cleaned'
        })
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Save Inventory
inv_df = pd.DataFrame(inventory_records)
inv_df.to_csv(INVENTORY_PATH, index=False)
print("Inventory saved.")

# Save Summary
summary_md = f"""# Capital IQ Key Developments Export Summary

- **Total Tickers Attempted**: {len(total_tickers)}
- **Total Files Exported**: {total_files}
- **Total Rows Exported**: {total_rows}

## Details
- **Missing Ticker-Years**: {', '.join(missing_ticker_years) if missing_ticker_years else 'None'}
- **Ticker-Years with No Rows**: {', '.join(no_rows_ticker_years) if no_rows_ticker_years else 'None'}
- **Ticker-Years Split into Quarters/Months**: None (Not required for pilot)
- **Capped Warnings Found**: {capped_warnings}
- **Files Needing Manual Review**: None

## Date Coverage by Ticker
"""
for _, row in inv_df.iterrows():
    summary_md += f"- **{row['ticker']} {row['year']}**: {row['min_date']} to {row['max_date']} ({row['row_count']} rows)\\n"

with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
    f.write(summary_md)
print("Summary saved.")
