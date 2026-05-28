# Sentiment-Augmented Quarterly Stock Prediction Poster (V2.1)

This directory contains the V2.1 source files for the academic poster, utilizing a hybrid two-column layout on a 48x36 inch landscape canvas.

## Export Instructions (Manual PDF)
1. Open `index.html` in Chrome or Edge.
2. Press `Ctrl+P` (or `Cmd+P`) to open the print dialog.
3. Configure the following settings:
   - **Destination**: Save as PDF
   - **Layout**: Landscape
   - **Paper Size**: Arch E (or Custom: 48in x 36in)
   - **Margins**: None
   - **Scale**: 100%
   - **Options**: Check "Background graphics"

## Export Instructions (Automated Playwright)
An automated Python script is provided to generate a high-quality PDF.

Requirements:
```bash
pip install playwright
playwright install chromium
```

Run the script from the root directory:
```bash
python scripts/export_poster_v2_1_pdf.py
```
This will generate `poster_v2_1.pdf` in the root directory.
