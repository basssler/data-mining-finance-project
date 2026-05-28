# Finance Project Poster V2

This is the updated, print-ready, HTML/CSS-based poster for the Sentiment-Augmented Quarterly Stock Prediction project.

## Poster Specifications
- **Size:** 48 inches wide × 36 inches tall
- **Orientation:** Landscape
- **Format:** HTML/CSS (Print to PDF)
- **Style:** Clean academic finance style with an emphasis on readable typography and the central pipeline visual.

## Exporting the Poster to PDF

### Option 1: Manual Browser Export
1. Open `index.html` in Chrome or Edge.
2. Press `Ctrl+P` (or `Cmd+P` on Mac) to open the Print dialog.
3. Apply the following settings:
   - **Destination:** Save as PDF
   - **Layout:** Landscape
   - **Paper Size:** Custom (if available) or Arch E / ANSI E (make sure it maps to 48x36 aspect ratio). The CSS `@page` rule enforces the 48x36in size automatically in modern browsers.
   - **Margins:** None
   - **Scale:** Default or 100%
   - **Options:** **Enable "Background graphics"** (Crucial for cards and colored boxes to appear).
4. Save the PDF.

### Option 2: Automated Export via Playwright
If you have Playwright installed, you can use the provided script to generate the PDF perfectly every time.
```bash
pip install playwright
playwright install chromium
python scripts/export_poster_v2_pdf.py
```
*(The script will output `poster_v2.pdf` in the root directory)*
