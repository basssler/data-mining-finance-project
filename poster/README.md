# Data Mining Finance Project Poster

This folder contains a fixed-size, print-ready HTML/CSS academic poster:

- `index.html`
- `styles.css`
- `assets/*.svg`

Poster size: 48 inches wide by 36 inches tall, landscape.

## Option 1: Browser Export

1. Open `poster/index.html` in Chrome or Edge.
2. Press `Ctrl+P`.
3. Destination: Save as PDF.
4. Layout: Landscape.
5. Paper size: 48in x 36in if available, otherwise custom.
6. Margins: None.
7. Scale: 100%.
8. Background graphics: enabled.
9. Save as `data_mining_finance_project_poster.pdf`.

## Option 2: Playwright Export

From the repository root, run:

```powershell
python scripts/export_poster_pdf.py
```

The script writes:

```text
poster/poster.pdf
```

Before printing, open the PDF at 100% zoom and confirm all body text is readable.
