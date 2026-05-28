import asyncio
import os
from playwright.async_api import async_playwright

async def export_png():
    # Calculate dimensions: 24in x 36in at 300 DPI for high quality print
    # Using standard 96 DPI for web viewport but scaling up for PNG
    width = 24 * 96
    height = 36 * 96

    poster_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "poster_v2", "index.html"))
    file_url = f"file:///{poster_path.replace(chr(92), '/')}"
    
    output_png = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "poster_v2.png"))

    print(f"Loading {file_url}...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # We need to set deviceScaleFactor to 3 or 4 to get a high-res image suitable for print (300dpi)
        # 96dpi * 3.125 = 300dpi
        page = await browser.new_page(
            viewport={'width': width, 'height': height},
            device_scale_factor=3
        )
        
        # Load the HTML file and wait for it to be completely ready
        await page.goto(file_url, wait_until="networkidle")
        
        print("Generating 24x36 inch PNG...")
        # Print to PNG
        await page.screenshot(
            path=output_png,
            full_page=True
        )
        
        await browser.close()
        
    print(f"Success! PNG saved to: {output_png}")

if __name__ == "__main__":
    asyncio.run(export_png())
