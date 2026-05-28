import asyncio
import os
from playwright.async_api import async_playwright

async def export_pdf():
    # 48in x 36in at 96 DPI
    width = 48 * 96
    height = 36 * 96

    poster_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "poster_v2_1", "index.html"))
    file_url = f"file:///{poster_path.replace(chr(92), '/')}"
    
    output_pdf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "poster_v2_1.pdf"))

    print(f"Loading {file_url}...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            viewport={'width': width, 'height': height}
        )
        
        await page.goto(file_url, wait_until="networkidle")
        
        print("Generating 48x36 inch PDF...")
        await page.pdf(
            path=output_pdf,
            width="48in",
            height="36in",
            print_background=True,
            page_ranges="1"
        )
        
        await browser.close()
        
    print(f"Success! PDF saved to: {output_pdf}")

if __name__ == "__main__":
    asyncio.run(export_pdf())
