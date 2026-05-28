import asyncio
import os
from playwright.async_api import async_playwright

async def export_png():
    # 48in x 36in at 96 DPI base, scaled up for high res
    width = 48 * 96
    height = 36 * 96

    poster_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "poster_v2_1", "index.html"))
    file_url = f"file:///{poster_path.replace(chr(92), '/')}"
    
    output_png = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "poster_v2_1.png"))

    print(f"Loading {file_url}...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Scale factor 3 gives ~300 DPI for high quality review
        page = await browser.new_page(
            viewport={'width': width, 'height': height},
            device_scale_factor=3
        )
        
        await page.goto(file_url, wait_until="networkidle")
        
        print("Generating 48x36 inch PNG...")
        await page.screenshot(
            path=output_png,
            full_page=True
        )
        
        await browser.close()
        
    print(f"Success! PNG saved to: {output_png}")

if __name__ == "__main__":
    asyncio.run(export_png())
