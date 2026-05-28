from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
POSTER_HTML = ROOT / "poster" / "index.html"
OUTPUT_PDF = ROOT / "poster" / "poster.pdf"


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1920, "height": 1440}, device_scale_factor=1)
        page.goto(POSTER_HTML.as_uri(), wait_until="networkidle")
        layout = page.evaluate(
            """
            () => {
                const poster = document.querySelector('.poster').getBoundingClientRect();
                const nodes = [...document.querySelectorAll('.card, .header, .grid, .column, img')];
                const overflow = nodes
                    .map((node) => {
                        const rect = node.getBoundingClientRect();
                        return {
                            tag: node.tagName.toLowerCase(),
                            cls: node.className || node.getAttribute('src') || '',
                            bottom: Math.round(rect.bottom - poster.bottom),
                            right: Math.round(rect.right - poster.right)
                        };
                    })
                    .filter((item) => item.bottom > 1 || item.right > 1);
                return {
                    posterWidth: Math.round(poster.width),
                    posterHeight: Math.round(poster.height),
                    scrollWidth: document.documentElement.scrollWidth,
                    scrollHeight: document.documentElement.scrollHeight,
                    overflow
                };
            }
            """
        )
        page.pdf(
            path=str(OUTPUT_PDF),
            width="48in",
            height="36in",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            prefer_css_page_size=True,
        )
        browser.close()
    print(
        "layout="
        f"{layout['posterWidth']}x{layout['posterHeight']}px "
        f"scroll={layout['scrollWidth']}x{layout['scrollHeight']}px "
        f"overflow_count={len(layout['overflow'])}"
    )
    if layout["overflow"]:
        print(layout["overflow"])
    print(f"wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
