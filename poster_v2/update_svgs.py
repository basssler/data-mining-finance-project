import os
import re

svg_dir = r"C:\Users\maxba\Documents\GitHub\data-mining-finance-project\poster_v2\assets"

color_map = {
    # Typography
    r'#0B1F3A': '#0f172a',
    r'#334E68': '#475569',
    r'#52606D': '#475569',
    
    # Fonts
    r'font-family="Arial,Helvetica,sans-serif"': 'font-family="Inter, Arial, sans-serif"',
    r'Arial,Helvetica,sans-serif': 'Inter, Arial, sans-serif',
    r'Arial': 'Inter',
    
    # Accents
    r'#059669': '#10b981',
    r'#D97706': '#f59e0b'
}

for filename in os.listdir(svg_dir):
    if not filename.endswith('.svg'):
        continue
        
    filepath = os.path.join(svg_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for old, new in color_map.items():
        content = re.sub(old, new, content, flags=re.IGNORECASE)
        
    if filename == "purged_validation_timeline.svg":
        # Check if the 2024 Holdout label exists, if not add it
        if "2024 Holdout</text>" not in content or "Holdout" not in content.split("Holdout", 1)[-1]:
             # Add label under the bar
             if 'text-anchor="middle">Validation</text>' in content:
                 content = content.replace(
                     '<text class="s" x="773" y="128" text-anchor="middle">Validation</text>',
                     '<text class="s" x="773" y="128" text-anchor="middle">Validation</text>\n    <text class="s" x="895" y="128" text-anchor="middle">2024 Holdout</text>'
                 )
        # To avoid clipping, make the SVG wider or move things
        content = re.sub(r'width="980" height="245" viewBox="0 0 980 245"', r'width="1000" height="245" viewBox="0 0 1000 245"', content)
        content = re.sub(r'<rect width="980" height="245" rx="22" fill="#F8FAFC"/>', r'<rect width="1000" height="245" rx="22" fill="#F8FAFC"/>', content)
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
print("Updated all SVGs.")
