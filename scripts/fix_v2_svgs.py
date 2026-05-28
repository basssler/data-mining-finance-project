import os
import re

svg_dir = r"C:\Users\maxba\Documents\GitHub\data-mining-finance-project\poster_v2\assets"

for filename in os.listdir(svg_dir):
    if not filename.endswith('.svg'):
        continue
        
    filepath = os.path.join(svg_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Fix Validation clipping
    if filename == "purged_validation_timeline.svg":
        # Check if already added
        if "2024 Holdout</text>" not in content or "Holdout" not in content.split("Holdout", 1)[-1]:
             if 'text-anchor="middle">Validation</text>' in content:
                 content = content.replace(
                     '<text class="s" x="773" y="128" text-anchor="middle">Validation</text>',
                     '<text class="s" x="773" y="128" text-anchor="middle">Validation</text>\n    <text class="s" x="915" y="128" text-anchor="middle">2024 Holdout</text>'
                 )
        content = re.sub(r'width="[0-9]+" height="245" viewBox="0 0 [0-9]+ 245"', r'width="1060" height="245" viewBox="0 0 1060 245"', content)
        content = re.sub(r'<rect width="[0-9]+" height="245" rx="22" fill="#F8FAFC"/>', r'<rect width="1060" height="245" rx="22" fill="#F8FAFC"/>', content)
        
    # Fix Ladder clipping
    if filename == "experiment_ladder.svg":
        content = re.sub(r'width="[0-9]+" height="490" viewBox="0 0 [0-9]+ 490"', r'width="1150" height="490" viewBox="0 0 1150 490"', content)
        content = re.sub(r'<rect width="[0-9]+" height="490" rx="22" fill="#F8FAFC"/>', r'<rect width="1150" height="490" rx="22" fill="#F8FAFC"/>', content)
            
    # Fix Leakage clipping
    if filename == "leakage_alignment_comparison.svg":
        content = re.sub(r'width="[0-9]+" height="360" viewBox="0 0 [0-9]+ 360"', r'width="1020" height="360" viewBox="0 0 1020 360"', content)
        content = re.sub(r'<rect width="[0-9]+" height="360" rx="22" fill="#F8FAFC"/>', r'<rect width="1020" height="360" rx="22" fill="#F8FAFC"/>', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
print("Fixed internal clipping in V2 SVGs.")
