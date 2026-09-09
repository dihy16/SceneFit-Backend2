import json
import os
import numpy as np
from PIL import Image

def analyze_outfits():
    with open('manifest.json', 'r', encoding='utf-8') as f:
        m = json.load(f)
    outfits = m['outfits']
    scenes = m['scenes']
    
    outfit_analysis = []
    for i, o in enumerate(outfits):
        img_path = os.path.join('data/drive_data/data/2d', os.path.basename(o['path']))
        img = Image.open(img_path).convert('RGBA').resize((150, 150))
        arr = np.array(img)
        alpha = arr[:, :, 3]
        fg_mask = alpha > 10
        if not np.any(fg_mask):
            mean_rgb = (0, 0, 0)
            brightness = 0
            sat = 0
        else:
            fg_rgb = arr[fg_mask][:, :3]
            mean_rgb = fg_rgb.mean(axis=0)
            brightness = float(mean_rgb.mean())
            max_c = fg_rgb.max(axis=1).astype(float)
            min_c = fg_rgb.min(axis=1).astype(float)
            sat = float(np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-5), 0).mean())
            
        r, g, b = [int(x) for x in mean_rgb]
        outfit_analysis.append({
            'index': i,
            'id': o['id'],
            'sha256': o['sha256'],
            'path': o['path'],
            'rgb': (r, g, b),
            'brightness': round(brightness, 1),
            'saturation': round(sat, 2)
        })
        
    print(f"Analyzed {len(outfit_analysis)} outfits.")
    with open('results/outfit_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(outfit_analysis, f, indent=2)

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    analyze_outfits()
