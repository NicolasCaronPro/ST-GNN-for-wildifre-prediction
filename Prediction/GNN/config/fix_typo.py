import os
import json
import glob

config_dir = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/config'
files = glob.glob(os.path.join(config_dir, '*burnedareaRoot*.json')) + glob.glob(os.path.join(config_dir, '*nbsinister*.json'))

count = 0
for filepath in files:
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            continue
            
    if data.get('name') == 'defualt_2024':
        data['name'] = 'default_2024'
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        count += 1

print(f"Fixed typo in {count} files.")
