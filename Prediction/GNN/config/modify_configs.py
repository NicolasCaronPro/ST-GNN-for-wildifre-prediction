import os
import json
import glob

config_dir = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/config'

files = glob.glob(os.path.join(config_dir, '*burnedareaRoot*.json')) + glob.glob(os.path.join(config_dir, '*nbsinister*.json'))

for filepath in files:
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except:
            continue
            
    modified = False
    
    if 'name' in data:
        data['name'] = 'defualt_2024'
        modified = True
        
    if 'models' in data and isinstance(data['models'], list):
        new_models = [m for m in data['models'] if m.get('type') not in ['LSTM', 'DilatedCNN']]
        if len(new_models) != len(data['models']):
            data['models'] = new_models
            modified = True
            
    if modified:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
print(f"Processed {len(files)} files.")
