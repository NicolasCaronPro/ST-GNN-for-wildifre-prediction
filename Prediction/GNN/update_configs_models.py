import json
import os
import glob
from pathlib import Path

def update_config_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        changed = False
        
        # Remove from root if exists
        if 'loss_param_search' in data:
            del data['loss_param_search']
            changed = True
            
        # Add to models
        if 'models' in data and isinstance(data['models'], list):
            for model in data['models']:
                if isinstance(model, dict):
                    # Check if key exists (even if False) to avoid overwriting if manually set differently (though unlikely here)
                    # But wait, I want to ADD it if missing.
                    if 'loss_param_search' not in model:
                        model['loss_param_search'] = False
                        changed = True
        
        if changed:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            #print(f"Updated {filepath}")
        else:
            #print(f"No changes needed for {filepath}")
            pass
            
    except Exception as e:
        print(f"Error updating {filepath}: {e}")

def main():
    # Use current working directory or specific path
    base_dir = Path('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN')
    config_dirs = ['config', 'config_firemen']
    
    count = 0
    for config_dir in config_dirs:
        dir_path = base_dir / config_dir
        # json_files = glob.glob(str(dir_path / '*.json'))
        # Recursive globs might be better if there were subdirs, but list_dir showed flat structure mostly.
        # list_dir showed no subdirs in config_firemen, but check config again.
        # config/config_article exists? No, list_dir of config showed 0 subdirs?
        # Wait, step 2485 showed "config_06_burnedareaRoot_bceloss.json" etc.
        # list_dir output: "Summary: This directory contains 0 subdirectories and 116 files." for config.
        # "Summary: This directory contains 0 subdirectories and 45 files." for config_firemen.
        
        json_files = list(dir_path.glob('*.json'))
        
        print(f"Found {len(json_files)} in {config_dir}")
        for json_file in json_files:
            update_config_file(json_file)
            count += 1
            
    print(f"Processed {count} files.")

if __name__ == "__main__":
    main()
