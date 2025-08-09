import argparse
import subprocess
import sys
from pathlib import Path
import os
import sys
import itertools

import sys
import itertools
import tempfile
import json  # ou json selon ton format
import shutil
from pathlib import Path

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)

from GNN.config_parser import ConfigParser

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_temp_config(config_data):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w')
    json.dump(config_data, tmp_file)
    tmp_file.close()
    return tmp_file.name

def main():
    parser = argparse.ArgumentParser(
        prog="train_model_from_config",
        description="Train multiple models based on a JSON configuration file",
    )
    parser.add_argument(
        "-c", "--config", default="config/config.json", help="Path to config file"
    )
    args = parser.parse_args()

    cfg = ConfigParser(args.config)
    models = cfg.get("models", [])
    script = cfg.get("train_script")
    
    if not script:
        print(f"No script provided")
    
    script_path = Path(__file__).parent / script
    
    print(cfg.scale, cfg.graphConstruct)
    
    if cfg.scale == 'search':
        scales = [2, 3, 4, 'departement']
        for scale in scales:
            print(f'{scale}')
            cfg.scale = scale
            if cfg.graphConstruct == 'search':
                if scale == 'departement':
                    updated_cfg = load_config(args.config)
                    updated_cfg['scale'] = scale
                    updated_cfg['graphConstruct'] = "None"

                    temp_config_path = save_temp_config(updated_cfg)
                    cmd = [sys.executable, str(script_path), "-c", temp_config_path]
                    try:
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as exc:
                        print(f"failed with exit code {exc.returncode}")
                else:
                    nb_attemps = [2, 3, 4, 5]
                    n_reduce_class = [3, 4 , 5, 6]
                    test = itertools.product(nb_attemps, n_reduce_class)
                    for a, r in test:
                        updated_cfg = load_config(args.config)
                        updated_cfg['scale'] = scale
                        updated_cfg['graphConstruct'] = f'risk-size-watershed-degree-a{a}-r{r}-t0.3'
                        
                        print(f'risk-size-watershed-degree-a{a}-r{r}-t0.3')
                        
                        temp_config_path = save_temp_config(updated_cfg)
                        cmd = [sys.executable, str(script_path), "-c", temp_config_path]
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as exc:
                            print(f"failed with exit code {exc.returncode}")
            else:
                updated_cfg = load_config(args.config)
                updated_cfg['scale'] = scale
                updated_cfg['graphConstruct'] = cfg.graphConstruct

                temp_config_path = save_temp_config(updated_cfg)
                cmd = [sys.executable, str(script_path), "-c", temp_config_path]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"failed with exit code {exc.returncode}")
    else:
        cmd = [sys.executable, str(script_path), "-c", args.config]

        print("Executing", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"failed with exit code {exc.returncode}")


if __name__ == "__main__":
    main()
