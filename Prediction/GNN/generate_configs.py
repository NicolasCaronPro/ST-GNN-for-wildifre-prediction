import json
import os

base_configs = [
    ("config/config_06_burnedareaRoot_bceloss.json", "config/"),
    ("config/config_06_nbsinister_bceloss.json", "config/"),
    ("config_firemen/config_06_nbsinister_bceloss.json", "config_firemen/"),
    ("config_firemen/config_06_ressource_bceloss.json", "config_firemen/"),
    ("config_firemen/config_06_timeintervention_bceloss.json", "config_firemen/")
]

new_losses = ["cll", "lmol", "prls"]

for base_path, out_dir in base_configs:
    if not os.path.exists(base_path):
        print(f"Base config not found: {base_path}")
        continue
        
    with open(base_path, 'r') as f:
        data = json.load(f)
        
    base_filename = os.path.basename(base_path)
    
    for loss in new_losses:
        new_filename = base_filename.replace("bceloss", loss)
        new_path = os.path.join(out_dir, new_filename)
        
        new_data = data.copy()
        
        # Modify model params
        if "models" in new_data:
            for model in new_data["models"]:
                model["loss"] = loss
                model["out_channels"] = 1
                model["task"] = "regression"
                # Remove loss parameter search if present, or set to False?
                # The user didn't specify, but `bceloss` base has it false.
        
        with open(new_path, 'w') as f:
            json.dump(new_data, f, indent=4)
        print(f"Created {new_path}")
