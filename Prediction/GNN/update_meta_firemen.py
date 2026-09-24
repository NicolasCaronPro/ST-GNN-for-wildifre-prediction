import json
import os

targets = ["nbsinister", "ressource", "timeintervention"]

for target in targets:
    new_file = f"config_firemen/config_06_{target}_meta_ranknet.json"
    if not os.path.exists(new_file):
        print(f"File not found: {new_file}")
        continue
        
    with open(new_file, 'r') as f:
        data = json.load(f)
    
    # Keep only the GRU model and set horizon to 0
    gru_models = [m for m in data.get("models", []) if m.get("type") == "GRU"]
    for m in gru_models:
        m["horizon"] = 0
        
    data["models"] = gru_models
    
    with open(new_file, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Updated {new_file}")
