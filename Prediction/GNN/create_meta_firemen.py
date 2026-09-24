import json
import os

targets = ["nbsinister", "ressource", "timeintervention"]

for target in targets:
    base_file = f"config_firemen/config_06_{target}_ranknet.json"
    if not os.path.exists(base_file):
        print(f"File not found: {base_file}")
        continue
        
    with open(base_file, 'r') as f:
        data = json.load(f)
        
    data["training_mode"] = "meta"
    
    for model in data["models"]:
        model["meta_task_column"] = "departement"
        model["meta_known_departments"] = "full"
        model["meta_random_tasks"] = True
        model["meta_n_tasks_per_iteration"] = "full"
        model["meta_query_ratio"] = 0.5
        model["meta_inner_lr"] = 0.01
        model["meta_inner_steps"] = 1
        model["meta_seed"] = 42
        
    new_file = f"config_firemen/config_06_{target}_meta_ranknet.json"
    with open(new_file, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Created {new_file}")
