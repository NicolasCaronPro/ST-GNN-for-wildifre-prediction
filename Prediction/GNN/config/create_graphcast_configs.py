import json
import os

config_dir = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/config'
base_nbsinister = os.path.join(config_dir, 'config_06_nbsinister_flwk.json')
base_burnedarea = os.path.join(config_dir, 'config_06_burnedareaRoot_flwk.json')

with open(base_nbsinister, 'r') as f:
    data_nb = json.load(f)

with open(base_burnedarea, 'r') as f:
    data_ba = json.load(f)

for data, target_str in [(data_nb, 'nbsinister-kmeans-5-Class-Dept'), (data_ba, 'burnedareaRoot-kmeans-5-Class-Dept')]:
    data['name'] = 'default'
    
    # GraphCastGRU models with loss flwk, pdegpd, flwk
    base_model = data['models'][0].copy() if data['models'] else {
        "type": "GRU", "kdays": 10, "nbfeatures": "all", "target": target_str,
        "task": "classification", "loss": "flwk", "out_channels": 5, "n_run": 1,
        "horizon": 0, "use_log": False, "params": {}, "use_temporal_as_edges": False,
        "mesh_file": None, "under_sampling": "search", "over_sampling": "full",
        "aggregation_method": "weighted", "federated_cluster": "cluster",
        "params_to_update": ["linear2"], "loss_param_search": False
    }
    
    model1 = base_model.copy()
    model1['type'] = 'GraphCastGRU'
    model1['loss'] = 'flwk'
    model1['target'] = target_str
    
    model2 = base_model.copy()
    model2['type'] = 'GraphCastGRU'
    model2['loss'] = 'pdegpd'
    model2['target'] = target_str
    
    model3 = base_model.copy()
    model3['type'] = 'GraphCastGRU'
    model3['loss'] = 'flwk'
    model3['target'] = target_str
    
    data['models'] = [model1, model2, model3]

with open(os.path.join(config_dir, 'config_graphcast_gru_nbsinister.json'), 'w') as f:
    json.dump(data_nb, f, indent=4)

with open(os.path.join(config_dir, 'config_graphcast_gru_burnedareaRoot.json'), 'w') as f:
    json.dump(data_ba, f, indent=4)

print("Created config_graphcast_gru_nbsinister.json and config_graphcast_gru_burnedareaRoot.json")
