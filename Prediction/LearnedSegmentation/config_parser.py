import json
from pathlib import Path

class ConfigParser:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get_train_departements(self):
        return self.config.get("train_departements", [])

    def get_test_departements(self):
        return self.config.get("test_departements", [])

    def get_new_test_departements(self):
        return self.config.get("new_test_departements", [])

    def get_train_flag(self):
        return self.config.get("train", True)

    def get_test_flag(self):
        return self.config.get("test", True)

    def get_frequency(self):
        return self.config.get("frequency", "full")

    def get_pipeline_params(self):
        return self.config.get("pipeline_params", {})

    def get_model_params(self):
        return self.config.get("model_params", {})

    def get_graph_flag(self):
        return self.config.get("graph", True)

    def get_target_variable(self):
        return self.config.get("target_variable", "risk")

    def get_target_type(self):
        return self.config.get("target_type", "binary")

    def get_scale(self):
        return self.config.get("scale", None)

    def get_dataset_name(self):
        return self.config.get("dataset_name", None)

    def get_graph_construct(self):
        return self.config.get("graph_construct", self.config.get("graphConstruct", None))

    def get_scale(self):
        return self.config.get("scale", None)
    
    def get_base(self):
        return self.config.get("base", None)

    def get_load_flag(self):
        return self.config.get("load", False)

    def get_n_clusters_node(self):
        return self.config.get("n_clusters_node", 4)
