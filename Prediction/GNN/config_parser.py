import json
from pathlib import Path

class ConfigParser:
    """Simple JSON configuration reader exposing keys as attributes."""

    def __init__(self, config_path: str):
        path = Path(config_path)
        with open(path, "r") as f:
            data = json.load(f)

        # expose raw data for flexible access
        self._data = data
        self.path = path
        self.name_config = self.path.name

        self.features = ["temp", "dwpt", "rhum", "prcp", "wdir", "wspd", "prec24h",
        "dc", "ffmc", "dmc", "nesterov", "munger", "kbdi",
        "isi", "angstroem", "bui", "fwi", "dailySeverityRating",
        "temp16", "dwpt16", "rhum16", "prcp16", "wdir16", "wspd16", "prec24h16",
        "days_since_rain", "sum_consecutive_rainfall",
        "sum_rain_last_7_days",
        "sum_snow_last_7_days", "snow24h", "snow24h16",
        "precipitationIndexN3", "precipitationIndexN5", "precipitationIndexN7",
        "elevation",
        "population",
        "sentinel",
        "foret_encoder",
        #"argile_encoder",
        "corine_encoder",
        "cluster_encoder",
        "bdroute_encoder",
        "id_encoder",
        "Geo",
        "foret",
        "bdroute",
        "corine",
        "Calendar"
      ]

        # commonly used fields
        self.train_features = data.get("train_features", [])
        self.kmeans_features = data.get("kmeans_features", self.features)
        self.train_years = data.get("train_years", [])
        self.val_years = data.get("val_years", [])
        self.test_years = data.get("test_years", [])
        self.train_departments = data.get("train_departments", [])
        self.test_departments = data.get("test_departments", [])
        
        self.graphConstruct = data.get("graphConstruct", None)
        self.graphConstruct = None if self.graphConstruct == "None" else self.graphConstruct

        # hyperparameters section
        self.hyperparameters = data.get("hyperparameters", {})
        self.epochs = self.hyperparameters.get("epochs")
        self.batch_size = self.hyperparameters.get("batch_size")
        self.lr = self.hyperparameters.get("lr")
        self.PATIENCE_CNT = self.hyperparameters.get("PATIENCE_CNT")
        self.CHECKPOINT = self.hyperparameters.get("CHECKPOINT")
        self.delta_lr = self.hyperparameters.get("delta_lr")
        self.PATIENCE_CNT_LR = self.hyperparameters.get("PATIENCE_CNT_LR")

        # expose all remaining entries as attributes
        for key, value in data.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def get(self, key, default=None):
        return self._data.get(key, default)
