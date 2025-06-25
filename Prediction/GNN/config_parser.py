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

        # commonly used fields
        self.features = data.get("features", [])
        self.kmeans_features = data.get("kmeans_features", self.features)
        self.train_years = data.get("train_years", [])
        self.val_years = data.get("val_years", [])
        self.test_years = data.get("test_years", [])
        self.train_departments = data.get("train_departments", [])
        self.test_departments = data.get("test_departments", [])

        # expose all remaining entries as attributes
        for key, value in data.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def get(self, key, default=None):
        return self._data.get(key, default)
