import argparse
import subprocess
import sys
from pathlib import Path

from GNN.config_parser import ConfigParser

SCRIPT_BY_TYPE = {
    "STGNN": "train_STGNN.py",
    "GRU": "train_gru.py",
    "TREES": "train_trees.py",
}


def main():
    parser = argparse.ArgumentParser(
        prog="train_any_model",
        description="Entry point to train any supported model using a config file",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/config.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    cfg = ConfigParser(args.config)
    models = cfg.get("models", [])

    for model in models:
        script = model.get("script")
        if not script:
            model_type = str(model.get("type", "")).upper()
            script = SCRIPT_BY_TYPE.get(model_type)
        if not script:
            print(f"Skipping model because no script or known type was provided: {model}")
            continue

        script_path = Path(__file__).parent / script
        cmd = [sys.executable, str(script_path), "-c", args.config]
        print("Executing", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"Training {model.get('name', model.get('type', 'unknown'))} failed with exit code {exc.returncode}"
            )


if __name__ == "__main__":
    main()
