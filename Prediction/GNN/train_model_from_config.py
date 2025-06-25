import argparse
import subprocess
import sys
from pathlib import Path
from GNN.config_parser import ConfigParser


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

    for m in models:
        script = m.get("script")
        if not script:
            print(f"Skipping model {m.get('name')} - no script provided")
            continue
        script_path = Path(__file__).parent / script
        cmd = [sys.executable, str(script_path), "-c", args.config]

        print("Executing", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Model {m.get('name')} failed with exit code {exc.returncode}")


if __name__ == "__main__":
    main()
