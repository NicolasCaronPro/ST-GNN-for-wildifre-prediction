import argparse
from pathlib import Path

from GNN.construct import init
from GNN.dataloader import (
    get_train_val_test_set,
    wrapped_train_deep_learning_1D,
    wrapped_train_deep_learning_1D_federated,
)
from GNN.config_parser import ConfigParser
from GNN.config import (
    device,
    PATIENCE_CNT,
    CHECKPOINT,
    Rewrite,
    encoding,
    METHODS_SPATIAL_TRAIN,
)
from GNN.tools import check_and_create_path, get_features_name_list
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog="train_any_model",
        description="Generic training entry point using a JSON config",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/config.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    cfg = ConfigParser(args.config)

    dataset_name = cfg.dataset
    sinister = cfg.sinister
    resolution = cfg.resolution
    name_exp = cfg.name

    name_dir = f"{dataset_name}/{sinister}/{resolution}/train/"
    dir_output = Path(name_dir)
    check_and_create_path(dir_output)

    df, graphScale, prefix, fp, features_selected = init(cfg, dir_output, "train_any")
    dir_output = dir_output / f"{cfg.sinisterEncoding}_{name_exp}"

    train_dataset, val_dataset, test_dataset, train_dataset_unscale, val_dataset_unscale, test_dataset_unscale, prefix, features_selected = get_train_val_test_set(
        graphScale,
        df,
        features_selected,
        cfg.train_departments,
        prefix,
        dir_output,
        cfg,
        cfg,
    )

    features_name, _ = get_features_name_list(graphScale.scale, cfg.features, METHODS_SPATIAL_TRAIN)
    features_selected_str = list(features_name)
    features_selected = np.arange(len(features_selected_str))

    global_params = {
        "graphScale": graphScale,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "features_selected": features_selected,
        "features_selected_str": features_selected_str,
        "device": device,
        "optimize_feature": cfg.optimizeFeature,
        "PATIENCE_CNT": PATIENCE_CNT,
        "CHECKPOINT": CHECKPOINT,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "scaling": cfg.scaling,
        "encoding": encoding,
        "prefix": prefix,
        "Rewrite": Rewrite,
        "dir_output": dir_output,
        "train_dataset_unscale": train_dataset_unscale,
        "graph": graphScale,
        "name_dir": name_dir,
        "graph_method": cfg.graph_method,
    }

    for m in cfg.get("models", []):
        info = f"{m['under_sampling']}_{m['over_sampling']}_{m['kdays']}_all_one_{m['target']}_{m['task']}_{m['loss']}"
        params = dict(global_params)
        params.update(
            {
                "model": m["type"],
                "infos": info,
                "out_channels": m["out_channels"],
                "n_run": m["n_run"],
                "custom_model_params": m.get("params"),
            }
        )

        if m.get("mesh_file"):
            params["mesh_file"] = m.get("mesh_file")
            params["use_temporal_as_edges"] = m.get("use_temporal_as_edges")
            params["torch_structure"] = "Model_gnn"
        else:
            params["mesh_file"] = None
            params["use_temporal_as_edges"] = None
            params["torch_structure"] = "Model_Torch"

        if cfg.training_mode == "federated":
            params["federated_cluster"] = m.get("federated_cluster", "departement")
            params["aggregation_method"] = "median"
            wrapped_train_deep_learning_1D_federated(params)
        else:
            wrapped_train_deep_learning_1D(params)


if __name__ == "__main__":
    main()
