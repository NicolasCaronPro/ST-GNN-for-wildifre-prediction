import argparse
from pathlib import Path

from GNN.construct import init
from GNN.dataloader import (
    get_train_val_test_set,
    wrapped_train_deep_learning_1D,
    wrapped_train_deep_learning_1D_federated,
    test_sklearn_api_model,
    test_dl_model,
)
from GNN.train import wrapped_train_sklearn_api_model
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

TREE_MODELS = {
    "xgboost",
    "lightgbm",
    "rf",
    "dt",
    "svm",
    "ngboost",
    "catboost",
    "lg",
    "gam",
    "poisson",
    "ordered",
}

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

    tree_model_names = []
    dl_model_names = []

    for m in cfg.get("models", []):
        info = (
            f"{m['under_sampling']}_{m['over_sampling']}_{m['kdays']}"
            f"_{m.get('nbfeatures', 'all')}_one_{m['target']}_{m['task']}_{m['loss']}"
        )
        model_name = f"{m['type']}_{info}"
        is_tree = m["type"].lower() in TREE_MODELS

        if is_tree:
            model_tuple = (model_name, None, None, None, m.get("n_run", 1))
            wrapped_train_sklearn_api_model(
                train_dataset=train_dataset.copy(deep=True),
                val_dataset=val_dataset.copy(deep=True),
                test_dataset=test_dataset.copy(deep=True),
                model=model_tuple,
                graph_method=cfg.graph_method,
                dir_output=dir_output / f"check_{cfg.scaling}/{prefix}/baseline",
                device="gpu",
                features=features_selected_str,
                autoRegression=False,
                training_mode=cfg.training_mode,
                do_grid_search=cfg.GridSearch,
                do_bayes_search=cfg.BayesSearch,
                scale=graphScale.scale,
            )
            tree_model_names.append(model_name)
        else:
            params = dict(global_params)
            params.update(
                {
                    "model": m["type"],
                    "infos": info,
                    "out_channels": m["out_channels"],
                    "n_run": m["n_run"],
                    "custom_model_params": m.get("params"),
                    "k_days": m.get("kdays", 0),
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
                params["federated_cluster"] = m.get("federated_cluster", "department")
                params["aggregation_method"] = "median"
                wrapped_train_deep_learning_1D_federated(params)
            else:
                wrapped_train_deep_learning_1D(params)

            dl_model_names.append(model_name)

    if cfg.doTest:
        dir_train = Path(name_dir)
        dir_test = Path(f"{dataset_name}/{sinister}/{resolution}/test/{cfg.sinisterEncoding}_{name_exp}")
        prefix_config = prefix
        prefix_kmeans = f"{cfg.nbpoint}_{graphScale.scale}_{graphScale.base}_{graphScale.graph_method}"

        if tree_model_names:
            test_sklearn_api_model(
                cfg,
                graphScale,
                test_dataset,
                test_dataset_unscale,
                "all",
                prefix,
                prefix_config,
                tree_model_names,
                dir_test / "all" / prefix,
                device,
                encoding,
                cfg.scaling,
                test_dataset.departement.unique(),
                dir_train,
                name_exp,
                dir_train / "check_none" / prefix_kmeans / "kmeans",
                cfg.KMEANS,
            )

        if dl_model_names:
            test_dl_model(
                cfg,
                graphScale,
                test_dataset,
                test_dataset_unscale,
                train_dataset_unscale,
                "all",
                features_selected_str,
                prefix,
                prefix_config,
                dl_model_names,
                dir_test / "all" / prefix,
                device,
                encoding,
                cfg.scaling,
                ["all"],
                dir_train,
                features_selected_str,
                dir_train / "check_none" / prefix_kmeans / "kmeans",
                name_exp,
                cfg.KMEANS,
                "temp",
            )


if __name__ == "__main__":
    main()
