import argparse
from pathlib import Path
import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory into sys.path
sys.path.insert(0, parent_dir)

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
from GNN.tools import check_and_create_path, get_features_name_list, read_object
from GNN.features import add_past_risk
import numpy as np
import pandas as pd

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
    QUICK = bool(cfg.get("quick", False))

    dataset_name = cfg.dataset
    sinister = cfg.sinister
    resolution = cfg.resolution
    name_exp = cfg.name

    name_dir = f"{dataset_name}/{sinister}/{resolution}/train/"
    dir_output = Path(name_dir)
    check_and_create_path(dir_output)

    if not QUICK:
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
        features_selected_str.append("Past_risk")
        features_selected_str.append("Past_burnedarea")
        features_selected = np.arange(len(features_selected_str))

        train_dataset = add_past_risk(
            train_dataset,
            "nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "risk",
        )
        test_dataset = add_past_risk(
            test_dataset,
            "nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "risk",
        )
        val_dataset = add_past_risk(
            val_dataset,
            "nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "risk",
        )

        train_dataset = add_past_risk(
            train_dataset,
            "burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "burnedarea",
        )
        test_dataset = add_past_risk(
            test_dataset,
            "burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "burnedarea",
        )
        val_dataset = add_past_risk(
            val_dataset,
            "burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "burnedarea",
        )
    else:
        prefix = f"full_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{cfg.graphConstruct}_{cfg.graph_method}"
        graphScale = read_object(f"graph_{cfg.scale}_{cfg.graphConstruct}_{cfg.graph_method}.pkl", dir_output)

        dir_output = dir_output / f"{cfg.sinisterEncoding}_{name_exp}"

        train_dataset = read_object(f"df_train_{prefix}.pkl", dir_output)
        val_dataset = read_object(f"df_val_{prefix}.pkl", dir_output)
        test_dataset = read_object(f"df_test_{prefix}.pkl", dir_output)

        train_dataset_unscale = read_object(f"df_unscaled_train_{prefix}.pkl", dir_output)
        val_dataset_unscale = read_object(f"df_unscaled_val_{prefix}.pkl", dir_output)
        test_dataset_unscale = read_object(f"df_unscaled_test_{prefix}.pkl", dir_output)

        features_selected_str = read_object(
            "features_importance.pkl",
            dir_output
            / "features_importance"
            / f"{cfg.nbpoint}_{getattr(cfg, 'k_days', 0)}_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{graphScale.base}_{graphScale.graph_method}",
        )
        if features_selected_str is not None:
            features_selected_str = list(np.asarray(features_selected_str)[:, 0])
        else:
            features_name, _ = get_features_name_list(graphScale.scale, cfg.features, METHODS_SPATIAL_TRAIN)
            features_selected_str = list(features_name)

        features_selected_str.append("Past_risk")
        features_selected_str.append("Past_burnedarea")
        features_selected = np.arange(len(features_selected_str))

        train_dataset = add_past_risk(
            train_dataset,
            "nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "risk",
        )
        test_dataset = add_past_risk(
            test_dataset,
            "nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "risk",
        )
        val_dataset = add_past_risk(
            val_dataset,
            "nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "risk",
        )

        train_dataset = add_past_risk(
            train_dataset,
            "burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "burnedarea",
        )
        test_dataset = add_past_risk(
            test_dataset,
            "burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "burnedarea",
        )
        val_dataset = add_past_risk(
            val_dataset,
            "burnedareaDaily-kmeans-5-Class-Dept-cubic-Specialized-Past",
            "burnedarea",
        )

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

    prefix = f"full_all_{cfg.NbFeatures}_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{cfg.graphConstruct}_{cfg.graph_method}"

    tree_model_names = []
    dl_model_names = []

    if cfg.doTrain:
        for m in cfg.get("models", []):
            info = (
                f"{m['under_sampling']}_{m['over_sampling']}_{m['kdays']}"
                f"_{m.get('nbfeatures', 'all')}_one_{m['target']}_{m['task']}_{m['loss']}"
            )
            model_name = f"{m['type']}_{info}"
            is_tree = m["type"].lower() in TREE_MODELS

            if is_tree:
                name = 'check_'+cfg.scaling + '/' + prefix + '/' + 'baseline'
                model_tuple = (model_name, None, None, None, m.get("n_run", 1))
                wrapped_train_sklearn_api_model(
                    train_dataset=train_dataset.copy(deep=True),
                    val_dataset=val_dataset.copy(deep=True),
                    test_dataset=test_dataset.copy(deep=True),
                    model=model_tuple,
                    graph_method=cfg.graph_method,
                    dir_output=dir_output / name,
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
                name = 'check_'+cfg.scaling + '/' + prefix + '/'
                params = dict(global_params)
                params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "dir_output" : dir_output / name
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
        host = "pc"
        dir_train = Path(name_dir)
        dir_test = Path(
            f"{dataset_name}/{sinister}/{resolution}/test/{cfg.sinisterEncoding}_{name_exp}"
        )
        prefix_config = prefix
        prefix_kmeans = (
            f"{cfg.nbpoint}_{graphScale.scale}_{graphScale.base}_{graphScale.graph_method}"
        )

        df_metrics = None

        if tree_model_names:
            metrics, _, _, _ = test_sklearn_api_model(
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
            df_metrics = pd.DataFrame.from_dict(metrics, orient="index").reset_index()

        if dl_model_names:
            metrics, _, _, _ = test_dl_model(
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
            if df_metrics is None:
                df_metrics = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
            else:
                df_metrics = pd.concat(
                    (df_metrics, pd.DataFrame.from_dict(metrics, orient="index").reset_index())
                )

        for dept in test_dataset.departement.unique(),:
            test_dataset_dept = test_dataset[
                test_dataset["departement"] == name2int[dept]
            ].reset_index(drop=True)
            if test_dataset_unscale is not None:
                test_dataset_unscale_dept = test_dataset_unscale[
                    test_dataset_unscale["departement"] == name2int[dept]
                ].reset_index(drop=True)
            else:
                test_dataset_unscale_dept = None

            if test_dataset_dept.shape[0] < 5:
                continue

            if tree_model_names:
                metrics, _, _, _ = test_sklearn_api_model(
                    cfg,
                    graphScale,
                    test_dataset_dept,
                    test_dataset_unscale_dept,
                    dept,
                    prefix,
                    prefix_config,
                    tree_model_names,
                    dir_test / dept / prefix,
                    device,
                    encoding,
                    cfg.scaling,
                    [dept],
                    dir_train,
                    name_exp,
                    dir_train / "check_none" / prefix_kmeans / "kmeans",
                    cfg.KMEANS,
                )
                if df_metrics is None:
                    df_metrics = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
                else:
                    df_metrics = pd.concat(
                        (
                            df_metrics,
                            pd.DataFrame.from_dict(metrics, orient="index").reset_index(),
                        )
                    )

            if dl_model_names:
                metrics, _, _, _ = test_dl_model(
                    cfg,
                    graphScale,
                    test_dataset_dept,
                    test_dataset_unscale_dept,
                    train_dataset_unscale,
                    dept,
                    features_selected_str,
                    prefix,
                    prefix_config,
                    dl_model_names,
                    dir_test / dept / prefix,
                    device,
                    encoding,
                    cfg.scaling,
                    [dept],
                    dir_train,
                    features_selected_str,
                    dir_train / "check_none" / prefix_kmeans / "kmeans",
                    name_exp,
                    cfg.KMEANS,
                    "temp",
                )
                if df_metrics is None:
                    df_metrics = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
                else:
                    df_metrics = pd.concat(
                        (
                            df_metrics,
                            pd.DataFrame.from_dict(metrics, orient="index").reset_index(),
                        )
                    )

        if df_metrics is not None:
            df_metrics.rename({"index": "Run"}, inplace=True, axis=1)
            df_metrics.reset_index(drop=True, inplace=True)
            check_and_create_path(dir_test / prefix)
            df_metrics.to_csv(dir_test / prefix / "df_metrics_any.csv")

if __name__ == "__main__":
    main()
