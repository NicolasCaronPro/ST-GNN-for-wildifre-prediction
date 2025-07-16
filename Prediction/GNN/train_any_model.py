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
    wrapped_train_sklearn_api_and_pytorch_voting_model,
    test_sklearn_api_model,
    test_dl_model,
    wrapped_train_deep_learning_1D_moonfederated,
    wrapped_train_deep_learning_1D_splittraining,
    wrapped_train_deep_learning_1D_protofederated,
    wrapped_train_deep_learning_1D_unique,
    wrapped_train_deep_learning_1D_alafederated
)
from GNN.train import wrapped_train_sklearn_api_model, wrapped_train_sklearn_api_voting_model, define_voting_dl_models
from GNN.config_parser import ConfigParser
from GNN.config import (
    device,
    PATIENCE_CNT,
    CHECKPOINT,
    Rewrite,
    encoding,
    METHODS_SPATIAL_TRAIN,
)
from GNN.tools import check_and_create_path, get_features_name_list, read_object, save_object
from GNN.discretization import post_process_model
from GNN.features import add_past_risk
from GNN.dico_departements import *
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

        train_dataset, val_dataset, test_dataset, train_dataset_unscale, val_dataset_unscale, test_dataset_unscale, prefix, features_selected_str = get_train_val_test_set(
            graphScale,
            df,
            features_selected,
            cfg.train_departments,
            prefix,
            dir_output,
            cfg,
            cfg,
        )

        dir_post_process = dir_output / 'post_process'
        post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)
        save_object(train_dataset, f"df_train_{prefix}.pkl", dir_output)
        save_object(val_dataset, f"df_val_{prefix}.pkl", dir_output)
        save_object(test_dataset, f"df_test_{prefix}.pkl", dir_output)

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

        if 'nbsinisterDaily-kmeans-5-Class-Dept-cubic-Specialized-Past' not in train_dataset.columns:
            dir_post_process = dir_output / 'post_process'
            post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)
            save_object(train_dataset, f"df_train_{prefix}.pkl", dir_output)
            save_object(val_dataset, f"df_val_{prefix}.pkl", dir_output)
            save_object(test_dataset, f"df_test_{prefix}.pkl", dir_output)

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
        
        """if "Past_risk" in cfg.features:
            features_selected_str.append("Past_risk")
        if "Past_burnedarea" in cfg.features:
            features_selected_str.append("Past_burnedarea")"""
        
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


    train_dataset['cluster-encoder'] = train_dataset['cluster_encoder']
    val_dataset['cluster-encoder'] = val_dataset['cluster_encoder']
    test_dataset['cluster-encoder'] = test_dataset['cluster_encoder']
    
    prefix = f"full_{cfg.NbFeatures}_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{cfg.graphConstruct}_{cfg.graph_method}"

    global_params = {
        "graphScale": graphScale,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "features_selected": features_selected,
        "features_selected_str": features_selected_str,
        "device": device,
        "optimize_feature": cfg.optimizeFeature,
        "PATIENCE_CNT": cfg.PATIENCE_CNT,
        "CHECKPOINT": cfg.CHECKPOINT,
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

    print(features_selected_str)

    for i, m in enumerate(cfg.get("models", [])):
        info = (
            f"{m['under_sampling']}_{m['over_sampling']}_{m['kdays']}"
            f"_{m.get('nbfeatures', 'all')}_one_{m['target']}_{m['task']}_{m['loss']}"
        )

        if cfg.training_mode == 'voting':
            voting_model = define_voting_dl_models(m['type'], m['kdays'], m['out_channels'], m['run'])[0]

        model_name = f"{m['type']}_{info}"
        is_tree = m["type"].lower() in TREE_MODELS

        if is_tree:
            name = 'check_'+cfg.scaling + '/' + prefix + '/' + 'baseline'
            model_tuple = (model_name, None, None, None, m.get("n_run", 1))
            if cfg.doTrain:
                if cfg.training_mode == "voting":
                    wrapped_train_sklearn_api_voting_model(train_dataset=train_dataset.copy(deep=True),
                                val_dataset=val_dataset.copy(deep=True),
                                test_dataset=test_dataset.copy(deep=True),
                                graph_method=cfg.graph_method,
                                dir_output=dir_output / name,
                                device='cpu',
                                autoRegression=False,
                                features=features_selected_str,
                                training_mode='normal',
                                do_grid_search=cfg.GridSearch,
                                do_bayes_search=cfg.BayesSearch,
                                model=voting_model,
                                scale=graphScale.scale)

                else:
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
                        training_mode='normal',
                        do_grid_search=cfg.GridSearch,
                        do_bayes_search=cfg.BayesSearch,
                        scale=graphScale.scale,
                    )
            if cfg.training_mode == 'normal':
                tree_model_names.append(model_name)
            elif cfg.trainin_mode == 'voting':
                config_weight = m.get('config_weight', 'soft-weight')
                num_test = m.get('num_test', [1,5,10,15,20, 'all'])
                for nt in num_test:
                    test_name = f'filter-{m["type"]}-{config_weight}-{nt}_{info}'
                    tree_model_names.append(test_name)
        else:
            params = dict(global_params)
            if cfg.doTrain:
                if m.get("mesh_file"):
                    params["mesh_file"] = m.get("mesh_file")
                    params["use_temporal_as_edges"] = m.get("use_temporal_as_edges")
                    params["torch_structure"] = "Model_gnn"
                else:
                    params["mesh_file"] = None
                    params["use_temporal_as_edges"] = None
                    params["torch_structure"] = "Model_Torch"
                    
                params["use_log"] = m.get('use_log', True)

                if cfg.training_mode == "federated":
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "dir_output" : dir_output,
                        "global_epochs" : cfg.hyperparameters['global_epochs'],
                        "patience_count_global" : cfg.hyperparameters['patience_count_global'],
                        "patience_count_local" : cfg.hyperparameters['PATIENCE_CNT'],
                        "use_log" : m.get('use_log', True)
                    }
                    )
                    params["federated_cluster"] = m.get("federated_cluster", "department")
                    params["aggregation_method"] = m.get('aggregation_method', "median")
                    wrapped_train_deep_learning_1D_federated(params)

                elif cfg.training_mode == "moonfederated":
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "temperature" : m.get('temperature'),
                        "smooth" : m.get('smooth'),
                        "dir_output" : dir_output,
                        "global_epochs" : cfg.hyperparameters['global_epochs'],
                        "patience_count_global" : cfg.hyperparameters['patience_count_global'],
                        "patience_count_local" : cfg.hyperparameters['PATIENCE_CNT'],
                        "use_log" : m.get('use_log', True)
                    }
                    )
                    params["federated_cluster"] = m.get("federated_cluster", "department")
                    params["aggregation_method"] = m.get('aggregation_method', "median")
                    wrapped_train_deep_learning_1D_moonfederated(params)
                
                elif cfg.training_mode == "alafederated":
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "temperature" : m.get('temperature'),
                        "smooth" : m.get('smooth'),
                        "dir_output" : dir_output,
                        "global_epochs" : cfg.hyperparameters['global_epochs'],
                        "patience_count_global" : cfg.hyperparameters['patience_count_global'],
                        "patience_count_local" : cfg.hyperparameters['PATIENCE_CNT'],
                        "use_log" : m.get('use_log', True)

                    }
                    )
                    params["federated_cluster"] = m.get("federated_cluster", "department")
                    params["aggregation_method"] = m.get('aggregation_method', "median")
                    params["eta"] = m.get('eta', 1.0)
                    params["layer_idx"] = m.get('layer_idx', 0)
                    wrapped_train_deep_learning_1D_alafederated(params)
                
                elif cfg.training_mode == "protofederated":
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "temperature" : m.get('temperature'),
                        "smooth" : m.get('smooth'),
                        "dir_output" : dir_output,
                        "global_epochs" : cfg.hyperparameters['global_epochs'],
                        "patience_count_global" : cfg.hyperparameters['patience_count_global'],
                        "patience_count_local" : cfg.hyperparameters['PATIENCE_CNT'],
                        "use_log" : m.get('use_log', True)

                    }
                    )
                    params["federated_cluster"] = m.get("federated_cluster", "department")
                    wrapped_train_deep_learning_1D_protofederated(params)

                elif cfg.training_mode == "splittraining":
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "temperature" : m.get('temperature'),
                        "smooth" : m.get('smooth'),
                        "dir_output" : dir_output,
                        "global_epochs" : cfg.hyperparameters['global_epochs'],
                        "patience_count_global" : cfg.hyperparameters['patience_count_global'],
                        "input_server_model": m.get('input_server_model'),
                        "use_log" : m.get('use_log', True)

                    }
                    )
                    params["federated_cluster"] = m.get("federated_cluster", "department")
                    params["cut_layer_name"] = m.get("cut_layer", None)
                    assert params["cut_layer_name"] is not None
                    wrapped_train_deep_learning_1D_splittraining(params)

                elif cfg.training_mode == 'voting':
                    params.update(
                    {
                        "model": voting_model,
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "dir_output" : dir_output,
                        "use_log" : m.get('use_log', True)

                    }
                    )
                    wrapped_train_sklearn_api_and_pytorch_voting_model(params)
                elif cfg.training_mode == 'unique':
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "dir_output" : dir_output,
                        "global_epochs" : cfg.hyperparameters['global_epochs'],
                        "patience_count_global" : cfg.hyperparameters['patience_count_global'],
                        "use_log" : m.get('use_log', True)
                    }
                    )
                    params["cluster"] = m.get("cluster", "department")
                    params["sub_training_mode"] = m.get("sub_training_mode", "normal")

                    if params["sub_training_mode"] == 'normal':
                        pass
                    elif params["sub_training_mode"] == 'federated':
                        params['federated_cluster'] = m.get('federated_cluster')
                        params['aggregation_method'] = m.get('aggregation_method')

                    elif params["sub_training_mode"] == 'moonfederated':
                        params['federated_cluster'] = m.get('federated_cluster')
                        params['aggregation_method'] = m.get('aggregation_method')
                        params["temperature"] = m.get('temperature')
                        params["smooth"] = m.get('smooth')

                    elif params["sub_training_mode"] == 'splittraining':
                        params['federated_cluster'] = m.get('federated_cluster')
                        params['cut_layer_name'] = m.get('cut_layer')
                        assert params["cut_layer_name"] is not None

                    else:
                        raise ValueError(f'{params["sub_training_mode"]} not implemented')
                    wrapped_train_deep_learning_1D_unique(params)
                else:
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "dir_output" : dir_output,
                        "use_log" : m.get("use_log", True)
                    }
                    )
                    print(params['dir_output'])
                    wrapped_train_deep_learning_1D(params)                

            if cfg.training_mode == 'normal':
                dl_model_names.append(model_name)
            elif cfg.training_mode == 'voting':
                config_weight = m.get('config_weight', 'soft-weight')
                num_test = m.get('num_test', [1,5,10,15,20, 'all'])
                for nt in num_test:
                    test_name = f'filter-{m["type"]}-{config_weight}-{nt}_{info}'
                    dl_model_names.append(test_name)
            elif cfg.training_mode == 'federated':
                test_name = f'federated-{m["type"]}-{m.get("federated_cluster", "department")}-{m.get("aggregation_method", "median")}_{info}'
                dl_model_names.append(test_name)
            elif cfg.training_mode == 'protofederated':
                test_name = f'Protofederated-{m["type"]}-{m.get("federated_cluster", "department")}_{info}'
                dl_model_names.append(test_name)
            elif cfg.training_mode == 'moonfederated':
                test_name = f'moonfederated-{m["type"]}-{m.get("federated_cluster", "department")}-{m.get("aggregation_method", "median")}_{info}'
                dl_model_names.append(test_name)
            elif cfg.training_mode == 'alafederated':
                test_name = f'alafederated-{m["type"]}-{m.get("federated_cluster", "department")}-{m.get("aggregation_method", "median")}_{info}'
                dl_model_names.append(test_name)
            elif cfg.training_mode == 'splittraining':
                test_name = f'SplitTraining-{m["type"]}-{m.get("federated_cluster", "department")}_{info}'
                dl_model_names.append(test_name)

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
                f'{cfg.sinisterEncoding}_{name_exp}',
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
                f'{cfg.sinisterEncoding}_{name_exp}',
                cfg.KMEANS,
                "temp",
            )
            if df_metrics is None:
                df_metrics = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
            else:
                df_metrics = pd.concat(
                    (df_metrics, pd.DataFrame.from_dict(metrics, orient="index").reset_index())
                )

        if cfg.doTestDepartement:
            for dept in cfg.test_departments:
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
                        f'{cfg.sinisterEncoding}_{name_exp}',
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
                        f'{cfg.sinisterEncoding}_{name_exp}',
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
            df_metrics['training_mode'] = cfg.training_mode
            check_and_create_path(dir_test / prefix)
            #if (dir_test / prefix / "df_metrics_any.csv").is_file():
            #    log_metrics = pd.read_csv(dir_test / prefix / "df_metrics_any.csv")
            #    df_metrics = pd.concat((log_metrics, df_metrics)).reset_index(drop=True)
            df_metrics.to_csv(dir_test / prefix / f"df_metrics_any_{cfg.name_config}.csv")

if __name__ == "__main__":
    main()