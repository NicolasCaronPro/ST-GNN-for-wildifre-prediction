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
    wrapped_train_deep_learning_1D_alafederated,
    wrapped_train_deep_learning_1D_dualtraining,
    wrapped_train_deep_learning_1D_distrib2classTraining,
    wrapped_train_deep_learning_2D,
    wrapped_train_deep_learning_distallation,
    test_fire_index_model
)

from GNN.statistical_model import Statistical_Model

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
from GNN.tools import (check_and_create_path, get_features_name_list,
                       read_object, save_object, get_features_selected_for_time_series_for_2D,
                       get_features_name_lists_2D, get_saison_encoding,
                       #compute_department_areas_km2_dict_wgs84_union
                       )

from GNN.discretization import post_process_model, get_post_process_model
from GNN.features import add_past_risk, shift_target
from GNN.dico_departements import *
import numpy as np
import pandas as pd
import geopandas as gpd
from GNN.tools import allDates

from tools import get_saison
from features import is_mediterranean_dept

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

        features_selected = np.arange(len(features_selected_str))
        
        train_dataset['burnedarea'] = train_dataset['burned_area'].values
        val_dataset['burnedarea'] = val_dataset['burned_area'].values
        test_dataset['burnedarea'] = test_dataset['burned_area'].values
        
        train_dataset['burnedareaRoot'] = train_dataset['burnedarea'].apply(lambda x : np.sqrt(x))
        val_dataset['burnedareaRoot'] = val_dataset['burnedarea'].apply(lambda x : np.sqrt(x))
        test_dataset['burnedareaRoot'] = test_dataset['burnedarea'].apply(lambda x : np.sqrt(x))

        post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)

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

        save_object(train_dataset, f"df_train_{prefix}.pkl", dir_output)
        save_object(val_dataset, f"df_val_{prefix}.pkl", dir_output)
        save_object(test_dataset, f"df_test_{prefix}.pkl", dir_output)
    else:
        prefix = f"full_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{cfg.graphConstruct}_{cfg.graph_method}"
        graphScale = read_object(f"graph_{cfg.scale}_{cfg.graphConstruct}_{cfg.graph_method}.pkl", dir_output)

        dir_output = dir_output / f"{cfg.sinisterEncoding}_{name_exp}"

        train_dataset = read_object(f"df_train_{prefix}.pkl", dir_output)
        val_dataset = read_object(f"df_val_{prefix}.pkl", dir_output)
        test_dataset = read_object(f"df_test_{prefix}.pkl", dir_output)
        
        train_dataset['burnedarea'] = train_dataset['burned_area'].values
        val_dataset['burnedarea'] = val_dataset['burned_area'].values
        test_dataset['burnedarea'] = test_dataset['burned_area'].values
        
        train_dataset['burnedareaRoot'] = train_dataset['burnedarea'].apply(lambda x : np.sqrt(x))
        val_dataset['burnedareaRoot'] = val_dataset['burnedarea'].apply(lambda x : np.sqrt(x))
        test_dataset['burnedareaRoot'] = test_dataset['burnedarea'].apply(lambda x : np.sqrt(x))
        
        train_dataset['area'] = 0
        val_dataset['area'] = 0
        test_dataset['area'] = 0
        #if 'area' not in train_dataset.columns:
        #if True:
        #    geo = gpd.read_file(f'regions/{sinister}/{dataset_name}/regions.geojson')
        #    areas = compute_department_areas_km2_dict_wgs84_union(geo, 'departement')
                        
        #    print(areas)

        #    train_dataset['area'] = 0.0
        #    for departement in train_dataset.departement.unique():
        #        train_dataset.loc[train_dataset[train_dataset['departement'] == departement].index, 'area'] = areas[int2name[departement]] 
        #        val_dataset.loc[val_dataset[val_dataset['departement'] == departement].index, 'area'] = areas[int2name[departement]] 
        #        test_dataset.loc[test_dataset[test_dataset['departement'] == departement].index, 'area'] = areas[int2name[departement]] 

        train_dataset['saison-encoding'] = train_dataset['date'].apply(get_saison_encoding)
        val_dataset['saison-encoding'] = val_dataset['date'].apply(get_saison_encoding)
        test_dataset['saison-encoding'] = test_dataset['date'].apply(get_saison_encoding)

        train_dataset['saison-mediterranean'] = train_dataset['saison'] + '-' + train_dataset['cluster-encoder'].astype(str)
        val_dataset['saison-mediterranean'] = val_dataset['saison'] + '-' + val_dataset['cluster-encoder'].astype(str)
        test_dataset['saison-mediterranean'] = test_dataset['saison'] + '-' + test_dataset['cluster-encoder'].astype(str)

        train_dataset['saison-cluster-encoder'] = train_dataset['saison'] + '-' + train_dataset['cluster-encoder'].astype(str)
        val_dataset['saison-cluster-encoder'] = val_dataset['saison'] + '-' + val_dataset['cluster-encoder'].astype(str)
        test_dataset['saison-cluster-encoder'] = test_dataset['saison'] + '-' + test_dataset['cluster-encoder'].astype(str)

        print('burnedareaRoot-kmeans-5-Class-Dept' not in train_dataset.columns)
        
        if 'burnedareaRoot-kmeans-5-Class-Dept' not in train_dataset.columns:
        #if True:
            dir_post_process = dir_output / 'post_process'
            post_process_model_dico, train_dataset, val_dataset, test_dataset, new_cols = post_process_model(train_dataset, val_dataset, test_dataset, dir_post_process, graphScale)
            save_object(train_dataset, f"df_train_{prefix}.pkl", dir_output)
            save_object(val_dataset, f"df_val_{prefix}.pkl", dir_output)
            save_object(test_dataset, f"df_test_{prefix}.pkl", dir_output)

        train_dataset['burnedarea-kmeans-5-Class-Dept'] = train_dataset['burned_area-kmeans-5-Class-Dept']
        val_dataset['burnedarea-kmeans-5-Class-Dept'] = val_dataset['burned_area-kmeans-5-Class-Dept']
        test_dataset['burnedarea-kmeans-5-Class-Dept'] = test_dataset['burned_area-kmeans-5-Class-Dept']
        
        train_dataset['timeintervention-kmeans-5-Class-Dept'] = train_dataset['time_intervention-kmeans-5-Class-Dept']
        val_dataset['timeintervention-kmeans-5-Class-Dept'] = val_dataset['time_intervention-kmeans-5-Class-Dept']
        test_dataset['timeintervention-kmeans-5-Class-Dept'] = test_dataset['time_intervention-kmeans-5-Class-Dept']

        train_dataset_unscale = read_object(f"df_unscaled_train_{prefix}.pkl", dir_output)
        val_dataset_unscale = read_object(f"df_unscaled_val_{prefix}.pkl", dir_output)
        test_dataset_unscale = read_object(f"df_unscaled_test_{prefix}.pkl", dir_output)

        features_selected_str = read_object(
            "features_importance.pkl",
            dir_output
            / "features_importance"
            / f"{cfg.nbpoint}_{getattr(cfg, 'k_days', 0)}_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{graphScale.base}_{graphScale.graph_method}",
        )
        print(features_selected_str)
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

        train_dataset['saison-encoding'] = train_dataset['date'].apply(get_saison_encoding)
        val_dataset['saison-encoding'] = val_dataset['date'].apply(get_saison_encoding)
        test_dataset['saison-encoding'] = test_dataset['date'].apply(get_saison_encoding)

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

        save_object(train_dataset, f"df_train_{prefix}.pkl", dir_output)
        save_object(val_dataset, f"df_val_{prefix}.pkl", dir_output)
        save_object(test_dataset, f"df_test_{prefix}.pkl", dir_output)
    
    train_dataset['nbsinister-binary'] = (train_dataset['nbsinister'] > 0).astype(int)
    val_dataset['nbsinister-binary'] = (val_dataset['nbsinister'] > 0).astype(int)
    test_dataset['nbsinister-binary'] = (test_dataset['nbsinister'] > 0).astype(int)
    
    prefix = f"full_all_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{cfg.graphConstruct}_{cfg.graph_method}"

    global_params = {
        "graphScale": graphScale,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
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
    stat_model_names = []

    for i, m in enumerate(cfg.get("models", [])):
        if m.get("type") != "fwi":
            info = (
                f"{m['under_sampling']}_{m['over_sampling']}_{m['kdays']}_{m.get('horizon', 0)}"
                f"_{m.get('nbfeatures', 'all')}_one_{m['target']}_{m['task']}_{m['loss']}"
            )

            if cfg.training_mode == 'voting':
                voting_model = define_voting_dl_models(m['type'], m['kdays'], m['horizon'], m['out_channels'], m['n_run'], m['loss'])[0]

            model_name = f"{m['type']}_{info}"
        is_tree = m["type"].lower() in TREE_MODELS
        post_process = get_post_process_model(train_dataset, 'kmeans', m['target'], 'departement', dir_log=dir_output / 'clusterers', n_clusters=5) if m.get('apply_discretization', False) else None 
        global_params['post_process'] = post_process
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
            elif cfg.training_mode == 'voting':
                config_weight = m.get('config_weight', 'soft-weight')
                num_test = m.get('num_test', list(np.arrange(1, 21)) + ['all'])
                for nt in num_test:
                    test_name = f'filter-{m["type"]}-{config_weight}-{nt}_{info}'
                    tree_model_names.append(test_name)
        elif m["type"].lower() == "fwi":
            model = Statistical_Model(m.get('column'), m.get('thresholds'), m.get('num_cluster'), m.get('target'), m.get('task'), m.get('col_id'))
            model.fit(train_dataset_unscale)
            save_object(model, f'{model.name}.pkl', dir_output / Path('check_'+cfg.scaling + '/' + prefix + '/' + 'baseline') / model.name)
            stat_model_names.append(model.name)
        else:
            params = dict(global_params)

            if cfg.training_mode == 'distrib2classtraining':
                train_dataset, _ = shift_target(train_dataset, m["target"], features_selected_str, m["task"], m['out_channels'])
                val_dataset, features_class = shift_target(val_dataset, m["target"], [], m["task"], m["out_channels"])
                test_dataset, _ = shift_target(test_dataset, m["target"], [], m["task"], m['out_channels'])
                prefix_save = f"full_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{cfg.graphConstruct}_{cfg.graph_method}"
                save_object(train_dataset, f"df_train_{prefix_save}.pkl", dir_output)
                save_object(val_dataset, f"df_val_{prefix_save}.pkl", dir_output)
                save_object(test_dataset, f"df_test_{prefix_save}.pkl", dir_output)
                features_selected = features_selected_str
                
                params["features_selected"] = features_selected
                params["features_selected_str"] = features_selected
                
                #params["features_selected_class"] = features_class
                #params["features_selected_str_class"] = features_class
                
            elif cfg.training_mode != "dualtraining":
                train_dataset, _ = shift_target(train_dataset, m["target"], features_selected_str, m["task"], m['out_channels'])
                val_dataset, _ = shift_target(val_dataset, m["target"], [], m["task"], m["out_channels"])
                test_dataset, _ = shift_target(test_dataset, m["target"], [], m["task"], m['out_channels'])
                prefix_save = f"full_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{cfg.graphConstruct}_{cfg.graph_method}"
                save_object(train_dataset, f"df_train_{prefix_save}.pkl", dir_output)
                save_object(val_dataset, f"df_val_{prefix_save}.pkl", dir_output)
                save_object(test_dataset, f"df_test_{prefix_save}.pkl", dir_output)
                features_selected = features_selected_str
                params["features_selected"] = features_selected
                params["features_selected_str"] = features_selected
            else:
                train_dataset, features_selected_str_occ = shift_target(train_dataset, m["target"], features_selected_str, m["task_occ"], m['out_channels_occ'])
                val_dataset, _ = shift_target(val_dataset, m["target"], [], m["task_occ"], m["out_channels_occ"])
                test_dataset, _ = shift_target(test_dataset, m["target"], [], m["task_occ"], m['out_channels_occ'])
                
                params["features_selected_occ"] = features_selected_str_occ
                params["features_selected_str_occ"] = features_selected_str_occ
                
                train_dataset, features_selected_str_num = shift_target(train_dataset, m["target"], features_selected_str, m["task_num"], m['out_channels_num'])
                val_dataset, _ = shift_target(val_dataset, m["target"], [], m["task_num"], m["out_channels_num"])
                test_dataset, _ = shift_target(test_dataset, m["target"], [], m["task_num"], m['out_channels_num'])

                params["features_selected_num"] = features_selected_str_num
                params["features_selected_str_num"] = features_selected_str_num

                prefix_save = f"full_{cfg.scale}_{getattr(cfg, 'days_in_futur', 0)}_{cfg.graphConstruct}_{cfg.graph_method}"
                save_object(train_dataset, f"df_train_{prefix_save}.pkl", dir_output)
                save_object(val_dataset, f"df_val_{prefix_save}.pkl", dir_output)
                save_object(test_dataset, f"df_test_{prefix_save}.pkl", dir_output)
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
                        "min_epochs": m.get('min_epochs', 1),
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "client_n_run": m.get("client_n_run", 1),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
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
                        "min_epochs": m.get('min_epochs', 1),
                        "client_n_run": m.get("client_n_run", 1),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
                        "temperature" : m.get('temperature', 1.0),
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
                        "min_epochs": m.get('min_epochs', 1),
                        "client_n_run": m.get("client_n_run", 1),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
                        "temperature" : m.get('temperature'),
                        "smooth" : m.get('smooth'),
                        "dir_output" : dir_output,
                        "global_epochs" : cfg.hyperparameters['global_epochs'],
                        "patience_count_global" : cfg.hyperparameters['patience_count_global'],
                        "patience_count_local" : cfg.hyperparameters['PATIENCE_CNT'],
                        "use_log" : m.get('use_log', True),
                        "params_to_update" : m.get("params_to_update", [])
                    }
                    )
                    assert len(m['params_to_update']) > 0
                    params["federated_cluster"] = m.get("federated_cluster", "department")
                    params["aggregation_method"] = m.get('aggregation_method', "median")
                    params["eta"] = m.get('eta', 0.1)
                    wrapped_train_deep_learning_1D_alafederated(params)
                
                elif cfg.training_mode == "protofederated":
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "min_epochs": m.get('min_epochs', 1),
                        "client_n_run": m.get("client_n_run", 1),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
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
                        "min_epochs": m.get('min_epochs', 1),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
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

                elif cfg.training_mode == "dualtraining":
                    
                    params.update(
                    {
                        "model": m["type"],
                        "task_type_num": m["task_num"],
                        "infos": info,
                        "min_epochs": m.get('min_epochs', 1),
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "target_num": m.get("target_num"),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
                        "dir_output" : dir_output,
                        "use_log" : m.get('use_log', True)

                    }
                    )
                    wrapped_train_deep_learning_1D_dualtraining(params)
                    
                elif cfg.training_mode == "distrib2classtraining":
                    
                    params.update(
                    {
                        "model": m["type"],
                        "loss_distrib" : m["loss_distrib"],
                        "loss_class" : m["loss_class"],
                        "infos": info,
                        "min_epochs": m.get('min_epochs', 1),
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
                        "dir_output" : dir_output,
                        "use_log" : m.get('use_log', True)
                    }
                    )
                    wrapped_train_deep_learning_1D_distrib2classTraining(params)

                elif cfg.training_mode == 'voting':
                    params.update(
                    {
                        "model": voting_model,
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "min_epochs": m.get('min_epochs', 1),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
                        "dir_output" : dir_output,
                        "use_log" : m.get('use_log', True)

                    }
                    )
                    wrapped_train_sklearn_api_and_pytorch_voting_model(
                                            train_dataset, val_dataset, test_dataset,
                                            voting_model, cfg.graph_method,
                                            dir_output,
                                            False,
                                            'normal',
                                            False,
                                            False,
                                            cfg.scale,
                                            params)
                    
                elif cfg.training_mode == 'unique':
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "min_epochs": m.get('min_epochs', 1),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
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
                elif cfg.training_mode == 'distillation':
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "min_epochs": m.get('min_epochs', 1),
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
                        "dir_output" : dir_output,
                        "use_log" : m.get("use_log", True),
                        "image_per_node" : m.get('image_per_node', None),
                        "name_exp" : name_exp
                    })

                    params['teacher_name'] = m.get('teacher', None)
                    params['teacher_loss'] = m.get('teacher_loss', None)
                    params['alpha'] = m.get('alpha', None)
                    params['temperature'] = m.get('temperature', None)
                    params['distillation_training_mode'] = m.get('distillation_training_mode', None)
                    assert params['teacher_name'] is not None
                    wrapped_train_deep_learning_distallation(params)

                else:
                    params.update(
                    {
                        "model": m["type"],
                        "infos": info,
                        "min_epochs": m.get('min_epochs', 1),
                        "out_channels": m["out_channels"],
                        "n_run": m["n_run"],
                        "custom_model_params": m.get("params"),
                        "k_days": m.get("kdays", 0),
                        "horizon": m.get("horizon", 0),
                        "dir_output" : dir_output,
                        "use_log" : m.get("use_log", True),
                        "image_per_node" : m.get('image_per_node', None),
                        "name_exp" : name_exp
                    }
                    )
                    if m.get('type') in ['ResNet', 'ConvLSTM']:
                        params['torch_structure'] = 'Model_CNN'
                        features_name_2D, newShape2D = get_features_name_lists_2D(6, cfg.train_features)
                        features_selected_str_2D = get_features_selected_for_time_series_for_2D(features_selected_str, features_name_2D, [], 'all')
                        params['features_name_2D'] = features_selected_str_2D
                        params['features_name_1D'] = features_selected_str
                        params['features'] = cfg.features
                        params['train_features'] = cfg.train_features
                        wrapped_train_deep_learning_2D(params)
                    else:
                        wrapped_train_deep_learning_1D(params)                

            if cfg.training_mode == 'normal':
                dl_model_names.append(model_name)
            elif cfg.training_mode == 'voting':
                config_weight = m.get('config_weight', 'soft-weight')
                num_test = m.get('num_test', [1,5,10,15,20, 'all', 'task'])
            
                for nt in num_test:
                    test_name = f'filter-{m["type"]}-{config_weight}-{nt}_{info}'
                    dl_model_names.append(test_name)
            
            elif cfg.training_mode == 'distillation':
                test_name = f"{m['type']}-{m.get('distillation_training_mode', None)}-{m.get('temperature', None)}-{m.get('alpha', None)}-{m.get('teacher', None)}_{info}"
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
            elif cfg.training_mode == 'dualtraining':
                test_name = f'DualTraining-{m["type"]}_{info}'
                dl_model_names.append(test_name)
            elif cfg.training_mode == 'distrib2classtraining':
                test_name = f'Distribution2Class-{m["type"]}_{info}'
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
                distallation=cfg.training_mode == 'distillation'
            )
            if df_metrics is None:
                df_metrics = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
            else:
                df_metrics = pd.concat(
                    (df_metrics, pd.DataFrame.from_dict(metrics, orient="index").reset_index())
                )
        
        if stat_model_names:
            metrics, _, _, _ = test_fire_index_model(
                cfg, graphScale, test_dataset.copy(deep=True),
                                test_dataset_unscale.copy(deep=True),
                                    "all",
                                    prefix,
                                    stat_model_names,
                                    dir_output / "all" / prefix,
                                    prefix_config,
                                    encoding,
                                    f'{cfg.sinisterEncoding}_{name_exp}',
                                    cfg.scaling,
                                    ["all"],
                                    dir_train,
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