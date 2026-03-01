import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path

from GNN.graph_structure import GraphStructure
from GNN.pytorch_model_class import Model_Torch

def create_synthetic_data(n_samples=100):
    df = pd.DataFrame({
        'f1_0': np.random.randn(n_samples),
        'f2_0': np.random.randn(n_samples),
        'nbsinister': np.random.randint(0, 5, n_samples),
        'departement': np.random.randint(0, 5, n_samples),
        'graph_id': np.arange(n_samples),
        'cluster-encoder': np.random.randint(0, 3, n_samples),
        'weight': np.ones(n_samples),
        'date': np.arange(n_samples),
        'latitude': np.random.randn(n_samples),
        'longitude': np.random.randn(n_samples),
        'fwi': np.random.randn(n_samples),
        'id': np.arange(n_samples),
        'scale': np.ones(n_samples),
        'saison-encoding': np.zeros(n_samples),
        'mediterranean': np.zeros(n_samples),
        'area': np.ones(n_samples),
        'time_intervention': np.zeros(n_samples),
        'ressource': np.zeros(n_samples),
        'burned_area': np.zeros(n_samples),
        'nbsinister_id': np.zeros(n_samples),
        'risk': np.zeros(n_samples),
        'DFE': np.zeros(n_samples)
    })
    return df

def test():
    # Fix seed for script level
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    df_train = create_synthetic_data(200)
    df_val = create_synthetic_data(50)
    df_test = create_synthetic_data(50)

    graph = GraphStructure(scale = None,
                 geo = None,
                 maxDist = 100,
                 numNei = 4,
                 resolution = "2x2",
                 graph_construct = None,
                 sinister = "dfe",
                 sinister_encoding = None,
                 dataset_name = "alpes",
                 train_departements = [6],
                 attempt = None,
                 reduce = None,
                 tol = None)

    # Make dummy graph methods happy if they exist
    graph.graph_method = 'node'
    graph.scale = 'departement'

    model_dir = Path('./test_optuna_logs/')
    model_dir.mkdir(exist_ok=True)

    print("--- Initialize Model_Torch ---")
    model = Model_Torch(
        model_name='NetMLP',
        nbfeatures=2,
        batch_size=16,
        lr=0.01,
        delta_lr=10,
        patience_cnt_lr=0,
        target_name='nbsinister',
        task_type='regression',
        out_channels=1,
        dir_log=model_dir,
        features_name=['f1_0', 'f2_0'],
        ks=0,
        loss='cllt',
        name='TestOptuna',
        device='cpu',
        under_sampling='full',
        over_sampling='full',
        n_run=1,
        training_mode='normal',
        loss_param_search=False
    )

    print("--- Create DataLoaders ---")
    model.create_train_val_test_loader(graph, df_train, df_val, df_test, epochs=2, PATIENCE_CNT=5, CHECKPOINT=5, features_importance=False)

    # We monkey patch model.suggest_loss_params to output FIXED PARAMS for OPTUNA.
    # And we also use the SAME PARAMS for train()
    fixed_params = {
        'beta': 3.6167258754794345, 't': 0.38808363994393985, 'wmed': 2.1181338709061728, 
        'wmin': 0.3090944162785197, 'wneg': 0.010696671944419034, 'gamma': 3.858781740471833, 
        'taugate': 0.10621789513165877, 'gatetemp': 0.07804890070629288, 'wkdecay': 'exp', 
        'wklambda': 0.3495008616795649, 'wkmin': 0.0026366397418353914, 'learngains': False, 
        'wfocal': 1.5494283305697185, 'wmu0': 0.4200314727662068, 'fgamma': 3.7215798979568424, 
        'falpha': 0.8563103452989596
    }
    
    def custom_suggest(trial, loss_name):
        return fixed_params
    
    model.suggest_loss_params = custom_suggest

    print("\n\n==== TRAINING WITH OPTUNA ====\n")
    model.loss_param_search = True
    
    # We monkeypatch the model's launch_batch to extract gradients and weights after backward
    original_launch_batch = model.launch_batch
    
    global iter_count, is_optuna_run
    iter_count = 0
    is_optuna_run = True
    optuna_records = []
    normal_records = []
    
    def custom_launch_batch(data, criterion, batch_type, do_update):
        global iter_count, is_optuna_run
        loss, loss_res = original_launch_batch(data, criterion, batch_type, do_update)
        
        # After backward step is taken in launch_train_loader, we can inspect weights.
        # But wait, original_launch_batch only does forward pass! The backward is in launch_train_loader.
        return loss, loss_res
        
    model.launch_batch = custom_launch_batch
    
    # We will just patch the optimizer step to see when weights diverge
    original_get_optimizer = model.get_optimizer
    def custom_get_optimizer(criterion):
        opt = original_get_optimizer(criterion)
        original_step = opt.step
        def custom_step(*args, **kwargs):
            global iter_count, is_optuna_run
            original_step(*args, **kwargs)
            # Record weights after step
            weight_sum = sum(p.sum().item() for p in model.model.parameters())
            if is_optuna_run:
                optuna_records.append(weight_sum)
            else:
                normal_records.append(weight_sum)
            iter_count += 1
        opt.step = custom_step
        return opt
        
    model.get_optimizer = custom_get_optimizer

    model.train_optuna(graph, PATIENCE_CNT=5, CHECKPOINT=5, epochs=2, verbose=True, new_model=True, n_trials=1, min_epochs=1)
    
    optuna_weights = {k: v.clone() for k, v in model.model.state_dict().items()}
    
    print("\n\n==== TRAINING WITHOUT OPTUNA ====\n")
    model.loss_param_search = False
    is_optuna_run = False
    iter_count = 0
    
    model.train(graph, PATIENCE_CNT=5, CHECKPOINT=5, epochs=2, verbose=True, new_model=True)
    
    normal_weights = {k: v.clone() for k, v in model.model.state_dict().items()}
    
    print("\n\n--- ITERATION COMPARISON ---")
    for i, (o_w, n_w) in enumerate(zip(optuna_records, normal_records)):
        if abs(o_w - n_w) > 1e-4:
            print(f"Iter {i} diff: Optuna={o_w}, Normal={n_w}")
        else:
            print(f"Iter {i} match: {o_w}")
    
    
    diff_count = 0
    for k in optuna_weights:
        if not torch.allclose(optuna_weights[k], normal_weights[k], atol=1e-6):
            diff_count += 1
            print(f"Difference found in layer: {k}")
            
    if diff_count == 0:
        print("\nSUCCESS: train and train_optuna produced identical weights!")
    else:
        print(f"\nDiscrepancy: {diff_count} layers have different weights between train and train_optuna.")
if __name__ == "__main__":
    test()
