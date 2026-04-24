import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

# Ensure current directory is at the front of sys.path to avoid conflicts with forecasting_models/discretization.py
current_gnn_dir = os.path.abspath('.')
if current_gnn_dir not in sys.path:
    sys.path.insert(0, current_gnn_dir)

# Also add Prediction directory for GNN sub-package imports
prediction_dir = os.path.abspath('..')
if prediction_dir not in sys.path:
    sys.path.insert(1, prediction_dir)

# Import discretization first to ensure we get the right one
try:
    import discretization
    from discretization import KMeansRiskZerosHandle, QuantileRiskZerosHandle
    import tools
    from tools import read_object, save_object
    print("Successfully imported discretization and tools.")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"sys.path: {sys.path[:5]}")
# Dataset Paths (Absolute paths provided by user)
BDIFF_FILE = Path('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/bdiff/firepoint/2x2/train/occurence_default/df_train_full_departement_0_None_node.pkl')
FIREMEN_FILE = Path('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/firemen/firepoint/2x2/train/occurence_01_06_25/df_train_full_3_0_risk-size-zonemeteo-degree-a3-r5-t0.3_node.pkl')

print(f"Loading BDIFF dataset from {BDIFF_FILE.name}...")
df_bdiff = pd.read_pickle(BDIFF_FILE)

print(f"Loading FIREMEN dataset from {FIREMEN_FILE.name}...")
df_firemen = pd.read_pickle(FIREMEN_FILE)

print("Data loading complete.")
def find_discretization_cols(df, target_name):
    """Finds the available discretization columns for a given target."""
    q_col = f"{target_name}-quantile-5-Class-Dept"
    k_col = f"{target_name}-kmeans-5-Class-Dept"
    
    found_q = q_col if q_col in df.columns else None
    found_k = k_col if k_col in df.columns else None
    
    if not found_q:
        variants = [c for c in df.columns if target_name in c and 'quantile' in c and '5-Class-Dept' in c]
        if variants: found_q = variants[0]
            
    if not found_k:
        variants = [c for c in df.columns if target_name in c and 'kmeans' in c and '5-Class-Dept' in c]
        if variants: found_k = variants[0]
    
    print(df[df[found_q] == 4].shape[0], df[df[found_k] == 4].shape[0])

    return found_q, found_k

targets = ['nbsinister', 'burnedareaRoot', 'timeintervention', 'ressource']

print("BDIFF Discretization Columns:")
bdiff_cols = {}
for t in targets:
    q, k = find_discretization_cols(df_bdiff, t)
    bdiff_cols[t] = (q, k)
    print(f"  {t}: Quantile={q}, KMeans={k}")

print("\nFIREMEN Discretization Columns:")
firemen_cols = {}
for t in targets:
    q, k = find_discretization_cols(df_firemen, t)
    firemen_cols[t] = (q, k)
    print(f"  {t}: Quantile={q}, KMeans={k}")
def plot_distributions(df, target_cols, dataset_name):
    for target, cols in target_cols.items():
        q_col, k_col = cols
        if not q_col and not k_col:
            continue

        print(f"\nPlotting distributions for: {target} ({dataset_name})")
        # Create a 2x2 grid: Top row=Full, Bottom row=Positive only
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Full Distribution
        if q_col:
            axes[0, 0].hist(df[target], bins=50, color='blue', alpha=0.7, density=True)
            axes[0, 0].set_xlabel(f"Value (Full)")
            axes[0, 0].set_ylabel("Density")
            
        if k_col:
            axes[0, 1].hist(df[target], bins=50, color='green', alpha=0.7, density=True)
            axes[0, 1].set_xlabel(f"Value (Full)")
            axes[0, 1].set_ylabel("Density")

        # Positive Only Distribution
        df_pos = df[df[target] > 0]
        if not df_pos.empty:
            if q_col:
                axes[1, 0].hist(df_pos[target], bins=50, color='blue', alpha=0.7, density=True)
                axes[1, 0].set_xlabel(f"Value (Positive Only)")
                axes[1, 0].set_ylabel("Density")
                
            if k_col:
                axes[1, 1].hist(df_pos[target], bins=50, color='green', alpha=0.7, density=True)
                axes[1, 1].set_xlabel(f"Value (Positive Only)")
                axes[1, 1].set_ylabel("Density")
        else:
            print(f"  Warning: No positive values for {target}")

        plt.tight_layout()
        
        # Save each figure to the Bureau directory
        save_path = f"/home/caron/Bureau/distribution_{dataset_name}_{target}.png"
        plt.savefig(save_path)
        print(f"  Figure saved to: {save_path}")
        
        plt.show()

print("BDIFF Distributions:")
plot_distributions(df_bdiff, bdiff_cols, "BDIFF")

print("\nFIREMEN Distributions:")
plot_distributions(df_firemen, firemen_cols, "FIREMEN")
BDIFF_MODEL = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/bdiff/firepoint/2x2/train/occurence_default/check_z-score/full_all_departement_0_None_node/GRU_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept-sum-3_classification_flwk/best.pt'
FIREMEN_MODEL = '/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/firemen/firepoint/2x2/train/occurence_062023/check_z-score/full_all_3_0_zonemeteo_node/GRU_search_full_10_0_all_one_nbsinister-quantile-5-Class-Dept_regression_pdegpd/run_1/best.pt'

def try_load_model(path, name):
    path = Path(path)
    if path.exists():
        print(f"Loading {name} model from {path.name}...")
        try:
            model_data = torch.load(path, map_location='cpu')
            print(f"  Successfully loaded {name} model data.")
            if isinstance(model_data, dict):
                print(f"  Keys: {list(model_data.keys())[:5]}...")
        except Exception as e:
            print(f"  Error loading {name} model: {e}")
    else:
        print(f"  Warning: {name} model path does not exist.")

try_load_model(BDIFF_MODEL, "BDIFF")
try_load_model(FIREMEN_MODEL, "FIREMEN")