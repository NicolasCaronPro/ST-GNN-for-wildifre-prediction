import sys
import os
import pickle
import pandas as pd
from pathlib import Path
import torch
import numpy as np
import argparse

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def graph_id2_num_zone(df, scale, graph_construct):

    df = df.copy()

    # --- Cas simple ---
    if scale == 'departement' or "zonemeteo" not in graph_construct:
        df['num_zone'] = df['graph_id'].values
        return df
    
    # --- Cas zonemeteo ---
    df['num_zone'] = df['graph_id'].values  # initialisation par défaut

    # ====== PARTIE DEPARTEMENT 6 ======
    mask_dep6 = df['departement'] == 6
    if np.any(mask_dep6):

        df_dep6 = df[mask_dep6]

        print(df_dep6)

        graph_ids_6 = np.sort(df_dep6['graph_id'].unique())
        num_zone_6 = [65, 62, 64, 61, 66, 67, 63]

        if len(graph_ids_6) != len(num_zone_6):
            print(f'Removing first id {graph_ids_6[0]}')
            graph_ids_6 = graph_ids_6[1:]

        dico_6 = {gi: num_zone_6[i] for i, gi in enumerate(graph_ids_6)}

        df.loc[mask_dep6, 'num_zone'] = df.loc[mask_dep6, 'graph_id'].map(dico_6)

    # ====== PARTIE AUTRES DEPARTEMENTS ======
    mask_other = df['departement'] != 6
    df_other = df[mask_other]

    graph_ids_other = np.sort(df_other['graph_id'].unique())
    dico_other = {gi: gi for gi in graph_ids_other}

    df.loc[mask_other, 'num_zone'] = df.loc[mask_other, 'graph_id'].map(dico_other)

    return df

def save_object(obj, filename: str, path: Path):
    check_and_create_path(path)
    with open(path / filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Configuration des chemins pour permettre les imports du module GNN
# Le script est placé dans Prediction/GNN/, donc le parent direct est Prediction/
# et le parent de Prediction est la racine du projet.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent # /home/caron/Bureau/ST-GNN-for-wildifre-prediction/
prediction_path = current_dir.parent      # /home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/

if str(prediction_path) not in sys.path:
    sys.path.append(str(prediction_path))

# On ajoute aussi la racine du projet au cas où
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Argument parsing
parser = argparse.ArgumentParser(description='Run SHAP analysis for wildfire prediction models')
parser.add_argument('--dataset', type=str, default='firemen', help='Dataset name')
parser.add_argument('--name', type=str, default='occurence_01_06_25', help='Run name')
parser.add_argument('--graph_construct', type=str, default='risk-size-zonemeteo-degree-a3-r5-t0.3', help='Graph construction string')
parser.add_argument('--target', type=str, default='nbsinister-quantile-5-Class-Dept', help='Target name')
parser.add_argument('--loss', type=str, default='flwk', help='Loss function name')
parser.add_argument('--model_type', type=str, default='GRU', help='Model type (e.g., GRU)')
parser.add_argument('--task_type', type=str, default='classification', help='Task type (classification/regression)')
parser.add_argument('--scale', type=int, default=3, help='Spatial scale')
parser.add_argument('--horizon', type=int, default=0, help='Prediction horizon')
parser.add_argument('--kdays', type=int, default=10, help='Number of days')
args = parser.parse_args()

dataset = args.dataset
name = args.name
graph_construct = args.graph_construct
target = args.target
loss = args.loss
model_type = args.model_type
task_type = args.task_type
scale = args.scale
horizon = args.horizon
kdays = args.kdays

model_name = f'{model_type}_search_full_{kdays}_{horizon}_all_one_{target}_{task_type}_{loss}'

# Chemins des fichiers
root_path = f"/Work/Users/ncaron/GNN/{dataset}/firepoint/2x2/train/{name}/check_z-score/full_all_{scale}_0_{graph_construct}_node/{model_name}"
#root_path = f"/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/{dataset}/firepoint/2x2/train/{name}/check_z-score/full_all_{scale}_0_{graph_construct}_node/{model_name}"
model_path = f"{root_path}/{model_name}.pkl"

print(f"Chargement du modèle: {model_path}")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {e}")
    sys.exit(1)
    
df_test = model.df_test.copy(deep=True)

# Définition du répertoire de sortie
output_dir = Path(model_path).parent / "shap_analysis"
os.makedirs(output_dir, exist_ok=True)

# Sélection d'un échantillon pour éviter des temps de calcul prohibitifs
# Vous pouvez ajuster le nombre d'échantillons ici.
#sample_size = min(2000, len(df_test))
#df_sample = df_test.sample(sample_size, random_state=42)

horizon = model.horizon
target = model.target_name
df_test = graph_id2_num_zone(df_test, scale=scale, graph_construct=graph_construct)
nz = df_test['num_zone'].unique()
depts = df_test['departement'].unique()

for H in range(1):
    
    device = 'cpu'
    model.device = device
    if hasattr(model, 'model') and model.model is not None:
        model.model.to(device)
    
    model.shapley_additive_explanation(
                df=df_test,
                outname=f'all_dept',
                dir_output=output_dir / f'all',
                mode='beeswarm',
                figsize=(15, 25),
                plot=True
            )
    
    for num_zone in nz:

        outname = f'{target}_h{H}_zone{num_zone}' 

        df_sample = df_test[df_test['num_zone'] == num_zone]
        dept = df_sample['departement'].unique()[0]
        sample_size = len(df_sample)

        print(f"Lancement de shapley_additive_explanation sur {num_zone} num_zone {sample_size} échantillons...")
        try:
            # On s'assure que le modèle utilise le bon device (CPU par défaut pour SHAP si GPU non dispo)
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = 'cpu'
            model.device = device
            if hasattr(model, 'model') and model.model is not None:
                model.model.to(device)

            model.shapley_additive_explanation(
                df=df_sample,
                outname=outname,
                dir_output=output_dir / f'{str(dept)}',
                mode='beeswarm',
                figsize=(15, 25),
                plot=True
            )
            print(f"Analyse terminée avec succès. Résultats dans: {output_dir}")
        except Exception as e:
            print(f"Erreur lors de l'exécution de SHAP: {e}")
            import traceback
            traceback.print_exc()

    for dept in depts:
        outname = f'{target}_h{H}_{dept}'

        df_sample = df_test[df_test['departement'] == dept]
        sample_size = len(df_sample)

        print(f"Lancement de shapley_additive_explanation sur {sample_size} dept {dept} échantillons...")
        try:
            # On s'assure que le modèle utilise le bon device (CPU par défaut pour SHAP si GPU non dispo)
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = 'cpu'
            model.device = device
            if hasattr(model, 'model') and model.model is not None:
                model.model.to(device)
                
            model.shapley_additive_explanation(
                df=df_sample,
                outname=outname,
                dir_output=output_dir / str(dept),
                mode='beeswarm',
                figsize=(15, 25),
                plot=True
            )
            print(f"Analyse terminée avec succès. Résultats dans: {output_dir}")
        except Exception as e:
            print(f"Erreur lors de l'exécution de SHAP: {e}")
            import traceback
            traceback.print_exc()

    #save_object(model, f'{model_name}.pkl', Path(root_path))