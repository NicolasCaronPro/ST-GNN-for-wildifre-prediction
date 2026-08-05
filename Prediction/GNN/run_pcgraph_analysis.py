"""Analyse et visualisation d'un modele PC-graph entraine.

Suit la convention des autres scripts d'analyse autonomes du projet
(run_shap_analysis.py, run_causal_analysis.py) : depickle l'objet
`PCGraphTraining` sauvegarde par l'entrainement et appelle son API publique
de visualisation (pytorch_model_pc_graph.py) :
  - plot_energy_convergence : verifie que `t_query` suffit a la convergence
    de la relaxation (diagnostic Fig. 15-17 de l'article PC-graph).
  - plot_topology            : graphe networkx de la topologie apprise
    (feature -> interne -> label), largeur d'arete = |theta| agrege.
  - feature_causal_ranking   : classement des features par force d'influence
    cumulee (chemins multi-sauts dans `theta`) sur chaque noeud de label.

Exemple :
    python3 run_pcgraph_analysis.py --dataset firemen --name 06_pcgraph \\
        --target nbsinister-kmeans-5-Class-Dept --loss pc \\
        --scale departement --graph_construct zonemeteo --kdays 5

Note : le PC-graph n'utilise aucune loss externe pour son apprentissage (cf.
pytorch_model_pc_graph.py) -- `--loss` ne sert ici qu'a reconstruire le nom
du dossier de sortie (`{model}_{infos}`), pas a selectionner un critere.
"""
import sys
import argparse
import pickle
from pathlib import Path

current_dir = Path(__file__).resolve().parent
prediction_path = current_dir.parent
project_root = prediction_path.parent
for p in (prediction_path, project_root):
    if str(p) not in sys.path:
        sys.path.append(str(p))

parser = argparse.ArgumentParser(description='Analyse/visualisation d un modele PC-graph entraine')
parser.add_argument('--dataset', type=str, default='firemen')
parser.add_argument('--name', type=str, default='06_pcgraph')
parser.add_argument('--graph_construct', type=str, default='zonemeteo')
parser.add_argument('--target', type=str, default='nbsinister-kmeans-5-Class-Dept')
parser.add_argument('--loss', type=str, default='pc')
parser.add_argument('--task_type', type=str, default='classification')
parser.add_argument('--scale', type=str, default='departement')
parser.add_argument('--horizon', type=int, default=0)
parser.add_argument('--kdays', type=int, default=5)
parser.add_argument('--n_hops', type=int, default=3, help='Nombre de sauts pour le classement causal')
parser.add_argument('--top_k', type=int, default=20, help='Nombre de features affichees dans le classement causal')
args = parser.parse_args()

model_name = (
    f'PCGraph_search_full_{args.kdays}_{args.horizon}_all_one_'
    f'{args.target}_{args.task_type}_{args.loss}'
)

root_path = (
    f"/Work/Users/ncaron/GNN/{args.dataset}/firepoint/2x2/train/{args.name}/"
    f"check_z-score/full_all_{args.scale}_0_{args.graph_construct}_node/{model_name}"
)
model_path = f"{root_path}/{model_name}.pkl"

print(f"Chargement du modele PC-graph : {model_path}")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Erreur lors du chargement du modele: {e}")
    sys.exit(1)

model.device = 'cpu'
if getattr(model, 'model', None) is not None:
    model.model.to('cpu')

output_dir = Path(model_path).parent / 'pcgraph_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

print('Verification de la convergence de la relaxation...')
model.plot_energy_convergence(dir_output=output_dir)

print('Visualisation de la topologie apprise...')
model.plot_topology(dir_output=output_dir)

print('Classement causal des features...')
df_ranking = model.feature_causal_ranking(n_hops=args.n_hops, top_k=args.top_k, dir_output=output_dir)
print(df_ranking.head(args.top_k))

print(f'Analyse terminee. Resultats dans : {output_dir}')
