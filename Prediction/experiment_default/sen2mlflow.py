import argparse
import os
import mlflow
import shutil
import matplotlib.pyplot as plt
from tools import *
import re

def send_figure(figures, output_path, name_fig):
    if isinstance(figures, list):  # Si plusieurs figures sont retournées
            for i, fig in enumerate(figures):
                fig_path = os.path.join(output_path, f"{name_fig}_{i}.png")
                fig.savefig(fig_path)
                mlflow.log_artifact(fig_path)
    else:  # Si une seule figure est retournée
        fig_path = os.path.join(output_path, f"{name_fig}.png")
        figures.savefig(fig_path)
        mlflow.log_artifact(fig_path)

def sanitize_path(path):
    # Removes any problematic characters from the path, such as slashes and spaces
    return re.sub(r'-', '_', path)

def send_folder_to_mlflow(base_path, dataset, experiment, name):
    """Envoie les fichiers du dossier dataset/experiment à MLflow sous le nom de experiment_name"""
    experiment_path = Path(base_path) / dataset / 'firepoint' / '2x2' / 'test' / experiment
    
    experiment_name = f"{experiment}"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=name):
        # Envoi des fichiers du dossier à MLflow
        for root, _, files in os.walk(experiment_path):
            for file in files:
                file_path = os.path.join(root, file)
                path_parts = root.split('/')[1:]  # Ignore the first part
                arte_path = '/'.join([sanitize_path(part) for part in path_parts])
                mlflow.log_artifact(file_path, arte_path)

        # Chargement et traitement des données
        df = load_all_metrics_files_dataset(base_path, [dataset], [experiment])
        df = parse_dataframe(df)
        
        df['Model'] = df['Model'] + ' ' + df['Target'] + ' ' + df['exp'] +  ' ' + df['Loss_function']
        
        # Sauvegarde et envoi des figures
        output_path = os.path.join(experiment_path, "figures")
        os.makedirs(output_path, exist_ok=True)

        # Création des graphiques
        figures = plot_column_comparison_bar(df, 'iou_class_hard', 'Scale')
        send_figure(figures, output_path, 'iou_class_hard')

        figures = plot_result(df, 'iou_class_hard', 'bdiff', 'all')
        send_figure(figures, output_path, 'iou_class_hard_curve')

        figures = plot_result(df, 'f1', 'bdiff', 'all')
        send_figure(figures, output_path, 'f1_curve')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Envoie les fichiers à MLflow et génère des visualisations.")
    parser.add_argument("-d", "--dataset", required=True, help="Chemin du dataset")
    parser.add_argument("-e", "--experiment", required=True, help="Nom de l'expérimentation")
    parser.add_argument("-n", "--name", required=True, help="Nom de l'expérience MLflow")

    args = parser.parse_args()
    
    base_path = Path('../GNN/')
    
    send_folder_to_mlflow(base_path, args.dataset, args.experiment, args.name)