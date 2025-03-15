import subprocess
import mlflow

# Configuration des paramètres pour le script à exécuter, y compris 'training_mode'
params = {
    '--name': 'test_geometry',           # -n test_geometry
    '--encoder': 'False',                # -e False
    '--sinister': 'firepoint',           # -s firepoint
    '--point': 'False',                  # -p False
    '--nbpoint': 'full',                 # -np full
    '--database': 'False',               # -d False
    '--graph': 'False',                  # -g False
    '--maxDate': '2023-01-01',           # -mxd 2023-01-01
    '--trainDate': '2022-01-01',         # -mxdv 2022-01-01
    '--featuresSelection': 'True',       # -f True
    '--database2D': 'False',             # -dd False
    '--scale': '4',                      # -sc 4
    '--dataset': 'firemen',              # -dataset firemen
    '--NbFeatures': 'all',               # -nf all
    '--optimizeFeature': 'False',        # -of False
    '--GridSearch': 'False',             # -gs False
    '--BayesSearch': 'False',            # -bs False
    '--doTest': 'True',                  # -test True
    '--doTrain': 'True',                 # -train True
    '--resolution': '2x2',               # -r 2x2
    '--pca': 'False',                    # -pca False
    '--KMEANS': 'False',                 # -kmeans False
    '--ncluster': '5',                   # -ncluster 5
    '--k_days': '0',                     # -k_days 0
    '--days_in_futur': '0',              # -days_in_futur 0
    '--scaling': 'z-score',              # -scaling z-score
    '--graphConstruct': 'risk-size-watershed',  # -graphConstruct risk-size-watershed
    '--sinisterEncoding': 'occurence',   # -sinisterEncoding occurence
    '--training_weight': 'weight_outlier', # -weights weight_outlier
    '--test_weight': 'weight_outlier',   # -test_weight (ajouté comme demandé)
    '--top_cluster': '3',                # -top_cluster 3 (pas défini dans la commande mais ajouté pour cohérence)
    '--graph_method': 'node',             # -graph_method node (par défaut)
    '--training_mode': 'standard',       # Nouveau paramètre --training_mode standard
}

# Initialisation de MLflow
mlflow.set_experiment("My_Experiment")  # Remplacez par le nom de votre expérience
with mlflow.start_run(run_name=params['--name']) as run:
    # Enregistrer les paramètres dans MLflow
    for key, value in params.items():
        if isinstance(value, list):
            mlflow.log_param(key.lstrip('-'), ','.join(value))
        else:
            mlflow.log_param(key.lstrip('-'), value)

    # Préparation de la commande pour exécuter le script Python
    command = ['python', 'train_trees.py']  # Nom du script à exécuter
    for key, value in params.items():
        if isinstance(value, list):
            command.append(key)
            command.extend(value)
        else:
            command.append(key)
            command.append(value)

    # Lancer le script avec les paramètres
    result = subprocess.run(command, capture_output=True, text=True)

    # Enregistrer les sorties du script dans MLflow
    mlflow.log_text(result.stdout, "stdout.txt")
    mlflow.log_text(result.stderr, "stderr.txt")

    # Enregistrer un tag indiquant la fin de l'exécution
    mlflow.set_tag("status", "completed" if result.returncode == 0 else "failed")
