import math
from matplotlib import figure
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path 

departements_list = [
    (1, 'Ain'), (2, 'Aisne'), (3, 'Allier'), (4, 'Alpes-de-Haute-Provence'),
    (5, 'Hautes-Alpes'), (6, 'Alpes-Maritimes'), (7, 'Ardeche'), (8, 'Ardennes'),
    (9, 'Ariege'), (10, 'Aube'), (11, 'Aude'), (12, 'Aveyron'),
    (13, 'Bouches-du-Rhone'), (14, 'Calvados'), (15, 'Cantal'), (16, 'Charente'),
    (17, 'Charente-Maritime'), (18, 'Cher'), (19, 'Correze'), (21, 'Cote-d-Or'),
    (22, 'Cotes-d-Armor'), (23, 'Creuse'), (24, 'Dordogne'), (25, 'Doubs'),
    (26, 'Drome'), (27, 'Eure'), (28, 'Eure-et-Loir'), (29, 'Finistere'),
    ('2A', 'Corse-du-Sud'), ('2B', 'Haute-Corse'), (30, 'Gard'), (31, 'Haute-Garonne'),
    (32, 'Gers'), (33, 'Gironde'), (34, 'Herault'), (35, 'Ille-et-Vilaine'),
    (36, 'Indre'), (37, 'Indre-et-Loire'), (38, 'Isere'), (39, 'Jura'),
    (40, 'Landes'), (41, 'Loir-et-Cher'), (42, 'Loire'), (43, 'Haute-Loire'),
    (44, 'Loire-Atlantique'), (45, 'Loiret'), (46, 'Lot'), (47, 'Lot-et-Garonne'),
    (48, 'Lozere'), (49, 'Maine-et-Loire'), (50, 'Manche'), (51, 'Marne'),
    (52, 'Haute-Marne'), (53, 'Mayenne'), (54, 'Meurthe-et-Moselle'), (55, 'Meuse'),
    (56, 'Morbihan'), (57, 'Moselle'), (58, 'Nievre'), (59, 'Nord'),
    (60, 'Oise'), (61, 'Orne'), (62, 'Pas-de-Calais'), (63, 'Puy-de-Dome'),
    (64, 'Pyrenees-Atlantiques'), (65, 'Hautes-Pyrenees'), (66, 'Pyrenees-Orientales'),
    (67, 'Bas-Rhin'), (68, 'Haut-Rhin'), (69, 'Rhone'), (70, 'Haute-Saone'),
    (71, 'Saone-et-Loire'), (72, 'Sarthe'), (73, 'Savoie'), (74, 'Haute-Savoie'),
    (75, 'Paris'), (76, 'Seine-Maritime'), (77, 'Seine-et-Marne'), (78, 'Yvelines'),
    (79, 'Deux-Sevres'), (80, 'Somme'), (81, 'Tarn'), (82, 'Tarn-et-Garonne'),
    (83, 'Var'), (84, 'Vaucluse'), (85, 'Vendee'), (86, 'Vienne'),
    (87, 'Haute-Vienne'), (88, 'Vosges'), (89, 'Yonne'), (90, 'Territoire de Belfort'),
    (91, 'Essonne'), (92, 'Hauts-de-Seine'), (93, 'Seine-Saint-Denis'), (94, 'Val-de-Marne'),
    (95, 'Val-d-Oise'), (971, 'Guadeloupe'), (972, 'Martinique'), (973, 'Guyane'),
    (974, 'La Reunion'), (976, 'Mayotte')
]

# Laurent steack

int2str = {code: name.lower().replace("'", "-") for code, name in departements_list}
int2strMaj = {code: name for code, name in departements_list}
int2name = {code: f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}" for code, name in departements_list}

str2int = {name.lower().replace("'", "-"): code for code, name in departements_list}
str2intMaj = {name: code for code, name in departements_list}
str2name = {name: f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}" for code, name in departements_list}

name2str = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": name for code, name in departements_list}
name2int = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": code for code, name in departements_list}
name2strlow = {f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": name.lower() for code, name in departements_list}
name2intstr = {
    f"departement-{str(code).zfill(2)}-{name.lower().replace(' ', '-')}": (f'0{code}' if code not in ['2A', '2B'] and int(code) < 10 else str(code))
    for code, name in departements_list
}

def plot_result(dff, metric, dataset, top='all'):
    """
    Affiche un graphique par Scale, avec les départements sur l'axe des X triés selon 'nbsinister'.
    Chaque courbe représente un modèle.

    :param dff: DataFrame contenant les colonnes ['Department', 'Scale', 'Model', 'nbsinister', metric]
    :param metric: Nom de la colonne contenant la métrique à afficher
    :param dataset: Nom du dataset à filtrer
    :param top: Nombre de départements à afficher (ou 'all' pour afficher tous les départements)
    """

    df = dff[dff['Department'] != 'all']
    df = df[df['Dataset'] == dataset].copy(deep=True)
    
    # Trier les départements par 'nbsinister' décroissant
    df_sorted = df.sort_values(by='nbsinister', ascending=False)
    df_sorted['Department'] = df_sorted['Department'].apply(lambda x: name2str[x])

    # Sélectionner les "top" départements si nécessaire
    if top != 'all':
        top = int(top)
        top_departments = df_sorted['Department'].unique()[:top]
        df_sorted = df_sorted[df_sorted['Department'].isin(top_departments)]

    #df_sorted['namex'] = df_sorted.apply(lambda x: f"{x['Department']}  {x['nbsinister']}", axis=1)

    # Assurer que tous les modèles ont les mêmes départements en X
    all_departments = df_sorted['Department'].unique()

    # Récupérer les échelles uniques
    scales = df_sorted['Scale'].unique()
    num_scales = len(scales)

    # Définir la disposition de la grille
    if num_scales > 3:
        cols = 3
        rows = math.ceil(num_scales / cols)
    else:
        cols = 1
        rows = num_scales

    fig, axes = plt.subplots(rows, cols, figsize=(25, 7 * rows), squeeze=False)
    axes = axes.flatten()

    for i, scale in enumerate(scales):
        ax = axes[i]
        df_scale = df_sorted[df_sorted['Scale'] == scale].copy()
        df_scale.drop_duplicates(subset=['Department', 'Model'], inplace=True)

        # Créer un pivot pour assurer l'alignement des départements sur X
        pivot_df = df_scale.pivot(index='Department', columns='Model', values=metric)
        pivot_df = pivot_df.reindex(all_departments)  # S'assurer que l'ordre des départements est respecté

        # Tracer les courbes par modèle
        sns.lineplot(
            data=pivot_df,
            markers=True,
            #palette='tab10',
            ax=ax
        )

        ax.set_xticks(range(len(all_departments)))
        ax.set_xticklabels(all_departments, rotation=90, ha='right')
        ax.set_title(f"Metric {metric} - Scale: {scale}")
        ax.set_xlabel("Department")
        ax.set_ylabel(metric)
        ax.legend(title="Model")
        ax.grid(True, linestyle='--', alpha=0.5)

    # Supprimer les axes inutilisés
    for i in range(num_scales, rows * cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    #plt.show()
    return fig

def plot_column_comparison(df, metric, col):
    """
    Affiche un graphique par département, avec les datasets sur l'axe des X.
    Chaque courbe représente un modèle. Seuls les départements et scales communs sont affichés.
    
    :param df: DataFrame contenant les colonnes ['Department', 'Scale', 'Dataset', 'Model', metric]
    :param metric: Nom de la colonne contenant la métrique à afficher
    """
    # Trouver les départements communs à tous les datasets
    common_departments = df.groupby('Department')[col].nunique()
    common_departments = common_departments[common_departments == df[col].nunique()].index
    
    # Trouver les scales communes à tous les datasets
    common_scales = df.groupby('Scale')[col].nunique()
    common_scales = common_scales[common_scales == df[col].nunique()].index
    
    df_filtered = df[(df['Department'].isin(common_departments)) & (df['Scale'].isin(common_scales))]
    
    # Récupérer les départements uniques
    departments = sorted(common_departments)
    num_departments = len(departments)
    
    # Définir la disposition de la grille
    cols = 3 if num_departments > 3 else 1
    rows = math.ceil(num_departments / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(25, 7 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i, department in enumerate(departments):
        ax = axes[i]
        df_department = df_filtered[df_filtered['Department'] == department]
        
        # Tracer les courbes par modèle
        sns.lineplot(
            data=df_department, 
            x=col, 
            y=metric, 
            hue='Model', 
            marker='o',
            palette='tab10',
            ax=ax
        )
        
        ax.set_title(f"Metric {metric} - Department: {department}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)
        ax.legend(title="Model")
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Supprimer les axes inutilisés
    for i in range(num_departments, rows * cols):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    #plt.show()
    return fig
    
def plot_column_comparison_bar(df, metric, col):
    """
    Affiche un bar plot par département, avec les datasets sur l'axe des X.
    Chaque groupe de barres représente un modèle. Seuls les départements communs sont affichés.

    :param df: DataFrame contenant les colonnes ['Department', 'Scale', 'Dataset', 'Model', metric]
    :param metric: Nom de la colonne contenant la métrique à afficher
    :param col: Colonne à utiliser pour l'axe X (ex: 'Dataset')
    """
    # Trouver les départements communs à tous les datasets
    common_departments = df.groupby('Department')[col].nunique()
    common_departments = common_departments[common_departments == df[col].nunique()].index

    df_filtered = df[df['Department'].isin(common_departments)]

    # Récupérer les départements uniques
    departments = sorted(common_departments)
    num_departments = len(departments)

    # Définir la disposition de la grille
    cols = 3 if num_departments > 3 else 1
    rows = math.ceil(num_departments / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(25, 7 * rows), squeeze=False)
    axes = axes.flatten()

    # Initialisation des handles et labels pour la légende globale
    handles, labels = None, None  

    for i, department in enumerate(departments):
        ax = axes[i]
        df_department = df_filtered[df_filtered['Department'] == department]

        # Tracer le bar plot
        barplot = sns.barplot(
            data=df_department,
            x=col,
            y=metric,
            hue='Model',
            palette='tab20',
            ax=ax
        )

        ax.set_title(f"Metric {metric} - Department: {department}")
        ax.set_xlabel(col)
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Récupérer la légende UNIQUEMENT lors du premier plot
        if handles is None or labels is None:
            handles, labels = ax.get_legend_handles_labels()

        # Supprimer la légende individuelle du subplot
        ax.get_legend().remove()

    # Supprimer les axes inutilisés
    for i in range(num_departments, rows * cols):
        fig.delaxes(axes[i])

    # Ajouter une légende globale
    if handles and labels:
        fig.legend(handles, labels, title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)

    plt.tight_layout()
    #plt.show()
    return fig

def load_all_metrics_files(path):
    """
    Parcourt tous les sous-dossiers de `path`, charge les fichiers `df_metrics.csv`,
    et les combine en un seul DataFrame global.

    Parameters:
    - path (Path): Le chemin du dossier racine à parcourir.

    Returns:
    - pd.DataFrame: Le DataFrame global combinant tous les fichiers `df_metrics.csv`.
    """
    path = Path(path)
    all_metrics = []  # Liste pour stocker tous les DataFrames

    # Parcourir tous les sous-dossiers et rechercher les fichiers df_metrics.csv
    for file in path.rglob('df_metrics*.csv'):
        try:
            # Charger le fichier CSV dans un DataFrame
            df = pd.read_csv(file)
            print(df)

            # Ajouter une colonne pour identifier la source du fichier
            df['source'] = str(file.parent)  # Ajouter le chemin du dossier parent

            # Ajouter le DataFrame à la liste
            all_metrics.append(df) 
        except Exception as e:
            print(f"Erreur lors du chargement de {file}: {e}")

    # Combiner tous les DataFrames en un seul DataFrame global
    if all_metrics:
        df_global = pd.concat(all_metrics, ignore_index=True)
    else:
        df_global = pd.DataFrame()  # Si aucun fichier trouvé, retourne un DataFrame vide

    return df_global

def load_all_metrics_files_dataset(base_path, datasets, experiments):
    """
    Charge plusieurs datasets pour une expérience donnée.
    
    :param base_path: Chemin de base vers les fichiers de métriques.
    :param datasets: Liste des datasets à charger.
    :param experiment: Nom de l'expérience à charger.
    :return: DataFrame concaténé de toutes les données.
    """
    all_dfs = []
    print(datasets, experiments)
    for i, dataset in enumerate(datasets):
        for expe in experiments: 
            path = Path(base_path) / dataset / 'firepoint' / '2x2' / 'test' / expe
            print(path)
            df = load_all_metrics_files(path)  # Assurez-vous que cette fonction est définie ailleurs
            df['Dataset'] = dataset  # Ajouter une colonne pour identifier le dataset
            df['exp'] = expe  # Ajouter une colonne pour identifier le dataset
            all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)

def parse_run_name(x):
    dico = {}
    vec = x.split('_')
    dico['Department'] = vec[0]
    dico['Model'] = vec[1]
    i = 2
    if dico['Model'] == 'fwi':
        i += 1
        dico['Target'] = 'indice'
    else:
        dico['under_sampling'] = vec[i]
        i += 1
        dico['over_sampling'] = vec[i]
        i += 1
        dico['weight'] = vec[i]
        i += 1
        dico['Target'] = vec[i]
        i += 1

    if dico['Model'] != 'fwi':
        dico['Task_type'] = vec[i]
        i += 1
        dico['loss'] = vec[i]
        i += 1
    else:
        dico['loss'] = None
        dico['Task_type'] = 'Indice'

    i += 1
    dico['kdays'] = vec[i]
    i += 1
    dico['Number_of_features'] = vec[i]
    i += 1
    dico['Scale'] = vec[i]
    i += 1
    dico['Days_in_futur'] = vec[i]
    i += 1
    dico['Base'] = vec[i]
    i += 1
    dico['Method'] = vec[i]
    i += 1
    if i == len(vec):
        return dico
    if vec[i] == 'kmeans':
        i += 1
        dico['kmeans_shift'] = vec[i]
        i += 1
        dico['kmeans_thresh'] = vec[i]
        i += 1
    return dico

def parse_dataframe(df):
    # Initialisation des colonnes avec des valeurs None
    df['Department'] = None
    df['Model'] = None
    df['Target'] = None
    df['Task_type'] = None
    df['Drop'] = None
    df['Loss_function'] = None
    df['under_sampling'] = None
    df['over_sampling'] = None
    df['kdays'] = None
    df['Number_of_features'] = None
    df['Scale'] = None
    df['Base'] = None
    df['Method'] = None
    df['Days_in_futur'] = None
    df['weight'] = None
    df['kmeans_thresh'] = None
    df['kmeans_shift'] = None

    # Boucle pour remplir les colonnes avec les valeurs de dico_parse
    for index, row in df.iterrows():
        dico_parse = parse_run_name(row['Run'])
        # Mise à jour de chaque colonne avec les valeurs du dictionnaire dico_parse
        df.loc[index, 'Department'] = dico_parse.get('Department')
        df.loc[index, 'Drop'] = dico_parse.get('Drop')
        df.loc[index, 'Model'] = dico_parse.get('Model')
        df.loc[index, 'Target'] = dico_parse.get('Target')
        df.loc[index, 'Task_type'] = dico_parse.get('Task_type')
        df.loc[index, 'Loss_function'] = dico_parse.get('loss')
        df.loc[index, 'under_sampling'] = dico_parse.get('under_sampling')
        df.loc[index, 'over_sampling'] = dico_parse.get('over_sampling')
        df.loc[index, 'kdays'] = dico_parse.get('kdays')
        df.loc[index, 'Number_of_features'] = dico_parse.get('Number_of_features')
        df.loc[index, 'Scale'] = dico_parse.get('Scale')
        df.loc[index, 'Base'] = dico_parse.get('Base')
        df.loc[index, 'Method'] = dico_parse.get('Method')
        df.loc[index, 'Days_in_futur'] = dico_parse.get('Days_in_futur')

        df.loc[index, 'weight'] = dico_parse.get('weight')
        df.loc[index, 'kmeans_thresh'] = dico_parse.get('kmeans_thresh', 0)
        df.loc[index, 'kmeans_shift'] = dico_parse.get('kmeans_shift', 0)

    return df