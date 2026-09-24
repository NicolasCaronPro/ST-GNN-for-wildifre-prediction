"""
Compare GRU model groups — saves figures to /home/caron/Bureau/figures_gru/
Groups:
  losses   — weightedcrossentropy, wkloss, pdegpd
  filter   — filter-GRU-soft-weight-5/10/20
  federated — federated-GRU-saison/cluster-encoder/departement/mediterranean (fltg)
  flwk     — flwk, flwki
"""

import pickle
import sys
import argparse

# Options de ligne de commande
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--expert', action='store_true',
                     help="Afficher le modèle expert FWI (operationnal_plot et saisonnalité).")
_parser.add_argument('--only-seasonal', action='store_true',
                     help="Ne produire que les tracés de signal : section 1 (signal annuel) et "
                          "section 8 (saisonnalité). Saute les métriques de classification "
                          "(sections 2-7), la calibration, le scoring et la heatmap opérationnelle.")
_parser.add_argument('--only-operational', action='store_true',
                     help="Ne produire que le score opérationnel : heatmap opérationnelle, "
                          "heatmap de couverture et table LaTeX. Saute tous les tracés de "
                          "signal, la calibration et les figures par modèle/cluster.")
_args, _ = _parser.parse_known_args()
SHOW_EXPERT      = _args.expert
ONLY_SEASONAL    = _args.only_seasonal
ONLY_OPERATIONAL = _args.only_operational
import numpy as np
import pandas as pd

import sys
if not hasattr(pd.core.indexes, 'numeric'):
    from pandas.core.indexes.api import Index
    import pandas.core.indexes as pci
    class NumericIndexCompat:
        Int64Index = Index
        Float64Index = Index
        UInt64Index = Index
    pci.numeric = NumericIndexCompat()
    sys.modules['pandas.core.indexes.numeric'] = pci.numeric

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.dates as mdates
import datetime as dt
import math

from pathlib import Path
import sys
_gnn_path = str(Path(".").resolve())
if _gnn_path not in sys.path:
    sys.path.insert(0, _gnn_path)
from forecasting_models.sklearn.score import Scoring

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ── allDates ──────────────────────────────────────────────────────────────────
def find_dates_between(start, end):
    s = dt.datetime.strptime(start, '%Y-%m-%d').date()
    e = dt.datetime.strptime(end,   '%Y-%m-%d').date()
    delta, date, res = dt.timedelta(days=1), s, []
    while date <= e:
        res.append(date.strftime('%Y-%m-%d'))
        date += delta
    return res

def iou_score(y_true, y_pred):
    """
    Calcule les scores (aire commune, union, sous-prédiction, sur-prédiction) entre deux signaux.

    Args:
        t (np.array): Tableau de temps ou indices (axe x).
        y_pred (np.array): Signal prédiction (rouge).
        y_true (np.array): Signal vérité terrain (bleu).

    Returns:
        dict: Dictionnaire contenant les scores calculés.
    """
    
    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    return intersection / union if union > 0 else 0

allDates = find_dates_between('2017-06-12', '2025-12-31')

dataset = 'bdiff'

if dataset == "firemen":
    expe = "occurence_01_06_25"
    graph_construct = "full_all_3_0_risk-size-zonemeteo-degree-a3-r5-t0.3_node"
else:
    expe = "occurence_default"
    graph_construct = "full_all_departement_0_None_node"

# ── constants ─────────────────────────────────────────────────────────────────
BASE = Path(f'/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/{dataset}/firepoint/2x2/test/{expe}/all/{graph_construct}')
BASE_OUT  = Path('/home/caron/Bureau/bdiff_ordinal_loss_2_operationel_horizon')
BASE_OUT.mkdir(parents=True, exist_ok=True)

# dossiers de sortie dédiés par famille d'expériences
STUDENT_OUT   = Path('/home/caron/Bureau/bdiff_student_distillation')
FEDERATED_OUT = Path('/home/caron/Bureau/bdiff_federated')

# clés de configuration d'un groupe (à ne jamais confondre avec un modèle)
META_KEYS = {'_target', '_out', '_split'}

N_CLASSES   = 5
if '2024' in expe:
    YEAR_FILTER = 2024
else:
    YEAR_FILTER = 2023

def make_cols(base_target, model_name):
    """Return (TARGET, SIGNAL_COL, PRED_COL, PROBA_COLS) for a given base target and model name."""
    _signal_col_map = {
        'nbsinister': 'nbsinister',
        'timeintervention': 'timeintervention',
        'ressource': 'ressource',
        'burnedareaRoot': 'burnedareaRoot',
    }
    
    target     = f'{base_target}-kmeans-5-Class-Dept'
    signal_col = next((v for k, v in _signal_col_map.items() if k in base_target), 'nbsinister')
    
    if model_name == 'expert':
        return target, signal_col, None, None
        
    #if 'cll' in model_name:
    #    pred_col   = f'prediction_{base_target}-quantile-5-Class-Dept_0'
    #    proba_cols = [f'prediction_{target}_0_C{c}' for c in range(N_CLASSES)]
    #else:
    pred_col   = f'prediction_{target}_0'
    proba_cols = [f'prediction_{target}_0_C{c}' for c in range(N_CLASSES)]
    return target, signal_col, pred_col, proba_cols

# ── model groups ──────────────────────────────────────────────────────────────
"""models_prefixes = {
    'GRU': 'GRU_search_full_10_0_all_one_',
    'DilatedCNN': 'DilatedCNN_search_full_10_0_all_one_',
    'LSTM': 'LSTM_search_full_10_0_all_one_',
    'NetMLP': 'NetMLP_search_full_0_0_all_one_',
    #'GraphCastGRU': 'GraphCastTime_search_full_10_0_all_one_'
}"""

models_prefixes = {
    'GRU': 'GRU_search_full_10_0_all_one_',
    'LSTM': 'LSTM_search_full_10_0_all_one_',
    'DilatedCNN': 'DilatedCNN_search_full_10_0_all_one_',
    'NetMLP': 'NetMLP_search_full_0_0_all_one_',
    #'GraphCastGRU': 'GraphCastTime_search_full_10_0_all_one_'
}

losses_suffixes = {
    'pdegpd': 'regression_pdegpd',
    #'flwki': 'classification_flwki-id{departement}',
    'flwk': 'classification_flwk',
    'cornloss': 'corn_cornloss',
    'fl': 'classification_fl',
    'weightedce': 'classification_weightedcrossentropy',
    'wkloss': 'classification_wkloss',
    'bceloss': 'classification_bceloss',
}

losses_suffixes = {
   #'ranknet_2': 'regression_ranknet-id{cluster}-nclusters{4}-alphatype{cluster}',
   'ranknet': 'regression_ranknet-id{node}-nclusters{95}',
    #'cll': 'classification_cll',
    'flwki': 'classification_flwki-id{departement}',
    'ccllt': 'regression_ccllt-id{node}-nclusters{95}',
    'ccllt-2': 'regression_ccllt-id{cluster}-iddept{cluster}-nclusters{4}-alphatype{cluster}',
    #'ccllt_warm5': 'regression_ccllt-id{node}-nclusters{30}-warmupes{5}',
    #'ccllt_nomu': 'regression_ccllt-id{node}-nclusters{30}-warmupes{3000}',
    #'ccllt_nocov': 'regression_ccllt-id{node}-nclusters{30}-wcoverage{0.0}',
    #'ccllt_nomid': 'regression_ccllt-id{node}-nclusters{30}-wmid{0.0}',
    #'ccllt_noanchor': 'regression_ccllt-id{node}-nclusters{30}-wmu0{0.0}',
}

if dataset == 'firemen':
    targets_list = ['nbsinister', 'ressource', "timeintervention"]
    #targets_list = ['ressource']
else:
    targets_list = ['nbsinister']
    #targets_list = ['burnedareaRoot']
    pass
    
GROUPS = {}

for target in targets_list:
    group_name = f"{target}_all_models"
    GROUPS[group_name] = {'_target': target}
    for m_name, m_prefix in models_prefixes.items():
        for l_name, l_suffix in losses_suffixes.items():
            if 'ranknet' in l_suffix or 'cll' in l_suffix or 'ccllt' in l_suffix:
                # ccllt doesn't use the clustered target name
                folder_name = f"{m_prefix}{target}_{l_suffix}"
            else:
                folder_name = f"{m_prefix}{target}-kmeans-5-Class-Dept_{l_suffix}"
                
            GROUPS[group_name][f"{m_name}_{l_name}"] = folder_name

"""GROUPS['filter'] = {
    '_target':       'nbsinister',
    'GRU_2':  'filter-GRU-soft-weight-2_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_3':  'filter-GRU-soft-weight-3_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_4':  'filter-GRU-soft-weight-4_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_5':  'filter-GRU-soft-weight-5_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_7':  'filter-GRU-soft-weight-7_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_10': 'filter-GRU-soft-weight-10_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_14': 'filter-GRU-soft-weight-14_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_18': 'filter-GRU-soft-weight-18_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_19': 'filter-GRU-soft-weight-19_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_20': 'filter-GRU-soft-weight-20_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'Confidence_20': 'studenMLP-Confidence-T3.0-A1.0-B1.0-filter-GRU-soft-weight-20_full_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'Normal_10': 'studenMLP-normal-T3.0-A0.2-filter-GRU-soft-weight-10_full_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
}"""

"""GROUPS['federated'] = {
    '_target':               'nbsinister',
    'GRU_saison':           'federated-GRU-saison-fltg_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_cluster-encoder':  'federated-GRU-cluster-encoder-fltg_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_departement':      'federated-GRU-departement-fltg_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'GRU_mediterranean':    'federated-GRU-mediterranean-fltg_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'DilatedCNN_saison':    'federated-DilatedCNN-saison-fltg_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'DilatedCNN_cluster-encoder': 'federated-DilatedCNN-cluster-encoder-fltg_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'DilatedCNN_departement': 'federated-DilatedCNN-departement-fltg_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
    'DilatedCNN_mediterranean': 'federated-DilatedCNN-mediterranean-fltg_search_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
}"""

#GROUPS['student'] = {
#    '_target':            'nbsinister',
#    'Confidence-20':      'studenMLP-Confidence-T3.0-A1.0-B1.0-filter-GRU-soft-weight-20_full_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
#    'normal-10':  'studenMLP-normal-T3.0-A0.2-filter-GRU-soft-weight-10_full_full_10_0_all_one_nbsinister-kmeans-5-Class-Dept_classification_wkloss',
#}

# ── studentMLP knowledge-distillation groups ──────────────────────────────────
# Les élèves MLP sont distillés depuis un ensemble enseignant filter-GRU-soft-weight-{n}.
# Le chiffre de 'soft-weight-{n}' est le num_test du vote soft-weight
# (cf. train_any_model.py:444, test_name = f'filter-{type}-{config_weight}-{nt}_...'),
# c.-à-d. LE NOMBRE DE MODÈLES ENSEIGNANTS agrégés pour produire les cibles molles.
#
# Les configurations ne sont pas codées en dur : elles sont découvertes sur disque,
# donc toute nouvelle run apparaît automatiquement dans l'analyse.
# Label = '<variante complète>-<mode>mode_<n>-teachers' : température, poids de perte,
# mode d'entraînement et nombre d'enseignants sont tous lisibles dans les sorties.
import re

STUDENT_TARGET = 'nbsinister'
STUDENT_TASK   = 'nbsinister-kmeans-5-Class-Dept_classification_wkloss'

# méthodes de distillation retenues (préfixe de la variante).
# None = toutes les méthodes présentes sur disque.
STUDENT_METHODS = ['normal', 'Confidence']

# modes d'entraînement retenus pour la référence sans distillation.
# None = tous les modes présents sur disque.
STUDENT_REF_MODES = ['search']

_STUDENT_RE = re.compile(
    r'^studenMLP-(?P<variant>.+)-filter-GRU-soft-weight-(?P<n>\d+)_(?P<mode>[a-z]+)_'
)

def _has_pred(folder):
    return bool(list((BASE / folder / 'H0').glob('*_all_pred.pkl')))

def discover_students(methods=STUDENT_METHODS):
    """Toutes les runs élève distillées présentes sur disque -> {(variante, mode, n): dossier}."""
    found = {}
    for d in sorted(BASE.glob(f'studenMLP-*-filter-GRU-soft-weight-*_{STUDENT_TASK}')):
        m = _STUDENT_RE.match(d.name)
        if not m or not _has_pred(d.name):
            continue
        variant = m.group('variant')
        if methods is not None and not any(variant.startswith(x) for x in methods):
            continue
        found[(variant, m.group('mode'), int(m.group('n')))] = d.name
    return found

def discover_student_references(modes=STUDENT_REF_MODES):
    """Élèves entraînés sans distillation -> même axe que les élèves, 0 enseignant."""
    refs = {}
    for d in sorted(BASE.glob(f'studenMLP_*_{STUDENT_TASK}')):
        mode = d.name.split('_')[1]
        if modes is not None and mode not in modes:
            continue
        if _has_pred(d.name):
            refs[f'MLP-noKD-{mode}mode_0-teachers'] = d.name
    return refs

if dataset == 'bdiff':
    students   = discover_students()
    references = discover_student_references()

    # un groupe par taille d'ensemble enseignant + un groupe global
    all_students = {'_target': STUDENT_TARGET, '_out': STUDENT_OUT}
    for n_teachers in sorted({n for _, _, n in students}):
        group_name = f'studentMLP_{n_teachers}-teachers'
        GROUPS[group_name] = {'_target': STUDENT_TARGET, '_out': STUDENT_OUT}
        for (variant, mode, n), folder in students.items():
            if n != n_teachers:
                continue
            label = f'{variant}-{mode}mode_{n}-teachers'
            GROUPS[group_name][label] = folder
            all_students[label] = folder
        GROUPS[group_name].update(references)

    all_students.update(references)
    GROUPS['studentMLP_all'] = all_students

# ── federated groups ──────────────────────────────────────────────────────────
# Nom de dossier : {prefixe}-{archi}-{schema}-{agregation}_{info}
#   prefixe     : l'algorithme fédéré (cf. train_any_model.py:909-918)
#                   federated     -> Fed   (agrégation simple)
#                   moonfederated -> MOON  (perte contrastive model-level)
#                   alafederated  -> ALA   (adaptive local aggregation)
#   archi       : GRU, DilatedCNN, ...
#   schema      : partitionnement des clients (saison, cluster-encoder,
#                 departement, mediterranean)
#   agregation  : fltg | weighted -> règle de fusion des poids côté serveur
# Le segment 'search' de {info} n'est pas un paramètre du modèle mais le mode de
# recherche d'hyperparamètres : il ne figure pas dans les labels.
FEDERATED_TASK = 'nbsinister-kmeans-5-Class-Dept_classification_wkloss'

# algorithme -> préfixe de dossier (None en valeur = famille absente du disque)
FEDERATED_ALGOS = {
    'Fed':  'federated',
    'MOON': 'moonfederated',
    'ALA':  'alafederated',
}

# architectures retenues (None = toutes celles présentes sur disque)
FEDERATED_ARCHS = None

_FED_AGG = 'fltg|weighted|median|mean|max'

def discover_federated(algos=FEDERATED_ALGOS, archs=FEDERATED_ARCHS):
    """Runs fédérées sur disque -> {(algo, archi, schema, agregation): dossier}."""
    found = {}
    for algo, prefix in algos.items():
        rx = re.compile(
            rf'^{prefix}-(?P<arch>[A-Za-z0-9]+)-(?P<scheme>.+)-(?P<agg>{_FED_AGG})_'
        )
        for d in sorted(BASE.glob(f'{prefix}-*_{FEDERATED_TASK}')):
            m = rx.match(d.name)
            if not m or not _has_pred(d.name):
                continue
            arch = m.group('arch')
            if archs is not None and arch not in archs:
                continue
            found[(algo, arch, m.group('scheme'), m.group('agg'))] = d.name
    return found

if dataset == 'bdiff':
    federated = discover_federated()

    # label = '<archi>-<schema>-<agregation>_<algo>' : le découpage aval donne
    # model = archi+schéma+agrégation, loss = algorithme fédéré. Les lignes de la
    # heatmap se regroupent donc par Fed / MOON / ALA.
    # '_split' : une heatmap par règle d'agrégation, sinon 48 lignes dans une figure.
    FEDERATED_SPLIT = ('fltg', 'weighted')

    all_federated = {'_target': STUDENT_TARGET, '_out': FEDERATED_OUT,
                     '_split': FEDERATED_SPLIT}
    for algo in FEDERATED_ALGOS:
        members = {f'{arch}-{scheme}-{agg}_{algo}': folder
                   for (a, arch, scheme, agg), folder in federated.items() if a == algo}
        if not members:
            continue
        GROUPS[f'federated_{algo}'] = {'_target': STUDENT_TARGET, '_out': FEDERATED_OUT,
                                       '_split': FEDERATED_SPLIT, **members}
        all_federated.update(members)

    if len(all_federated) > 3:
        GROUPS['federated_all'] = all_federated

# auto-assign colors per group
_PALETTE = ['#E07B39', '#3A86FF', '#2EC4B6', '#CC2936', '#8338EC', '#FB5607']

def make_colors(labels):
    return {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(labels)}

CLUSTER_BACKGROUNDS = {
    "cluster_0": "#EAF3FF",  # bleu très clair
    "cluster_1": "#EEF9E8",  # vert très clair
    "cluster_2": "#FFF1E3",  # orange très clair
    "cluster_3": "#F2EAFE",  # violet très clair
}

def plot_operational_heatmap(
    df: pd.DataFrame,
    title: str = "Performance opérationnelle en fonction du schéma",
    k_cols=("k1", "k2", "k3", "k4"),
    recall_col="recall",
    iou_col="iou",
    loss_col="loss",
    model_col="model",
    cluster_col="cluster",
    cluster_backgrounds=None,
    figsize=(22, 13),
    save_path=None,
    dpi=300,
    show_expert=None,
):
    """
    df attendu au format long :
    columns = [loss, model, cluster, k1, k2, k3, k4, recall]
    """
    cluster_backgrounds = cluster_backgrounds or CLUSTER_BACKGROUNDS

    df = df.copy()
    clusters = list(df[cluster_col].drop_duplicates())
    losses = list(df[loss_col].drop_duplicates())

    # Modèles par loss, dans l'ordre d'apparition
    if show_expert is None:
        show_expert = SHOW_EXPERT
    rows = []
    for loss in losses:
        models = list(df.loc[df[loss_col] == loss, model_col].drop_duplicates())
        # Filtrer expert selon show_expert
        if not show_expert:
            models = [m for m in models if m != 'expert']
        for model in models:
            rows.append((loss, model))

    # Force expert to be first si demandé
    if show_expert:
        rows.sort(key=lambda x: 0 if x[1] == 'expert' else 1)
    else:
        rows.sort(key=lambda x: 1 if x[1] == 'expert' else 0)

    n_rows = len(rows)
    n_clusters = len(clusters)
    n_metrics = len(k_cols) + 2

    # Colonnes de gauche dimensionnées sur les libellés les plus longs : sinon les
    # noms débordent sur les cellules de métriques. ~0.11 unité par caractère à
    # fontsize 10 ; la figure s'élargit d'autant pour garder des cellules de
    # taille constante, et grandit en hauteur quand il y a beaucoup de lignes.
    _loss_w  = max(1.0, max((len(str(l)) for l, _ in rows), default=6) * 0.11)
    _model_w = max(1.0, max((len(str(m)) for _, m in rows), default=10) * 0.11)
    _left_w  = _loss_w + _model_w
    _loss_x, _model_x = _loss_w / 2, _loss_w + _model_w / 2
    _span    = _left_w + n_clusters * n_metrics
    figsize  = (figsize[0] * _span / (2.0 + n_clusters * n_metrics),
                max(figsize[1], 0.75 * n_rows + 4))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, _span)
    ax.set_ylim(0, n_rows + 3)
    ax.axis("off")

    # Normalisations communes — vmax = max score atteint dans le groupe
    k_values = df[list(k_cols)].to_numpy().ravel()
    k_min = np.nanmin(k_values)
    k_max = np.nanmax(k_values)

    # Anchor colormap to actual data range (not symmetric)
    # vmin < 0 < vmax required by TwoSlopeNorm
    k_norm = TwoSlopeNorm(
        vmin=min(k_min, -1e-9),
        vcenter=0,
        vmax=max(k_max, 1e-9),
    )
    recall_norm = Normalize(vmin=0, vmax=1)
    iou_norm = Normalize(vmin=0, vmax=1)

    cmap_k = plt.cm.RdYlGn
    cmap_recall = plt.cm.Blues
    cmap_iou = plt.cm.Purples

    # Paramètres visuels
    left_w = _left_w
    cell_w = 1.0
    cell_h = 0.75
    header_h = 1.25
    top_y = n_rows + 1.2

    # Titre
    ax.text(
        left_w + n_clusters * n_metrics / 2,
        n_rows + 2.5,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # En-têtes gauche
    ax.text(_loss_x, top_y - 0.6, "Schéma", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(_model_x, top_y - 0.6, "Modèle", ha="center", va="center", fontsize=10, fontweight="bold")

    # En-têtes clusters
    for c_idx, cluster in enumerate(clusters):
        x0 = left_w + c_idx * n_metrics
        bg = cluster_backgrounds.get(cluster, "#F7F7F7")

        ax.add_patch(
            Rectangle(
                (x0, 0.4),
                n_metrics,
                n_rows + header_h,
                facecolor=bg,
                edgecolor="0.65",
                linewidth=1.2,
                zorder=0,
            )
        )

        ax.text(
            x0 + n_metrics / 2,
            top_y + 0.15,
            cluster,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        for j, col in enumerate(list(k_cols) + [recall_col, iou_col]):
            ax.text(
                x0 + j + 0.5,
                top_y - 0.65,
                "Recall" if col == recall_col else ("IoU" if col == iou_col else col),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    # Lignes
    y = n_rows
    previous_loss = None
    
    for row_idx, (loss, model) in enumerate(rows):
        y_pos = n_rows - row_idx - 0.1

        if loss != previous_loss:
            ax.plot([0, left_w + n_clusters * n_metrics], [y_pos + 0.45, y_pos + 0.45], color="0.55", lw=0.8)
            ax.text(_loss_x, y_pos, loss, ha="center", va="center", fontsize=10, fontweight="bold")
            previous_loss = loss

        ax.text(_model_x, y_pos, model, ha="center", va="center", fontsize=10, fontweight="bold")

        for c_idx, cluster in enumerate(clusters):
            x0 = left_w + c_idx * n_metrics

            sub = df[
                (df[loss_col] == loss)
                & (df[model_col] == model)
                & (df[cluster_col] == cluster)
            ]

            if sub.empty:
                values = [np.nan] * n_metrics
            else:
                values = sub.iloc[0][list(k_cols) + [recall_col, iou_col]].to_list()

            for j, val in enumerate(values):
                is_recall = j == len(k_cols)
                is_iou = j == len(k_cols) + 1

                if pd.isna(val):
                    color = "#FFFFFF"
                    text = "--"
                else:
                    if is_recall:
                        color = cmap_recall(recall_norm(val))
                    elif is_iou:
                        color = cmap_iou(iou_norm(val))
                    else:
                        color = cmap_k(k_norm(val))
                        
                    val_round = round(val, 2)
                    if val_round == 0.0 and val != 0.0:
                        val_round = 0.01 if val > 0 else -0.01
                    text = f"{val_round:.2f}"

                ax.add_patch(
                    Rectangle(
                        (x0 + j, y_pos - cell_h / 2),
                        cell_w,
                        cell_h,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=1.0,
                    )
                )

                ax.text(
                    x0 + j + 0.5,
                    y_pos,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white" if (is_recall or is_iou) and not pd.isna(val) and val > 0.65 else "black",
                )

    # Colorbars
    sm_k = plt.cm.ScalarMappable(norm=k_norm, cmap=cmap_k)
    sm_r = plt.cm.ScalarMappable(norm=recall_norm, cmap=cmap_recall)
    sm_i = plt.cm.ScalarMappable(norm=iou_norm, cmap=cmap_iou)

    cax1 = fig.add_axes([0.15, 0.055, 0.2, 0.018])
    cbar1 = fig.colorbar(sm_k, cax=cax1, orientation="horizontal")
    cbar1.set_label("Scores ordinaux k1–k4", fontsize=10)

    cax2 = fig.add_axes([0.4, 0.055, 0.2, 0.018])
    cbar2 = fig.colorbar(sm_r, cax=cax2, orientation="horizontal")
    cbar2.set_label("Recall", fontsize=10)

    cax3 = fig.add_axes([0.65, 0.055, 0.2, 0.018])
    cbar3 = fig.colorbar(sm_i, cax=cax3, orientation="horizontal")
    cbar3.set_label("IoU", fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

def plot_horizons_score_recall_heatmap(
    df: pd.DataFrame,
    title: str = "Heatmaps Score / Recall vs Horizon",
    k_cols=("score_k1", "score_k2", "score_k3", "score_k4"),
    recall_col="recall",
    save_path=None,
    dpi=300,
    show_expert=None,
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm, Normalize
    
    df = df.copy()
    
    if show_expert is None:
        show_expert = SHOW_EXPERT
        
    losses = list(df['loss'].drop_duplicates())
    rows = []
    for loss in losses:
        models_for_loss = list(df.loc[df['loss'] == loss, 'model'].drop_duplicates())
        if not show_expert:
            models_for_loss = [m for m in models_for_loss if m != 'expert']
        for model in models_for_loss:
            rows.append((loss, model))
            
    if show_expert:
        rows.sort(key=lambda x: 0 if x[1] == 'expert' else 1)
    else:
        rows.sort(key=lambda x: 1 if x[1] == 'expert' else 0)
        
    def format_label(lbl):
        if '_' in lbl:
            parts = lbl.split('_', 1)
            m_type = parts[0]
            l_type = parts[1]
            if l_type.lower() == 'ccllt':
                l_type = 'CCLLT'
            elif l_type.lower() == 'ranknet':
                l_type = 'RankNet'
            else:
                l_type = l_type.capitalize()
            return f"{m_type}-{l_type}"
        return lbl

    horizons = sorted(df['horizon'].unique())
    n_horizons = len(horizons)
    n_rows = len(rows)
    if n_rows == 0 or n_horizons == 0:
        return
        
    metrics = list(k_cols) + [recall_col]
    metric_titles = ["k1", "k2", "k3", "k4", "Recall"]
    n_metrics = len(metrics)
    
    k_values = df[list(k_cols)].values
    k_min = np.nanmin(k_values)
    k_max = np.nanmax(k_values)
    
    k_norm = TwoSlopeNorm(
        vmin=min(k_min, -1e-9),
        vcenter=0,
        vmax=max(k_max, 1e-9),
    )
    recall_norm = Normalize(vmin=0, vmax=1)
    
    cmap_k = plt.cm.RdYlGn
    cmap_recall = plt.cm.Blues
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(22, max(4, n_rows * 0.8 + 2)))
    if n_metrics == 1: axes = [axes]
    
    for idx, (metric, m_title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        is_recall = (metric == recall_col)
        
        mat = np.full((n_rows, n_horizons), np.nan)
        for i, (loss, m) in enumerate(rows):
            for j, h in enumerate(horizons):
                sub = df[(df['model'] == m) & (df['horizon'] == h)]
                if not sub.empty:
                    mat[i, j] = sub.iloc[0][metric]
                    
        norm = recall_norm if is_recall else k_norm
        cmap = cmap_recall if is_recall else cmap_k
            
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')
        
        for i in range(n_rows):
            for j in range(n_horizons):
                val = mat[i, j]
                if not np.isnan(val):
                    text_color = "white" if is_recall and val > 0.65 else "black"
                    val_round = round(val, 2)
                    if val_round == 0.0 and val != 0.0:
                        val_round = 0.01 if val > 0 else -0.01
                    ax.text(j, i, f"{val_round:.2f}", ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
                    
        ax.set_title(m_title, fontsize=12)
        ax.set_xticks(np.arange(n_horizons))
        ax.set_xticklabels([f"H{int(h)}" for h in horizons])
        
        prev_loss = None
        for i, (loss, m) in enumerate(rows):
            if prev_loss is not None and loss != prev_loss:
                ax.axhline(i - 0.5, color='black', linewidth=1.5)
            prev_loss = loss
        
        if idx == 0:
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels([format_label(m) for l, m in rows])
        else:
            ax.set_yticks([])
            
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# ── helpers ───────────────────────────────────────────────────────────────────
def load_pkl(model_name, h="H0"):
    folder = BASE / model_name / h
    pkls = list(folder.glob('*_all_pred.pkl'))
    if not pkls:
        return None
    assert len(pkls) == 1, f'Expected 1 pkl, got {len(pkls)} in {folder}'
    with open(pkls[0], 'rb') as f:
        return pickle.load(f)
    
def load_group(models_dict):
    dfs = {}
    for label, name in models_dict.items():
        try:
            df = load_pkl(name)
            if df is None:
                continue
            df = df.copy()
            df['datetime'] = df['date'].astype(int).apply(lambda i: pd.Timestamp(allDates[i]))
            dfs[label] = df[df['datetime'].dt.year == YEAR_FILTER].reset_index(drop=True)
        except Exception as e:
            print(f'  [SKIP] {label}: {e}')
    return dfs

def get_season(d):
    m = d.month
    if m in [12, 1, 2]:  return 'Winter'
    elif m in [3, 4, 5]: return 'Spring'
    elif m in [6, 7, 8]: return 'Summer'
    else:                return 'Autumn'

def plot_dt(df, season):
    d = df.copy()
    if season == 'Winter':
        dec = d['datetime'].dt.month == 12
        d.loc[dec, 'datetime'] = d.loc[dec, 'datetime'] - pd.DateOffset(years=1)
    return d.sort_values('datetime')

def plot_calibration_with_background(probs, y_true, n_bins=10, figsize=(25, 5), average='micro'):
    probs  = np.asarray(probs, float)
    y_true = np.asarray(y_true, int)
    N, K   = probs.shape
    bins    = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, K, figsize=figsize)
    if K == 1:
        axes = [axes]
    eces, stats = [], {}
    for c in range(K):
        p = probs[:, c]
        t = (y_true == c).astype(float)
        conf_means, freq_means, counts = [], [], []
        ece_c, used_bins = 0.0, 0
        for b in range(n_bins):
            lo, hi = bins[b], bins[b + 1]
            mask = (p >= lo) & (p < hi if b < n_bins - 1 else p <= hi)
            nb = int(mask.sum())
            counts.append(nb)
            if nb == 0:
                conf_means.append(np.nan); freq_means.append(np.nan); continue
            cm = p[mask].mean(); fm = t[mask].mean()
            conf_means.append(cm); freq_means.append(fm)
            if average == 'micro':
                ece_c += (nb / N) * abs(fm - cm)
            else:
                ece_c += abs(fm - cm); used_bins += 1
        if average == 'macro' and used_bins > 0:
            ece_c /= used_bins
        eces.append(ece_c)
        stats[f'ECE_class_{c}'] = float(ece_c)
        conf_means = np.asarray(conf_means); freq_means = np.asarray(freq_means); counts = np.asarray(counts)
        valid = np.isfinite(conf_means) & np.isfinite(freq_means)
        ax = axes[c]
        ax.plot([0, 1], [0, 1], 'k:', lw=1)
        ax.plot(conf_means[valid], freq_means[valid], 'o-', lw=2, label=f'ECE = {ece_c:.3f}')
        ax.set_xlabel(f'P(Y = {c})'); ax.set_ylabel(f'Observed freq. (Y = {c})')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.1); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.text(0.02, 0.97, f'Class {c}', transform=ax.transAxes, fontweight='bold', va='top', fontsize=9)
        ax2 = ax.twinx()
        ax2.bar(centers, counts, width=(bins[1] - bins[0]) * 0.9, alpha=0.12)
        ax2.set_yticks([])
    stats['ECE_macro'] = float(np.nanmean(eces))
    fig.tight_layout()
    return fig, axes, stats

# ── main loop over groups ─────────────────────────────────────────────────────
RUN_ONLY = None  # set to a list of group names to restrict, e.g.
                 # ['studentMLP_all'] or ['studentMLP_10-teachers', 'studentMLP_14-teachers', 'studentMLP_20-teachers']
all_scoring_rows = []

for group_name, _group_dict in GROUPS.items():
    if RUN_ONLY is not None and group_name not in RUN_ONLY:
        continue
    print(f'\n{"="*60}')
    print(f'  Group: {group_name}')
    print(f'{"="*60}')

    base_target = _group_dict.get('_target', 'nbsinister')
    group_out   = Path(_group_dict.get('_out', BASE_OUT))
    group_split = tuple(_group_dict.get('_split', ()))
    models_dict = {k: v for k, v in _group_dict.items() if k not in META_KEYS}
    TARGET, NBSIN_COL, _, _ = make_cols(base_target, list(models_dict.values())[0])
    print(f'  Target: {TARGET}')

    dfs_loaded = load_group(models_dict)
    if not dfs_loaded:
        print('  No data loaded, skipping.')
        continue

    # Prepare column mappings for loaded models
    pred_col_map = {}
    proba_cols_map = {}
    for label, m_path in models_dict.items():
        _, _, pc, pb = make_cols(base_target, m_path)
        pred_col_map[label] = pc
        proba_cols_map[label] = pb

    # === EXPERT BASELINE CALCULATION ===
    df_ref_full = list(dfs_loaded.values())[0].copy()
    zone_col_temp = 'departement' if dataset == 'firemen' else 'cluster-encoder'
    
    expert_fwi_annuel = pd.Series(index=df_ref_full.index, dtype=float)
    if zone_col_temp in df_ref_full.columns and 'fwi_mean' in df_ref_full.columns:
        for z in df_ref_full[zone_col_temp].unique():
            mask_z = df_ref_full[zone_col_temp] == z
            vals = df_ref_full.loc[mask_z, 'fwi_mean'].copy()
            if len(vals.dropna()) > 5:
                try:
                    classes = pd.qcut(vals.rank(method='first'), q=[0.0, 0.50, 0.75, 0.95, 0.99, 1.0], labels=[0, 1, 2, 3, 4])
                    expert_fwi_annuel[mask_z] = classes.astype(float)
                except Exception:
                    expert_fwi_annuel[mask_z] = 0.0
            else:
                expert_fwi_annuel[mask_z] = 0.0
    df_ref_full['EXPERT_FWI_ANNUEL'] = expert_fwi_annuel

    models_dict_orig = {k: v for k, v in _group_dict.items() if k not in META_KEYS}

    for run_type in ['annuel', 'estival']:
        # Reset models_dict to avoid accumulating expert keys across iterations
        models_dict = dict(models_dict_orig)

        print(f'\n  --- Run: {run_type} ---')
        OUT = group_out / dataset / base_target / run_type
        OUT.mkdir(parents=True, exist_ok=True)

        dfs = {}
        for label, df in dfs_loaded.items():
            if run_type == 'estival':
                mask = ((df['datetime'].dt.month == 6) & (df['datetime'].dt.day >= 15)) | \
                       (df['datetime'].dt.month == 7) | \
                       (df['datetime'].dt.month == 8) | \
                       ((df['datetime'].dt.month == 9) & (df['datetime'].dt.day <= 25))
                dfs[label] = df[mask].copy()
            else:
                dfs[label] = df.copy()

        # Build expert model for the current run_type
        df_exp = df_ref_full.copy()
        if run_type == 'estival':
            mask_estival = ((df_exp['datetime'].dt.month == 6) & (df_exp['datetime'].dt.day >= 15)) | \
                           (df_exp['datetime'].dt.month == 7) | \
                           (df_exp['datetime'].dt.month == 8) | \
                           ((df_exp['datetime'].dt.month == 9) & (df_exp['datetime'].dt.day <= 25))
            df_exp = df_exp[mask_estival].copy()

        PRED_COL_EXP = f"prediction_{TARGET}_expert"
        PROBA_COLS_EXP = [f"{PRED_COL_EXP}_C{c}" for c in range(N_CLASSES)]
        df_exp[PRED_COL_EXP] = df_exp['EXPERT_FWI_ANNUEL']

        # Ensure the TARGET column exists in df_exp (for non-nbsinister targets)
        if TARGET not in df_exp.columns:
            # Try to copy from the first loaded model df filtered to same dates
            ref_df = list(dfs.values())[0] if dfs else None
            if ref_df is not None and TARGET in ref_df.columns:
                df_exp = df_exp.merge(
                    ref_df[['datetime', zone_col_temp, TARGET]].drop_duplicates(),
                    on=['datetime', zone_col_temp], how='left'
                )
            else:
                print(f'  [SKIP expert] TARGET column {TARGET} not found in ref df.')
                continue

        # --- Expert FWI (all departments) ---
        df_exp_fwi = df_exp.copy()
        df_exp_fwi[PRED_COL_EXP] = df_exp_fwi['EXPERT_FWI_ANNUEL']
        preds_fwi = df_exp_fwi[PRED_COL_EXP].fillna(0).clip(0, 4).astype(int)
        for c in range(5):
            df_exp_fwi[PROBA_COLS_EXP[c]] = (preds_fwi == c).astype(float)
        dfs['expert_fwi_mean'] = df_exp_fwi
        models_dict['expert_fwi_mean'] = 'expert'
        pred_col_map['expert_fwi_mean'] = PRED_COL_EXP
        proba_cols_map['expert_fwi_mean'] = PROBA_COLS_EXP

        # --- Expert DFE (dept 6 only, estival firemen) ---
        if run_type == 'estival' and dataset == 'firemen' and 'DFE' in df_exp.columns:
            mask_dept6 = df_exp[zone_col_temp].astype(str).str.endswith('6')
            df_exp_dfe = df_exp[mask_dept6].copy()
            if len(df_exp_dfe) > 0:
                PRED_COL_DFE = f"prediction_{TARGET}_expert_dfe"
                PROBA_COLS_DFE = [f"{PRED_COL_DFE}_C{c}" for c in range(5)]
                df_exp_dfe[PRED_COL_DFE] = df_exp_dfe['DFE'].fillna(0).astype(float)
                preds_dfe = df_exp_dfe[PRED_COL_DFE].clip(0, 4).astype(int)
                for c in range(5):
                    df_exp_dfe[PROBA_COLS_DFE[c]] = (preds_dfe == c).astype(float)
                dfs['expert_DFE'] = df_exp_dfe
                models_dict['expert_DFE'] = 'expert'
                pred_col_map['expert_DFE'] = PRED_COL_DFE
                proba_cols_map['expert_DFE'] = PROBA_COLS_DFE

        if any(len(d) == 0 for d in dfs.values()):
            print('  No data for this period, skipping.')
            continue

        COLORS = make_colors(list(dfs.keys()))
        ref_label = list(dfs.keys())[0]
        n = len(dfs)

        for label in dfs:
            dfs[label]['season'] = dfs[label]['datetime'].apply(get_season)

        # défini par la section 6 ; valeur de repli quand les sections 2-7 sont sautées
        has_probas = False

        # section 1 : tracé du signal, inutile pour le seul score opérationnel
        if not ONLY_OPERATIONAL:
            # ── Section 1: Aggregated temporal signal ─────────────────────────────────
            all_daily_means = []
            for label, df in dfs.items():
                if not SHOW_EXPERT and models_dict.get(label) == 'expert':
                    continue
                PRED_COL = pred_col_map[label]
                daily = df.groupby('datetime').agg(
                    pred_class=(PRED_COL, 'mean'),
                    true_class=(TARGET,   'mean'),
                ).reset_index()
                all_daily_means.append(daily['pred_class'].values)
                all_daily_means.append(daily['true_class'].values)
            combined = np.concatenate(all_daily_means)
            margin = 0.1 * (combined.max() - combined.min())
            class_ymin = combined.min() - margin
            class_ymax = combined.max() + margin

            fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), sharex=True)
            if n == 1: axes = [axes]

            for ax, (label, df) in zip(axes, dfs.items()):
                PRED_COL = pred_col_map[label]
                daily = df.groupby('datetime').agg(
                    signal_sum=(NBSIN_COL, 'sum'),
                    pred_class=(PRED_COL,  'mean'),
                    true_class=(TARGET,    'mean'),
                ).reset_index()
                ax2 = ax.twinx()
                ax.fill_between(daily['datetime'], daily['signal_sum'], alpha=0.2, color='gray', label=f'{NBSIN_COL} (sum)')
                ax.plot(daily['datetime'], daily['signal_sum'], color='gray', lw=0.8)
                ax2.plot(daily['datetime'], daily['true_class'], color='black',       lw=1.2, linestyle='--', label='true class')
                ax2.plot(daily['datetime'], daily['pred_class'], color=COLORS[label], lw=1.5, linestyle='-',  label=f'predicted — {label}')
                ax2.set_ylim(class_ymin, class_ymax)
                ax.set_ylabel(f'{NBSIN_COL} (sum)', fontsize=8)
                ax2.set_ylabel('kmeans class', fontsize=8)
                ax.text(0.01, 0.95, label, transform=ax.transAxes, fontweight='bold', va='top', fontsize=9)
                lines1, l1 = ax.get_legend_handles_labels()
                lines2, l2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, l1 + l2, fontsize=7, loc='upper right')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            fig.savefig(OUT / f'{group_name}_1_temporal_signal.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {group_name}_1_temporal_signal.png')

        # sections 2 à 7 : métriques de classification, inutiles pour les seuls tracés
        if not (ONLY_SEASONAL or ONLY_OPERATIONAL):
            # ── Section 2: Class distribution ─────────────────────────────────────────
            classes = np.arange(N_CLASSES)
            fig, ax = plt.subplots(figsize=(12, 5))
            w = 0.8 / (n + 1)
            offsets = np.linspace(-(n * w) / 2, (n * w) / 2, n)
            ref_dist = None
            for (label, df), offset in zip(dfs.items(), offsets):
                PRED_COL = pred_col_map[label]
                pred_dist = df[PRED_COL].value_counts(normalize=True).reindex(classes, fill_value=0)
                if ref_dist is None:
                    ref_dist = df[TARGET].value_counts(normalize=True).reindex(classes, fill_value=0)
                ax.bar(classes + offset, pred_dist.values, w * 0.9, label=label, color=COLORS[label], alpha=0.8)
            ax.bar(classes + offsets[-1] + w, ref_dist.values, w * 0.9, label='true', color='#333333', alpha=0.5)
            ax.set_xticks(classes); ax.set_xlabel('kmeans class'); ax.set_ylabel('Relative frequency')
            ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            fig.savefig(OUT / f'{group_name}_2_class_distribution.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {group_name}_2_class_distribution.png')

            # ── Section 3: Confusion matrices ─────────────────────────────────────────
            ncols = min(n, 3)
            nrows = (n + ncols - 1) // ncols
            fig, axes_cm = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
            axes_flat = np.array(axes_cm).flatten()
            for ax, (label, df) in zip(axes_flat, dfs.items()):
                PRED_COL = pred_col_map[label]
                y_true = df[TARGET].astype(int)
                y_pred = df[PRED_COL].astype(int)
                cm   = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)), normalize='true')
                disp = ConfusionMatrixDisplay(cm, display_labels=list(range(N_CLASSES)))
                disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='.2f')
                ax.set_title(label, fontsize=8)
            for ax in axes_flat[n:]:
                ax.set_visible(False)
            plt.tight_layout()
            fig.savefig(OUT / f'{group_name}_3_confusion_matrices.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {group_name}_3_confusion_matrices.png')

            # ── Section 4: Global metrics ──────────────────────────────────────────────
            rows = []
            for label, df in dfs.items():
                PRED_COL = pred_col_map[label]
                y_true = df[TARGET].astype(int); y_pred = df[PRED_COL].astype(int)
                rows.append({
                    'model':       label,
                    'accuracy':    accuracy_score(y_true, y_pred),
                    'f1_macro':    f1_score(y_true, y_pred, average='macro',    zero_division=0),
                    'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                    'rec_macro':   recall_score(y_true, y_pred, average='macro',    zero_division=0),
                    'prec_macro':  precision_score(y_true, y_pred, average='macro', zero_division=0),
                    'iou':  iou_score(y_true, y_pred),
                })
            metrics_df = pd.DataFrame(rows).set_index('model')
            fig, ax = plt.subplots(figsize=(8, 0.5 * n + 1.5))
            ax.axis('off')
            tbl = ax.table(
                cellText=metrics_df.round(4).values,
                rowLabels=metrics_df.index,
                colLabels=metrics_df.columns,
                loc='center', cellLoc='center',
            )
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.5)
            plt.tight_layout()
            fig.savefig(OUT / f'{group_name}_4_global_metrics.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {group_name}_4_global_metrics.png')

            # ── Section 5: F1 per class ────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(13, 5))
            x = np.arange(N_CLASSES); w = 0.8 / n
            for i, (label, df) in enumerate(dfs.items()):
                PRED_COL = pred_col_map[label]
                y_true = df[TARGET].astype(int); y_pred = df[PRED_COL].astype(int)
                f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0)
                offset = (i - (n - 1) / 2) * w
                ax.bar(x + offset, f1_per_class, w * 0.9, label=label, color=COLORS[label], alpha=0.85, edgecolor='white', linewidth=0.5)
            ax.set_xticks(x); ax.set_xticklabels([f'Class {c}' for c in range(N_CLASSES)])
            ax.set_ylabel('F1-score'); ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            fig.savefig(OUT / f'{group_name}_5_f1_per_class.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {group_name}_5_f1_per_class.png')

            try:
                # ── Section 6: Predicted probabilities (softmax) per true class ───────────
                has_probas = any(proba_cols_map[lbl][0] in df.columns and df[proba_cols_map[lbl][0]].notna().any() for lbl, df in dfs.items())
                if has_probas:
                    fig, axes_p = plt.subplots(N_CLASSES, n, figsize=(5 * n, 3 * N_CLASSES), sharey=False)
                    if n == 1: axes_p = axes_p.reshape(N_CLASSES, 1)
                    for col, (label, df) in enumerate(dfs.items()):
                        for true_cls in range(N_CLASSES):
                            ax = axes_p[true_cls, col]
                            PROBA_COLS = proba_cols_map[label]
                            mask = df[TARGET].astype(int) == true_cls
                            sub  = df.loc[mask, PROBA_COLS]
                            ax.boxplot(
                                [sub[c].dropna().values for c in PROBA_COLS],
                                labels=[f'C{i}' for i in range(N_CLASSES)],
                                patch_artist=True,
                                boxprops=dict(facecolor=COLORS[label], alpha=0.6),
                            )
                            ax.set_ylabel('P(class)', fontsize=7); ax.set_ylim(0, 1); ax.tick_params(labelsize=6)
                            ax.text(0.02, 0.96, f'{label} | cls {true_cls}', transform=ax.transAxes, fontsize=6, va='top')
                    plt.tight_layout()
                    fig.savefig(OUT / f'{group_name}_6_softmax_per_class.png', dpi=120, bbox_inches='tight')
                    plt.close(fig)
                    print(f'  Saved: {group_name}_6_softmax_per_class.png')
            except Exception as e:
                has_probas = False
                pass

            # ── Section 7: Per-department accuracy & MSE ──────────────────────────────
            dept_rows = []
            for label, df in dfs.items():
                PRED_COL = pred_col_map[label]
                y_true = df[TARGET].astype(int); y_pred = df[PRED_COL].astype(int)
                tmp = df[['departement']].copy()
                tmp['err'] = (y_true - y_pred) ** 2; tmp['correct'] = (y_true == y_pred).astype(int)
                agg = tmp.groupby('departement').agg(mse=('err', 'mean'), acc=('correct', 'mean'))
                agg['model'] = label; dept_rows.append(agg.reset_index())
            dept_df   = pd.concat(dept_rows)
            pivot_mse = dept_df.pivot(index='departement', columns='model', values='mse')
            pivot_acc = dept_df.pivot(index='departement', columns='model', values='acc')
            fig, axes_d = plt.subplots(2, 1, figsize=(18, 10))
            pivot_mse.plot(kind='bar', ax=axes_d[0], color=[COLORS[c] for c in pivot_mse.columns], alpha=0.8, width=0.8)
            axes_d[0].set_xlabel(''); axes_d[0].set_ylabel('MSE'); axes_d[0].tick_params(axis='x', labelsize=6); axes_d[0].legend()
            pivot_acc.plot(kind='bar', ax=axes_d[1], color=[COLORS[c] for c in pivot_acc.columns], alpha=0.8, width=0.8)
            axes_d[1].set_xlabel('Department'); axes_d[1].set_ylabel('Accuracy'); axes_d[1].tick_params(axis='x', labelsize=6); axes_d[1].legend()
            plt.tight_layout()
            fig.savefig(OUT / f'{group_name}_7_per_dept_metrics.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {group_name}_7_per_dept_metrics.png')

        # section 8 : tracé de saisonnalité, inutile pour le seul score opérationnel
        if not ONLY_OPERATIONAL:
            # ── Section 8: Seasonal signal ────────────────────────────────────────────
            seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
            fig, axes_s = plt.subplots(2, 2, figsize=(14, 9))
            for ax, season in zip(axes_s.flat, seasons):
                for label, df in dfs.items():
                    if not SHOW_EXPERT and models_dict.get(label) == 'expert':
                        continue
                    PRED_COL = pred_col_map[label]
                    sub   = plot_dt(df[df['season'] == season], season)
                    daily = sub.groupby('datetime').agg(pred_class=(PRED_COL, 'mean'), true_class=(TARGET, 'mean')).reset_index()
                    ax.plot(daily['datetime'], daily['pred_class'], color=COLORS[label], lw=1.2, label=label, alpha=0.8)
                ref       = plot_dt(dfs[ref_label][dfs[ref_label]['season'] == season], season)
                ref_daily = ref.groupby('datetime')[TARGET].mean().reset_index()
                ax.plot(ref_daily['datetime'], ref_daily[TARGET], color='black', lw=1.5, linestyle='--', label='true class', alpha=0.7)
                ax.text(0.01, 0.97, season, transform=ax.transAxes, fontweight='bold', va='top', fontsize=10)
                ax.set_ylabel('kmeans class (mean)')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
                ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(OUT / f'{group_name}_8_seasonal.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {group_name}_8_seasonal.png')

        if ONLY_SEASONAL:
            continue

        # ── Section 9: Calibration ────────────────────────────────────────────────
        if has_probas and not ONLY_OPERATIONAL:
            ece_rows = []
            for label, df in dfs.items():
                PROBA_COLS = proba_cols_map[label]
                probs_arr = df[PROBA_COLS].values
                y_true_arr = df[TARGET].astype(int).values
                if np.all(np.isnan(probs_arr)):
                    continue
                fig_cal, _, stats = plot_calibration_with_background(
                    probs_arr, y_true_arr, n_bins=10, figsize=(22, 4), average='micro',
                )
                fig_cal.suptitle(label, fontsize=11, fontweight='bold', y=1.01)
                fig_cal.savefig(OUT / f'{group_name}_9_calibration_{label}.png', dpi=100, bbox_inches='tight')
                plt.close(fig_cal)
                print(f'  Saved: {group_name}_9_calibration_{label}.png')
                row = {'model': label, 'ECE_macro': stats['ECE_macro']}
                row.update({k: v for k, v in stats.items() if k.startswith('ECE_class')})
                ece_rows.append(row)
            if ece_rows:
                ece_df = pd.DataFrame(ece_rows).set_index('model')
                print('\nECE summary:')
                print(ece_df.round(4).to_string())


        # ── Section 10: Scoring per zone ──────────────────────────────────────────
        is_firemen = 'firemen' in str(BASE)
        dataset_name = 'firemen' if is_firemen else 'bdiff'
        path_baseline = Path('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction') / dataset_name
        
        # if firemen lacks department, fallback to firemen2 just in case
        if is_firemen and not list(path_baseline.glob('*departement*.pkl')):
            path_baseline = Path('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/firemen2')

        _target_name_map = {
            'nbsinister': 'Fire',
            'timeintervention': 'Time',
            'ressource': 'Ressource',
            'burnedareaRoot': 'BurnedArea',
        }
        simple_target = _target_name_map.get(base_target, 'BurnedArea')
        zone_col = 'departement' if is_firemen else 'cluster-encoder'
        
        scoring_rows = []
        
        for label, df in dfs.items():
            PRED_COL = pred_col_map[label]
            if zone_col not in df.columns:
                print(f"  [SKIP] Column {zone_col} not in dataframe for model {label}.")
                continue
                
            unique_zones = [z for z in df[zone_col].unique() if pd.notna(z)]
            if not is_firemen:
                unique_zones = sorted(unique_zones)
            else:
                unique_zones = sorted(unique_zones)
                if base_target == 'timeintervention':
                    unique_zones = [z for z in unique_zones if int(z) != 1]
                
            for z in unique_zones:
                sub = df[df[zone_col] == z].copy()
                if len(sub) == 0: continue
                
                y_true_cls = sub[TARGET].astype(int).values
                y_true_raw = sub[NBSIN_COL].fillna(0).values
                y_pred_float = sub[PRED_COL].astype(float).values
                y_pred_int = sub[PRED_COL].astype(int).values
                dates = sub['datetime'].astype(str).values
                
                if 'node' in sub.columns:
                    zones_fe = sub['node'].values
                elif 'num_zone' in sub.columns:
                    zones_fe = sub['num_zone'].values
                elif 'departement' in sub.columns:
                    zones_fe = sub['departement'].values
                else:
                    zones_fe = np.zeros(len(y_true_raw))
                
                # Construct baseline scorer path
                if zone_col == 'cluster-encoder':
                    z_idx = unique_zones.index(z)
                    scorer_path = path_baseline / f"scorer_{simple_target}__cluster_encoder__c{z_idx}.pkl"
                else:
                    scorer_path = path_baseline / f"scorer_{simple_target}__departement__{int(z)}.pkl"
                    
                if not scorer_path.exists():
                    print(f"  [SKIP] Scorer file not found: {scorer_path}")
                    continue
                    
                try:
                    import pickle
                    with open(scorer_path, "rb") as _f:
                        sc = pickle.load(_f)
                except Exception as e:
                    print(f"  [ERROR] Loading scorer {scorer_path}: {e}")
                    continue
                
                try:
                    # Use evaluation_scoring matching the notebook exactly
                    sh, sl, cov, adj, smc, _, _ = sc.evaluation_scoring(
                        y_pred_float, y_true_raw,
                        dates, zones_fe,
                        df_spline=5, min_n=1, reference=False
                    )
                    rec = recall_score(y_true_raw > 0, y_pred_float > 0, zero_division=0)
                    
                    # Individual recall (per class) uses integer thresholded predictions
                    rec_indiv = recall_score(y_true_cls, y_pred_int, average=None, labels=list(range(N_CLASSES)), zero_division=0)
                    iou_val = iou_score(y_true_cls, y_pred_int)
                    
                    row = {'model': label, 'zone': z}
                    for k in [1, 2, 3, 4]:
                        row[f'score_k{k}'] = float(adj.get(k, np.nan))
                    row['recall_bin'] = rec
                    for c in range(N_CLASSES):
                        row[f'recall_class_{c}'] = rec_indiv[c]
                    scoring_rows.append(row)
                    
                    row_all = {
                        'target': base_target,
                        'loss': label.split('_', 1)[1] if '_' in label else group_name,
                        'model': label.split('_')[0] if '_' in label else label,
                        'cluster': f"cluster_{unique_zones.index(z)}" if not is_firemen else f"cluster_{z}",
                        'k1': float(adj.get(1, np.nan)),
                        'k2': float(adj.get(2, np.nan)),
                        'k3': float(adj.get(3, np.nan)),
                        'k4': float(adj.get(4, np.nan)),
                        'cov_k1': cov.get(1, 0),
                        'cov_k2': cov.get(2, 0),
                        'cov_k3': cov.get(3, 0),
                        'cov_k4': cov.get(4, 0),
                        'recall': rec,
                        'iou': iou_val,
                        'run_type': run_type,
                        'out_root': str(group_out),
                        # '*' : ligne sans jeton de split (référence, expert) -> reprise
                        # dans chaque heatmap au lieu d'en former une à part
                        'split': next((s for s in group_split if s in label),
                                      '*' if group_split else ''),
                    }
                    all_scoring_rows.append(row_all)

                    # figures par modèle et par cluster : coûteuses, hors score opérationnel
                    if not ONLY_OPERATIONAL:
                        title_suffix = f"{label}_{zone_col}_{z}"
                        try:
                            sc._plot_matrice(
                                ypred=y_pred_float, ytrue=y_true_raw, dates=dates, zones=zones_fe,
                                title=f"Matrice Transition - {title_suffix}",
                                dir_output=OUT, normalize_with_reference=False
                            )
                        except Exception as e:
                            pass
                    
                        try:
                            sc._plot_fixed_effects(
                                ypred=y_pred_float, ytrue=y_true_raw, dates=dates, zones=zones_fe,
                                title=f"Fixed Effects - {title_suffix}",
                                dir_output=OUT
                            )
                        except Exception as e:
                            pass
                    
                        # New plot request: _plot()
                        try:
                            sc._plot(
                                ypred=y_pred_float, ytrue=y_true_raw, dates=dates, zones=zones_fe,
                                title=f"Scoring Fit - {title_suffix}",
                                dir_output=OUT
                            )
                        except Exception as e:
                            pass

                except Exception as e:
                    print(f"  [ERROR] Computing Scoring metrics for {label} {zone_col}={z}: {e}")
                    
        if scoring_rows:
            scoring_df = pd.DataFrame(scoring_rows)
            scoring_df.to_csv(OUT / f'{group_name}_10_scoring_metrics.csv', index=False)
            print(f"  Saved: {group_name}_10_scoring_metrics.csv")
            
            metrics_to_plot = ['score_k1', 'score_k2', 'score_k3', 'score_k4', 'recall_bin'] + [f'recall_class_{c}' for c in range(N_CLASSES)]
            for m in ([] if ONLY_OPERATIONAL else metrics_to_plot):
                if m in scoring_df.columns and scoring_df[m].notna().any():
                    try:
                        pivot_m = scoring_df.pivot(index='zone', columns='model', values=m)
                        fig_m, ax_m = plt.subplots(figsize=(max(8, len(scoring_df['zone'].unique())*1.5), 5))
                        pivot_m.plot(kind='bar', ax=ax_m, color=[COLORS.get(c, '#333333') for c in pivot_m.columns], width=0.8, alpha=0.85)
                        ax_m.set_title(f'{m} per {zone_col}')
                        ax_m.set_ylabel(m)
                        ax_m.legend(fontsize=8, title='Model')
                        fig_m.tight_layout()
                        out_m = OUT / f'{group_name}_10_plot_{m}.png'
                        fig_m.savefig(out_m, dpi=120, bbox_inches='tight')
                        plt.close(fig_m)
                        print(f"  Saved: {out_m.name}")
                    except Exception as e:
                        print(f"  [SKIP] Plotting {m}: {e}")

        # ── Section 11: Multi-Horizon Analysis ─────────────────────────────────────────
        horizon_rows = []
        for label, model_folder in models_dict.items():
            if model_folder == 'expert': continue
            try:
                parts = model_folder.split('_')
                horizon_max = int(parts[4])
            except (IndexError, ValueError):
                continue
                
            if horizon_max == 0: continue
            
            sc = None
            if len(dfs) > 0 and zone_col in dfs[list(dfs.keys())[0]].columns:
                z0 = [z for z in dfs[list(dfs.keys())[0]][zone_col].unique() if pd.notna(z)][0]
                if zone_col == 'cluster-encoder':
                    z_idx = unique_zones.index(z0) if 'unique_zones' in locals() and z0 in unique_zones else 0
                    scorer_path = path_baseline / f"scorer_{simple_target}__cluster_encoder__c{z_idx}.pkl"
                else:
                    scorer_path = path_baseline / f"scorer_{simple_target}__departement__{int(z0)}.pkl"
                if scorer_path.exists():
                    import pickle
                    with open(scorer_path, "rb") as _f:
                        sc = pickle.load(_f)
            
            if sc is None:
                print(f"  [SKIP Horizon] Could not load scorer for model {label}")
                continue
            
            model_horizons_data = []
            
            for h in range(horizon_max + 1):
                df_h = load_pkl(model_folder, f"H{h}")
                if df_h is None: continue
                
                df_h['datetime'] = df_h['date'].astype(int).apply(lambda i: pd.Timestamp(allDates[i]))
                df_h = df_h[df_h['datetime'].dt.year == YEAR_FILTER].reset_index(drop=True)
                
                if run_type == 'estival':
                    mask = ((df_h['datetime'].dt.month == 6) & (df_h['datetime'].dt.day >= 15)) | \
                           (df_h['datetime'].dt.month == 7) | \
                           (df_h['datetime'].dt.month == 8) | \
                           ((df_h['datetime'].dt.month == 9) & (df_h['datetime'].dt.day <= 25))
                    df_h = df_h[mask].copy()
                if len(df_h) == 0: continue
                
                PRED_COL = pred_col_map.get(label, f"prediction_{TARGET}_{h}")
                if PRED_COL not in df_h.columns:
                    PRED_COL = f"prediction_{TARGET}_{h}"
                    if PRED_COL not in df_h.columns:
                        continue
                        
                y_true_raw = df_h[NBSIN_COL].fillna(0).values
                y_pred_float = df_h[PRED_COL].astype(float).values
                dates_h = df_h['datetime'].astype(str).values
                
                if 'node' in df_h.columns: zones_fe = df_h['node'].values
                elif 'num_zone' in df_h.columns: zones_fe = df_h['num_zone'].values
                elif 'departement' in df_h.columns: zones_fe = df_h['departement'].values
                else: zones_fe = np.zeros(len(y_true_raw))
                
                try:
                    sh, sl, cov, adj, smc, mu, mu_dense = sc.evaluation_scoring(
                        y_pred_float, y_true_raw, dates_h, zones_fe,
                        df_spline=5, min_n=1, reference=False
                    )
                    
                    y_true_cls = np.clip(y_true_raw, 0, 4).astype(int)
                    y_pred_int = np.clip(np.round(y_pred_float), 0, 4).astype(int)
                    rec = recall_score(y_true_cls > 0, y_pred_int > 0, zero_division=0)
                    
                    h_row = {'model': label, 'loss': label.split('_', 1)[1] if '_' in label else group_name, 'horizon': h, 'zone': 'Global', 'recall': float(rec)}
                    for k in [1, 2, 3, 4]:
                        h_row[f'score_k{k}'] = float(adj.get(k, np.nan))
                        h_row[f'cov_k{k}'] = float(cov.get(k, np.nan))
                    horizon_rows.append(h_row)
                    model_horizons_data.append((h, mu, mu_dense, y_pred_float))
                except Exception as e:
                    print(f"  [ERROR Horizon {h} Global] for {label}: {e}")
                
                # Per department eval
                for z in np.unique(zones_fe):
                    mask = zones_fe == z
                    if mask.sum() < 2: continue
                    try:
                        _, _, cov_z, adj_z, _, _, _ = sc.evaluation_scoring(
                            y_pred_float[mask], y_true_raw[mask], dates_h[mask], zones_fe[mask],
                            df_spline=5, min_n=1, reference=False
                        )
                        
                        y_true_cls_z = np.clip(y_true_raw[mask], 0, 4).astype(int)
                        y_pred_int_z = np.clip(np.round(y_pred_float[mask]), 0, 4).astype(int)
                        rec_z = recall_score(y_true_cls_z > 0, y_pred_int_z > 0, zero_division=0)
                        
                        h_row_z = {'model': label, 'loss': label.split('_', 1)[1] if '_' in label else group_name, 'horizon': h, 'zone': z, 'recall': float(rec_z)}
                        for k in [1, 2, 3, 4]:
                            h_row_z[f'score_k{k}'] = float(adj_z.get(k, np.nan))
                            h_row_z[f'cov_k{k}'] = float(cov_z.get(k, np.nan))
                        horizon_rows.append(h_row_z)
                    except Exception as e:
                        pass
                    
            if len(model_horizons_data) > 0:
                n_h = len(model_horizons_data)
                cols = min(3, n_h)
                rows_plot = math.ceil(n_h / cols) if cols > 0 else 1
                fig_h, axes_h = plt.subplots(rows_plot, cols, figsize=(cols*6, rows_plot*5))
                if n_h == 1: axes_h = np.array([axes_h])
                axes_h = axes_h.flatten()
                
                for idx, (h, mu, mu_dense, yp) in enumerate(model_horizons_data):
                    ax = axes_h[idx]
                    predicted_classes = np.unique(np.clip(np.round(yp), 0, 4).astype(int))
                    mu_filtered = {k: v for k, v in mu.items() if k in predicted_classes}
                    
                    if mu_dense is not None and len(mu_dense) > 0:
                        x_dense = np.linspace(0.0, 4.0, len(mu_dense))
                        ax.plot(x_dense, mu_dense, color='blue', linewidth=2, alpha=0.8, label='Spline Fit')
                    if mu_filtered and len(mu_filtered) > 0:
                        levels = sorted(mu_filtered.keys())
                        mu_vals = [mu_filtered[l] for l in levels]
                        ax.scatter(levels, mu_vals, color='red', s=80, label='Discrete Means', zorder=5)
                        for l, v in zip(levels, mu_vals):
                            ax.annotate(f"{v:.3f}", (l, v), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red', fontweight='bold')
                    ax.set_title(f"Horizon {h}")
                    ax.set_xlabel("Predicted Class")
                    ax.set_xticks(predicted_classes)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.legend()
                    
                for idx in range(n_h, len(axes_h)):
                    axes_h[idx].set_visible(False)
                    
                fig_h.suptitle(f"Fit Monotone par Horizon - {label} ({run_type})", fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                out_h = OUT / f'{group_name}_11_horizons_fit_{label}.png'
                fig_h.savefig(out_h, dpi=120, bbox_inches='tight')
                plt.close(fig_h)
                print(f"  Saved: {out_h.name}")
                
        if len(horizon_rows) > 0:
            import math
            import matplotlib.pyplot as plt
            import matplotlib.lines as mlines
            
            df_horizons = pd.DataFrame(horizon_rows)
            
            # Calculate and add Average across departments
            df_depts = df_horizons[df_horizons['zone'] != 'Global']
            if not df_depts.empty:
                numeric_cols = [c for c in df_depts.columns if c.startswith('score_') or c.startswith('cov_') or c == 'recall']
                df_avg = df_depts.groupby(['model', 'loss', 'horizon'])[numeric_cols].mean().reset_index()
                df_avg['zone'] = 'Average'
                df_horizons = pd.concat([df_horizons, df_avg], ignore_index=True)
            
            models_types = df_horizons['model'].apply(lambda x: x.split('_')[0] if '_' in x else x).unique()
            losses = df_horizons['loss'].unique()
            
            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(models_types))))
            model_color_map = {m: c for m, c in zip(models_types, colors)}
            
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', '+', 'x']
            linestyles = ['-', '--', '-.', ':']
            loss_style_map = {}
            for i, l in enumerate(losses):
                loss_style_map[l] = (markers[i % len(markers)], linestyles[i % len(linestyles)])
                
            def get_model_type(lbl):
                return lbl.split('_')[0] if '_' in lbl else lbl
            
            for z in df_horizons['zone'].unique():
                sub_df = df_horizons[df_horizons['zone'] == z]
                if z == "Global":
                    z_name = "Global"
                elif z == "Average":
                    z_name = "Average"
                else:
                    z_name = f"dept_{int(float(z))}"
                
                # --- Plot Score k and Recall ---
                fig_sk, axes_sk = plt.subplots(2, 3, figsize=(22, 12))
                for k in [1, 2, 3, 4]:
                    ax = axes_sk.flatten()[k-1]
                    for label in sub_df['model'].unique():
                        sub_h = sub_df[sub_df['model'] == label].sort_values('horizon')
                        m_type = get_model_type(label)
                        loss_type = sub_h['loss'].iloc[0]
                        ax.plot(
                            sub_h['horizon'], sub_h[f'score_k{k}'], 
                            marker=loss_style_map[loss_type][0],
                            linestyle=loss_style_map[loss_type][1],
                            color=model_color_map[m_type],
                            label=f"{m_type} ({loss_type})"
                        )
                    ax.set_title(f"Score k={k} vs Horizon - {z_name}")
                    ax.set_xlabel("Horizon (jours)")
                    ax.set_ylabel(f"Score k{k}")
                    ax.set_ylim(-0.5, 1.0)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    # Deduplicate legend
                    handles, labels_leg = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels_leg, handles))
                    ax.legend(by_label.values(), by_label.keys(), fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Add Recall Plot
                ax_rec = axes_sk.flatten()[4]
                for label in sub_df['model'].unique():
                    sub_h = sub_df[sub_df['model'] == label].sort_values('horizon')
                    m_type = get_model_type(label)
                    loss_type = sub_h['loss'].iloc[0]
                    ax_rec.plot(
                        sub_h['horizon'], sub_h['recall'], 
                        marker=loss_style_map[loss_type][0],
                        linestyle=loss_style_map[loss_type][1],
                        color=model_color_map[m_type],
                        label=f"{m_type} ({loss_type})"
                    )
                ax_rec.set_title(f"Recall vs Horizon - {z_name}")
                ax_rec.set_xlabel("Horizon (jours)")
                ax_rec.set_ylabel("Recall")
                ax_rec.set_ylim(0, 1.0)
                ax_rec.grid(True, linestyle='--', alpha=0.6)
                handles, labels_leg = ax_rec.get_legend_handles_labels()
                by_label = dict(zip(labels_leg, handles))
                ax_rec.legend(by_label.values(), by_label.keys(), fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Hide 6th axis
                axes_sk.flatten()[5].set_visible(False)

                plt.tight_layout()
                out_sk = OUT / f'{group_name}_11_horizons_{z_name}_score_k_and_recall.png'
                fig_sk.savefig(out_sk, dpi=120, bbox_inches='tight')
                plt.close(fig_sk)
                print(f"  Saved: {out_sk.name}")
                
                # --- Plot Score k and Recall Heatmap ---
                out_hm = OUT / f'{group_name}_11_horizons_{z_name}_score_recall_heatmap.png'
                try:
                    plot_horizons_score_recall_heatmap(
                        sub_df,
                        title=f"Heatmaps Score / Recall vs Horizon — {z_name}",
                        save_path=out_hm,
                        show_expert=SHOW_EXPERT
                    )
                    print(f"  Saved heatmap: {out_hm.name}")
                except Exception as e:
                    print(f"  [ERROR] Plotting horizon heatmap for {z_name}: {e}")
                
                # --- Plot Coverage k ---
                fig_cov, axes_cov = plt.subplots(2, 2, figsize=(16, 12))
                for k in [1, 2, 3, 4]:
                    ax = axes_cov.flatten()[k-1]
                    for label in sub_df['model'].unique():
                        sub_h = sub_df[sub_df['model'] == label].sort_values('horizon')
                        m_type = get_model_type(label)
                        loss_type = sub_h['loss'].iloc[0]
                        ax.plot(
                            sub_h['horizon'], sub_h[f'cov_k{k}'], 
                            marker=loss_style_map[loss_type][0],
                            linestyle=loss_style_map[loss_type][1],
                            color=model_color_map[m_type],
                            label=f"{m_type} ({loss_type})"
                        )
                    ax.set_title(f"Coverage k={k} vs Horizon - {z_name}")
                    ax.set_xlabel("Horizon (jours)")
                    ax.set_ylabel(f"Coverage k{k}")
                    ax.grid(True, linestyle='--', alpha=0.6)
                    handles, labels_leg = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels_leg, handles))
                    ax.legend(by_label.values(), by_label.keys(), fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                out_cov = OUT / f'{group_name}_11_horizons_{z_name}_coverage_curve.png'
                fig_cov.savefig(out_cov, dpi=120, bbox_inches='tight')
                plt.close(fig_cov)
                print(f"  Saved: {out_cov.name}")
            
            # Heatmap global
            df_cov_heat = df_horizons[(df_horizons['horizon'] == 0) & (df_horizons['zone'] == 'Global')].copy()
            if not df_cov_heat.empty:
                df_cov_heat['cluster'] = "Global"
                df_cov_heat['k1'] = df_cov_heat['cov_k1']
                df_cov_heat['k2'] = df_cov_heat['cov_k2']
                df_cov_heat['k3'] = df_cov_heat['cov_k3']
                df_cov_heat['k4'] = df_cov_heat['cov_k4']
                df_cov_heat['recall'] = np.nan
                df_cov_heat['iou'] = np.nan
                
                try:
                    out_cov_heat = OUT / f'{group_name}_11_horizons_coverage_heatmap.png'
                    plot_operational_heatmap(
                        df_cov_heat,
                        title=f"Coverage des transitions par Horizon ({run_type})",
                        k_cols=("k1", "k2", "k3", "k4"),
                        loss_col="loss",
                        model_col="model",
                        cluster_col="cluster",
                        save_path=out_cov_heat,
                        show_expert=SHOW_EXPERT
                    )
                    print(f"  Saved heatmap: {out_cov_heat.name}")
                except Exception as e:
                    print(f"  [ERROR] Plotting coverage heatmap: {e}")

print('\nDone. All figures saved under', BASE_OUT, 'and', STUDENT_OUT, '(groupes studentMLP)')



def plot_coverage_heatmap(
    df: pd.DataFrame,
    title: str = "Coverage opérationnel en fonction du schéma",
    k_cols=("cov_k1", "cov_k2", "cov_k3", "cov_k4"),
    model_col="model",
    loss_col="loss",
    cluster_col="cluster",
    cluster_backgrounds=None,
    figsize=(22, 13),
    save_path=None,
    dpi=300,
    show_expert=None,
):
    """
    df attendu au format long :
    columns = [loss, model, cluster, cov_k1, cov_k2, cov_k3, cov_k4]
    """
    cluster_backgrounds = cluster_backgrounds or CLUSTER_BACKGROUNDS

    df = df.copy()
    clusters = list(df[cluster_col].drop_duplicates())
    losses = list(df[loss_col].drop_duplicates())

    if show_expert is None:
        show_expert = SHOW_EXPERT
    rows = []
    for loss in losses:
        models = list(df.loc[df[loss_col] == loss, model_col].drop_duplicates())
        if not show_expert:
            models = [m for m in models if m != 'expert']
        for model in models:
            rows.append((loss, model))

    if show_expert:
        rows.sort(key=lambda x: 0 if x[1] == 'expert' else 1)
    else:
        rows.sort(key=lambda x: 1 if x[1] == 'expert' else 0)

    n_rows = len(rows)
    n_clusters = len(clusters)
    n_metrics = len(k_cols)

    # Colonnes de gauche dimensionnées sur les libellés les plus longs : sinon les
    # noms débordent sur les cellules de métriques. ~0.11 unité par caractère à
    # fontsize 10 ; la figure s'élargit d'autant pour garder des cellules de
    # taille constante, et grandit en hauteur quand il y a beaucoup de lignes.
    _loss_w  = max(1.0, max((len(str(l)) for l, _ in rows), default=6) * 0.11)
    _model_w = max(1.0, max((len(str(m)) for _, m in rows), default=10) * 0.11)
    _left_w  = _loss_w + _model_w
    _loss_x, _model_x = _loss_w / 2, _loss_w + _model_w / 2
    _span    = _left_w + n_clusters * n_metrics
    figsize  = (figsize[0] * _span / (2.0 + n_clusters * n_metrics),
                max(figsize[1], 0.75 * n_rows + 4))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, _span)
    ax.set_ylim(0, n_rows + 3)
    ax.axis("off")

    k_values = df[list(k_cols)].to_numpy().ravel()
    k_min = 0
    k_max = np.nanmax(k_values) if not np.isnan(np.nanmax(k_values)) else 1

    k_norm = Normalize(vmin=k_min, vmax=k_max)
    cmap_k = plt.cm.Greens

    left_w = _left_w
    cell_w = 1.0
    cell_h = 0.75
    header_h = 1.25
    top_y = n_rows + 1.2

    ax.text(
        left_w + n_clusters * n_metrics / 2,
        n_rows + 2.5,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(_loss_x, top_y - 0.6, "Schéma", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(_model_x, top_y - 0.6, "Modèle", ha="center", va="center", fontsize=10, fontweight="bold")

    for c_idx, cluster in enumerate(clusters):
        x0 = left_w + c_idx * n_metrics
        bg = cluster_backgrounds.get(cluster, "#F7F7F7")

        ax.add_patch(
            Rectangle(
                (x0, 0.4),
                n_metrics,
                n_rows + header_h,
                facecolor=bg,
                edgecolor="0.65",
                linewidth=1.2,
                zorder=0,
            )
        )

        ax.text(
            x0 + n_metrics / 2,
            top_y,
            cluster,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        for j, col in enumerate(k_cols):
            ax.text(
                x0 + j + 0.5,
                top_y - 0.8,
                col.replace('cov_', ''),
                ha="center",
                va="center",
                fontsize=9,
                fontstyle="italic",
            )

    previous_loss = None
    for row_idx, (loss, model) in enumerate(rows):
        y_pos = n_rows - row_idx
        if loss != previous_loss:
            ax.text(_loss_x, y_pos, loss, ha="center", va="center", fontsize=10, fontweight="bold")
            ax.axhline(y_pos + 0.5, xmin=0, xmax=1.5, color="black", linewidth=1.5)
            previous_loss = loss

        ax.text(_model_x, y_pos, model, ha="center", va="center", fontsize=10, fontweight="bold")

        for c_idx, cluster in enumerate(clusters):
            x0 = left_w + c_idx * n_metrics

            sub = df[
                (df[loss_col] == loss)
                & (df[model_col] == model)
                & (df[cluster_col] == cluster)
            ]

            if sub.empty:
                values = [np.nan] * n_metrics
            else:
                values = sub.iloc[0][list(k_cols)].to_list()

            for j, val in enumerate(values):
                if pd.isna(val):
                    color = "#FFFFFF"
                    text = "--"
                else:
                    color = cmap_k(k_norm(val))
                    text = f"{val:.0f}"

                ax.add_patch(
                    Rectangle(
                        (x0 + j, y_pos - cell_h / 2),
                        cell_w,
                        cell_h,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=1.0,
                    )
                )

                ax.text(
                    x0 + j + 0.5,
                    y_pos,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="white" if not pd.isna(val) and val > (k_max * 0.6) else "black",
                )

    sm_k = plt.cm.ScalarMappable(norm=k_norm, cmap=cmap_k)
    cax1 = fig.add_axes([0.4, 0.055, 0.2, 0.018])
    cbar1 = fig.colorbar(sm_k, cax=cax1, orientation="horizontal")
    cbar1.set_label("Coverage (nombre de paires utilisées)", fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

def save_operational_latex(
    df: pd.DataFrame,
    out_path,
    k_cols=("k1", "k2", "k3", "k4"),
    recall_col="recall",
    iou_col="iou",
    loss_col="loss",
    model_col="model",
    cluster_col="cluster",
    caption: str = "Performance opérationnelle",
    label: str = "tab:operational",
):
    """
    Sauvegarde le DataFrame de scoring opérationnel en tableau LaTeX.

    Le tableau est structuré comme la heatmap :
      Schéma | Modèle | Cluster | k1 | k2 | k3 | k4 | Recall | IoU
    """
    df = df.copy()
    metric_cols = list(k_cols) + [recall_col, iou_col]

    # Colonnes disponibles (recall/iou peuvent être NaN)
    available_metrics = [c for c in metric_cols if c in df.columns]

    # Trier : expert en premier, puis par loss/model
    df["_sort"] = df[model_col].apply(lambda m: 0 if m == "expert" else 1)
    df = df.sort_values(["_sort", loss_col, model_col, cluster_col]).drop(columns="_sort")

    rows_latex = []
    prev_loss = None
    for _, row in df.iterrows():
        loss_val = row[loss_col]
        if loss_val != prev_loss:
            if prev_loss is not None:
                rows_latex.append(r"\midrule")
            prev_loss = loss_val

        vals = []
        for m in available_metrics:
            v = row.get(m, float("nan"))
            if pd.isna(v):
                vals.append("--")
            else:
                v_round = round(v, 2)
                if v_round == 0.0 and v != 0.0:
                    v_round = 0.01 if v > 0 else -0.01
                vals.append(f"{v_round:.2f}")

        rows_latex.append(
            " & ".join([
                str(row[loss_col]),
                str(row[model_col]),
                str(row[cluster_col]),
            ] + vals) + r" \\"
        )

    col_headers = (
        ["Schéma", "Modèle", "Cluster"]
        + [("Recall" if c == recall_col else ("IoU" if c == iou_col else c)) for c in available_metrics]
    )
    n_cols = len(col_headers)
    col_spec = "ll l" + " r" * (n_cols - 3)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(col_headers) + r" \\",
        r"\midrule",
    ]
    lines += rows_latex
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved LaTeX table: {out_path}")


if all_scoring_rows:
    df_all = pd.DataFrame(all_scoring_rows)
    for root in df_all['out_root'].unique():
      root_splits = sorted(s for s in df_all.loc[df_all['out_root'] == root, 'split'].unique()
                           if s not in ('', '*'))
      for split in (root_splits or ['']):
       for tgt in df_all['target'].unique():
        for rt in df_all['run_type'].unique():
            keep = df_all['split'].isin([split, '*'] if split else [''])
            sub_df = df_all[(df_all['out_root'] == root) & keep &
                            (df_all['target'] == tgt) & (df_all['run_type'] == rt)].copy()
            # un même modèle peut venir de plusieurs groupes (références partagées)
            sub_df = sub_df.drop_duplicates(subset=['loss', 'model', 'cluster'])
            if not sub_df.empty:
                suffix = f'_{split}' if split else ''
                out_dir = Path(root) / dataset / tgt / rt
                out_dir.mkdir(parents=True, exist_ok=True)
                sub_df.to_csv(out_dir / f'operational_scores_{tgt}_{rt}{suffix}.csv', index=False)

                # ── PNG heatmap ────────────────────────────────────────────
                out_heat = out_dir / f'operational_heatmap_{tgt}_{rt}{suffix}.png'
                try:
                    plot_operational_heatmap(
                        sub_df,
                        title=f"Performance opérationnelle ({tgt}) - {rt.capitalize()}"
                              + (f" - {split}" if split else ""),
                        loss_col="loss",
                        model_col="model",
                        cluster_col="cluster",
                        save_path=out_heat,
                        show_expert=SHOW_EXPERT
                    )
                    print(f"  Saved heatmap: {out_heat}")
                except Exception as e:
                    print(f"  [ERROR] Plotting heatmap for {tgt} ({rt}): {e}")

                # ── Coverage heatmap ─────────────────────────────────────────
                out_cov = out_dir / f'coverage_heatmap_{tgt}_{rt}{suffix}.png'
                try:
                    plot_coverage_heatmap(
                        sub_df,
                        title=f"Coverage opérationnel ({tgt}) - {rt.capitalize()}",
                        loss_col="loss",
                        model_col="model",
                        cluster_col="cluster",
                        save_path=out_cov,
                        show_expert=SHOW_EXPERT
                    )
                    print(f"  Saved coverage heatmap: {out_cov}")
                except Exception as e:
                    print(f"  [ERROR] Plotting coverage heatmap for {tgt} ({rt}): {e}")

                # ── LaTeX table ────────────────────────────────────────────
                out_tex = out_dir / f'operational_heatmap_{tgt}_{rt}{suffix}.tex'
                try:
                    save_operational_latex(
                        sub_df,
                        out_path=out_tex,
                        caption=f"Performance opérationnelle ({tgt}) — {rt.capitalize()}",
                        label=f"tab:operational_{tgt}_{rt}",
                    )
                except Exception as e:
                    print(f"  [ERROR] Saving LaTeX table for {tgt} ({rt}): {e}")


