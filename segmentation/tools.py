from ast import Not
from email.mime import image
from threading import local
from scipy.ndimage import generic_filter
import datetime as dt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import plotly.express as px
import plotly.io as pio
from hdbscan import HDBSCAN, approximate_predict
import math
import rasterio
import rasterio.features
import rasterio.warp
import scipy.stats
import sys
import warnings
from osgeo import gdal, ogr
from pathlib import Path
from scipy import ndimage as ndi
from skimage import measure, segmentation, morphology
from skimage.segmentation import watershed
from skimage import transform
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import SpectralClustering
from scipy.spatial import distance as d
from sklearn.cluster import KMeans
from skimage import io, color, filters, measure, morphology
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import scipy.ndimage as ndimage
import logging
import cv2
from arborescence import *
from array_fet import *
import geopandas as gpd
import pandas as pd
import xarray as xr
from dtaidistance import dtw
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

# Handler pour afficher les logs dans le terminal
streamHandler = logging.StreamHandler(stream=sys.stdout)
streamHandler.setFormatter(logFormatter)
logger.addHandler(streamHandler)

def find_dates_between(start, end):
    start_date = dt.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end, '%Y-%m-%d').date()

    delta = dt.timedelta(days=1)
    date = start_date
    res = []
    while date < end_date:
            res.append(date.strftime("%Y-%m-%d"))
            date += delta
    return res

allDates = find_dates_between('2017-06-12', '2024-06-29')

resolutions = {'2x2' : {'x' : 0.02875215641173088,'y' :  0.020721094073767096}}

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        logger.info(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def create_larger_scale_bin(input, bin, influence, raster):
    binImageScale = np.full(bin.shape, np.nan)
    influenceImageScale = np.full(influence.shape, np.nan)
    timeScale = np.full(influence.shape, np.nan)

    clusterID = np.unique(input)
    for di in range(bin.shape[-1]):
        for id in clusterID:
            mask = (input == id)
            if np.any(influence[mask, di] > 0):
                binImageScale[mask, di] = np.nansum(bin[mask, di])
                influenceImageScale[mask, di] = np.nansum(influence[mask, di])
            else:
                binImageScale[mask, di] = 0
                influenceImageScale[mask, di] = 0
                timeScale[mask, di] = 0

    return binImageScale, influenceImageScale

def order_class(predictor, pred, min_values=0):
    res = np.zeros(pred[~np.isnan(pred)].shape[0], dtype=int)
    cc = predictor.cluster_centers_.reshape(-1)
    classes = np.arange(cc.shape[0])
    ind = np.lexsort([cc])
    cc = cc[ind]
    classes = classes[ind]
    for c in range(cc.shape[0]):
        mask = np.argwhere(pred == classes[c])
        res[mask] = c
    return res + min_values

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def find_neighbor_by_size(res, label, min_cluster_size, max_cluster_size, mask_label_ori, dilated_image, mask_label, neighbor_labels):
    neighbors_size = sorted(
        [[neighbor_label, np.sum(res == neighbor_label)] for neighbor_label in neighbor_labels],
        key=lambda x: x[1]  # trie par la somme (ordre croissant)
    )
    
    best_neighbor = None
    find_neighbor = False
    
    # Mode basé sur la taille des clusters
    max_neighbor_size = -math.inf
    for nei, neighbor in enumerate(neighbors_size):
        if neighbor[0] == label:
            continue
        neighbor_size = neighbor[1] + np.sum(res == label)

        # Vérifier si le voisin satisfait min_cluster_size
        if neighbor_size > min_cluster_size:
            # Vérifier si la taille reste sous max_cluster_size
            if neighbor_size < max_cluster_size:
                dilate = False
                res[mask_label_ori] = neighbor[0]
                dilated_image[mask_label] = neighbor[0]
                logger.info(f'Use neighbord label {label} -> {neighbor[0]}')
                label = neighbor[0]
                find_neighbor = True
                break
            
            best_neighbor = neighbor[0]
            max_neighbor_size = neighbor_size
            break

        # Enregistrer le plus grand voisin si min_cluster_size n'est pas atteint
        if neighbor_size > max_neighbor_size:
            best_neighbor = neighbor[0]
            max_neighbor_size = neighbor_size
    
    return best_neighbor, max_neighbor_size, find_neighbor

from scipy.spatial.distance import braycurtis

def find_neighbor_by_BrayCurtis_similarity(res, features, label, min_cluster_size, max_cluster_size, mask_label_ori, dilated_image, mask_label, neighbor_labels):
    # Calcul des features moyennes de la zone courante
    features_label = np.nanmean(features[:, res == label], axis=1)
    find_neighbor = False
    # Liste triée des voisins par distance de Bray–Curtis
    neighbors_braycurtis = sorted(
        [
            [
                neighbor_label,
                np.sum(res == neighbor_label), # taille du cluster voisin
                braycurtis(features_label, np.nanmean(features[:, res == neighbor_label], axis=1))
            ]
            for neighbor_label in neighbor_labels
        ],
        key=lambda x: x[1]  # tri par distance de Bray–Curtis
    )
    dilate = True
    max_neighbor_size = -math.inf
    for nei, neighbor in enumerate(neighbors_braycurtis):
        if neighbor[0] == label:
            continue
        
        neighbor_size = neighbor[1] + np.sum(res == label)

        # Vérifier si le voisin satisfait min_cluster_size
        if neighbor_size > min_cluster_size:
            # Vérifier si la taille reste sous max_cluster_size
            if neighbor_size < max_cluster_size:
                dilate = False
                res[mask_label_ori] = neighbor[0]
                dilated_image[mask_label] = neighbor[0]
                logger.info(f'Use neighbord label {label} -> {neighbor[0]}')
                label = neighbor[0]
                find_neighbor = True
            
            best_neighbor = neighbor[0]
            max_neighbor_size = neighbor_size
            break

        # Enregistrer le plus grand voisin si min_cluster_size n'est pas atteint
        if neighbor_size > max_neighbor_size:
            best_neighbor = neighbor[0]
            max_neighbor_size = neighbor_size
    
    return best_neighbor, max_neighbor_size, find_neighbor, dilate

def merge_adjacent_clusters(image, mode='size', min_cluster_size=0, max_cluster_size=math.inf, exclude_label=None, background=-1, features=None, nb_attempt=3):
    """
    Fusionne les clusters adjacents dans une image en fonction de critères définis.
    
    Paramètres :
    - image : Image labellisée contenant des clusters.
    - mode : Critère de fusion ('size', 'time_series_similarity', 'time_series_similarity_fast').
    - min_cluster_size : Taille minimale d'un cluster avant fusion.
    - max_cluster_size : Taille maximale autorisée après fusion.
    - oridata : Données supplémentaires utilisées pour la fusion basée sur des séries temporelles (facultatif).
    - exclude_label : Label à exclure de la fusion.
    - background : Label représentant le fond (par défaut -1).
    """

    # Copie de l'image d'entrée pour éviter de la modifier directement
    labeled_image = np.copy(image)

    # Obtenir les propriétés des régions labellisées
    regions = measure.regionprops(labeled_image)
    # Trier les régions par taille croissante
    regions = sorted(regions, key=lambda r: r.area)

    # Masque pour stocker les labels mis à jour après fusion
    res = np.copy(labeled_image)

    # Liste des labels qui ont été modifiés
    changed_labels = []

    fix_label = []

    # Longueur initiale des régions
    len_regions = len(regions)
    i = 0

    # Boucle pour traiter chaque région
    while i < len_regions:
        region = regions[i]

        # Vérifier si le cluster est à exclure ou est un fond
        if region.label == exclude_label or region.label == background:
            # On conserve ces clusters tels quels
            res[labeled_image == region.label] = region.label
            i += 1
            continue

        label = region.label
        if label in fix_label:
            i += 1
            continue

        # Si le label a déjà été modifié, passer au suivant
        #if label in changed_labels:
        #    i += 1
        #    continue

        # Vérifier la taille du cluster actuel
        ones = np.argwhere(res == label).shape[0]
        if ones < min_cluster_size:
            # Si la taille est inférieure au minimum, essayer de fusionner avec un voisin
            nb_test = 0
            find_neighbor = False
            dilated_image = np.copy(res)
            while nb_test < nb_attempt and not find_neighbor:

                # Trouver les voisins du cluster actuel
                mask_label = dilated_image == label
                mask_label_ori = res == label
                neighbors = segmentation.find_boundaries(mask_label, connectivity=1, mode='outer', background=background)
                neighbor_labels = np.unique(dilated_image[neighbors])
                # Exclure les labels indésirables
                neighbor_labels = neighbor_labels[(neighbor_labels != exclude_label) & (neighbor_labels != background) & (neighbor_labels != label)]
                dilate = True
                changed_labels.append(label)

                if len(neighbor_labels) > 0:
                    # Trier les voisins par taille
                    neighbors_size = sorted(
                        [[neighbor_label, np.sum(res == neighbor_label)] for neighbor_label in neighbor_labels],
                        key=lambda x: x[1]  # trie par la somme (ordre croissant)
                    )

                    best_neighbor = None

                    if mode == 'size':
                        # Mode basé sur la taille des clusters
                        max_neighbor_size = -math.inf
                        for nei, neighbor in enumerate(neighbors_size):
                            if neighbor[0] == label:
                                continue
                            neighbor_size = neighbor[1] + np.sum(res == label)

                            # Vérifier si le voisin satisfait min_cluster_size
                            if neighbor_size > min_cluster_size:
                                # Vérifier si la taille reste sous max_cluster_size
                                if neighbor_size < max_cluster_size:
                                    dilate = False
                                    res[mask_label_ori] = neighbor[0]
                                    dilated_image[mask_label] = neighbor[0]
                                    logger.info(f'Use neighbord label {label} -> {neighbor[0]}')
                                    label = neighbor[0]
                                    find_neighbor = True
                                    break
                                
                                best_neighbor = neighbor[0]
                                max_neighbor_size = neighbor_size
                                break

                            # Enregistrer le plus grand voisin si min_cluster_size n'est pas atteint
                            if neighbor_size > max_neighbor_size:
                                best_neighbor = neighbor[0]
                                max_neighbor_size = neighbor_size

                    elif mode == 'timeSeriesSimilarity':
                        # Mode basé sur la similarité de séries temporelles (DTW)
                        assert features is not None
                        time_series_data = np.nansum(features[dilated_image == label], axis=0).reshape(-1, 1)
                        best_neighbord = None
                        min_dst = math.inf  # Cherche à minimiser la distance
                        dst_thresh = 1000

                        for neighbor in neighbors_size:
                            if neighbor[0] == label:
                                continue

                            time_series_data_neighbor = np.nansum(features[dilated_image == neighbor[0]], axis=0).reshape(-1, 1)
                            distance = dtw.distance(time_series_data, time_series_data_neighbor)

                            if distance < min_dst and distance < dst_thresh:
                                best_neighbord = neighbor[0]
                                min_dst = distance

                        if best_neighbord is not None:
                            dilate = False
                            res[mask_label_ori] = best_neighbord
                            logger.info(f'label {label} -> {best_neighbord}')
                            changed_labels.append(label)
                            label = best_neighbord
                            find_neighbor = True

                    elif mode == 'time_series_similarity_fast':
                        # Mode basé sur une version rapide de DTW
                        assert features is not None
                        time_series_data = np.nansum(features[dilated_image == label], axis=0).reshape(-1, 1)
                        best_neighbord = None
                        min_simi = math.inf  # Cherche à minimiser la similarité
                        dst_thresh = 100

                        for neighbor in neighbors_size:
                            time_series_data_neighbor = np.nansum(features[dilated_image == neighbor[0]], axis=0).reshape(-1, 1)
                            _, simi = dtw_functions.dtw(time_series_data, time_series_data_neighbor, local_dissimilarity=d.euclidean)

                            if simi < min_simi and simi < dst_thresh:
                                best_neighbord = neighbor[0]
                                min_simi = simi

                        if best_neighbord is not None:
                            dilate = False
                            res[mask_label_ori] = best_neighbord
                            changed_labels.append(label)
                            label = best_neighbord
                            find_neighbor = True
                    
                    elif mode == 'BrayCurtis':
                        assert features is not None
                        best_neighbor, max_neighbor_size, find_neighbor, dilate = find_neighbor_by_BrayCurtis_similarity(res, features, label, min_cluster_size, max_cluster_size, mask_label_ori, dilated_image, mask_label, neighbor_labels)
                
                    # Si aucun voisin ne satisfait les critères, utiliser le plus grand
                    if not find_neighbor and best_neighbor is not None:
                        if max_neighbor_size < max_cluster_size:
                            res[mask_label] = best_neighbor
                            dilated_image[mask_label] = best_neighbor
                            dilate = False
                            logger.info(f'Use biggest neighbord label {label} -> {best_neighbor}')
                            label = best_neighbor
                            find_neighbor = True
                            # Si la taille après fusion dépasse la taille maximal, appliquer l'érosion (peut être ne pas fusionner)
                            if max_neighbor_size < max_cluster_size:
                                mask_label = dilated_image == label
                                ones = np.argwhere(mask_label == 1).shape[0]
                                while ones > max_cluster_size:
                                    mask_label = morphology.erosion(mask_label, morphology.disk(3))
                                    ones = np.argwhere(mask_label == 1).shape[0]

                # Si aucun voisin trouvé, dilater la région
                if dilate:
                    mask_label = morphology.dilation(mask_label, morphology.square(3))
                    dilated_image[(mask_label)] = label
                    nb_test += 1

                if not dilate:
                    break
                
            # Si aucun voisin trouvé après nb_attempt, supprimer ou conserver la région
            if not find_neighbor:
                if ones < min_cluster_size:
                    mask_label = dilated_image == label
                    ones = np.argwhere(mask_label == 1).shape[0] 
                    # Si l'objet dilaté ne vérifie pas la condition minimum
                    if ones < min_cluster_size:
                        res[mask_label] = 0
                        logger.info(f'Remove label {region.label}')
                    else:
                        # Si l'objet dilaté ne vérifie pas la condition maximum
                        while ones > max_cluster_size:
                            mask_label = morphology.erosion(mask_label, morphology.square(3))
                            ones = np.argwhere(mask_label == 1).shape[0]
                        
                        res[mask_label] = region.label
                        logger.info(f'Keep label dilated {region.label}')
                        fix_label.append(region.label)

            # Mettre à jour les régions pour tenir compte des changements
            regions = measure.regionprops(res)
            regions = sorted(regions, key=lambda r: r.area)
            len_regions = len(regions)
            i = 0
            print(dilate, find_neighbor, ones, nb_test, min_cluster_size, max_cluster_size)
            continue

        else:
            mask_label = res == region.label
            mask_before_erosion = np.copy(mask_label)
            while ones > max_cluster_size:
                mask_label = morphology.erosion(mask_label, morphology.square(3))
                ones = np.argwhere(mask_label == 1).shape[0]

            res[mask_before_erosion & ~mask_label] = 0

            # Si le cluster est assez grand, on le conserve tel quel
            logger.info(f'Keep label {region.label}')

        print(f'{ones} {min_cluster_size}, {max_cluster_size}')    
        i += 1

    return res

import hdbscan

def cluster_image_pixels(image, max_cluster_size, min_cluster_size, background):
    """
    Clusterise les pixels valides d'une image en utilisant HDBSCAN,
    tout en filtrant les clusters pour qu'ils soient entre min_cluster_size et max_cluster_size.

    :param image: np.ndarray (H, W, C) ou (H, W)
    :param max_cluster_size: int
    :param min_cluster_size: int
    :param background: list - Liste de pixels (ex: [[0,0,0]]) à ignorer
    :return: np.ndarray (H, W) - Image des labels (-1 = bruit ou hors taille)
    """
    img = np.array(image)
    if img.ndim == 2:
        img = img[..., np.newaxis]

    h, w, c = img.shape
    coords = []
    valid_positions = []

    for y in range(h):
        for x in range(w):
            pixel = img[y, x]
            if np.any(np.isnan(pixel)) or any(np.array_equal(pixel, b) for b in background):
                continue
            coords.append([y, x])
            valid_positions.append((y, x))

    if not coords:
        return np.full((h, w), -1, dtype=int)

    coords = np.array(coords)

    # Appliquer HDBSCAN
    clusterer = clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,  # limite appliquée
        cluster_selection_method='eom'  # obligatoire pour que max_cluster_size soit pris en compte
    )
    labels = clusterer.fit_predict(coords)

    # Filtrage des clusters par taille
    filtered_labels = np.full_like(labels, -1)
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    valid_clusters = [label for label, count in zip(unique, counts) if min_cluster_size <= count <= max_cluster_size]

    for i, label in enumerate(labels):
        if label in valid_clusters:
            filtered_labels[i] = label

    # Génération de l'image de clusters
    cluster_image = np.full((h, w), -1, dtype=int)
    for idx, (y, x) in enumerate(valid_positions):
        cluster_image[y, x] = filtered_labels[idx]

    return cluster_image

def find_clusters(image, threshold, clusters_to_ignore=None, background=0):
    """
    Traverse the clusters in an image and return the clusters whose size is greater than a given threshold.
    
    :param image: np.array, 2D image with values representing the clusters
    :param threshold: int, minimum size of the cluster to be considered
    :param background: int, value representing the background (default: 0)
    :param clusters_to_ignore: list, list of clusters to ignore (default: None)
    :return: list, list of cluster IDs whose size is greater than the threshold
    """
    # Initialize the list of valid clusters to return
    valid_clusters = []
    
    # If no clusters to ignore are provided, initialize with an empty list
    if clusters_to_ignore is None:
        clusters_to_ignore = []
    
    # Create a mask where the background is ignored
    mask = image != background
    
    # Label the clusters in the image
    cluster_ids = np.unique(image[mask])
    cluster_ids = cluster_ids[~np.isnan(cluster_ids)]
    
    # Traverse each cluster and check its size
    for cluster_id in cluster_ids:
        # Skip the cluster if it's in the ignore list
        if cluster_id == clusters_to_ignore:
            continue
        
        # Calculate the size of the cluster
        cluster_size = np.sum(image == cluster_id)
        
        # If the cluster size exceeds the threshold, add it to the list
        if cluster_size > threshold:
            valid_clusters.append(cluster_id)
    
    return valid_clusters

def split_large_clusters(image, size_threshold, min_cluster_size, wanted_size, background):
    labeled_image = np.copy(image)
    
    regions = measure.regionprops(labeled_image)
    new_labeled_image = np.copy(labeled_image)
    changes_made = False

    for region in regions:
        if region.label in background:
            continue

        original_size = region.area

        if original_size > size_threshold:
            minr, minc, maxr, maxc = region.bbox
            region_mask = (labeled_image[minr:maxr, minc:maxc] == region.label)
            coords = np.column_stack(np.nonzero(region_mask))

            if len(coords) > 1:
                # Appliquer KMeans
                clusterer = KMeans(n_clusters=2, random_state=42, n_init=10).fit(coords)
                labels = clusterer.labels_

                # Calcul des tailles des deux sous-clusters
                size_1 = np.sum(labels == 0)
                size_2 = np.sum(labels == 1)

                # Vérifier si le split améliore la proximité aux tailles voulues
                original_diff = abs(original_size - wanted_size)
                split_diff = abs(size_1 - wanted_size) + abs(size_2 - wanted_size)

                #if split_diff < original_diff and size_1 >= min_cluster_size and size_2 >= min_cluster_size:
                if split_diff < original_diff:
                    # Appliquer le split
                    new_label_1 = new_labeled_image.max() + 1
                    new_label_2 = new_label_1 + 1
                    full_region = new_labeled_image[minr:maxr, minc:maxc]

                    new_region = np.where(region_mask, full_region, 0)
                    # On assigne en place dans la région uniquement là où le masque est actif
                    new_region[region_mask] = np.where(labels == 0, new_label_1, new_label_2)
                    new_labeled_image[minr:maxr, minc:maxc][region_mask] = new_region[region_mask]

                    changes_made = True

    if changes_made:
        # Appel récursif pour gérer les divisions multiples
        new_labeled_image = split_large_clusters(
            new_labeled_image, size_threshold, min_cluster_size, wanted_size, background
        )

    return new_labeled_image

def relabel_clusters(cluster_labels, started):
    """
    Réorganise les labels des clusters pour qu'ils soient séquentiels et croissants.
    """
    unique_labels = np.sort(np.unique(cluster_labels[~np.isnan(cluster_labels)]))

    relabeled_image = np.copy(cluster_labels)
    for ncl, cl in enumerate(unique_labels):
        relabeled_image[cluster_labels == cl] = ncl + started
    return relabeled_image

def load_features(variables, sdate_year, edate_year, dir_data):

    vec_x = []
    
    for var in variables:
        if var in cems_variables:
            values = read_object(f'{var}raw.pkl', dir_data)
            assert values is not None
            values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
            values = np.mean(values, axis=2)
            vec_x.append(values)

        elif var == 'population' or var == 'elevation':
            values = read_object(f'{var}.pkl', dir_data)
            assert values is not None
            values = values.reshape((values.shape[0], values.shape[1]))
            vec_x.append(values)

        elif var == 'foret':
            values = read_object(f'{var}.pkl', dir_data)
            assert values is not None
            for i, var2 in enumerate(foret_variables):
                values2 = values[i]
                vec_x.append(values2)

        elif var == 'cosia':
            values = read_object(f'{var}.pkl', dir_data)
            assert values is not None
            for i, var2 in enumerate(cosia_variables):
                values2 = values[i]
                vec_x.append(values2)

        elif var == 'air':
            assert values is not None
            for i, var2 in enumerate(air_variables):
                values = read_object(f'{var2}raw.pkl', dir_data)
                assert values is not None
                values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                values = np.mean(values, axis=2)
                vec_x.append(values)

        elif var == 'sentinel':
            values = read_object(f'{var}.pkl', dir_data)
            assert values is not None
            for i, var2 in enumerate(sentinel_variables):
                values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                vec_x.append(values2)

        elif var == 'vigicrues':
            assert values is not None
            for i, var2 in enumerate(vigicrues_variables):
                values = read_object(f'vigicrues{var2}.pkl', dir_data)
                assert values is not None
                values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                values = np.mean(values, axis=2)
                vec_x.append(values)

        elif var == 'nappes':
            for i, var2 in enumerate(nappes_variables):
                values = read_object(f'{var2}.pkl', dir_data)
                assert values is not None
                values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                values = np.mean(values, axis=2)
                vec_x.append(values)

        elif var == 'highway':
            values = read_object(f'osmnx.pkl', dir_data)
            assert values is not None
            for i, var2 in enumerate(osmnx_variables):
                values2 = values[i]
                vec_x.append(values2)

        elif var == 'dynamic_world':
            values = read_object(f'{var}.pkl', dir_data)
            assert values is not None
            for i, var2 in enumerate(dynamic_world_variables):
                values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                vec_x.append(values2)
        else:
            raise ValueError(f'Unknow variable {var}')
        
    return np.asarray(vec_x)

def binary_closing_id(mask, selem):

    from skimage.morphology import binary_closing

    # Appliquer binary_closing pour chaque ID unique
    unique_ids = np.unique(mask[~np.isnan(mask)])
    processed_mask = np.full(mask.shape, fill_value=np.nan)

    for uid in unique_ids:
        mask_binary = (mask == uid)
        closed_mask = binary_closing(mask_binary, selem)
        processed_mask[closed_mask] = uid  # Réinjecter les valeurs
        
    return processed_mask

def count_pixels_in_france_deg_square(res_km=2, deg_size=0.25, lat_deg=46.5):
    """
    Calcule le nombre de pixels (res_km x res_km) dans un carré deg_size x deg_size degrés,
    situé au centre de la France (latitude 46.5°N par défaut).

    Args:
        res_km (float): Taille d’un pixel en kilomètres (par défaut 2 km).
        deg_size (float): Taille du carré en degrés (par défaut 0.25°).
        lat_deg (float): Latitude (par défaut 46.5°N, centre de la France).

    Returns:
        tuple: (n_rows, n_cols, total_pixels)
    """
    # Longueur d’un degré de latitude (quasi constant)
    km_per_deg_lat = 111.32

    # Longueur d’un degré de longitude dépendant de la latitude
    #km_per_deg_lon = 111.32 * math.cos(math.radians(lat_deg))
    km_per_deg_lon = 111.32

    # Dimensions du carré en km
    height_km = deg_size * km_per_deg_lat
    width_km = deg_size * km_per_deg_lon

    # Nombre de pixels
    n_rows = int(height_km // res_km)
    n_cols = int(width_km // res_km)
    total_pixels = n_rows * n_cols

    return n_rows, n_cols, total_pixels

from scipy.stats import pearsonr, spearmanr

def compute_pixelwise_correlation_matrix(target, features, IDs, method='pearson', image=True):
    """
    Calcule la corrélation entre chaque feature (par pixel) et la target pour chaque ID spatial (valeurs positives uniquement).

    Args:
        target (np.ndarray): Image 2D (H, W)
        features (np.ndarray): Image 3D (H, W, n_feat)
        IDs (np.ndarray): Image 2D (H, W) avec des IDs (valeurs entières, 0 = à ignorer)
        method (str): 'pearson' ou 'spearman'

    Returns:
        np.ndarray: Matrice (n_feat, n_ids) des corrélations
    """
    if image:
        if target.shape != IDs.shape or target.shape != features.shape[:2]:
            raise ValueError("target, IDs, et features doivent avoir des dimensions spatiales compatibles")

        H, W, n_feat = features.shape
    else:
        H, n_feat = features.shape

    unique_ids = np.unique(IDs)
    n_ids = len(unique_ids)

    corr_func = pearsonr if method == 'pearson' else spearmanr
    corr_matrix = np.full((n_feat, n_ids), np.nan)  # initialiser avec NaN

    for j, uid in enumerate(unique_ids):
        mask = (IDs == uid) & ~(np.isnan(IDs))

        if np.sum(mask) < 2:
            continue  # corrélation impossible avec < 2 points
        
        y = target[mask]
        for i in range(n_feat):
            if features.ndim == 3:
                x = features[:, :, i][mask]
            else:
                x = features[:, i][mask]

            y_c = y[~np.isnan(x) & ~np.isnan(y)]
            x = x[~np.isnan(x) & ~np.isnan(y)]
            if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                continue
            corr, _ = corr_func(x, y_c)
            if not np.isnan(corr):
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = np.nan

    #plot_correlation_with_maps_and_summary(corr_matrix, IDs, all_features_name, id_labels=unique_ids, method=method)
    return corr_matrix

from array_fet import *

def get_features_name_list(scale, features, methods):
    foretint2str = {
    '0': 'PasDeforet',
    '1': 'Châtaignier',
    '2': 'Chênes décidus',
    '3': 'Chênes sempervirents',
    '4': 'Conifères',
    '5': 'Douglas',
    '6': 'Feuillus',
    '7': 'Hêtre',
    '8': 'Mélèze',
    '9': 'Mixtes',
    '10': 'NC',
    '11': 'NR',
    '12': 'Pin à crochets, pin cembro',
    '13': 'Pin autre',
    '14': 'Pin d\'Alep',
    '15': 'Pin laricio, pin noir',
    '16': 'Pin maritime',
    '17': 'Pin sylvestre',
    '18': 'Pins mélangés',
    '19': 'Peuplier',
    '20': 'Robinier',
    '21': 'Sapin, épicéa'
    }

    osmnxint2str = {
    '0' : 'PasDeRoute',
    '1':'motorway',
    '2': 'primary',
    '3': 'secondary',
    '4': 'tertiary', 
    '5': 'path'}
    features_name = []
    if scale == 0:
        methods = ['mean']
    for var in features:
        if var == 'Calendar':
            features_name += calendar_variables
        elif var == 'air':
            features_name += air_variables
        elif var in landcover_variables:
            features_name += [f'{var}_{met}' for met in methods]
        elif var == 'sentinel':
            features_name += [f'{v}_{met}' for v in sentinel_variables for met in methods]
        elif var == "foret":
            features_name += [f'{foretint2str[v]}_{met}' for v in foret_variables for met in methods]
        elif var == 'dynamicWorld':
            features_name += [f'{v}_{met}' for v in dynamic_world_variables for met in methods]
        elif var == 'cosia':
            features_name += [f'{v}_{met}' for v in cosia_variables for met in methods]
        elif var == 'highway':
            features_name += [f'{osmnxint2str[v]}_{met}' for v in osmnx_variables for met in methods]
        elif var == 'Geo':
            features_name += geo_variables
        elif var == 'vigicrues':
            features_name += [f'{v}_{met}' for v in vigicrues_variables for met in methods]
        elif var == 'nappes':
            features_name += [f'{v}_{met}' for v in nappes_variables for met in methods]
        elif var == 'Historical':
            features_name += [f'{v}' for v in historical_variables]
        elif var == 'AutoRegressionReg':
            features_name += [f'AutoRegressionReg-{v}' for v in auto_regression_variable_reg]
        elif var == 'AutoRegressionBin':
            features_name +=  [f'AutoRegressionBin-{v}' for v in auto_regression_variable_bin]
        elif var == 'elevation':
            features_name += [f'{v}_{met}' for v in elevation_variables for met in methods]
        elif var == 'population':
            features_name += [f'{v}_{met}' for v in population_variabes for met in methods]
        elif var == 'region_class':
            features_name += [var]
        elif var == 'Past_risk' or var == 'Past_bunredarea':
            features_name += [var]
        elif var in varying_time_variables_name:
            features_name += [var]
        elif var == 'temporal_prediction' or var == 'spatial_prediction':
            features_name += [var]
        elif var.find('frequencyratio') != -1:
            features_name += [var]
        elif var in cluster_encoder:
            features_name += [var]
        else:
            features_name += [f'{var}_{met}' for met in methods]

    return features_name, len(features_name)

import seaborn as sns
from matplotlib.patches import Patch
def plot_correlation_with_maps_and_summary(corr_matrix, IDs, dir_output, feature_names=None, id_labels=None, method='pearson'):
    """
    Affiche :
    - la matrice de corrélation (n_feat x n_ids),
    - la carte des IDs (avec légende),
    - un barplot des moyennes par feature,
    - un boxplot unique regroupant toutes les corrélations.

    Args:
        corr_matrix (np.ndarray): Matrice (n_feat, n_ids) des corrélations.
        IDs (np.ndarray): Image 2D (H, W), avec IDs > 0.
        feature_names (list[str], optional): Noms des features.
        id_labels (list[int], optional): Liste des IDs.
        method (str): 'pearson' ou 'spearman'.
    """
    n_feat, n_ids = corr_matrix.shape
    feature_names = feature_names if feature_names is not None else [f"feat_{i}" for i in range(n_feat)]
    id_labels = id_labels if id_labels is not None else [f"ID {i}" for i in range(n_ids)]

    fig, axs = plt.subplots(2, 2, figsize=(35, 35))

    # 1. Heatmap des corrélations
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        xticklabels=id_labels, yticklabels=feature_names, ax=axs[0, 0]
    )
    axs[0, 0].set_title(f"Matrice de corrélation ({method})")
    axs[0, 0].set_xlabel("ID")
    axs[0, 0].set_ylabel("Feature")

    # 3. Barplot : moyenne des corrélations par feature
    # Filtrer les corrélations nulles ou NaN avant la moyenne
    filtered_corr_matrix = np.where(corr_matrix == 0, np.nan, corr_matrix)
    mean_corr = np.nanmean(np.abs(filtered_corr_matrix), axis=1)

    # Barplot avec moyennes filtrées
    axs[1, 0].bar(feature_names, mean_corr, color='steelblue')
    axs[1, 0].set_title("Corrélation moyenne par feature (hors zéros)")
    axs[1, 0].set_ylabel("Corrélation moyenne")
    axs[1, 0].set_xticks(range(n_feat))
    axs[1, 0].set_xticklabels(feature_names, rotation=90)
    
    sns.boxplot(data=[np.abs(mean_corr)], ax=axs[1, 1])
    axs[1, 1].set_title("Distribution globale des corrélations (hors zéros)")
    axs[1, 1].set_ylabel("Corrélation")
    axs[1, 1].set_xticklabels(["Toutes features"])

    plt.tight_layout()
    plt.savefig(dir_output / f'correlation_{method}.png')
    plt.close('all')

def frequency_ratio_per_feature(target, IDs, features, image):
    if image:
        if target.shape != IDs.shape or target.shape != features.shape[:2]:
            raise ValueError("target, IDs, et features doivent avoir des dimensions spatiales compatibles")

        H, W, n_feat = features.shape
    else:
        H, n_feat = features.shape

    unique_ids = np.unique(IDs)
    #unique_ids = unique_ids[unique_ids > 0]  # ignorer les IDs non positifs
    n_ids = len(unique_ids)

    corr_matrix = np.full((n_feat, n_ids), np.nan)  # initialiser avec NaN

    for j, uid in enumerate(unique_ids):
        mask = (IDs == uid) & ~(np.isnan(IDs))

        if np.sum(mask) < 2:
            continue  # corrélation impossible avec < 2 points
        
        y = target[mask]
        for i in range(n_feat):
            if features.ndim == 3:
                x = features[:, :, i][mask]
            else:
                x = features[:, i][mask]

            y_c = y[~np.isnan(x) & ~np.isnan(y)]
            x = x[~np.isnan(x) & ~np.isnan(y)]
            if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                continue
            
            FF_t = np.sum(y_c)
            Area_t = y_c.shape[0]

            if FF_t == 0 or Area_t == 0:
                corr_matrix[i, j] = np.nan
            else:            
                Area_i = np.argwhere(x > 0).shape[0]
                FF_i = np.sum(y_c[np.argwhere(x > 0)][:, 0])

                FireOcc = FF_i / FF_t
                Area = Area_i / Area_t

                if Area == 0:
                    corr_matrix[i, j] = np.nan
                else:    
                    FR = FireOcc / Area
                    corr_matrix[i, j] = FR
                
    #plot_correlation_with_maps_and_summary(corr_matrix, IDs, all_features_name, id_labels=unique_ids, method=method)
    return corr_matrix

from itertools import product

def compare_experiment(departements, scales, nb_attempts, n_reduce_classes, bases, method, name_output):
    
    path = Path(f'/media/caron/X9 Pro/travaille/Thèse/segmentation/')

    features_name = ['foret', 'elevation', 'population', 'cosia', 'highway']
    corr_exp = []
    exp_name = []
    all_features_name = get_features_name_list('departement', features_name, [''])[0]
    for scale, nb_attempt, n_reduce_class, base in product(scales, nb_attempts, n_reduce_classes, bases):
        targets_list = []
        features_list = []
        ids_list = []
        name_id = []
        path_file = path / f'{base}/s{scale}_a{nb_attempt}_r{n_reduce_class}/raster/'

        if (nb_attempt is None or n_reduce_class is None) and base != 'risk-regular':
            continue

        if base == 'risk-regular' and (nb_attempt is not None or n_reduce_class is not None):
            continue
        
        print(f'############################### {base}/s{scale}_a{nb_attempt}_r{n_reduce_class} ###############################')

        for dept in departements:
            file = read_object(f'{dept}rasterScale{scale}_{base}_node.pkl', path_file)
            if file is None:
                continue
            root_features = Path(f'/media/caron/X9 Pro/travaille/Thèse/csv/{dept}/raster/2x2')
            features = load_features(features_name, '2022-06-01', '2022-10-01', root_features)
            features = np.moveaxis(features, 0, 2)
            target = read_object(f'{dept}binScale0.pkl', root_target)
            assert target is not None
            target = np.nansum(target, axis=2)
            targets_list.append(target.reshape(-1, 1))
            features_list.append(features.reshape(-1, len(all_features_name)))
            ids_list.append(file.reshape(-1, 1))
            name_id.append(dept)

        if len(targets_list) > 0 :
            targets_list = np.concatenate(targets_list, axis=0)[:, 0]
            ids_list = np.concatenate(ids_list, axis=0)[:, 0]
            features_list = np.concatenate(features_list, axis=0)
            if method == 'fr':
                corr = frequency_ratio_per_feature(targets_list, ids_list, features_list, False)
            else:
                corr = compute_pixelwise_correlation_matrix(targets_list, features_list, ids_list, method,  image=False)
                plot_correlation_with_maps_and_summary(corr, name_id, path_file, all_features_name, id_labels=None, method=method)
            corr = np.nanmean(np.abs(corr), axis=-1)
            corr_exp.append(corr)
            exp_name.append(f'{base}/s{scale}_a{nb_attempt}_r{n_reduce_class}')

    if len(corr_exp) > 0:
        data_vars = {'name': ('experiment', exp_name)}
        for i, fet in enumerate(all_features_name):
            data_vars[fet] = ('experiment', [c[i] for c in corr_exp])

        ds = xr.Dataset(data_vars)
        save_object(ds, f'{name_output}_{method}.pkl', path)