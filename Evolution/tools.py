from pathlib import Path
import pickle
from turtle import color
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import measure, segmentation, morphology, filters
import math
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from scipy.ndimage import generic_filter
from skimage.transform import resize
from scipy.interpolate import griddata
import datetime as dt
from logger import *

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def save_object_torch(obj, filename : str, path : Path):
    check_and_create_path(path)
    torch.save(obj, path/filename)

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        logger.info(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def nanmax_pool_same_resolution(image, kernel_size=3):
    """
    Applique un max pooling sur l'image sans changer la résolution, en ignorant les NaN.
    
    Parameters:
        image (np.ndarray): Image 2D (avec potentiellement des NaN)
        kernel_size (int): Taille du noyau (doit être impair)
    
    Returns:
        np.ndarray: Image filtrée
    """
    def nanmax_func(values):
        return np.nanmax(values)

    return generic_filter(
        image,
        function=nanmax_func,
        size=kernel_size,
        mode='nearest'  # gère les bords
    )

from skimage.util import view_as_windows

def nanmax_pool_reduce(image, kernel_size=3):
    """
    Réduit la taille de l'image en appliquant un max pooling, en ignorant les NaN.
    
    Parameters:
        image (np.ndarray): Image 2D avec potentiellement des NaN.
        kernel_size (int): Taille du pooling (doit diviser image.shape exactement ou s'arrondira)
    
    Returns:
        np.ndarray: Image réduite (downsampled)
    """
    # Découpe en fenêtres
    k = kernel_size
    new_shape = (
        image.shape[0] // k,
        image.shape[1] // k
    )

    # Coupe l'image pour qu'elle soit divisible par kernel_size
    trimmed = image[:new_shape[0] * k, :new_shape[1] * k]

    # Vue glissante en blocs non chevauchants
    windows = view_as_windows(trimmed, (k, k), step=k)

    # Applique nanmax à chaque bloc
    pooled = np.nanmax(windows, axis=(2, 3))

    return pooled

import seaborn as sns

def plot_feature_distributions(images, features, dir_out: Path, titles):
    """
    images: list of 3D np.ndarray, shape: (num_features, H, W)
    features: list of str, names of the features (length == num_features)
    dir_out: Path where to save the plots
    """
    dir_out.mkdir(parents=True, exist_ok=True)

    num_features = len(features)
    num_images = len(images)
    palette = sns.color_palette("husl", num_images)
    
    for f_idx, feature_name in enumerate(features):
        feature_values_by_image = []
        image_labels = []

        for i, img in enumerate(images):
            assert img.shape[0] == num_features, f"Inconsistent number of features in image {i} (expected {num_features})"
            flattened = img[f_idx].flatten()
            cleaned = flattened[~np.isnan(flattened)]  # remove NaN

            if cleaned.size > 0:
                feature_values_by_image.append(cleaned)
                image_labels.extend([f"{titles[i]}"] * len(cleaned))

        all_values = np.concatenate(feature_values_by_image)

        plt.figure(figsize=(12, 6))
        sns.boxplot(x=image_labels, y=all_values, palette=palette[:len(feature_values_by_image)])
        plt.title(f"Distribution of Feature: {feature_name}")
        plt.xlabel("Image Index")
        plt.ylabel("Feature Value")
        plt.xticks(rotation=45)

        # Save plot
        filename = f"{feature_name.replace(' ', '_')}_distribution.png"
        plt.tight_layout()
        plt.savefig(dir_out / filename)
        plt.close()
        #print(f"✅ Saved: {dir_out / filename}")

def extract_bbox_region(binary_image):
    """
    Extrait la sous-image correspondant à la bounding box des pixels à 1.

    Args:
        binary_image (np.ndarray): Image 2D contenant des 0 et des 1

    Returns:
        np.ndarray: Image 2D contenant uniquement la région de la bounding box (sous-image)
                    ou un tableau vide si aucun pixel à 1
    """
    assert binary_image.ndim == 2, "L'image doit être 2D"

    indices = np.argwhere(binary_image == 1)

    if indices.size == 0:
        return np.array([], dtype=binary_image.dtype)  # Aucun pixel à 1, retourne tableau vide

    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)

    return binary_image[y_min:y_max + 1, x_min:x_max + 1]

def plot_matches_on_images(image_matches_list, titles, cmap='plasma', point_color='blue', figsize=(12, 10), name='matches_plot', dir_output=None, colorbar=True):
    """
    Affiche des correspondances pour une liste d'images avec une colorbar commune en dessous.
    NaN dans les images sont affichés comme transparents.

    Parameters:
        image_matches_list (list): Liste de tableaux image (np.ndarray)
        titles (list): Liste de titres pour chaque image
        point_color (str): Couleur des points (non utilisé ici)
        figsize (tuple): Taille globale de la figure
        dir_output (Path or None): Dossier où sauvegarder l'image. Si None, ne sauvegarde pas.
    """
    num_plots = len(image_matches_list)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    if colorbar:
        vmin = np.nanmin([np.nanmin(img) for img in image_matches_list])
        vmax = np.nanmax([np.nanmax(img) for img in image_matches_list])

    im = None
    for idx, img in enumerate(image_matches_list):
        ax = axes[idx]
        if colorbar:
            im = ax.imshow(img, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(img, cmap=cmap, origin='upper')
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_title(titles[idx])
        ax.set_xticks([])
        ax.set_yticks([])

    # Masquer les axes inutilisés
    for ax in axes[num_plots:]:
        ax.axis('off')

    # Ajustement du layout
    fig.subplots_adjust(wspace=0.2, hspace=0.3, bottom=0.15)

    # Ajouter une colorbar horizontale en bas
    if colorbar:
        if im:
            cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # [left, bottom, width, height]
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label("Number of fire")

    # Sauvegarde si demandé
    if dir_output:
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        output_path = dir_output / f"{name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    #plt.show()
    plt.close('all')

def merge_adjacent_clusters(image, nb_attempt=3, mode='size', min_cluster_size=0, max_cluster_size=math.inf, exclude_label=None, background=-1):
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

    # Nombre d'essais pour dilater un cluster avant abandon
    nb_attempt = 3

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
                best_neighbor = None
                mask_label = dilated_image == label
                mask_label_ori = res == label
                neighbors = segmentation.find_boundaries(mask_label, connectivity=1, mode='outer', background=background)
                neighbor_labels = np.unique(dilated_image[neighbors])
                
                    #print(neighbor_labels)
                
                # Exclure les labels indésirables
                neighbor_labels = neighbor_labels[(neighbor_labels != exclude_label) & (neighbor_labels != background) & (neighbor_labels != label)]

                dilate = True
                changed_labels.append(label)

                if len(neighbor_labels) > 0:
                    # Trier les voisins par taille
                    #neighbors_size = np.sort([[neighbor_label, np.sum(res == neighbor_label)] for neighbor_label in neighbor_labels], axis=0)
                    #neighbors_size = np.sort([[neighbor_label, np.sum(res == neighbor_label)] for neighbor_label in neighbor_labels], axis=1)
                    
                    neighbors_size = sorted(
                        [[neighbor_label, np.sum(res == neighbor_label)] for neighbor_label in neighbor_labels],
                        key=lambda x: x[1]  # trie par la somme (ordre croissant)
                    )

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
                                    dilated_image[dilated_image == label] = neighbor[0]
                                    print(f'Use neighbord label {label} -> {neighbor[0]}')
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

                        # Si aucun voisin ne satisfait les critères, utiliser le plus grand
                        if not find_neighbor and best_neighbor is not None:
                            if max_neighbor_size < max_cluster_size:
                                res[mask_label] = best_neighbor
                                dilated_image[dilated_image == label] = best_neighbor
                                dilate = False
                                print(f'Use biggest neighbord label {label} -> {best_neighbor}')
                                label = best_neighbor
                                find_neighbor = True
                                
                                # Si la taille après fusion dépasse la taille maximal, appliquer l'érosion (peut être ne pas fusionner)
                                """if max_neighbor_size < max_cluster_size:
                                    mask_label = dilated_image == label
                                    ones = np.argwhere(mask_label == 1).shape[0]
                                    while ones > max_cluster_size:
                                        mask_label = morphology.erosion(mask_label, morphology.disk(3))
                                        ones = np.argwhere(mask_label == 1).shape[0]"""
                                break

                # Si aucun voisin trouvé, dilater la région
                if dilate:
                    mask_label = morphology.dilation(mask_label, morphology.square(3))
                    dilated_image[dilated_image == label] = label
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
                        print(f'Remove label {region.label}')
                    else:
                        # Si l'objet dilaté ne vérifie pas la condition maximum
                        while ones > max_cluster_size:
                            mask_label = morphology.erosion(mask_label, morphology.square(3))
                            ones = np.argwhere(mask_label == 1).shape[0]
                        
                        res[mask_label] = region.label
                        print(f'Keep label dilated {region.label}')

            # Mettre à jour les régions pour tenir compte des changements
            regions = measure.regionprops(res)
            #if label == 7 or label == 11:
            #    print(np.unique(res))
            regions = sorted(regions, key=lambda r: r.area)
            len_regions = len(regions)
            i = 0
            continue
        else:
            mask_label = res == region.label
            mask_before_erosion = np.copy(mask_label)
            while ones > max_cluster_size:
                mask_label = morphology.erosion(mask_label, morphology.square(3))
                ones = np.argwhere(mask_label == 1).shape[0]

            res[mask_before_erosion & ~mask_label] = background

            # Si le cluster est assez grand, on le conserve tel quel
            print(f'Keep label {region.label}')
            
        i += 1
        
    return res

def my_watershed(data, valid_mask, apply_kmeans=True):

        if apply_kmeans:
                reducor = KMeans(n_clusters=4, n_init=10)
                reducor.fit(data[valid_mask].reshape(-1,1))
                data[valid_mask] = reducor.predict(data[valid_mask].reshape(-1,1))
                data[valid_mask] = order_class(reducor, data[valid_mask])
                print(np.unique(data[valid_mask]))
        data[~valid_mask] = 0

        data[valid_mask] = morphology.erosion(data, morphology.square(1))[valid_mask]

        # High Fire region
        # Détection des contours avec l'opérateur Sobel
        edges = filters.sobel(data)

        # Créer une carte de distance
        distance = np.full(data.shape, fill_value=0.0)
        distance = ndi.distance_transform_edt(edges)

        # Marquer les objets (régions connectées) dans l'image
        local_maxi = np.full(data.shape, fill_value=0)
        markers = np.full(data.shape, fill_value=0)
        local_maxi = morphology.local_maxima(distance)
        markers, _ = ndi.label(local_maxi)

        # Appliquer la segmentation Watershed
        pred = watershed(-data, markers, mask=data, connectivity=1)
        return pred

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

def split_large_clusters(image, size_threshold, min_cluster_size, background):
    labeled_image = np.copy(image)
    
    # Obtenir les propriétés des régions labellisées
    regions = measure.regionprops(labeled_image)
    
    # Initialiser une image pour les nouveaux labels après division
    new_labeled_image = np.copy(labeled_image)
    changes_made = False

    for region in regions:

        if region.label in background:
            continue
        
        if region.area > size_threshold:
            # Si la région est plus grande que le seuil, la diviser
            
            # Extraire le sous-image du cluster
            minr, minc, maxr, maxc = region.bbox
            region_mask = (labeled_image[minr:maxr, minc:maxc] == region.label)
            
            # Obtenir les coordonnées des pixels du cluster
            coords = np.column_stack(np.nonzero(region_mask))
            # Appliquer K-means pour diviser en 2 clusters
            if len(coords) > 1:  # Assurez-vous qu'il y a suffisamment de points pour appliquer K-means
                clusterer = KMeans(n_clusters=2, random_state=42, n_init=10).fit(coords)
                #clusterer = HDBSCAN(min_cluster_size=size_threshold).fit(coords)
                labels = clusterer.labels_
                
                # Créer deux nouveaux labels
                new_label_1 = new_labeled_image.max() + 1
                new_label_2 = new_labeled_image.max() + 2
                
                # Assigner les nouveaux labels aux pixels correspondants
                new_labeled_image[minr:maxr, minc:maxc][region_mask] = np.where(labels == 0, new_label_1, new_label_2)
                
                changes_made = True
    
    # Si des changements ont été effectués, vérifier s'il y a des clusters à fusionner
    if changes_made:
        new_labeled_image = split_large_clusters(new_labeled_image, size_threshold, min_cluster_size, background)
    
    return new_labeled_image

def resize_multiband_array(array, new_shape, preserve_dtype=True, interpolation_order=1):
    """
    Redimensionne un tableau numpy multi-bandes (H, W, C) vers une nouvelle taille, en ignorant les NaN.

    Paramètres :
        array (np.ndarray): Array d'entrée de forme (H, W, C) ou (H, W).
        new_shape (tuple): Nouvelle taille (new_height, new_width).
        preserve_dtype (bool): Si True, conserve le type original.
        interpolation_order (int): 0=nearest, 1=bilinear, etc.

    Retourne :
        np.ndarray: Array redimensionné à shape (new_height, new_width[, C]).
    """
    original_dtype = array.dtype
    new_height, new_width = new_shape

    def resize_ignore_nan(band):
        nan_mask = np.isnan(band)
        filled = np.copy(band)
        filled[nan_mask] = 0  # ou band[~nan_mask].mean() si tu veux interpoler sur moyenne

        resized_band = resize(filled, (new_height, new_width), order=interpolation_order,
                              preserve_range=True, anti_aliasing=False)

        # Resize du masque (valeurs entre 0 et 1)
        resized_mask = resize(~nan_mask, (new_height, new_width), order=0,
                              preserve_range=True, anti_aliasing=False)

        # Remettre les NaN là où le masque est nul
        resized_band[resized_mask < 0.5] = np.nan
        return resized_band

    if array.ndim == 2:
        resized = resize_ignore_nan(array)
    elif array.ndim == 3:
        bands = [resize_ignore_nan(array[:, :, i]) for i in range(array.shape[2])]
        resized = np.stack(bands, axis=2)
    else:
        raise ValueError("L'array doit être en 2D ou 3D (H, W, C)")

    if preserve_dtype:
        resized = resized.astype(original_dtype)

    resized[np.isnan(resized)] = np.nan
    return resized

from skimage.transform import resize

def resize_1d_array_ignore_nan(array, new_length, preserve_dtype=True, interpolation_order=1):
    """
    Redimensionne un tableau 1D vers une nouvelle taille, en ignorant les NaN.

    Paramètres :
        array (np.ndarray): Tableau 1D d'entrée.
        new_length (int): Longueur cible après redimensionnement.
        preserve_dtype (bool): Si True, conserve le type d'origine.
        interpolation_order (int): Ordre de l'interpolation (0 = nearest, 1 = linear, etc.)

    Retourne :
        np.ndarray: Tableau 1D redimensionné avec gestion des NaN.
    """
    if array.ndim != 1:
        raise ValueError("L'array doit être 1D.")

    original_dtype = array.dtype

    nan_mask = np.isnan(array)
    filled = np.copy(array)
    filled[nan_mask] = 0  # ou np.nanmean(array) si tu préfères

    # Redimensionner les données
    resized = resize(filled, (new_length,), order=interpolation_order,
                     preserve_range=True, anti_aliasing=False)

    # Redimensionner le masque (inverse pour repérer où il y avait du vrai contenu)
    resized_mask = resize(~nan_mask, (new_length,), order=0,
                          preserve_range=True, anti_aliasing=False)

    if preserve_dtype:
        resized = resized.astype(original_dtype)

    return resized

def interpolate_image_to_match_target(base_image, target_image, interpolation_order=1, preserve_dtype=True):
    """
    Redimensionne une image source pour qu'elle ait la même taille que l'image cible.

    Args:
        base_image (np.ndarray): Image d'origine (2D ou 3D).
        target_image (np.ndarray): Image de référence pour la taille.
        interpolation_order (int): Ordre de l'interpolation (0=nearest, 1=bilinear, 3=bicubic).
        preserve_dtype (bool): Si True, conserve le type d'origine à la fin.

    Returns:
        np.ndarray: Image interpolée à la taille de l'image cible.
    """
    original_dtype = base_image.dtype
    target_shape = target_image.shape[:2]  # (H, W)

    def resize_ignore_nan(band):
        nan_mask = np.isnan(band)
        filled = np.copy(band)
        filled[nan_mask] = 0

        resized_band = resize(filled, target_shape, order=interpolation_order,
                              preserve_range=True, anti_aliasing=False)

        resized_mask = resize(~nan_mask, target_shape, order=0,
                              preserve_range=True, anti_aliasing=False)

        resized_band[resized_mask < 0.5] = np.nan
        return resized_band

    if base_image.ndim == 2:
        resized = resize_ignore_nan(base_image)
    elif base_image.ndim == 3:
        resized = np.stack([resize_ignore_nan(base_image[:, :, i]) for i in range(base_image.shape[2])], axis=2)
    else:
        raise ValueError("L'image de base doit être en 2D ou 3D (H, W[, C])")

    if preserve_dtype:
        resized = resized.astype(original_dtype)

    return resized

def interpolate_image_3d(image, exclude_mask=None, method='linear'):
    """
    Interpole les NaN d'une image 3D (C, H, W), canal par canal,
    en excluant les pixels masqués (True = exclus).

    Parameters:
        image (np.ndarray): Image 3D de forme (C, H, W) avec des NaN.
        exclude_mask (np.ndarray): Masque booléen 2D (H, W), True = ne pas interpoler.
        method (str): Méthode d'interpolation ('linear', 'nearest', 'cubic').

    Returns:
        np.ndarray: Image 3D interpolée.
    """
    assert image.ndim == 3, "L'image doit être 3D (C, H, W)"
    C, H, W = image.shape
    if exclude_mask is None:
        exclude_mask = np.zeros((H, W), dtype=bool)
    assert exclude_mask.shape == (H, W), "Le masque doit avoir la forme (H, W)"

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    result = np.copy(image)

    for c in range(C):
        channel = image[c]
        valid_mask = ~np.isnan(channel) & ~exclude_mask
        nan_mask = np.isnan(channel) & ~exclude_mask

        if np.any(nan_mask):
            points = np.stack((X[valid_mask], Y[valid_mask]), axis=-1)
            values_known = channel[valid_mask]
            points_interp = np.stack((X[nan_mask], Y[nan_mask]), axis=-1)
            interpolated = griddata(points, values_known, points_interp, method=method)

            channel_interp = channel.copy()
            channel_interp[nan_mask] = interpolated
            result[c] = channel_interp
            
    return result

def interpolate_image_2d(image, exclude_mask=None, method='nearest'):
    """
    Interpole les NaN d'une image 2D (H, W), en excluant les pixels masqués (True = exclus).

    Parameters:
        image (np.ndarray): Image 2D de forme (H, W) avec des NaN.
        exclude_mask (np.ndarray): Masque booléen 2D (H, W), True = ne pas interpoler.
        method (str): Méthode d'interpolation ('linear', 'nearest', 'cubic').

    Returns:
        np.ndarray: Image 2D interpolée.
    """
    assert image.ndim == 2, "L'image doit être 2D (H, W)"
    H, W = image.shape

    if exclude_mask is None:
        exclude_mask = np.zeros((H, W), dtype=bool)
    assert exclude_mask.shape == (H, W), "Le masque doit avoir la forme (H, W)"

    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    valid_mask = ~np.isnan(image) & ~exclude_mask
    nan_mask = np.isnan(image) & ~exclude_mask

    result = image.copy()

    if np.any(nan_mask):
        points = np.stack((X[valid_mask], Y[valid_mask]), axis=-1)
        values_known = image[valid_mask]
        points_interp = np.stack((X[nan_mask], Y[nan_mask]), axis=-1)

        interpolated = griddata(points, values_known, points_interp, method=method)
        result[nan_mask] = interpolated

    return result

def remove_nan_pixels(features, target):
    N, C, H, W, T = features.shape

    # Mise à plat
    X = features.reshape(N, C * T, -1).T  # shape: (H*W, C*T)
    y = target.flatten()

    # Détection des NaN
    nan_mask = torch.any(torch.isnan(X), axis=1) | torch.any(torch.isnan(y), axis=1)

    # Inversion → pixels valides
    valid_mask = ~nan_mask

    # Nettoyage
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    return X_clean, y_clean

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