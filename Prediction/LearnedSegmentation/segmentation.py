
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import logging
import math
from skimage import morphology, filters, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsRegressor
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

# Add parent directory to path to import GNN modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'GNN'))

# Mock dgl to avoid import errors if not needed
from unittest.mock import MagicMock
import types
dgl_mock = types.ModuleType('dgl')
dgl_mock.DGLGraph = MagicMock()
sys.modules['dgl'] = dgl_mock

dgl_nn_mock = types.ModuleType('dgl.nn')
sys.modules['dgl.nn'] = dgl_nn_mock

dgl_nn_pytorch_mock = types.ModuleType('dgl.nn.pytorch')
dgl_nn_pytorch_mock.GATConv = MagicMock()
dgl_nn_pytorch_mock.GraphConv = MagicMock()
sys.modules['dgl.nn.pytorch'] = dgl_nn_pytorch_mock
sys.modules['dgl.nn.pytorch.conv'] = MagicMock()

sys.modules['dgl.nn.functional'] = MagicMock()
sys.modules['dgl.function'] = MagicMock()
sys.modules['dgl.convert'] = MagicMock()

blitz_mock = types.ModuleType('blitz')
sys.modules['blitz'] = blitz_mock
sys.modules['blitz.modules'] = MagicMock()
sys.modules['blitz.utils'] = MagicMock()
sys.modules['blitz.losses'] = MagicMock()

sys.modules['pygam'] = MagicMock()

sys.modules['ngboost'] = MagicMock()
sys.modules['ngboost.distns'] = MagicMock()
sys.modules['ngboost.scores'] = MagicMock()

skopt_mock = types.ModuleType('skopt')
skopt_mock.BayesSearchCV = MagicMock()
skopt_mock.Optimizer = MagicMock()
sys.modules['skopt'] = skopt_mock
sys.modules['skopt.space'] = MagicMock()

sys.modules['osmnx'] = MagicMock()

sys.modules['dtaidistance'] = MagicMock()
sys.modules['dtaidistance.dtw'] = MagicMock()

sys.modules['dtwParallel'] = MagicMock()

from GNN.tools import (
    read_object, save_object, check_and_create_path, relabel_clusters,
    count_pixels_in_france_deg_square, merge_adjacent_clusters, find_clusters,
    split_large_clusters, frequency_ratio, order_class, allDates, iou_binary, to_binary_mask
)
from GNN.arborescence import rootDisk, root_target
from GNN.weigh_predictor import Predictor

logger = logging.getLogger(__name__)

class Segmentation:
    def __init__(self, scale, base, attempt, reduce, tol, susecptibility_variables=None, train_departements=None,
                 resolution='2x2', dataset_name='fire_risk', sinister='firepoint', sinister_encoding='occurence'):
        
        self.scale = scale
        self.base = base
        self.attempt = attempt
        self.reduce = reduce
        self.tol = tol
        self.susecptibility_variables = susecptibility_variables
        self.train_departements = train_departements if train_departements is not None else []
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.sinister = sinister
        self.sinister_encoding = sinister_encoding

        self.max_target_value = None
        self.dispersions = {}

    def create_geometry_with_watershed(self, dept, vec_base, path, sinister, dataset_name,
                                       sinister_encoding, resolution, mask, node_already_predicted, train_date, data, GT=None):
        
        dir_data = rootDisk / 'csv' / dept / 'raster' / resolution
        
        dir_target_bin = root_target / sinister / dataset_name / sinister_encoding / 'bin' / resolution
        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
        raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
        
        assert raster is not None
        raster = raster[0]
        pred = np.full(raster.shape, fill_value=np.nan)
        valid_mask = (raster != -1) & (~np.isnan(raster))

        self.max_target_value = None

        vb = vec_base[0]
        mode = vec_base[1]

        # Load data_bin (nbsinister)
        data_bin = self.process_input_data(dept, dir_target_bin, train_date)

        if data.ndim == 3:
            data = np.nansum(data, axis=2)

        if data is None:
            logger.info(f'Can t find {vb}')
            exit(1)

        if self.max_target_value is None:
            self.max_target_value = np.nanmax(data)
        else:
            self.max_target_value = max(self.max_target_value, np.nanmax(data))
            
        oridata = np.copy(data)
        
        self._save_feature_image(path, dept, 'sum', data, raster, 0, self.max_target_value)

        # ---------------------------------------------------------------------
        # OPTIMISATION NELDER-MEAD (si demandé)
        # ---------------------------------------------------------------------
        if self.attempt == 'search' and self.reduce == 'search':
            logger.info("Starting Grid Search optimization for (a, r)...")

            # Binaire de la vérité terrain
            B_bin = to_binary_mask(data_bin)

            # Historique pour le plot
            history = []

            # Calcul des tailles de clusters (logique extraite de create_cluster)
            if 'degree' in self.base:
                size = count_pixels_in_france_deg_square(deg_size=float(f'0.{self.scale}'))[-1]
                max_cluster_size = int(size + (self.tol * size))
                min_cluster_size = int(size - (self.tol * size))
            else:
                min_cluster_size = 1 + 3 * self.scale * (self.scale + 1)
                max_cluster_size = (int)(min_cluster_size * 2.5)

            best_score = -float('inf')
            best_a = 2
            best_r = 1

            # Grid Search Loop
            # attempt (r) : 1 à 10
            # reduce (a) : 2 à 10
            range_r = range(1, 11)
            range_a = range(2, 11)

            for a in range_a:
                # 1) my_watershed avec 'a' (reduce)
                # On travaille sur une copie de data pour ne pas écraser l'original
                data_tmp = np.copy(oridata)
                pred_ws = self.my_watershed(dept, data_tmp, valid_mask, raster, path, vb, 'optim_temp', reduce=a)
                
                for r in range_r:
                    # 2) merge_adjacent_clusters avec 'r' (attempt)
                    S_raw = merge_adjacent_clusters(pred_ws, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
                                                    features=None, mode=mode, exclude_label=0, background=-1,
                                                    nb_attempt=r)

                    # 3) Calcul IoU
                    S = to_binary_mask(np.asarray(S_raw))
                    score = iou_binary(B_bin, S)
                    
                    # Sauvegarde historique
                    history.append((a, r, score))

                    # Mise à jour du meilleur score
                    if score > best_score:
                        best_score = score
                        best_a = a
                        best_r = r

            logger.info(f"Grid Search finished. Best a={best_a}, Best r={best_r}, Best IoU={best_score}")

            # On fixe les valeurs pour la suite de l'exécution
            final_reduce = best_a
            final_attempt = best_r
        else:
            # Pas d'optimisation, on utilise les valeurs de self
            final_reduce = self.reduce
            final_attempt = self.attempt

        # ---------------------------------------------------------------------
        # EXECUTION FINALE (avec paramètres optimisés ou fixés)
        # ---------------------------------------------------------------------
        
        # 1) my_watershed
        # On repart de oridata propre
        data = np.copy(oridata)
        pred = self.my_watershed(dept, data, valid_mask, raster, path, vb, 'pred', reduce=final_reduce)
        
        if GT is not None:
            self._save_feature_image(path, dept, 'gt_sum', GT, raster, 0, self.max_target_value)
            pred_GT = self.my_watershed(dept, GT, valid_mask, raster, path, vb, 'gt', reduce=final_reduce)

        # Merge and split clusters
        umarker = np.unique(pred)
        umarker = umarker[(umarker != 0) & ~(np.isnan(umarker))]
        risk_image = np.full(oridata.shape, fill_value=np.nan)
        for m in umarker:
            mask_temp = (pred == m)
            risk_image[mask_temp] = np.sum(oridata[mask_temp])
        self._save_feature_image(path, dept, 'pred_risk', risk_image, raster)

        pred[~valid_mask] = -1
        bin_data = read_object(f'{dept}binScale0.pkl', dir_target_bin)
        assert bin_data is not None

        # 2) create_cluster
        _, pred = self.create_cluster(pred, dept, path, self.scale, mode, bin_data, raster, valid_mask, 'pred', attempt=final_attempt)

        try:
            # Analyse dispersion of fire regions
            self.dispersions = {}
            clusters = np.unique(pred[valid_mask])
            for cluster_id in clusters:
                # Extraire les pixels du cluster actuel
                cluster_pixels = np.argwhere(pred == cluster_id)
                
                # Calculer le centroïde du cluster
                centroid = np.mean(cluster_pixels, axis=0)
                
                # Calculer la distance de chaque pixel au centroïde
                distances = cdist(cluster_pixels, [centroid])
                
                # Calculer la dispersion (écart-type des distances)
                self.dispersions[cluster_id] = np.std(distances)
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return pred

        #logger.info(f'Cluster dispersion {self.dispersions}')

    def create_cluster(self, pred, dept, path, scale, mode, bin_data, raster, valid_mask, type, attempt=None, print=True):

        if bin_data.ndim == 2:
            bin_data = np.expand_dims(bin_data, axis=2)
        
        # Gestion de attempt
        if attempt is None:
            if self.attempt == 'search':
                raise ValueError("self.attempt est 'search', vous devez fournir une valeur explicite pour 'attempt'.")
            attempt = self.attempt

        if 'degree' in self.base:
            if isinstance(self.scale, int) or isinstance(self.scale, str):
                size = count_pixels_in_france_deg_square(deg_size=float(f'0.{self.scale}'))[-1]
                max_cluster_size = int(size + (self.tol * size))
                min_cluster_size = int(size - (self.tol * size))
                s = float(f'0.{self.scale}')
            else:
                size = count_pixels_in_france_deg_square(deg_size=self.scale)[-1]
                max_cluster_size = int(size + (self.tol * size))
                min_cluster_size = int(size - (self.tol * size))
                s = self.scale
        else:
            min_cluster_size = 1 + 3 * scale * (scale + 1)
            max_cluster_size = (int)(min_cluster_size * 2.5)

        if mode == 'size':
            
            pred = merge_adjacent_clusters(pred, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
            features=None, mode=mode, exclude_label=0, background=-1,
            nb_attempt=attempt)

            valid_cluster = find_clusters(pred, min_cluster_size, 0, -1)
            self._save_feature_image(path, dept, 'pred_merge', pred, raster)

            logger.info(np.unique(pred))
            mask_valid = np.isin(pred, valid_cluster)
            if print:
                logger.info(f'{dept} : We found {len(valid_cluster)} to build geometry.')
                logger.info(f'Number of fire inside regions {np.nansum(bin_data[mask_valid])}')
                logger.info(f'Number of fire outside regions {np.nansum(bin_data[~mask_valid])}')

            valid_cluster = [val + 1 for val in valid_cluster]
            pred[valid_mask] += 1
            pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, size, valid_cluster)
            valid_cluster = [val - 1 for val in valid_cluster]
            pred[valid_mask] -= 1
            
            pred = pred.astype(float)
            self._save_feature_image(path, dept, f'{type}_split', pred, raster)
        
        elif mode == 'fireArea':
            
            pred = merge_adjacent_clusters(pred, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
            features=None, mode=mode, exclude_label=0, background=-1,
            nb_attempt=attempt)

            valid_cluster = find_clusters(pred, min_cluster_size, 0, -1)
            self._save_feature_image(path, dept, 'pred_merge', pred, raster)
            
            pred = pred.astype(float)
            self._save_feature_image(path, dept, f'{type}_split', pred, raster)

        elif mode == 'time_series_similarity':
            pred = merge_adjacent_clusters(pred, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, mode=mode, features=bin_data, exclude_label=0, background=-1, nb_attempt=attempt)
            valid_cluster = find_clusters(pred, min_cluster_size, 0, -1)
            self._save_feature_image(path, dept, f'{type}_merge', pred, raster)

            logger.info(np.unique(pred))
            logger.info(f'{dept} : We found {len(valid_cluster)} to build geometry.')
            mask_valid = np.isin(pred, valid_cluster)
            logger.info(f'Number of fire inside regions {np.nansum(bin_data[mask_valid])}')
            logger.info(f'Number of fire outside regions {np.nansum(bin_data[~mask_valid])}')

            valid_cluster = [val + 1 for val in valid_cluster]
            pred[valid_mask] += 1
            pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, size, valid_cluster)
            valid_cluster = [val - 1 for val in valid_cluster]
            pred[valid_mask] -= 1

        elif mode == 'time_series_similarity_fast':
            pred[~valid_mask] = -1
            pred = merge_adjacent_clusters(pred, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, mode=mode, features=bin_data, exclude_label=0, background=-1, nb_attempt=attempt)
            valid_cluster = find_clusters(pred, math.inf, 0, -1)
            self._save_feature_image(path, dept, f'{type}_merge', pred, raster)

            logger.info(np.unique(pred))
            logger.info(f'{dept} : We found {len(valid_cluster)} to build geometry.')
            mask_valid = np.isin(pred, valid_cluster)
            logger.info(f'Number of fire inside regions {np.nansum(bin_data[mask_valid])}')
            logger.info(f'Number of fire outside regions {np.nansum(bin_data[~mask_valid])}')

            valid_cluster = [val + 1 for val in valid_cluster]
            pred[valid_mask] += 1
            pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, size, valid_cluster)
            valid_cluster = [val - 1 for val in valid_cluster]
            pred[valid_mask] -= 1

        sum_fr = 0
        bin_data_sum = np.nansum(bin_data, axis=2)
        for cluster in valid_cluster:
            fr_cluster = frequency_ratio(bin_data_sum[~np.isnan(bin_data_sum)], np.argwhere((pred[~np.isnan(bin_data_sum)] == cluster)))
            sum_fr += fr_cluster
            if print:
                logger.info(f'{cluster} frequency ratio -> {fr_cluster}')
        
        if len(valid_cluster) == 0:
            sum_fr = 0
        else:
            sum_fr = sum_fr / len(valid_cluster)
        if print:
            logger.info(f'Mean fr {sum_fr}')

        return sum_fr, pred

    def my_watershed(self, dept, data, valid_mask, raster, path, vb, image_type, reduce=None):
        
        # Si reduce n'est pas fourni, on utilise self.reduce (sauf si c'est 'search', auquel cas il faut le fournir)
        if reduce is None:
            if self.reduce == 'search':
                raise ValueError("self.reduce est 'search', vous devez fournir une valeur explicite pour 'reduce'.")
            reduce = self.reduce

        reducor = Predictor(n_clusters=reduce)
        reducor.fit(data[valid_mask].reshape(-1,1))
        data[valid_mask] = reducor.predict(data[valid_mask].reshape(-1,1))
        data[valid_mask] = order_class(reducor, data[valid_mask])
        data[~valid_mask] = 0

        data[valid_mask] = morphology.erosion(data, morphology.square(1))[valid_mask]
        self._save_feature_image(path, dept, f'{vb}_{image_type}', data, raster)

        # High Fire region
        # Détection des contours avec l'opérateur Sobel
        edges = filters.sobel(data)
        self._save_feature_image(path, dept, f'edges_{image_type}', edges, raster)

        # Créer une carte de distance
        distance = np.full(data.shape, fill_value=0.0)
        distance = ndi.distance_transform_edt(edges)
        self._save_feature_image(path, dept, f'distance_{image_type}', distance, raster)

        # Marquer les objets (régions connectées) dans l'image
        local_maxi = np.full(data.shape, fill_value=0)
        markers = np.full(data.shape, fill_value=0)
        local_maxi = morphology.local_maxima(distance)
        markers, _ = ndi.label(local_maxi)

        # Appliquer la segmentation Watershed
        pred = watershed(-data, markers, mask=data, connectivity=1)
        self._save_feature_image(path, dept, f'pred_watershed_{image_type}', pred, raster)
        pred_save = np.copy(pred).astype(float)
        pred_save[np.isnan(raster)] = np.nan
        save_object(pred_save, f'watershed_{dept}.pkl', path / 'features_geometry')
        return pred

    def _save_feature_image(self, path, dept, vb, image, raster, mini=None, maxi=None):
        data = np.copy(image)
        check_and_create_path(path / 'features_geometry' / f'{self.scale}_{self.base}_node' / dept) # Assuming graph_method='node' or generic
        data = data.astype(float)
        data[np.isnan(raster)] = np.nan
        plt.figure(figsize=(15, 15))
        if mini is None:
            mini = np.nanmin(image)
        if maxi is None:
            maxi = np.nanmax(image)
        img = plt.imshow(data, vmin=mini, vmax=maxi)
        plt.colorbar(img)
        plt.title(vb)
        plt.savefig(path / 'features_geometry' / f'{self.scale}_{self.base}_node' / dept / f'{vb}.png')
        plt.close('all')

    def process_input_data(self, vb, dept, dir_target, dir_target_bin, dir_data, valid_mask, raster, train_date, path):
        
        GT = None

        if train_date is None:
            train_date = allDates
            
        if vb == 'risk':
            data = read_object(f'{dept}Influence.pkl', dir_target)
            if data is None or dept not in self.train_departements:
                if data is not None:
                    GT = np.copy(data)
                    GT = GT[:, :, [allDates.index(date) for date in train_date]]
                    GT = np.nansum(GT, axis=2)
                if data is None:
                    self.predict_susecptibility_map([dept], self.susecptibility_variables, dir_data, path / 'predict_map')
                    data = read_object(f'{dept}Influence.pkl', path / 'predict_map')
            else:
                data = data[:, :, [allDates.index(date) for date in train_date]]
                data = np.nansum(data, axis=2)
        elif vb == 'nbsinister':
            data = read_object(f'{dept}binScale0.pkl', dir_target_bin)
            if data is None or dept not in self.train_departements:
                if data is None:
                    self.predict_susecptibility_map([dept], self.susecptibility_variables, dir_data, path / 'predict_map')
                    data = read_object(f'{dept}binScale0.pkl', path / 'predict_map')
            else:
                data = data[:, :, [allDates.index(date) for date in train_date]]
                data = np.nansum(data, axis=2)
        elif vb == 'geometry':
            width, height = raster.shape[1], raster.shape[0]
            positions = np.array([[x, y] for y in range(height) for x in range(width)])
            X_image = np.zeros((height, width), dtype=float)
            Y_image = np.zeros((height, width), dtype=float)
            
            # Remplir les images avec les indices x et y
            for pos in positions:
                x, y = pos
                X_image[y, x] = x  # Indice selon x
                Y_image[y, x] = y  # Indice selon y
            
            X_image[~valid_mask] = np.nan
            Y_image[~valid_mask] = np.nan
            res = np.stack((X_image, Y_image), axis=2)
            return res
        else:
            dir_encoder = path / 'Encoder'
            data = read_object(f'{vb}.pkl', dir_data)
            if data is None:
                exit(1)
            if vb == 'foret_landcover':
                encoder = read_object('encoder_foret.pkl', dir_encoder)
            elif vb == 'dynamic_world_landcover':
                encoder = read_object('encoder_landcover.pkl', dir_encoder)
            else:
                encoder = None

            if encoder is not None:
                values = data[valid_mask].reshape(-1)
                data[valid_mask] = encoder.transform(values).values.reshape(-1)

            if len(data.shape) > 2:
                data = np.nansum(data, axis=2)

        data[~valid_mask] = np.nan
        if GT is not None:
            GT[~valid_mask] = np.nan

        return data, GT
