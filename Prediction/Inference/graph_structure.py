from re import L
from matplotlib import axis
import numpy as np
from pandas import Categorical
np.random.seed(11)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn import pipeline
import math
import geopandas as gpd 
from shapely import unary_union
from astropy.convolution import convolve, Box2DKernel
from shapely import Point
from dataloader import *
from skimage import io, color, filters, measure, morphology
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import scipy.ndimage as ndimage
from astropy.convolution import convolve_fft
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial.distance import cdist

# Create graph structure from corresponding geoDataframe
class GraphStructure():
    def __init__(self, scale : int,
                 geo : gpd.GeoDataFrame,
                 maxDist: float,
                 numNei : int,
                 resolution : str,
                 graph_construct : str,
                 sinister : str,
                 sinister_encoding : str,
                 dataset_name : str,
                 train_departements : list,
                 graph_method : str == 'node'):

        for col in ['latitude', 'longitude', 'geometry', 'departement']:
            if col not in geo.columns:
                logger.info(f'{col} not in geo columns. Please send a correct geo dataframe')
                exit(2)

        self.nodes = None # All nodes in graph (N, 6) [id, long, lat, dep]
        self.edges = None # All edges in graph (2 x E)
        self.temporalEdges = None # Temporal edges maximum (2 * D * k) where D is the number of dates available,
                                    #K is the number of link by date (minimum 1 for self connection)
        self.model = None # GNN model
        self.train_kmeans = None # Boolean to trace if we creating new scale
        self.scale = scale # current scale define by numUniqueNode // 6
        self.oriLatitudes = geo.latitude # all original latitude of interest
        self.oriLongitude = geo.longitude # all original longitude of interest
        self.oriLen = self.oriLatitudes.shape[0] # len of data
        self.oriGeometry = geo['geometry'].values # original geometry
        self.oriIds = geo['scale0'].values
        self.orihexid = geo['hex_id'].values
        self.departements = geo.departement # original dept of each geometry
        self.maxDist = maxDist # max dist to link two nodes
        self.numNei = numNei # max connect neighbor a node can have
        self.sinister = None # type of sinister
        self.clusterer = {} # 
        self.drop_department = [] # the department with no fire region
        self.sinister_regions = [] # fire region id
        self.sequences_month = None
        self.historical_risk = None
        self.resolution = resolution
        self.sinister = sinister
        self.base = graph_construct
        self.sinister_encoding = sinister_encoding
        self.dataset_name = dataset_name
        self.train_departements = train_departements
        self.graph_method = graph_method
        self.susceptility_mapper = None

    def _create_sinister_region(self, base: str, path: Path, sinister: str, dataset_name:str, sinister_encoding : str, resolution, train_date) -> None:
        udept = np.unique(self.departements)
        if '-' in base:
            vec_base = base.split('-')
            self.base = base
        else:
            vec_base = [base]
            self.base = base

        self.train_kmeans = True
        node_already_predicted = 0
        self.numCluster = 0
        self.resolution = resolution
        
        logger.info(f'Create node via {vec_base}')
        
        self.ids = np.full(self.oriLatitudes.shape[0], fill_value=np.nan)
        self.graph_ids = np.full(self.oriLatitudes.shape[0], fill_value=np.nan)

        for dept in udept:
            mask = self.departements == dept
            logger.info(f'######################### {dept} ####################')
            if self.scale == 'departement':
                assert self.base == 'None'
                dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
                raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
                assert raster is not None
                raster = raster[0]
                pred = np.full(raster.shape, fill_value=np.nan)
                pred[~np.isnan(raster)] = node_already_predicted
                # Process post-watershed results
                self._post_process_result(pred, raster, mask, node_already_predicted, 'graph')
                self._save_feature_image(path, dept, 'pred_final', pred, raster)
            else:
                if 'clustering' in vec_base:
                    self.create_geometry_with_clustering(dept, vec_base, path, sinister, dataset_name, sinister_encoding, resolution, mask, node_already_predicted, train_date)
                elif 'watershed' in vec_base:
                    self.create_geometry_with_watershed(dept, vec_base, path, sinister, dataset_name, sinister_encoding, resolution, mask, node_already_predicted, train_date)
                elif 'regular' in vec_base:
                    self.create_geometry_with_regular(dept, vec_base, path, sinister, dataset_name, sinister_encoding, resolution, mask, node_already_predicted)

            current_cluster = np.nanmax(self.graph_ids[mask][~np.isnan(self.graph_ids[mask])]) + 1
            logger.info(f'{dept} Unique cluster : {np.unique(self.graph_ids[mask])}, {current_cluster}. {node_already_predicted}')
            node_already_predicted = current_cluster
            
        self.oriIds = self.oriIds[~np.isnan(self.graph_ids)]
        self.oriLatitudes = self.oriLatitudes[~np.isnan(self.graph_ids)].reset_index(drop=True)
        self.oriGeometry = self.oriGeometry[~np.isnan(self.graph_ids)]
        self.oriLongitude = self.oriLongitude[~np.isnan(self.graph_ids)].reset_index(drop=True)
        self.departements = self.departements[~np.isnan(self.graph_ids)].reset_index(drop=True)
        self.graph_ids = self.graph_ids[~np.isnan(self.graph_ids)].astype(int)
        self.graph_ids = relabel_clusters(self.graph_ids, 0)
        self.numCluster = np.unique(self.graph_ids).shape[0]
        self.oriLen = self.oriLatitudes.shape[0]
        self.numCluster = np.shape(np.unique(self.graph_ids))[0]

        self._raster(path=path, sinister=sinister, base=base, resolution=resolution, train_date=train_date, dataset_name=dataset_name, sinister_encoding=sinister_encoding)

    def create_geometry_with_clustering(self, dept, vec_base, path, sinister, dataset_name, sinister_encoding, resolution, mask, node_already_predicted, train_date):
        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
        dir_data = rootDisk / 'csv' / dept / 'raster' / resolution
        dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution
        dir_target_bin = root_target / sinister / dataset_name / sinister_encoding / 'bin' / resolution
        raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)

        if raster is None:
            exit(1)
        raster = raster[0]
        pred = np.full(raster.shape, fill_value=-1)
        self.clusterer[dept] = {}
        values = []

        #raster = remove_0_risk_pixel(dir_target, dir_target_bin, raster, dept, 'risk', 0)
        valid_mask = (raster != -1) & (~np.isnan(raster))

        for vb in vec_base:
            if vb == 'clustering':
                continue
            data, GT = self._process_base_data(vb, dept, dir_target, dir_target_bin, dir_data, valid_mask, raster, train_date, path)
            if data is None:
                continue
            if vb == 'geometry':
                values.append(data[:, :, 0])
                values.append(data[:, :, 1])
            else:
                reducor = Predictor(n_clusters=3, name='risk_reducor')
                reducor.fit(data[valid_mask].reshape(-1,1))
                data[valid_mask] = reducor.predict(data[valid_mask].reshape(-1,1))
                data[valid_mask] = order_class(reducor, data[valid_mask])
                data[~valid_mask] = 0
                valid_data = data != 0
                if 'pred_mask' not in locals():
                    pred_mask = valid_data
                else:
                    pred_mask = pred_mask & valid_data

                values[0][~pred_mask] = 0
                values[1][~pred_mask] = 0

                self._save_feature_image(path, dept, 'valid_data', valid_data, raster)

            if vb != 'geometry':
                self._save_feature_image(path, dept, vb, data, raster)

        values = np.asarray(values)
        self._save_feature_image(path, dept, 'latitude', values[0], raster)
        self._save_feature_image(path, dept, 'longitude', values[1], raster)
        pred_mask = values[0] > 0
        values = values[:, pred_mask]
        values = np.moveaxis(values, 0, 1)
        min_cluster_size = 1 + 3 * self.scale * (self.scale + 1)
        max_cluster_size = int(min_cluster_size * 2.5)

        model = HDBSCAN()
        pred[pred_mask] = model.fit_predict(values)
        self._save_feature_image(path, dept, 'pred_pp', pred, raster)

        # Merge an split
        pred[~valid_mask] = -1
        pred = merge_adjacent_clusters(pred, min_cluster_size, 0, -1)
        valid_cluster = find_clusters(pred, min_cluster_size, 0, -1)
        self._save_feature_image(path, dept, 'pred_plot', pred, raster)
        logger.info(f'{dept} : We found {len(valid_cluster)} to build geometry.')
        pred += 1
        pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, -1)
        pred -= 1
        pred = pred.astype(float)
        pred[~valid_mask] = np.nan
        self._save_feature_image(path, dept, 'pred', pred, raster)

        # Process post-watershed results
        self._post_process_result(pred, raster, mask, node_already_predicted)
        self._save_feature_image(path, dept, 'pred_final', pred, raster)

    def my_watershed(self, dept, data, valid_mask, raster, path, vb, image_type):
        reducor = Predictor(n_clusters=4, name='risk_reducor')
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
        return pred

    def create_geometry_with_watershed(self, dept, vec_base, path, sinister, dataset_name,
                                       sinister_encoding, resolution, mask, node_already_predicted, train_date):
        
        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
        dir_data = rootDisk / 'csv' / dept / 'raster' / resolution
        dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution
        dir_target_bin = root_target / sinister / dataset_name / sinister_encoding / 'bin' / resolution
        raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
        assert raster is not None
        raster = raster[0]
        pred = np.full(raster.shape, fill_value=np.nan)
        valid_mask = (raster != -1) & (~np.isnan(raster))

        self.max_target_value = None

        vb = vec_base[0]
        mode = vec_base[1]

        data, GT = self._process_base_data(vb, dept, dir_target, dir_target_bin, dir_data, valid_mask, raster, train_date, path)
        if data is None:
            logger.info(f'Can t find {vb}')
            exit(1)

        if self.max_target_value is None:
            self.max_target_value = np.nanmax(data)
        else:
            self.max_target_value = max(self.max_target_value, np.nanmax(data))
        oridata = np.copy(data)
        
        self._save_feature_image(path, dept, 'sum', data, raster, 0, self.max_target_value)

        pred = self.my_watershed(dept, data, valid_mask, raster, path, vb, 'pred')
        if GT is not None:
            self._save_feature_image(path, dept, 'gt_sum', GT, raster, 0, self.max_target_value)
            pred_GT = self.my_watershed(dept, GT, valid_mask, raster, path, vb, 'gt')

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

        if self.scale == -1:
            best_fr = 0
            best_scale = 0
            frs = []
            scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for scale in scales:
                fr_scale, _ = self.create_cluster(pred, dept, path, scale, mode, bin_data, raster, valid_mask, mask, node_already_predicted)
                frs.append(fr_scale)
                if fr_scale > best_fr:
                    best_fr = fr_scale
                    best_scale = scale

            check_and_create_path(path / 'scale_optimization')
            plt.figure(figsize=(15,5))
            plt.plot(scales, frs)
            plt.xlabel('Scales')
            plt.ylabel('Frequency ratio')
            plt.savefig(path / 'scale_optimization' / f'{dept}_scale_optimization.png')
            plt.close('all')
            _, pred = self.create_cluster(pred, dept, path, best_scale, mode, bin_data, raster, valid_mask, 'pred')
        else:
            _, pred = self.create_cluster(pred, dept, path, self.scale, mode, bin_data, raster, valid_mask, 'pred')

        if GT is not None:
            _, pred_GT = self.create_cluster(pred_GT, dept, path, self.scale, mode, bin_data, raster, valid_mask, 'gt')

            iou = calculate_iou(pred_GT, pred)

            logger.info(f'{dept} -> IoU : {iou}')
            if MLFLOW:
                existing_run = get_existing_run(f'segmentation_{dept}_{self.susceptility_mapper_name}')
                if existing_run:
                    mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
                else:
                    mlflow.start_run(run_name=f'segmentation_{dept}_{self.susceptility_mapper_name}', nested=True)
                mlflow.log_metric('IoU', iou)
                mlflow.end_run()

        # Process post-watershed results
        pred = self._post_process_result(pred, raster, mask, node_already_predicted, 'graph') 
        self._save_feature_image(path, dept, 'pred_final', pred, raster)
        
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

        logger.info(f'Cluster dispersion {self.dispersions}')

    def create_cluster(self, pred, dept, path, scale, mode, bin_data, raster, valid_mask, type):
        
        min_cluster_size = 1 + 3 * scale * (scale + 1)
        max_cluster_size = (int)(min_cluster_size * 2.5)
        
        if mode == 'size':
            pred = merge_adjacent_clusters(pred, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, oridata=None, mode=mode, exclude_label=0, background=-1)
            valid_cluster = find_clusters(pred, min_cluster_size, 0, -1)
            self._save_feature_image(path, dept, 'pred_merge', pred, raster)

            logger.info(np.unique(pred))
            logger.info(f'{dept} : We found {len(valid_cluster)} to build geometry.')
            mask_valid = np.isin(pred, valid_cluster)
            logger.info(f'Number of fire inside regions {np.nansum(bin_data[mask_valid])}')
            logger.info(f'Number of fire outside regions {np.nansum(bin_data[~mask_valid])}')

            valid_cluster = [val + 1 for val in valid_cluster]
            pred[valid_mask] += 1
            pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, valid_cluster)
            valid_cluster = [val - 1 for val in valid_cluster]
            pred[valid_mask] -= 1
            
            pred = pred.astype(float)
            self._save_feature_image(path, dept, f'{type}_split', pred, raster)

        elif mode == 'time_series_similarity':
            pred = merge_adjacent_clusters(pred, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, mode=mode, oridata=bin_data, exclude_label=0, background=-1)
            valid_cluster = find_clusters(pred, min_cluster_size, 0, -1)
            self._save_feature_image(path, dept, f'{type}_merge', pred, raster)

            logger.info(np.unique(pred))
            logger.info(f'{dept} : We found {len(valid_cluster)} to build geometry.')
            mask_valid = np.isin(pred, valid_cluster)
            logger.info(f'Number of fire inside regions {np.nansum(bin_data[mask_valid])}')
            logger.info(f'Number of fire outside regions {np.nansum(bin_data[~mask_valid])}')

            valid_cluster = [val + 1 for val in valid_cluster]
            pred[valid_mask] += 1
            pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, valid_cluster)
            valid_cluster = [val - 1 for val in valid_cluster]
            pred[valid_mask] -= 1

        elif mode == 'time_series_similarity_fast':
            pred[~valid_mask] = -1
            pred = merge_adjacent_clusters(pred, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, mode=mode, oridata=bin_data, exclude_label=0, background=-1)
            valid_cluster = find_clusters(pred, math.inf, 0, -1)
            self._save_feature_image(path, dept, f'{type}_merge', pred, raster)

            logger.info(np.unique(pred))
            logger.info(f'{dept} : We found {len(valid_cluster)} to build geometry.')
            mask_valid = np.isin(pred, valid_cluster)
            logger.info(f'Number of fire inside regions {np.nansum(bin_data[mask_valid])}')
            logger.info(f'Number of fire outside regions {np.nansum(bin_data[~mask_valid])}')

            valid_cluster = [val + 1 for val in valid_cluster]
            pred[valid_mask] += 1
            pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, valid_cluster)
            valid_cluster = [val - 1 for val in valid_cluster]
            pred[valid_mask] -= 1

        sum_fr = 0
        bin_data_sum = np.nansum(bin_data, axis=2)
        for cluster in valid_cluster:
            fr_cluster = frequency_ratio(bin_data_sum[~np.isnan(bin_data_sum)], np.argwhere((pred[~np.isnan(bin_data_sum)] == cluster)))
            sum_fr += fr_cluster
            logger.info(f'{cluster} frequency ratio -> {fr_cluster}')
        
        if len(valid_cluster) == 0:
            sum_fr = 0
        else:
            sum_fr = sum_fr / len(valid_cluster)
        logger.info(f'Mean fr {sum_fr}')

        return sum_fr, pred

    def create_geometry_with_regular(self, dept, vec_base, path, sinister, dataset_name,
                                       sinister_encoding, resolution, mask, node_already_predicted):
        
        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
        dir_data = rootDisk / 'csv' / dept / 'raster' / resolution
        dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution
        dir_target_bin = root_target / sinister / dataset_name / sinister_encoding / 'bin' / resolution
        raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
        assert raster is not None
        raster = raster[0]
        pred = np.full(raster.shape, fill_value=1)
        valid_mask = (raster != -1) & (~np.isnan(raster))
        pred[~valid_mask] = -1
        
        self.max_target_value = None

        min_cluster_size = 1 + 3 * self.scale * (self.scale + 1)
        max_cluster_size = (int)(min_cluster_size * 2.5)
        
        pred = split_large_clusters(pred, max_cluster_size, min_cluster_size, [-1])
        pred -= 1
        pred = pred.astype(float)
        pred[~valid_mask] = np.nan
        self._save_feature_image(path, dept, 'pred', pred, raster)

        # Process post-watershed results
        self._post_process_result(pred, raster, mask, node_already_predicted, 'graph')
        self._save_feature_image(path, dept, 'pred_final', pred, raster)

    def _process_base_data(self, vb, dept, dir_target, dir_target_bin, dir_data, valid_mask, raster, train_date, path):
        
        GT = None

        if vb == 'risk':
            data = read_object(f'{dept}Influence.pkl', dir_target)
            if data is None or dept not in self.train_departements:
                if data is not None:
                    GT = np.copy(data)
                    GT = GT[:, :, :allDates.index(train_date)]
                    GT = np.nansum(GT, axis=2)
                data = read_object(f'{dept}Influence.pkl', path / 'predict_map')
                if data is None:
                    self.predict_susecptibility_map([dept], self.susecptibility_variables, dir_data, path / 'predict_map')
                    data = read_object(f'{dept}Influence.pkl', path / 'predict_map')
            else:
                data = data[:, :, :allDates.index(train_date)]
                data = np.nansum(data, axis=2)
        elif vb == 'nbsinister':
            data = read_object(f'{dept}binScale0.pkl', dir_target_bin)
            if data is None or dept not in self.train_departements:
                data = read_object(f'{dept}Influence.pkl', path / 'predict_map')
                if data is None:
                    self.predict_susecptibility_map([dept], self.susecptibility_variables, dir_data, path / 'predict_map')
                    data = read_object(f'{dept}binScale0.pkl', path / 'predict_map')
            else:
                data = data[:, :, :allDates.index(train_date)]
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
    
    def _raster2(self, path, n_pixel_y, n_pixel_x, resStr, doBin, base, sinister, dataset_name, sinister_encoding, train_date):

        dir_bin = root_target / sinister / dataset_name / sinister_encoding / 'bin' / resStr
        dir_time = root_target / sinister / dataset_name / sinister_encoding / 'time_intervention' / resStr

        dir_bin_bdiff = root_target / sinister / 'bdiff' / sinister_encoding / 'bin' / resStr
        #dir_time_bdiff = root_target / sinister / dataset_name / sinister_encoding / 'time_intervention' / resStr

        dir_target = root_target / sinister / dataset_name /  sinister_encoding / 'log' / resStr
        dir_target_bdiff = root_target / sinister / 'bdiff' / sinister_encoding / 'log' / resStr

        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resStr

        if len(self.drop_department) == self.departements.unique().shape[0]:
            logger.info('All department were droped, try with a smaller minimum scale. If it still does work, try with department == scale')
            exit(1)

        for dept in np.unique(self.departements):

            if dept in self.drop_department:
                continue

            #if (path / 'raster' / f'{dept}rasterScale{self.scale}_{base}_{self.graph_method}.pkl').is_file():
            #    if self.graph_method == 'node':
            #        self.ids = np.copy(self.graph_ids)
            #    continue

            maskDept = np.argwhere(self.departements == dept)
            geo = gpd.GeoDataFrame(index=np.arange(maskDept.shape[0]), geometry=self.oriGeometry[maskDept[:,0]])
            """geo[f'scale{self.scale}'] = self.graph_ids[maskDept]
            mask, _, _ = rasterization(geo, n_pixel_y, n_pixel_x, f'scale{self.scale}', Path('log'), 'ori')
            mask = mask[0]"""
            outputName = f'{dept}rasterScale0.pkl'
            raster = read_object(outputName, dir_raster)
            assert raster is not None
            raster = raster[0]

            mask = np.full(raster.shape, fill_value = np.nan)
            for uid in np.unique(raster[~np.isnan(raster)]):
                mask[raster == uid] = self.graph_ids[self.oriIds == uid]

            logger.info(f'{dept, mask.shape, raster.shape}')

            mask[np.isnan(raster)] = np.nan

            if self.graph_method == 'graph':
                check_and_create_path(path / 'raster' / 'graph_method')
                raster_node = np.full(raster.shape, fill_value=np.nan)
                self.graph_node_dict = {}
                uids = np.unique(mask)
                uids = uids[~np.isnan(uids)]
                graph_node_cluster = HDBSCAN(min_cluster_size=5)
                #graph_node_cluster = KMeans(n_clusters=3, random_state=42)
                num_cluster = 0
                for id in uids:
                    mask_id = mask == id
                    coords = np.column_stack(np.nonzero(mask_id))
                    result = graph_node_cluster.fit_predict(coords)
                    raster_node[(mask_id) & (np.isnan(raster_node))] = result
                    
                    raster_node = np.where((raster_node == -1), np.nan, raster_node)

                    # Exemple de tableau raster avec des valeurs -1 remplacées par NaN
                    """raster_node_with_nan = np.where((raster_node == -1) & (mask_id), np.nan, raster_node)

                    # Calculer une carte de distance pour chaque NaN vers le point le plus proche non-NaN
                    # Cette méthode remplit les NaN par les valeurs les plus proches

                    nan_mask = np.isnan(raster_node_with_nan)  # Masque des NaN
                    filled_raster = raster_node_with_nan.copy()  # Copie du tableau original

                    nearest_indices = ndimage.distance_transform_edt(
                        nan_mask,
                        return_distances=False,
                        return_indices=True
                    )

                    # Utiliser les indices pour remplir les NaN avec les valeurs les plus proches
                    filled_raster = raster_node_with_nan.copy()
                    filled_raster[nan_mask] = raster_node_with_nan[tuple(nearest_indices[:, nan_mask])]

                    # Remettre à jour raster_node
                    raster_node[mask_id] = filled_raster[mask_id]
                    """
                    
                    raster_node[mask_id] = raster_node[mask_id] + num_cluster
                    num_cluster += np.unique(raster_node[mask_id & ~np.isnan(raster_node)]).shape[0]

                    plt.figure(figsize=(15,5))
                    plt.imshow(raster_node)
                    plt.savefig(path / 'raster' / 'graph_method' / f'{dept}_{self.scale}_{self.base}_{self.graph_method}_{id}.png')
                    plt.close()

                if 'node_already_predicted' not in locals():
                    node_already_predicted = 0
                raster_node = self._post_process_result(raster_node, raster, self.departements == dept, node_already_predicted, 'node')
                current_nodes = np.unique(self.ids[self.departements == dept][~np.isnan(self.ids[self.departements == dept])]).shape[0]
                logger.info(f'{dept} Unique node in graph : {np.unique(self.ids[self.departements == dept])}, {current_nodes}. {node_already_predicted}')
                node_already_predicted += current_nodes

                uids = np.unique(self.ids[self.departements == dept])
                for id in uids:
                    graph_values = self.graph_ids[self.ids == id]
                    if np.unique(graph_values).shape[0] > 1:
                        major_id = np.bincount(graph_values).argmax()
                        self.graph_ids[self.ids == id] = major_id

                """geo[f'scale{self.scale}'] = self.graph_ids[maskDept]
                mask, _, _ = rasterization(geo, n_pixel_y, n_pixel_x, f'scale{self.scale}', Path('log'), 'ori')
                mask = mask[0]
                mask[np.isnan(raster)] = np.nan"""
                mask = np.full(raster.shape, fill_value = np.nan)
                for uid in np.unique(raster[~np.isnan(raster)]):
                    mask[raster == uid] = self.graph_ids[self.oriIds == uid]
                save_object(raster_node, f'{dept}rasterScale{self.scale}_{base}_{self.graph_method}_node.pkl', path / 'raster')
                save_object(mask, f'{dept}rasterScale{self.scale}_{base}_{self.graph_method}.pkl', path / 'raster')
            else:
                self.ids = np.copy(self.graph_ids)
                raster_node = np.copy(mask)
                save_object(mask, f'{dept}rasterScale{self.scale}_{base}_{self.graph_method}_node.pkl', path / 'raster')
                save_object(mask, f'{dept}rasterScale{self.scale}_{base}_{self.graph_method}.pkl', path / 'raster')
           
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[~np.isnan(unique_ids)]
            plt.figure(figsize=(15, 5))
            plt.imshow(mask, label='ID')

            # Annotate each unique ID on the image
            for unique_id in unique_ids:
                # Find the positions of the current ID in the mask
                positions = np.column_stack(np.where(mask == unique_id))
                
                # Calculate the center of these positions
                center_y, center_x = np.mean(positions, axis=0).astype(int)
                
                # Place the text at the center
                plt.text(center_x, center_y, f'{unique_id}', color='white', fontsize=12, ha='center', va='center')

            plt.title(dept)
            plt.savefig(path / 'raster' / f'{dept}_{self.scale}_{self.base}_{self.graph_method}.png')
            plt.close('all')

            if self.graph_method == 'graph':
                plt.figure(figsize=(15, 5))
                plt.imshow(raster_node, label='node')

                plt.title(dept)
                plt.savefig(path / 'raster' / f'{dept}_{self.scale}_{self.base}_{self.graph_method}_node.png')
                plt.close('all')

            if doBin:
                outputName = f'{dept}binScale0.pkl'
                if dept not in ['departement-01-ain', 'departement-25-doubs', 'departement-78-yvelines', 'departement-69-rhone'] and dataset_name == 'firemen':
                    logger.info(f'Load {dept} from bdiff')
                    bin = read_object(outputName, dir_bin_bdiff)
                    outputName = f'{dept}Influence.pkl'
                    influence = read_object(outputName, dir_target_bdiff)
                    time = np.zeros(influence.shape)
                else:
                    bin = read_object(outputName, dir_bin)
                    outputName = f'{dept}Influence.pkl'
                    influence = read_object(outputName, dir_target)

                    outputName = f'{dept}timeScale0.pkl'
                    time = read_object(outputName, dir_target)

                if time is None:
                    time = np.zeros(influence.shape)

                binImageScale, influenceImageScale, timeScale = create_larger_scale_bin(mask, bin, influence, time, raster)
                save_object(binImageScale, f'{dept}binScale{self.scale}_{base}_{self.graph_method}.pkl', path / 'bin')
                save_object(influenceImageScale, f'{dept}InfluenceScale{self.scale}_{base}_{self.graph_method}.pkl', path / 'influence')
                save_object(timeScale, f'{dept}timeScale{self.scale}_{base}_{self.graph_method}.pkl', path / 'time_intervention')

                binImageScale, influenceImageScale, timeScale, = create_larger_scale_bin(raster_node, bin, influence, time, raster)
                save_object(binImageScale, f'{dept}binScale{self.scale}_{base}_{self.graph_method}_node.pkl', path / 'bin')
                save_object(influenceImageScale, f'{dept}InfluenceScale{self.scale}_{base}_{self.graph_method}_node.pkl', path / 'influence')
                save_object(timeScale, f'{dept}timeScale{self.scale}_{base}_{self.graph_method}_node.pkl', path / 'time_intervention')

            self.numCluster = np.shape(np.unique(self.ids))[0]
            
    def _raster(self, path : Path,
                sinister : str, 
                dataset_name: str,
                sinister_encoding,
                resolution : str,
                base: str,
                train_date) -> None:
        """
        Create new raster mask
        """
        check_and_create_path(path)
        check_and_create_path(path / 'raster')
        check_and_create_path(path / 'bin')
        check_and_create_path(path / 'proba')

        self._raster2(path, resolutions[resolution]['y'], resolutions[resolution]['x'], resolution, True,
                      base, sinister, train_date=train_date, dataset_name=dataset_name, sinister_encoding=sinister_encoding)

    def _save_feature_image(self, path, dept, vb, image, raster, mini=None, maxi=None):
        data = np.copy(image)
        check_and_create_path(path / 'features_geometry' / f'{self.scale}_{self.base}_{self.graph_method}' / dept)
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
        plt.savefig(path / 'features_geometry' / f'{self.scale}_{self.base}_{self.graph_method}' / dept / f'{vb}.png')
        plt.close('all')

    def _post_process_result(self, pred, raster, mask, node_already_predicted, graph_or_node='node'):

        """if self.scale != 'departement':
            valid_mask = (pred != -1) & (~np.isnan(pred))

            # Merge small clusters if needed
            min_cluster_size = 1 + 3 * self.scale * (self.scale + 1)
            pred2 = np.copy(pred)
            pred2[np.isnan(pred)] = -1
            pred[valid_mask] = merge_adjacent_clusters(pred2.astype(int), min_cluster_size, None, -1)[valid_mask]"""

        array = np.full(self.ids.shape[0], fill_value=np.nan)
        pred = relabel_clusters(pred, node_already_predicted)
        X = np.unique(raster)
        X = X[~np.isnan(X)]
        for node in X:
            mask_node = self.oriIds == node
            mask2 = raster == node
            array[mask_node] = pred[mask2]

       # Incorporate nan hexagone
        if True in np.isnan(array[mask]):

            valid_mask = ~np.isnan(array[mask])
            invalid_mask = np.isnan(array[mask])

            if True in valid_mask and True in invalid_mask:
                #valid_coords = np.column_stack(np.where(valid_mask))
                valid_coords = np.asarray(list(zip(self.oriLongitude[mask][valid_mask], self.oriLatitudes[mask][valid_mask])))
                valid_values = array[mask][valid_mask]

                # Coordonnées des pixels à remplacer
                invalid_coords = np.asarray(list(zip(self.oriLongitude[mask][invalid_mask], self.oriLatitudes[mask][invalid_mask])))
                #invalid_coords = np.column_stack(np.where(invalid_mask))

                # Utilisation de KNeighbors pour prédire les valeurs manquantes
                knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
                knn.fit(valid_coords, valid_values)

                predicted_values = knn.predict(invalid_coords)

                predicted_values = np.round(predicted_values)

                array[np.where(mask)[0][invalid_mask]] = predicted_values.astype(int)

        if graph_or_node == 'node': # graph = node
            self.ids[mask] = array[mask]
        elif graph_or_node == 'graph': # nodes per graph
            self.graph_ids[mask] = array[mask]
        else:
            logger.info(f'Unknow graph_or_node value {graph_or_node}')
            exit(1)
            
        #self.ids[mask] = relabel_clusters(self.ids[mask], node_already_predicted)
        return pred

    def _predict_node_graph_with_position(self, array : list, depts = None) -> np.array:
        assert self.clusterer is not None
        if depts is None:
            array2 = np.empty((np.asarray(array).shape[0], 5))
            array2[:, longitude_index:latitude_index+1] = np.asarray(array).reshape(-1,2)
            depts = self._assign_department(array2)[:, -1]
            
        array = np.asarray(array)
        udept = np.unique(depts)
        res = np.full((len(array), 2), fill_value=np.nan)
        for dept in udept:
            if dept is None:
                continue
            mask = np.argwhere(depts == dept)[:, 0]
            x = array[mask]
            if self.base == 'geometry':
                res[mask] = self.clusterer[int2name[dept]]['algo'].predict(x) + self.clusterer[int2name[dept]]['node_already_predicted']
            else:
                mask_ori = self.departements == int2name[dept]

                existing_data = np.asarray(list(zip(self.oriLongitude[mask_ori], self.oriLatitudes[mask_ori]))).reshape(-1,2)
                existing_ids = self.ids[mask_ori]
                existing_graph_ids = self.graph_ids[mask_ori]

                nbrs = NearestNeighbors(n_neighbors=1).fit(existing_data)
                ux = np.unique(x, axis=0)
                neighbours = nbrs.kneighbors(ux, return_distance=True)
                neighbours = neighbours[1]
                for i, xx in enumerate(ux):
                    neighbours_x = neighbours[i]
                    mask_x = np.argwhere((array[:, 0] == xx[0]) & (array[:, 1] == xx[1]))[:, 0]
                    m1 = self.oriLongitude[mask_ori] == existing_data[int(neighbours_x)][0]
                    m2 = self.oriLatitudes[mask_ori] == existing_data[int(neighbours_x)][1]
                    m3 = m1 & m2
                    res[mask_x, 0] = np.unique(existing_graph_ids[m3])[0]
                    res[mask_x, 1] = np.unique(existing_ids[m3])[0]
                    
        return res[:, 0], res[:, 1]
    
    def susecptibility_map_individual_pixel_feature(self, dept, year, variables, target_value, sdate_year, edate_year, raster, dir_data, dir_output):

        if target_value is not None:
            target_value = target_value.flatten()
            
            risk_arg = np.argwhere((target_value > 0) & (~np.isnan(target_value)))[:, 0]
            mask_X = list(risk_arg)
            non_risk_arg = np.argwhere(target_value == 0)[:, 0]
            mask_X += list(np.random.choice(non_risk_arg, min(non_risk_arg.shape[0], risk_arg.shape[0]), replace=False))
            mask_X = np.asarray(mask_X).reshape(-1,1)

            y = list(target_value[mask_X[:, 0]])
        vec_x = []
        
        assert raster is not None
        for var in variables:
            if var in cems_variables:
                values = read_object(f'{var}raw.pkl', dir_data)
                assert values is not None
                values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                values = np.mean(values, axis=2)
                self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var}', values, raster)
                values = values.flatten()
                vec_x.append(values[mask_X[:, 0]])

            elif var == 'population' or var == 'elevation':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                values = values.reshape((values.shape[0], values.shape[1]))
                self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var}', values, raster)
                values = values.flatten()
                vec_x.append(values[mask_X[:, 0]])

            elif var == 'foret':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(foret_variables):
                    values2 = values[i]
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    values2 = values2.flatten()
                    vec_x.append(values2[mask_X[:, 0]])

            elif var == 'air':
                assert values is not None
                for i, var2 in enumerate(air_variables):
                    values = read_object(f'{var2}raw.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    values = np.mean(values, axis=2)
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values, raster)
                    values = values.flatten()
                    vec_x.append(values[mask_X[:, 0]])

            elif var == 'sentinel':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(sentinel_variables):
                    values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    values2 = values2.flatten()
                    vec_x.append(values2[mask_X[:, 0]])

            elif var == 'vigicrues':
                assert values is not None
                for i, var2 in enumerate(vigicrues_variables):
                    values = read_object(f'vigicrues{var2}.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'vigicrues{var2}', values, raster)
                    values = np.mean(values, axis=2)
                    values = values.flatten()
                    vec_x.append(values[mask_X[:, 0]])

            elif var == 'nappes':
                for i, var2 in enumerate(nappes_variables):
                    values = read_object(f'{var2}.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values, raster)
                    values = np.mean(values, axis=2)
                    values = values.flatten()
                    vec_x.append(values[mask_X[:, 0]])

            elif var == 'osmnx':
                values = read_object(f'osmnx.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(osmnx_variables):
                    values2 = values[i]
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    values2 = values2.flatten()
                    vec_x.append(values2[mask_X[:, 0]])

            elif var == 'dynamic_world':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(dynamic_world_variables):
                    values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    values2 = values2.flatten()
                    vec_x.append(values2[mask_X[:, 0]])
            else:
                raise ValueError(f'Unknow variable {var}')
            
        vec_x = np.asarray(vec_x)
        if 'X' not in locals():
            X = vec_x
        else:
            #print(vec_x.shape)
            #print(X.shape)
            X = np.concatenate((X, vec_x), axis=0)

        if target_value is not None:
            return X, y
        return X
    
    def susecptibility_map_all_image(self, dept, year, variables, target_value, sdate_year, edate_year, raster, dir_data, dir_output):
        
        height = 64
        if target_value is not None:
            y = np.copy(target_value)
            y[np.isnan(y)] = 0
            y = resize_no_dim(y, height, height)

        vec_x = []
        
        assert raster is not None
        for var in variables:
            if var in cems_variables:
                values = read_object(f'{var}raw.pkl', dir_data)
                assert values is not None
                values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                values = np.mean(values, axis=2)
                self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var}', values, raster)
                vec_x.append(values)

            elif var == 'population' or var == 'elevation':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                values = values.reshape((values.shape[0], values.shape[1]))
                self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var}', values, raster)
                vec_x.append(values)

            elif var == 'foret':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(foret_variables):
                    values2 = values[i]
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{foretint2str[var2]}', values2, raster)
                    vec_x.append(values2)

            elif var == 'cosia':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(cosia_variables):
                    values2 = values[i]
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    vec_x.append(values2)

            elif var == 'air':
                assert values is not None
                for i, var2 in enumerate(air_variables):
                    values = read_object(f'{var2}raw.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    values = np.mean(values, axis=2)
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values, raster)
                    vec_x.append(values)

            elif var == 'sentinel':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(sentinel_variables):
                    values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    vec_x.append(values2)

            elif var == 'vigicrues':
                assert values is not None
                for i, var2 in enumerate(vigicrues_variables):
                    values = read_object(f'vigicrues{var2}.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    values = np.mean(values, axis=2)
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'vigicrues{var2}', values, raster)
                    vec_x.append(values)

            elif var == 'nappes':
                for i, var2 in enumerate(nappes_variables):
                    values = read_object(f'{var2}.pkl', dir_data)
                    assert values is not None
                    values = values[:, :, allDates.index(sdate_year):allDates.index(edate_year)]
                    values = np.mean(values, axis=2)
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values, raster)
                    vec_x.append(values)

            elif var == 'osmnx':
                values = read_object(f'osmnx.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(osmnx_variables):
                    values2 = values[i]
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{osmnxint2str[var2]}', values2, raster)
                    vec_x.append(values2)

            elif var == 'dynamic_world':
                values = read_object(f'{var}.pkl', dir_data)
                assert values is not None
                for i, var2 in enumerate(dynamic_world_variables):
                    values2 = np.mean(values[i, :, :, allDates.index(sdate_year):allDates.index(edate_year)], axis=2)
                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{var2}', values2, raster)
                    vec_x.append(values2)
            else:
                raise ValueError(f'Unknow variable {var}')
            
        vec_x = np.asarray(vec_x)
        vec_x[np.isnan(vec_x)] = 0
        vec_x_2 = np.zeros((vec_x.shape[0], height, height))
        for band in range(vec_x.shape[0]):
            vec_x_2[band] = resize_no_dim(vec_x[band], height, height)
        if 'X' not in locals():
            X = vec_x_2
        else:
            X = np.concatenate((X, vec_x_2), axis=0)

        if target_value is not None:
            return X, y
        else:
            return X
    
    def train_susecptibility_map(self, model_config, departements, variables, target, train_date, val_date, root_data, root_target, dir_output):
        self.susceptility_mapper = None
        self.susecptibility_variables = variables
        self.susceptility_target = target
        self.max_target_value = None
        self.mapper_config = model_config

        dir_target = root_target / 'log' / self.resolution
        dir_target_bin = root_target / 'bin' / self.resolution
        dir_raster = root_target / 'raster' / self.resolution

        group_month = [
            [2, 3, 4, 5],    # Medium season
            [6, 7, 8, 9],    # High season
            [10, 11, 12, 1]  # Low season
        ]

        years = np.arange(2018, 2024)

        logger.info('Creating Susceptibilty mapper')

        y_train = []
        y_test = []
        y_val = []
        dept_test = []
        years_test = []

        if not (dir_output / 'susecptibility_map_features' / model_config['type'] / 'y_train.pkl').is_file():

            for dept in departements:
                if dept in self.drop_department:
                    continue
                logger.info(f'{dept}')

                dir_data = root_data / dept / 'raster' / self.resolution
                raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
                assert raster is not None
                raster = raster[0]

                for year in years:

                    if target == 'risk':
                        target_value = read_object(f'{dept}Influence.pkl', dir_target)
                    elif target == 'nbsinister':
                        target_value = read_object(f'{dept}binScale0.pkl', dir_target_bin)
                
                    assert target_value is not None

                    sdate_year = f'{year}-06-01'
                    edate_year = f'{year}-10-01'
                    
                    if allDates[0] > edate_year:
                            continue
                    if allDates[-1] < sdate_year:
                            continue
                    
                    if sdate_year < allDates[0]:
                        sdate_year = allDates[0]

                    if edate_year > allDates[-1]:
                        edate_year = allDates[-1]

                    if dept == 'departement-69-rhone' and year > 2022:
                        continue

                    if edate_year < train_date:
                        set_type = 'train'
                    elif edate_year < val_date:
                        set_type = 'val'
                    else:
                        set_type = 'test'

                    if dept not in self.train_departements:
                        set_type = 'test'
                        
                    target_value = target_value[:, :, allDates.index(sdate_year):allDates.index(edate_year)]

                    check_and_create_path(dir_output / 'susecptibility_map_features' / str(year))

                    # Calculer la somme de target_value le long de la troisième dimension (axis=2)
                    sum_values = np.sum(target_value, axis=2)

                    if self.max_target_value is None or np.nanmax(sum_values) > self.max_target_value:
                        self.max_target_value = np.nanmax(sum_values)

                    self._save_feature_image(dir_output / 'susecptibility_map_features' / str(year), dept, f'{target}', sum_values, raster)

                    if model_config['type'] in sklearn_model_list:
                        X_, y_ = self.susecptibility_map_individual_pixel_feature(dept, year, variables, sum_values, sdate_year, edate_year, raster, dir_data, dir_output)
                        if set_type == 'train':
                            y_train += list(np.asarray([y_]))
                            if 'X_train' not in locals():
                                X_train = X_
                            else:
                                X_train = np.concatenate((X_train, X_), axis=0)
                        elif set_type == 'val':
                            y_val += list(np.asarray([y_]))
                            if 'X_val' not in locals():
                                X_val = X_
                            else:
                                X_val = np.concatenate((X_val, X_), axis=0)
                        elif set_type == 'test':
                            y_test += list(np.asarray([y_]))
                            if 'X_test' not in locals():
                                X_test = X_
                            else:
                                X_test = np.concatenate((X_test, X_), axis=0)
                    else:
                        X_, y_ = self.susecptibility_map_all_image(dept, year, variables, sum_values, sdate_year, edate_year, raster, dir_data, dir_output)
                        X_ = X_[np.newaxis, :, :, :]
                        if set_type == 'train':
                            y_train += list(np.asarray([y_]))
                            if 'X_train' not in locals():
                                X_train = X_
                            else:
                                X_train = np.concatenate((X_train, X_), axis=0)
                        elif set_type == 'val':
                            y_val += list(np.asarray([y_]))
                            if 'X_val' not in locals():
                                X_val = X_
                            else:
                                X_val = np.concatenate((X_val, X_), axis=0)
                        elif set_type == 'test':
                            dept_test.append(dept)
                            years_test.append(year)
                            y_test += list(np.asarray([y_]))
                            if 'X_test' not in locals():
                                X_test = X_
                            else:
                                X_test = np.concatenate((X_test, X_), axis=0)

            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
            y_val = np.asarray(y_val)
            X_train = np.asarray(X_train)
            X_val = np.asarray(X_val)
            X_test = np.asarray(X_test)

            save_object(y_train, 'y_train.pkl', dir_output / 'susecptibility_map_features' / model_config['type'])
            save_object(y_test, 'y_test.pkl', dir_output / 'susecptibility_map_features' / model_config['type'])
            save_object(y_val, 'y_val.pkl', dir_output / 'susecptibility_map_features' / model_config['type'])
            save_object(X_train, 'X_train.pkl', dir_output / 'susecptibility_map_features' / model_config['type'])
            save_object(X_val, 'X_val.pkl', dir_output / 'susecptibility_map_features' / model_config['type'])
            save_object(X_test, 'X_test.pkl', dir_output / 'susecptibility_map_features' / model_config['type'])
            save_object(dept_test, 'dept_test.pkl', dir_output / 'susecptibility_map_features' / model_config['type'])
            save_object(years_test, 'years_test.pkl', dir_output / 'susecptibility_map_features' / model_config['type'])

        else:
            base_path = dir_output / 'susecptibility_map_features' / model_config['type']
            y_train = read_object('y_train.pkl', base_path)
            y_test = read_object('y_test.pkl', base_path)
            y_val = read_object('y_val.pkl', base_path)
            X_train = read_object('X_train.pkl', base_path)
            X_val = read_object('X_val.pkl', base_path)
            X_test = read_object('X_test.pkl', base_path)
            dept_test = read_object('dept_test.pkl', base_path)
            years_test = read_object('years_test.pkl', base_path)

        logger.info('################# Susceptibility map dataset #########################')
        logger.info(f'positive : {y_train[y_train > 0].shape}, zero : {y_train[y_train == 0].shape}')

        if model_config['type'] in sklearn_model_list:
            self.susceptility_mapper = get_model(model_type=model_config['type'], name=model_config['name'],
                                device=model_config['device'], task_type=model_config['task'], params=model_config['params'], loss=model_config['loss'])
            
            self.susceptility_mapper.fit(X_train, y_train)

            score = self.susceptility_mapper.score(X_test, y_test)
            logger.info(f'Score obtained in susecptibility mapping {score}')
        else:
            # Initialisation des nouveaux tableaux avec des zéros
            T, H, W = y_train.shape
            new_y_train = np.zeros((T, H, W, 9), dtype=y_train.dtype)

            T, H, W = y_val.shape
            new_y_val = np.zeros((T, H, W, 9), dtype=y_val.dtype)

            T, H, W = y_test.shape
            new_y_test = np.zeros((T, H, W, 9), dtype=y_test.dtype)

            # Copie des données du tableau de base dans les bandes -1 et -2
            new_y_train[..., -1] = y_train
            new_y_train[..., -2] = y_train

            new_y_val[..., -1] = y_val
            new_y_val[..., -2] = y_val

            new_y_test[..., -1] = y_test
            new_y_test[..., -2] = y_test

            # Mise de la bande -4 à 1
            new_y_train[..., weight_index] = 1
            new_y_val[..., weight_index] = 1
            new_y_test[..., weight_index] = 1

            logger.info(f'{X_train.shape}, {X_val.shape}, {X_test.shape}')
            logger.info(f'{new_y_train.shape}, {new_y_val.shape}, {new_y_test.shape}')

            self.susceptibility_scaler = StandardScaler()
            self.susceptibility_scaler.fit(X_train.reshape(-1, X_train.shape[1]))
            X_train = self.susceptibility_scaler.transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
            X_val = self.susceptibility_scaler.transform(X_val.reshape(-1, X_train.shape[1])).reshape(X_val.shape)
            X_test = self.susceptibility_scaler.transform(X_test.reshape(-1, X_train.shape[1])).reshape(X_test.shape)

            logger.info(f'{np.max(X_train)} {np.max(X_val)} {np.max(X_test)}')

            data_augmentation_transform = RandomFlipRotateAndCrop(proba_flip=0.5, size_crop=32, max_angle=180)

            # Appliquez les transformations aux ensembles d'entraînement et de validation
            """train_dataset = AugmentedInplaceGraphDataset(
                X_train, y_train, [], transform=data_augmentation_transform, device=device)
            val_dataset = AugmentedInplaceGraphDataset(
                X_val, y_val, [], transform=data_augmentation_transform, device=device)
            test_dataset = AugmentedInplaceGraphDataset(
                X_test, y_test, [], transform=None, device=device)"""  # Pas de data augmentation sur le test
            
            train_dataset = InplaceGraphDataset(
                X_train, new_y_train, [], leni=len(X_train), device=device)
            val_dataset = InplaceGraphDataset(
                X_val, new_y_val, [], leni=len(X_val), device=device)
            test_dataset = InplaceGraphDataset(
                X_test, new_y_test, [], leni=len(X_test), device=device)

            train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=val_dataset.__len__(), shuffle=False)
            test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

            check_and_create_path(dir_output / 'susecptibility_map_features' / model_config['type'])

            features = np.arange(0, X_train.shape[1])

            save_graph_method = self.graph_method
            self.graph_method = 'node'
            params = {'train_loader' : train_loader,
                      'val_loader' : val_loader,
                      'test_loader' : test_loader,
                      'graph': self,
                      'optimize_feature' : False,
                      'PATIENCE_CNT': 100,
                      'CHECKPOINT' : CHECKPOINT,
                      'lr': 0.001,
                      'epochs' : 1000,
                      'loss_name': model_config['loss'],
                      'dir_output': dir_output / 'susecptibility_map_features' / model_config['type'],
                      'features_selected_str' : features,
                      'features_selected' : features,
                      'modelname' : model_config['type'],
                      'target_name' : target,
                      'autoRegression' : False,
                      'k_days' : 0,
                      'custom_model_params': model_config['params']
                      }
            
            self.susceptility_mapper_name = model_config['type']
            
            if model_config['type'] in models_hybrid:
                self.susceptility_mapper, _ = make_model(model_config['type'], len(features), len(features), self, dropout, 'relu', 0, target == 'binary', device, num_lstm_layers, model_config['params'])
            else:
                self.susceptility_mapper, _ = make_model(model_config['type'], len(features), len(features), self, dropout, 'relu', 0, target == 'binary', device, num_lstm_layers, model_config['params'])

            if (dir_output / 'susecptibility_map_features' / model_config['type'] / 'best.pt').is_file():
                self.susceptility_mapper.load_state_dict(torch.load(dir_output / 'susecptibility_map_features' / model_config['type'] / 'best.pt', map_location=device, weights_only=True), strict=False)

            else:
                train(params)
                self.susceptility_mapper.load_state_dict(torch.load(dir_output / 'susecptibility_map_features' / model_config['type'] / 'best.pt', map_location=device, weights_only=True), strict=False)

            self.graph_method = save_graph_method

            if MLFLOW:
                exp_name = f"{self.dataset_name}_train"
                experiments = client.search_experiments()

                if exp_name not in list(map(lambda x: x.name, experiments)):
                    tags = {
                        "mlflow.note.content": "This experiment is an example of how to use mlflow. The project allows to predict housing prices in california.",
                    }

                    client.create_experiment(name=exp_name, tags=tags)
                    
                mlflow.set_experiment(exp_name)
                existing_run = get_existing_run('Preprocessing')
                if existing_run:
                    mlflow.start_run(run_id=existing_run.info.run_id)
                else:
                    mlflow.start_run(run_name='Preprocessing')

            ####### test ########

            check_and_create_path(dir_output / 'susecptibility_map_features' / model_config['type'] / 'test')

            self.susceptility_mapper.eval()
            with torch.no_grad():
                mae_func = torch.nn.L1Loss(reduce='none')
                itest = 0
                mae = 0
                for data in test_loader:
                    try:
                        X, y, _, _ = data
                    except:
                        X, y, _ = data
                    
                    logger.info(f'{torch.max(X)}')
                    output = self.susceptility_mapper(X, None)
                
                    for b in range(y.shape[0]):
                        loss = mae_func(output[b, 0], y[b, :, :, -1])
                        yb = y[b].detach().cpu().numpy()
                        outputb = output[b].detach().cpu().numpy()

                        logger.info(f'loss {itest} {dept_test[itest]} {years_test[itest]}: {loss.item()} {np.max(outputb)}')

                        if MLFLOW:
                            existing_run = get_existing_run(f'susceptibility_{dept_test[itest]}_{years_test[itest]}_{model_config["type"]}')
                            if existing_run:
                                mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
                            else:
                                mlflow.start_run(run_name=f'susceptibility_{dept_test[itest]}_{years_test[itest]}_{model_config["type"]}', nested=True)
                            
                            mlflow.log_metric('MAE', loss.item())

                        mae += loss.item()

                        fig, ax = plt.subplots(1, 2, figsize=(15,5))
                        raster = read_object(f'{dept_test[itest]}rasterScale0.pkl', dir_raster)
                        assert raster is not None
                        raster = raster[0]
                        
                        output_image = resize_no_dim(outputb[0], raster.shape[0], raster.shape[1])
                        y_image = resize_no_dim(yb[:,  :, -1], raster.shape[0], raster.shape[1])

                        output_image[np.isnan(raster)] = np.nan
                        y_image[np.isnan(raster)] = np.nan

                        maxi = max(np.nanmax(output_image), np.nanmax(y_image))

                        ax[0].set_title('Prediction map')
                        ax[0].imshow(output_image, vmin=0, vmax=maxi)
                        ax[1].set_title('Ground truth')
                        ax[1].imshow(y_image, vmin=0, vmax=maxi)
                        plt.tight_layout()
                        plt.savefig(dir_output / 'susecptibility_map_features' /  model_config['type'] / 'test' / f'{dept_test[itest]}_{years_test[itest]}')
                        plt.close('all')
                        if MLFLOW:
                            mlflow.log_figure(fig, f'_{dept_test[itest]}_{years_test[itest]}.png')
                            mlflow.end_run()
                        itest += 1

                logger.info(f'MAE on test set : {mae / itest}')
            
    def predict_susecptibility_map(self, departements, variables, dir_data, dir_output):
        assert self.susceptility_mapper is not None
        assert self.sinister is not None

        check_and_create_path(dir_output)

        years = np.arange(2018, 2024)

        for dept in departements:
            
            dir_raster = root_target / self.sinister / self.dataset_name /  self.sinister_encoding / 'raster' / self.resolution
            raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
            assert raster is not None
            raster = raster[0]
            mask_X = np.argwhere(~np.isnan(raster))
            susceptibility_map = np.zeros(raster.shape)

            for year in years:

                test_season_start = f'{year}-06-01'
                test_season_end = f'{year}-10-01'

                index_start = test_season_start
                index_end = test_season_end
                
                assert raster is not None

                check_and_create_path(dir_output / 'susecptibility_map_features' / str(year))

                if self.mapper_config['type'] in sklearn_model_list:
                    X = self.susecptibility_map_individual_pixel_feature(dept, year, variables, None, index_start, index_end, raster, dir_data, dir_output)
                else:
                    X = self.susecptibility_map_all_image(dept, year, variables, None, index_start, index_end, raster, dir_data, dir_output)

                if self.mapper_config['type'] in sklearn_model_list:
                    vec_node = np.asarray(X).T
                    predictions = self.susceptility_mapper.predict(vec_node)
                    susceptibility_map_flat = susceptibility_map.flatten()
                    susceptibility_map_flat[mask_X[:, 0]] = predictions
                    susceptibility_map += susceptibility_map_flat.reshape(raster.shape)[0]
                else:
                    self.susceptility_mapper.eval()
                    with torch.no_grad():
                        X = X[np.newaxis, :, :, :]
                        X = self.susceptibility_scaler.transform(X.reshape(-1, X.shape[1])).reshape(X.shape)
                        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
                        predictions = self.susceptility_mapper(X_tensor, None)
                        if torch.is_tensor(predictions):
                            predictions = predictions.detach().cpu().numpy()
                            print(predictions.shape)
                        logger.info(f'{np.max(X)} {torch.max(X_tensor)} {np.max(predictions)}')
                        
                        predictions = resize_no_dim(predictions[0][0], raster.shape[0], raster.shape[1])
                        susceptibility_map += predictions

                if self.susceptility_target == 'risk':
                    self._save_feature_image(dir_output / str(year), dept, f'Influence_{year}', predictions, raster, 0, np.nanmax(predictions))
                else:
                    self._save_feature_image(dir_output / str(year), dept, f'binScale0_{year}', predictions, raster, 0, np.nanmax(predictions))

            if self.susceptility_target == 'risk':
                save_object(susceptibility_map, f'{dept}Influence.pkl', dir_output)
            else:
                save_object(susceptibility_map, f'{dept}binScale0.pkl', dir_output)

    def _clusterize_node(self, departements, variables, train_date, path, root_data):
        self.node_cluster = {}
        vec = []
        value_per_node = {}
        self.variables_for_cluster = variables
        
        for dept in departements:
            if dept in self.drop_department:
                continue

            dir_data = root_data / dept / 'raster' / self.resolution
            raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}.pkl', path / 'raster')
            assert raster is not None
            nodes = np.unique(raster[~np.isnan(raster)])
            for node in nodes:
                vec_node = []
                for var in variables:
                    if var in cems_variables:
                        values = read_object(f'{var}raw.pkl', dir_data)
                        assert values is not None
                        values = values[:, :, :allDates.index(train_date)]
                        vec_node.append(np.nanmean(values[raster == node], axis=1))

                    elif var == 'population' or var == 'elevation':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        values = values.reshape((values.shape[0], values.shape[1]))
                        vec_node.append(np.nanmean(values[raster == node]))

                    elif var == 'foret':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate(foret_variables):
                            vec_node.append(np.mean(values[i, raster == node]))

                    elif var == 'air':
                        assert values is not None
                        for i, var2 in enumerate(air_variables):
                            values = read_object(f'{var2}raw.pkl', dir_data)
                            assert values is not None
                            vec_node.append(np.nanmean(values[raster == node]))

                    elif var == 'sentinel':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate(sentinel_variables):
                            vec_node.append(np.nanmean(values[i][raster == node]))

                    elif var == 'vigicrues':
                        for i, var2 in enumerate(vigicrues_variables):
                            values = read_object(f'vigicrues{var2}.pkl', dir_data)
                            assert values is not None
                            vec_node.append(np.nanmean(values[raster == node]))

                    elif var == 'dynamic_world':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate(dynamic_world_variables):
                            vec_node.append(np.nanmean(values[i][raster == node]))

                    elif var == 'osmnx':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate(osmnx_variables):
                            vec_node.append(np.nanmean(values[i][raster == node]))

                value_per_node[node] = vec_node
                vec.append(np.asarray(vec_node))
        vec = np.asarray(vec)

        scaler = StandardScaler()
        vec = scaler.fit_transform(vec)

        if vec.shape[1] > 10:

            ### Reduction de dimension
            self.pca_cluster = None
            pca =  PCA(n_components=10, svd_solver='full')
            components = pca.fit_transform(vec)
            logger.info(f'99 % : {find_n_component(0.99, pca)}')
            logger.info(f'95 % : {find_n_component(0.95, pca)}')
            logger.info(f'90 % : {find_n_component(0.90, pca)}')
            logger.info(f'85 % : {find_n_component(0.85, pca)}')
            logger.info(f'75 % : {find_n_component(0.75, pca)}')
            logger.info(f'70 % : {find_n_component(0.70, pca)}')
            logger.info(f'60 % : {find_n_component(0.60, pca)}')
            logger.info(f'50 % : {find_n_component(0.50, pca)}')

            test_variances =  [0.99, 0.95, 0.90, 0.85, 0.80, 0.7, 0.6, 0.5]

            for test_variance in test_variances:
                    n_components = find_n_component(test_variance, pca)
                    if n_components > 0:
                        self.pca_cluster = PCA(n_components=n_components, svd_solver='full')
                        break
                
            if self.pca_cluster is None:
                raise ValueError(f'Can t find component that satisfied {test_variances}')
                exit(1)

            vec = self.pca_cluster.fit_transform(vec)
        else:
            self.pca_cluster = None

        ### Clustering
        #self.cluster_node_model = HDBSCAN(min_cluster_size=2, prediction_data=True)
        self.cluster_node_model = KMeans(n_clusters=10)
        
        all_clusters = self.cluster_node_model.fit_predict(vec)
        logger.info(f'{np.unique(all_clusters)}')

        for key, value in value_per_node.items():
            if self.pca_cluster is not None:
                pca_val = self.pca_cluster.transform(scaler.transform([value]))
            else:
                pca_val = [value]
            if isinstance(self.cluster_node_model, HDBSCAN):
                self.node_cluster[key] = approximate_predict(self.cluster_node_model,pca_val)[0]
            else:
                self.node_cluster[key] = self.cluster_node_model.predict(pca_val)

        check_and_create_path(path / 'time_series_clustering')
        for dept in departements:

            if dept in self.drop_department:
                continue

            plt.figure(figsize=(15,5))
            raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}.pkl', path / 'raster')
            assert raster is not None

            time_series_image = np.full(raster.shape, fill_value=np.nan)
            nodes = np.unique(raster[~np.isnan(raster)])
            for node in nodes:
                time_series_image[raster == node] = self.node_cluster[node]
                
            img = plt.imshow(time_series_image, vmin=np.min(all_clusters), vmax= np.max(all_clusters), cmap='jet')
            plt.savefig(path / 'time_series_clustering' / f'{dept}_{self.scale}_{self.base}.png')
            save_object(time_series_image, f'{dept}_{self.scale}_{self.base}.pkl', path / 'time_series_clustering')
            plt.close('all')
            save_object(time_series_image, f'{dept}_{self.scale}.pkl', path / 'time_series_clustering')

        logger.info(f'Time series clustering {self.node_cluster}')

    def predict_cluster_node(self, departements, path, root_data):
        """
        Charge les données des nœuds pour chaque département et prédit le cluster auquel ils appartiennent.

        :param departements: list, liste des départements à traiter
        :param path: Path, chemin du répertoire des données
        :param root_data: Path, chemin du répertoire de base pour les données spécifiques aux départements
        :return: dict, avec les départements comme clés et les prédictions de clusters pour chaque nœud comme valeurs
        """
        if self.pca_cluster is None or self.cluster_node_model is None:
            raise ValueError("Le modèle PCA ou HDBSCAN n'est pas encore initialisé. Veuillez d'abord exécuter _clusterize_node().")
        
        all_predictions = {}
        for dept in departements:
            dir_data = root_data / dept / 'raster' / 'resolution'
            vec_node = []
            raster = read_object(f'{dept}rasterScale.pkl', path / 'raster')
            assert raster is not None
            nodes = np.unique(raster[~np.isnan(raster)])
            
            for node in nodes:
                vec_node_per_node = []
                for var in self.variables_for_cluster:
                    if var in ['population', 'elevation']:
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        values = values.reshape((values.shape[0], values.shape[1]))
                        vec_node_per_node.append(np.nanmean(values[raster == node]))
                    elif var == 'sentinel':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([sentinel_variables]):
                            vec_node_per_node.append(np.nanmean(values[i, raster == node]))
                    elif var == 'vigicrues':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([vigicrues_variables]):
                            vec_node_per_node.append(np.nanmean(values[raster == node]))
                    elif var == 'dynamic_world':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([dynamic_world_variables]):
                            vec_node_per_node.append(np.nanmean(values[i, raster == node]))
                    elif var == 'foret':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([foret_variables]):
                            vec_node_per_node.append(np.nanmean(values[i, raster == node]))
                    elif var == 'air':
                        for i, var2 in enumerate([air_variables]):
                            values = read_object(f'{var2}raw.pkl', dir_data)
                            assert values is not None
                            vec_node_per_node.append(np.nanmean(values[raster == node]))
                    elif var == 'osmnx':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate(osmnx_variables):
                            vec_node.append(np.nanmean(values[i][raster == node]))

                vec_node.append(np.asarray(vec_node_per_node))
                
            new_data = np.asarray(vec_node)
            
            # Transformation des données du nœud avec le modèle PCA appris
            new_data_transformed = self.pca_cluster.transform(new_data)
            
            # Utilisation de HDBSCAN pour faire la prédiction sur les nouveaux nœuds
            cluster_predictions = [approximate_predict(self.cluster_node_model, data.reshape(1, -1)) for data in new_data_transformed]
            all_predictions[dept] = cluster_predictions
            
            # Mise à jour du dictionnaire self.node_cluster pour stocker les clusters pour chaque nœud
            for node, prediction in zip(nodes, cluster_predictions):
                self.node_cluster[node] = prediction

    def _clusterize_node_with_target(self, departements, variables, target, train_date, path, root_data, root_target,):
        self.variables_for_cluster = variables

        dir_target = root_target / 'log' / self.resolution
        dir_target_bin = root_target / 'bin' / self.resolution
        dir_raster = path / 'raster'
        vec_target = []
        vec = []
        
        self.values_per_node = {}
        self.node_cluster = {}
        target_per_node = {}
        for dept in departements:
            if dept in self.drop_department:
                continue

            if target == 'risk':
                target_value = read_object(f'{dept}Influence.pkl', dir_target)
            elif target == 'nbsinister':
                target_value = read_object(f'{dept}binScale0.pkl', dir_target_bin)

            assert target_value is not None
            target_value = target_value[:, :, :allDates.index(train_date)]

            dir_data = root_data / dept / 'raster' / self.resolution
            raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}.pkl', dir_raster)
            assert raster is not None
            nodes = np.sort(np.unique(raster[~np.isnan(raster)]))

            for node in nodes:
                vec_node = []
                for var in variables:
                    if var in cems_variables:
                        values = read_object(f'{var}raw.pkl', dir_data)
                        assert values is not None
                        values = values[:, :, :allDates.index(train_date)]
                        vec_node.append(np.nanmean(values[raster == node]))
                    elif var in ['population', 'elevation']:
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        values = values.reshape((values.shape[0], values.shape[1]))
                        vec_node.append(np.nanmean(values[raster == node]))
                    elif var == 'sentinel':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([sentinel_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node, :allDates.index(train_date)]))
                    elif var == 'vigicrues':
                        for i, var2 in enumerate([vigicrues_variables]):
                            values = read_object(f'vigicrues{var2}.pkl', dir_data)
                            assert values is not None
                            vec_node.append(np.nanmean(values[raster == node, :allDates.index(train_date)]))
                    elif var == 'dynamic_world':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([dynamic_world_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node, :allDates.index(train_date)]))
                    elif var == 'foret':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([foret_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node]))
                    elif var == 'air':
                        for i, var2 in enumerate([air_variables]):
                            values = read_object(f'{var2}raw.pkl', dir_data)
                            assert values is not None
                            vec_node.append(np.nanmean(values[raster == node]))
                    elif var == 'osmnx':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate(osmnx_variables):
                            vec_node.append(np.nanmean(values[i][raster == node]))

                target_node = np.nansum(target_value[raster == node])

                vec.append(np.asarray(vec_node).reshape(1, -1))

                self.values_per_node[node] = np.asarray(vec_node).reshape(1, -1)

                target_per_node[node] = target_node
                vec_target.append(target_node)

        vec = np.concatenate(vec, axis=0)
        
        ### Clustering
        self.cluster_node_model = Predictor(n_clusters=3)
        self.knearestKneighbours = NearestNeighbors(n_neighbors=1).fit(vec)

        self.cluster_node_model.fit(np.asarray(vec_target).reshape(-1,1))
        self.all_clusters = self.cluster_node_model.predict(np.asarray(vec_target).reshape(-1,1))
        self.all_clusters = order_class(self.cluster_node_model, self.all_clusters)
        logger.info(f'{np.unique(self.all_clusters)}')

        for key, value in target_per_node.items():
            self.node_cluster[key] = self.cluster_node_model.predict(np.asarray([value]))
            self.node_cluster[key] = order_class(self.cluster_node_model, np.asarray([self.node_cluster[key]]))[0]

        check_and_create_path(path / 'time_series_clustering')
        for dept in departements:

            if dept in self.drop_department:
                continue

            plt.figure(figsize=(15,5))
            raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}_{self.graph_method}.pkl', path / 'raster')
            assert raster is not None

            time_series_image = np.full(raster.shape, fill_value=np.nan)
            nodes = np.unique(raster[~np.isnan(raster)])
            for node in nodes:
                time_series_image[raster == node] = self.node_cluster[node]
                
            img = plt.imshow(time_series_image, vmin=0, vmax=4, cmap='jet')
            plt.savefig(path / 'time_series_clustering' / f'{dept}_{self.scale}_{self.base}_{self.graph_method}.png')
            save_object(time_series_image, f'{dept}_{self.scale}_{self.base}_{self.graph_method}.pkl', path / 'time_series_clustering')
            plt.close('all')
            save_object(time_series_image, f'{dept}_{self.scale}_{self.graph_method}.pkl', path / 'time_series_clustering')

        logger.info(f'Time series clustering {self.node_cluster}')

    def _clusterize_node_with_time_series(self, departements, variables, target, train_date, path, root_data, root_target):
        self.variables_for_cluster = variables

        dir_target = root_target / 'log' / self.resolution
        dir_target_bin = root_target / 'bin' / self.resolution
        dir_raster = path / 'raster'
        vec_target = []
        vec = []
        
        self.values_per_node = {}
        self.node_cluster = {}
        target_per_node = {}
        for dept in departements:
            if dept in self.drop_department:
                continue

            if target == 'risk':
                target_value = read_object(f'{dept}Influence.pkl', dir_target)
            elif target == 'nbsinister':
                target_value = read_object(f'{dept}binScale0.pkl', dir_target_bin)

            assert target_value is not None
            target_value = target_value[:, :, allDates.index('2018-01-01'):allDates.index(train_date)]

            dir_data = root_data / dept / 'raster' / self.resolution
            raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}_{self.graph_method}.pkl', dir_raster)
            assert raster is not None
            nodes = np.sort(np.unique(raster[~np.isnan(raster)]))

            for node in nodes:
                vec_node = []
                for var in variables:
                    if var in cems_variables:
                        values = read_object(f'{var}raw.pkl', dir_data)
                        assert values is not None
                        values = values[:, :, :allDates.index(train_date)]
                        vec_node.append(np.nanmean(values[raster == node]))
                    elif var in ['population', 'elevation']:
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        values = values.reshape((values.shape[0], values.shape[1]))
                        vec_node.append(np.nanmean(values[raster == node]))
                    elif var == 'sentinel':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([sentinel_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node, :allDates.index(train_date)]))
                    elif var == 'vigicrues':
                        for i, var2 in enumerate(vigicrues_variables):
                            values = read_object(f'vigicrues{var2}.pkl', dir_data)
                            assert values is not None
                            vec_node.append(np.nanmean(values[raster == node, :allDates.index(train_date)]))
                    elif var == 'dynamic_world':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([dynamic_world_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node, :allDates.index(train_date)]))
                    elif var == 'foret':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([foret_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node]))
                    elif var == 'air':
                        for i, var2 in enumerate([air_variables]):
                            values = read_object(f'{var2}raw.pkl', dir_data)
                            assert values is not None
                            vec_node.append(np.nanmean(values[raster == node]))
                    elif var == 'osmnx':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate(osmnx_variables):
                            vec_node.append(np.nanmean(values[i][raster == node]))

                target_node = np.nansum(target_value[raster == node], axis=0)
               
                target_per_node[node] = target_node
                vec_target.append(target_node)

                vec.append(np.asarray(vec_node).reshape(1, -1))
                self.values_per_node[node] = np.asarray(vec_node).reshape(1, -1)

        vec = np.concatenate(vec, axis=0)
        vec_target = np.asarray(vec_target)

        ### Clustering
        self.cluster_node_model = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=10, random_state=42)
        self.knearestKneighbours = NearestNeighbors(n_neighbors=1).fit(vec)

        self.cluster_node_model.fit(vec_target)
        self.all_clusters = self.cluster_node_model.predict(vec_target)
        logger.info(f'{np.unique(self.all_clusters)}')

        for key, value in target_per_node.items():
            self.node_cluster[key] = self.cluster_node_model.predict(value.reshape(1, -1))

        check_and_create_path(path / 'time_series_clustering')
        for dept in departements:

            if dept in self.drop_department:
                continue

            plt.figure(figsize=(15,5))
            raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}_{self.graph_method}.pkl', path / 'raster')
            assert raster is not None

            time_series_image = np.full(raster.shape, fill_value=np.nan)
            nodes = np.unique(raster[~np.isnan(raster)])
            for node in nodes:
                time_series_image[raster == node] = self.node_cluster[node]
                
            img = plt.imshow(time_series_image, vmin=0, vmax=2, cmap='jet')
            plt.savefig(path / 'time_series_clustering' / f'{dept}_{self.scale}_{self.base}_{self.graph_method}.png')
            save_object(time_series_image, f'{dept}_{self.scale}_{self.base}_{self.graph_method}.pkl', path / 'time_series_clustering')
            plt.close('all')
            save_object(time_series_image, f'{dept}_{self.scale}_{self.base}_{self.graph_method}.pkl', path / 'time_series_clustering')

        logger.info(f'Time series clustering {self.node_cluster}')

    def find_closest_cluster(self, departements, train_date, path, root_data):
        """
        Charge les données des nœuds pour chaque département et prédit le cluster auquel ils appartiennent.

        :param departements: list, liste des départements à traiter
        :param path: Path, chemin du répertoire des données
        :param root_data: Path, chemin du répertoire de base pour les données spécifiques aux départements
        :return: dict, avec les départements comme clés et les prédictions de clusters pour chaque nœud comme valeurs
        """
        if  self.cluster_node_model is None:
            raise ValueError("Le modèle KMEANS n'est pas encore initialisé. Veuillez d'abord exécuter _clusterize_node().")
        
        for dept in departements:

            dir_data = root_data / dept / 'raster' / self.resolution

            raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}_{self.graph_method}.pkl', path / 'raster')
            assert raster is not None

            nodes = np.unique(raster[~np.isnan(raster)])
            for node in nodes:
                vec_node = []
                for var in self.variables_for_cluster:
                    if var in cems_variables:
                        values = read_object(f'{var}raw.pkl', dir_data)
                        assert values is not None
                        values = values[:, :, :allDates.index(train_date)]
                        vec_node.append(np.nanmean(values[raster == node]))
                    elif var in ['population', 'elevation']:
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        values = values.reshape((values.shape[0], values.shape[1]))
                        vec_node.append(np.nanmean(values[raster == node]))
                    elif var == 'sentinel':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([sentinel_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node, :allDates.index(train_date)]))
                    elif var == 'vigicrues':
                        assert values is not None
                        for i, var2 in enumerate(vigicrues_variables):
                            values = read_object(f'vigicrues{var2}.pkl', dir_data)
                            assert values is not None
                            vec_node.append(np.nanmean(values[raster == node, :allDates.index(train_date)]))
                    elif var == 'dynamic_world':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([dynamic_world_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node, :allDates.index(train_date)]))
                    elif var == 'foret':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate([foret_variables]):
                            vec_node.append(np.nanmean(values[i, raster == node]))
                    elif var == 'air':
                        for i, var2 in enumerate([air_variables]):
                            values = read_object(f'{var2}raw.pkl', dir_data)
                            assert values is not None
                            vec_node.append(np.nanmean(values[raster == node]))
                    elif var == 'osmnx':
                        values = read_object(f'{var}.pkl', dir_data)
                        assert values is not None
                        for i, var2 in enumerate(osmnx_variables):
                            vec_node.append(np.nanmean(values[i][raster == node]))

                nn = self.knearestKneighbours.kneighbors(np.asarray(vec_node).reshape(1,-1), return_distance=False)
                self.node_cluster[node] = self.all_clusters[nn[0][0]]
                
        check_and_create_path(path / 'time_series_clustering')
        for dept in departements:

            if dept in self.drop_department:
                continue

            plt.figure(figsize=(15,5))
            raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}_{self.graph_method}.pkl', path / 'raster')
            assert raster is not None

            time_series_image = np.full(raster.shape, fill_value=np.nan)
            nodes = np.unique(raster[~np.isnan(raster)])
            for node in nodes:
                time_series_image[raster == node] = self.node_cluster[node]
                
            img = plt.imshow(time_series_image, vmin=0, vmax=4, cmap='jet')
            plt.savefig(path / 'time_series_clustering' / f'{dept}_{self.scale}_{self.base}_{self.graph_method}.png')
            save_object(time_series_image, f'{dept}_{self.scale}_{self.base}_{self.graph_method}.pkl', path / 'time_series_clustering')
            plt.close('all')
            save_object(time_series_image, f'{dept}_{self.scale}_{self.graph_method}.pkl', path / 'time_series_clustering')

    def _read_kmeans(self, path : Path) -> None:
        assert self.scale != 0
        self.train_kmeans = False
        name = 'kmeans'+str(self.scale)+'.pkl'
        self.numCluster = self.oriLen // (self.scale * 6)
        X = list(zip(self.oriLongitude, self.oriLatitudes))
        self.clusterer = pickle.load(open(path / name, 'rb'))
        self.ids = self.clusterer.predict(X)

    def _create_predictor(self, dataset, start, end, dir, target_spec):
        dir_predictor = root_graph / dir / 'influenceClustering'

        self.predictors = {}
        self.predictors[self.scale] = {}
        self.predictors['departement'] = {}

        all_values = []

        # Scale Shape
        logger.info(f'######### {self.scale} scale ##########')
        for dep in np.unique(self.departements):
            if dep in self.drop_department:
                continue
            #influence = read_object(f'{dep}InfluenceScale{self.scale}_{self.base}.pkl', dir_influence)
            #binValues = read_object(f'{dep}binScale{self.scale}_{self.base}.pkl', dir_bin)

            influence = dataset[(dataset['departement'] == name2int[dep]) & \
                                 (dataset['date'] >= allDates.index(start)) & (dataset['date'] < allDates.index(end))]['risk'].values
            binValues = dataset[(dataset['departement'] == name2int[dep]) & \
                                 (dataset['date'] >= allDates.index(start)) & (dataset['date'] < allDates.index(end))]['nbsinister'].values

            #influence = influence[:,:,allDates.index(start):allDates.index(end)]
            #binValues = binValues[:,:,allDates.index(start):allDates.index(end)]

            values = np.unique(influence[~np.isnan(influence)])
            if values.shape[0] == 0:
                continue
            if values.shape[0] >= 5:
                predictor = Predictor(5, name=dep)
                predictor.fit(np.asarray(values))
            else:
                predictor = Predictor(values.shape[0], name=dep)
                predictor.fit(np.asarray(values))
                
            all_values.extend(values)
            check_class(influence, binValues, predictor)
            save_object(predictor, f'{dep}Predictor{self.scale}_{self.base}.pkl', path=dir_predictor)
            
            self.predictors[self.scale][dep] = predictor
            
            dataset.loc[dataset[dataset['departement'] == name2int[dep]].index, f'class_risk_{target_spec}'] = \
                order_class(predictor, predictor.predict(dataset[(dataset['departement'] == name2int[dep])]['risk'].values))
            
        predictor = Predictor(5, type='fix', name=f'general_{self.scale}')
        predictor.fit(np.asarray(all_values))
        save_object(predictor, f'general_predictor_{self.scale}_{self.base}_{self.graph_method}.pkl', path=dir_predictor)

        if self.scale == 'departement':
            return dataset
        
        logger.info('######### Departement Scale ##########')
        all_values = []

        #if (dir_predictor / f'{dep}PredictorDepartement.pkl').is_file():
        #    return
        # Departement
        for dep in np.unique(self.departements):
            if dep in self.drop_department:
                continue
            #binValues = read_object(dep+'binScale0.pkl', dir_bin)
            #influence = read_object(dep+'Influence.pkl', dir_target)
            
            #influence = influence[:,:,allDates.index(start):allDates.index(end)]
            #binValues = binValues[:,:,allDates.index(start):allDates.index(end)]

            influence = dataset[(dataset['departement'] == name2int[dep]) & \
                                (dataset['date'] >= allDates.index(start)) & \
                                (dataset['date'] < allDates.index(end))].groupby('date')['risk'].sum().reset_index()['risk'].values
            binValues = dataset[(dataset['departement'] == name2int[dep]) & \
                                (dataset['date'] >= allDates.index(start)) & \
                                  (dataset['date'] < allDates.index(end))].groupby('date')['nbsinister'].sum().reset_index()['nbsinister'].values
            
            if influence.shape[0] == 0:
                continue
            if influence.shape[0] >= 5:
                predictor = Predictor(5, name=dep)
                predictor.fit(np.asarray(influence))
            else:
                predictor = Predictor(influence.shape[0], name=dep)
                predictor.fit(np.asarray(influence))

            all_values.extend(influence)
            
            check_class(influence, binValues, predictor)
            predictor.log(logger)
            save_object(predictor, f'{dep}PredictorDepartement.pkl', path=dir_predictor)

            gp = dataset[dataset['departement'] == name2int[dep]].groupby('date')['risk'].sum().reset_index()
            gp['departement'] = name2int[dep]
            date_values = dataset[dataset['departement'] == name2int[dep]].groupby('date')['risk'].sum().reset_index()['risk'].values
            date_class = order_class(predictor, predictor.predict(date_values))
            gp['class_dep'] = date_class

            # Get the code for the department 'dep' from the 'name2int' mapping
            dep_code = name2int[dep]

            # Create a mask for the department 'dep' in 'dataset'
            mask_dep = dataset['departement'] == dep_code

            # Extract the subset of 'dataset' for department 'dep'
            dataset_dep = dataset.loc[mask_dep].copy()

            # Perform the merge with 'gp' on 'date' and 'departement' for department 'dep'
            dataset_dep = dataset_dep.merge(
                gp[['departement', 'date', 'class_dep']],
                on=['departement', 'date'],
                how='left',
            )

            # Update the original 'dataset' with the updated values for department 'dep'
            #dataset.update(dataset_dep)
            self.predictors['departement'][dep] = predictor

        predictor = Predictor(5, type='fix', name=f'general_departement')
        predictor.fit(np.asarray(all_values))
        save_object(predictor, f'general_predictor_departement_{self.base}.pkl', path=dir_predictor)

        return dataset
        
    def _create_nodes_list(self) -> None:
        """
        Create nodes list in the form (N, 4) where N is the number of Nodes [ID, longitude, latitude, departement of centroid]
        """
        logger.info('Creating node list')
        self.nodes = np.full((self.numCluster, 6), np.nan)
        self.uniqueIDS = np.sort(np.unique(self.ids).astype(int))
        for id in self.uniqueIDS:
            mask = np.argwhere(id == self.ids)
            graph_id = self.graph_ids[mask][:, 0][0]
            if self.scale != 0:
                nodeGeometry = unary_union(self.oriGeometry[mask[:,0]]).centroid
            else:
                nodeGeometry = self.oriGeometry[mask[:,0]].centroid

            dept_in_node = self.departements.values[mask[:,0]]
            valuesDep, counts = np.unique(dept_in_node, return_counts=True)
            ind = np.argmax(counts)
            self.nodes[id] = np.asarray([graph_id, id, float(nodeGeometry.x), float(nodeGeometry.y), name2int[valuesDep[ind]], -1])

    def _create_edges_list(self) -> None:
        """
        Create edges list
        """
        assert self.nodes is not None
        logger.info('Creating edges list')
        self.edges = None # (2, E) where E is the number of edges in the graph
        src_nodes = []
        target_nodes = []
        if self.uniqueIDS.shape[0] == 1:
            self.edges = np.asarray([[self.uniqueIDS[0], self.uniqueIDS[0]]])
        for id in self.uniqueIDS:
            if self.graph_method == 'graph':
                graph_id = np.unique(self.graph_ids[self.ids == id])[0]
                closest = self._distances_from_node(self.nodes[id], self.nodes[(self.nodes[:, graph_id_index] == graph_id) & (self.nodes[:, id_index] != id)])
            else:
                closest = self._distances_from_node(self.nodes[id], self.nodes[(self.nodes[:, id_index] != id)])
            closest = closest[closest[:, 1].argsort()]
            for i in range(min(self.numNei, len(closest))):
                if closest[i, 1] > self.maxDist:
                    break
                src_nodes.append(id)
                target_nodes.append(closest[i,0])

            src_nodes.append(id)
            target_nodes.append(id)

        self.edges = np.row_stack((src_nodes, target_nodes)).astype(int)

    def _create_temporal_edges_list(self, dates : list, k_days : int):
        self.k_days = k_days
        self.temporalEdges = None
        src_date = []
        target_date = []
        leni = len(dates)
        for i, _ in enumerate(dates):
            for k in range(k_days+1):
                if i + k > leni:
                    break
                src_date.append(i)
                target_date.append(i + k)

        self.temporalEdges = np.row_stack((src_date, target_date)).astype(int)

    def _distances_from_node(self, node : np.array,
                            otherNodes: np.array) -> np.array:
        """
        Calculate the distance bewten node and all the other nodes
        """
        res = np.full((otherNodes.shape[0], 2), math.inf)
        ignore = [node[id_index]]
        for index, (_, id, lon, lat, _, _) in enumerate(otherNodes):
            if id in ignore:
                continue
            distance = haversine((node[longitude_index], node[latitude_index]), [lon, lat])
            res[index, 0] = id
            res[index, 1] = distance
            ignore.append(id)
        return res
    
    def _assign_department(self, nodes : np.array) -> np.array:
        uniqueDept = np.unique(self.departements)
        dico = {}
        for key in uniqueDept:
            mask = np.argwhere(self.departements.values == key)
            geoDep = unary_union(self.oriGeometry[mask[:,0]])
            dico[key] = geoDep

        def find_dept(x, y, dico):
            for key, value in dico.items():
                try:
                    if value.contains(Point(float(x), float(y))):
                        return name2int[key]
                except Exception as e:
                    logger.info(f'graph_assign_department : {e}, x={x}, y={y}')
 
            return np.nan

        nodes[:,departement_index] = np.asarray([find_dept(node[longitude_index], node[latitude_index], dico) for node in nodes])

        return nodes
    
    def _assign_latitude_longitude(self, nodes : np.array) -> np.array:
        logger.info('Assign latitude longitude')

        uniqueNode = np.unique(nodes[:,id_index])
        for uN in uniqueNode:
            maskNode = np.argwhere(nodes[:,0] == uN)
            if maskNode.shape[0] == 0:
                continue
            nodes[maskNode[:,0],longitude_index:latitude_index+1] = self.nodes[np.argwhere(self.nodes[:,id_index] == uN)[:,0]][0][longitude_index:latitude_index+1]
        return nodes

    def _plot(self, nodes : np.array, time=0, dir_output = None) -> None:
        """
        Plotting
        """
        dis = np.unique(nodes[:,date_index])[:time+1].astype(int)
        if time > 0:
            ax = plt.figure(figsize=(15,15)).add_subplot(projection='3d')
        else:
            _, ax = plt.subplots(figsize=(15,15))
        if self.edges is not None:
            uniqueIDS = np.unique(nodes[:,id_index]).astype(int)
            for id in uniqueIDS:
                connection = self.edges[1][np.argwhere((self.edges[0] == id) & (np.isin(self.edges[1], nodes[:,id_index])))[:,0]]
                for target in connection:
                    target = int(target)
                    if target in uniqueIDS:
                        if time > 0:
                            for ti in range(time + 1):
                                ax.plot([dis[ti], dis[ti]],
                                        [nodes[nodes[:,id_index] == id][0][longitude_index], nodes[nodes[:,id_index] == target][0][longitude_index]],
                                        [nodes[nodes[:,id_index] == id][0][latitude_index], nodes[nodes[:,id_index] == target][0][latitude_index]],
                                    color='black')
                        else:
                            ax.plot([nodes[nodes[:,id_index] == id][0][longitude_index], nodes[nodes[:,id_index] == target][0][longitude_index]],
                                    [nodes[nodes[:,id_index] == id][0][latitude_index], nodes[nodes[:,id_index] == target][0][latitude_index]], color='black')
        if time > 0:
            if self.temporalEdges is not None:
                nb = 0
                uniqueDate = np.unique(nodes[:,date_index]).astype(int)
                uniqueIDS = np.unique(nodes[:,id_index]).astype(int)
                for di in uniqueDate:
                    connection = self.temporalEdges[1][np.argwhere((self.temporalEdges[0] == di))[:,0]]
                    for target in connection:
                        target = int(target)
                        if target in uniqueDate:
                            for id in uniqueIDS:
                                ax.plot([di, target],
                                         [nodes[nodes[:,0] == id][0][longitude_index], nodes[nodes[:,id_index] == id][0][longitude_index]],
                                         [nodes[nodes[:,0] == id][0][latitude_index], nodes[nodes[:,id_index] == id][0][latitude_index]],
                                        color='red')
                    nb += 1
                    if nb == time:
                        break
        
        node_x = nodes[nodes[:,date_index] == dis[0]][:,longitude_index]
        node_y = nodes[nodes[:,date_index] == dis[0]][:,latitude_index]
        for ti in range(1, time+1):
            node_x = np.concatenate((node_x, nodes[nodes[:,date_index] == dis[ti]][:,longitude_index]))
            node_y = np.concatenate((node_y, nodes[nodes[:,date_index] == dis[ti]][:,latitude_index]))
   
        if time > 0:
            node_z = nodes[nodes[:,date_index] == dis[0]][:,date_index]
            for ti in range(1, time+1):
                node_z = np.concatenate((node_z, nodes[nodes[:,date_index] == dis[ti]][:,4]))

            ax.scatter(node_z, node_x, node_y)
            ax.set_xlabel('Date')
            ax.set_ylabel('Longitude')
            ax.set_zlabel('Latitude')
        else:
            node_x = nodes[:,longitude_index]
            node_y = nodes[:,latitude_index]
            for i, n in enumerate(nodes):
                ax.annotate(str(n[id_index]), (n[longitude_index] + 0.001, n[latitude_index] + 0.001))
            ax.scatter(x=node_x, y=node_y, s=60)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        if dir_output is not None:
            name = f'graph_{self.scale}_{self.base}_{self.graph_method}.png'
            plt.savefig(dir_output / name)

        plt.close('all')
            
    def _plot_risk(self, mode, dir_output, path=None):
        check_and_create_path(dir_output / f'{self.base}_{self.scale}')
        dir_output_ = deepcopy(dir_output)
        dir_output = dir_output / f'{self.base}_{self.scale}'
        if mode == 'time_series_class':
            graph_geometry = []
            encoder = read_object(f'encoder_cluster_{self.scale}_{self.base}.pkl', dir_output_ / 'Encoder')
            fig, ax = plt.subplots(1, figsize=(30,30))
            departements = self.departements.unique()
            for dept in departements:
                if dept in self.drop_department:
                    continue
                geo_dept = self.oriGeometry[self.departements == dept]
                geo_dept = gpd.GeoDataFrame(geo_dept, geometry=geo_dept)
                geo_dept['id'] = self.graph_ids[self.departements == dept]
                geo_dept['hex_id'] = self.orihexid[self.departements == dept]
                uids = geo_dept.id.unique()
                geo_dept['time_series_class'] = np.nan
                for id in uids:
                    if float(id) not in self.node_cluster.keys():
                        continue
                    #geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'time_series_class'] = encoder.transform([self.node_cluster[float(id)]])[0]
                    geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'time_series_class'] = self.node_cluster[float(id)]

                graph_geometry.append(geo_dept)
            
            graph_geometry = pd.concat(graph_geometry).reset_index(drop=True)
            hexagones_france = gpd.read_file(root / 'csv/france/data/geo/hexagones_france.gpkg')
            hexagones_france = hexagones_france.set_index('hex_id').join(graph_geometry.set_index('hex_id')['time_series_class'], on='hex_id').reset_index()
            hexagones_france['latitude'] = hexagones_france['geometry'].apply(lambda x : float(x.centroid.y))
            hexagones_france['longitude'] = hexagones_france['geometry'].apply(lambda x : float(x.centroid.x))

            # Vérification des valeurs manquantes dans 'time_series_risk'
            missing_mask = hexagones_france['time_series_class'].isna()
            # Si des valeurs manquantes existent, on les interpole
            if missing_mask.any():
                # Récupérer les points non nuls (valides) pour l'interpolation
                valid_points = ~missing_mask
                points = hexagones_france.loc[valid_points, ['longitude', 'latitude']].values
                values = hexagones_france.loc[valid_points, 'time_series_class'].values
                
                # Points où les valeurs sont manquantes
                missing_points = hexagones_france.loc[missing_mask, ['longitude', 'latitude']].values

                # Interpolation 2D avec griddata
                interpolated_values = griddata(points, values, missing_points, method='nearest', fill_value=0)

                # Remplir les valeurs manquantes dans 'time_series_class'
                hexagones_france.loc[missing_mask, 'time_series_class'] = interpolated_values.astype(int)

            hexagones_france.plot(ax=ax, column='time_series_class', cmap='jet', legend=True, categorical=True)
            plt.savefig(dir_output / f'{mode}.png')

            plt.close('all')
        elif mode == 'time_series_risk':
            graph_geometry = []
            fig, ax = plt.subplots(1, figsize=(30,30))
            departements = self.departements.unique()
            for dept in departements:
                if dept in self.drop_department:
                    continue
                geo_dept = self.oriGeometry[self.departements == dept]
                geo_dept = gpd.GeoDataFrame(geo_dept, geometry=geo_dept)
                geo_dept['id'] = self.graph_ids[self.departements == dept]
                uids = geo_dept.id.unique()
                geo_dept['time_series_risk'] = np.nan
                for id in uids:
                    if float(id) not in self.node_cluster.keys():
                        continue
                    geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'time_series_risk'] = self.node_cluster[float(id)]

                graph_geometry.append(geo_dept)
            graph_geometry = pd.concat(graph_geometry).reset_index(drop=True)

            # Vérification des valeurs manquantes dans 'time_series_risk'
            missing_mask = graph_geometry['time_series_risk'].isna()

            # Si des valeurs manquantes existent, on les interpole
            if missing_mask.any():
                # Récupérer les points non nuls (valides) pour l'interpolation
                valid_points = ~missing_mask
                points = graph_geometry.loc[valid_points, ['longitude', 'latitude']].values
                values = graph_geometry.loc[valid_points, 'time_series_risk'].values
                
                # Points où les valeurs sont manquantes
                missing_points = graph_geometry.loc[missing_mask, ['longitude', 'latitude']].values

                # Interpolation 2D avec griddata
                interpolated_values = griddata(points, values, missing_points, method='linear')

                # Remplir les valeurs manquantes dans 'time_series_risk'
                graph_geometry.loc[missing_mask, 'time_series_risk'] = interpolated_values

            graph_geometry.plot(ax=ax, column='time_series_risk', cmap='jet', legend=True)
            plt.savefig(dir_output / f'{mode}.png')
            plt.close('all')
        else:
            assert path is not None
            try:
                graph_geometry = []
                index_start = allDates.index(mode.split('_')[0])
                index_end = allDates.index(mode.split('_')[1]) + 1
                dir_target = path / 'influence'
                dir_target_bin = path / 'bin'
                dir_raster = path / 'raster'
                departements = self.departements.unique()
                for dept in departements:
                    if dept in self.drop_department:
                        continue
                    values = read_object(f'{dept}InfluenceScale{self.scale}_{self.base}.pkl', dir_target)
                    bin_values = read_object(f'{dept}binScale{self.scale}_{self.base}.pkl', dir_target_bin)
                    raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}.pkl', dir_raster)
                    if values is None or bin_values is None or raster is None:
                        continue
                    values = values[:, :, index_start:index_end]
                    bin_values = bin_values[:, :, index_start:index_end]
                    values = np.sum(values, axis=2)
                    bin_values = np.sum(bin_values, axis=2)
                    geo_dept = self.oriGeometry[self.departements == dept]
                    geo_dept = gpd.GeoDataFrame(geo_dept, geometry=geo_dept)
                    geo_dept['id'] = self.graph_ids[self.departements == dept]
                    geo_dept['risk'] = np.nan
                    geo_dept['bin'] = np.nan
                    uids = geo_dept.id.unique()
                    for id in uids:
                        if float(id) not in raster:
                            continue
                        geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'risk'] = values[raster == id][0]
                        geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'bin'] = bin_values[raster == id][0]

                    graph_geometry.append(geo_dept)
                graph_geometry = pd.concat(graph_geometry).reset_index(drop=True)

                fig, ax = plt.subplots(1, figsize=(30,30))
                graph_geometry.plot(ax=ax, column='risk', cmap='jet', legend=True)
                plt.savefig(dir_output / f'{mode}_risk.png')
                plt.close('all')
                fig, ax = plt.subplots(1, figsize=(30,30))
                graph_geometry.plot(ax=ax, column='bin', cmap='jet', legend=True)
                plt.savefig(dir_output / f'{mode}_bin.png')
                plt.close('all')
            except Exception as e:
                logger.info(f'Canno t produce map for {mode}')
                logger.info(f'{e}')

    def _plot_risk_departement(self, mode, dir_output, path=None):
        check_and_create_path(dir_output / f'{self.base}_{self.scale}')
        dir_output = dir_output / f'{self.base}_{self.scale}'
        if mode == 'time_series_class':
            graph_geometry = []
            fig, ax = plt.subplots(1, figsize=(30,30))
            departements = self.departements.unique()
            for dept in departements:
                if dept in self.drop_department:
                    continue
                geo_dept = self.oriGeometry[self.departements == dept]
                geo_dept = gpd.GeoDataFrame(geo_dept, geometry=geo_dept)
                geo_dept['id'] = self.graph_ids[self.departements == dept]
                uids = geo_dept.id.unique()
                geo_dept['time_series_class'] = np.nan
                for id in uids:
                    if float(id) not in self.time_series_class.keys():
                        continue
                    geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'time_series_class'] = self.time_series_class[float(id)]['class'][0]

                geo_dept['time_series_class'] = geo_dept['time_series_class'].mean()
                graph_geometry.append(geo_dept)
            
            graph_geometry = pd.concat(graph_geometry).reset_index(drop=True)
            graph_geometry.plot(ax=ax, column='time_series_class', cmap='jet', legend=True)
            plt.savefig(dir_output / f'{mode}_deparement.png')
            plt.close('all')
        elif mode == 'time_series_risk':
            graph_geometry = []
            fig, ax = plt.subplots(1, figsize=(30,30))
            departements = self.departements.unique()
            for dept in departements:
                if dept in self.drop_department:
                    continue
                geo_dept = self.oriGeometry[self.departements == dept]
                geo_dept = gpd.GeoDataFrame(geo_dept, geometry=geo_dept)
                geo_dept['id'] = self.graph_ids[self.departements == dept]
                uids = geo_dept.id.unique()
                geo_dept['time_series_risk'] = np.nan
                for id in uids:
                    if float(id) not in self.time_series_class.keys():
                        continue
                    geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'time_series_risk'] = self.time_series_class[float(id)]['value']

                geo_dept['time_series_risk'] = geo_dept['time_series_risk'].sum()
                graph_geometry.append(geo_dept)

            graph_geometry = pd.concat(graph_geometry).reset_index(drop=True)
            graph_geometry.plot(ax=ax, column='time_series_risk', cmap='jet', legend=True)
            plt.savefig(dir_output / f'{mode}_departement.png')
            plt.close('all')
        else:
            assert path is not None
            try:
                graph_geometry = []
                index_start = allDates.index(mode.split('_')[0])
                index_end = allDates.index(mode.split('_')[1]) + 1
                dir_target = path / 'influence'
                dir_target_bin = path / 'bin'
                dir_raster = path / 'raster'
                departements = self.departements.unique()
                for dept in departements:
                    if dept in self.drop_department:
                        continue
                    values = read_object(f'{dept}InfluenceScale{self.scale}_{self.base}.pkl', dir_target)
                    bin_values = read_object(f'{dept}binScale{self.scale}_{self.base}.pkl', dir_target_bin)
                    raster = read_object(f'{dept}rasterScale{self.scale}_{self.base}.pkl', dir_raster)
                    if values is None or bin_values is None or raster is None:
                        continue
                    values = values[:, :, index_start:index_end]
                    bin_values = bin_values[:, :, index_start:index_end]
                    values = np.sum(values, axis=2)
                    bin_values = np.sum(bin_values, axis=2)
                    geo_dept = self.oriGeometry[self.departements == dept]
                    geo_dept = gpd.GeoDataFrame(geo_dept, geometry=geo_dept)
                    geo_dept['id'] = self.graph_ids[self.departements == dept]
                    geo_dept['risk'] = np.nan
                    geo_dept['bin'] = np.nan
                    uids = geo_dept.id.unique()
                    for id in uids:
                        if float(id) not in raster:
                            continue
                        geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'risk'] = values[raster == id][0]
                        geo_dept.loc[geo_dept[geo_dept['id'] == id].index, 'bin'] = bin_values[raster == id][0]

                    geo_dept['risk'] = geo_dept['risk'].sum()
                    geo_dept['bin'] = geo_dept['bin'].sum()
                    graph_geometry.append(geo_dept)
                graph_geometry = pd.concat(graph_geometry).reset_index(drop=True)
                
                fig, ax = plt.subplots(1, figsize=(30,30))
                graph_geometry.plot(ax=ax, column='risk', cmap='jet', legend=True)
                plt.savefig(dir_output / f'{mode}_risk_departement.png')
                plt.close('all')
                
                fig, ax = plt.subplots(1, figsize=(30,30))
                graph_geometry.plot(ax=ax, column='bin', cmap='jet', legend=True)
                plt.savefig(dir_output / f'{mode}_bin_departement.png')
                plt.close('all')
            except Exception as e:
                logger.info(f'Canno t produce map for {mode}')
                logger.info(f'{e}')

    def _set_model(self, model) -> None:
        """
        model : the neural network model. Train or not
        """
        self.model = model

    def _set_model_from_path(self, path : Path) -> None:
        self.modelPath = path

    def _predict_np_array(self, X : np.array, edges : np.array, device : torch.device) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        tensorX = torch.tensor(X, dtype=torch.float32, device=device)
        tensorEdges = torch.tensor(edges, dtype=torch.float32, device=device)
        return self.model(tensorX, tensorEdges)

    def _predict_tensor(self, X : torch.tensor, edges : torch.tensor) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            return self.model(X, edges)

    def _predict_dataset(self, X) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            test_loader = DataLoader(X, 1)
            for i, data in enumerate(test_loader, 0):
                inputs, _, edges = data
                output = self.model(inputs, edges)
                return output
            
    def compute_mean_sequence(self, dataframe, database_name, maxdate, target_spe='0_0'):
        """
        Calculates the average size of continuous date sequences without interruption for each point.
        Updates the dataframe with the calculated risk values.

        Parameters:
        - dataframe: pandas DataFrame containing the data with columns 'date', 'month', 'id', 'nbsinister'.
        - doPast: boolean indicating whether to consider only past data in calculations.

        Returns:
        - dataframe: pandas DataFrame updated with the risk values.
        """
        
        ############################### Calculating the kernel size #####################################
        # Ensure that nodes are initialized
        assert self.nodes is not None

        # Define month groups representing different seasons or periods
        group_month = [
            [2, 3, 4, 5],    # Medium season
            [6, 7, 8, 9],    # High season
            [10, 11, 12, 1]  # Low season
        ]

        nbsinister_col = f'nbsinister_{target_spe}'

        logger.info(f'{nbsinister_col} -> {dataframe[nbsinister_col].sum()}')

        # Names corresponding to each group of months
        name = ['medium', 'high', 'low']

        dataframe['month_non_encoder'] = dataframe['date'].apply(lambda x : int(allDates[int(x)].split('-')[1]))

        # Sort the dataframe by date
        dataframe = dataframe.sort_values(by=['graph_id', 'date'])
        #self.sequences_month = None

        points = np.unique(self.nodes[:, graph_id_index])

        self.sequences_month = {}

        if database_name == 'firemen':
            td = 3  # Time delta in days to consider continuity
        elif database_name == 'bdiff':
            td = 3
        else:
            td = 3
        # Get unique points from self.nodes

        fire_dataframe = dataframe[dataframe[nbsinister_col] > 0].reset_index()

        # Iterate over each group of months with its index
        for i, months in enumerate(group_month):

            # Filter the dataframe for the current group of months
            df = fire_dataframe[fire_dataframe['month_non_encoder'].isin(months)]

            # Initialize the dictionary for the current season
            self.sequences_month[name[i]] = {}

            for point in points:
                # Initialize the dictionary for the current point
                self.sequences_month[name[i]][point] = {}
                self.sequences_month[name[i]][point]['dates'] = []
                self.sequences_month[name[i]][point]['mean_size'] = 1

                # Get the data for the current point

                df_point = df[df['graph_id'] == point]

                if len(df_point) == 0:
                    continue

                uids = df_point['id'].unique()

                df_point = df_point[df_point['id'] == uids[0]]

                # Group the data by date for the current point
                for date, group in df_point.groupby('date'):

                    # If no sequences exist yet, create a new one
                    if not self.sequences_month[name[i]][point]['dates']:
                        self.sequences_month[name[i]][point]['dates'].append([date])
                        continue

                    find_seq = False

                    # Iterate over existing sequences to see if the date continues any of them
                    for sei, seq in enumerate(self.sequences_month[name[i]][point]['dates']):
                        # Check if the date continues the sequence (within td days)
                        if (date - seq[-1]) <= td:
                            # Avoid duplicate dates
                            if date in seq:
                                continue
                            # Add the date to the sequence
                            seq.append(date)
                            find_seq = True

                            # Update the sequence in the list
                            self.sequences_month[name[i]][point]['dates'][sei] = seq
                            break
                    # If the date does not continue any existing sequence, create a new one
                    if not find_seq:
                        self.sequences_month[name[i]][point]['dates'].append([date])

                # Calculate the average size of sequences for the current point and season
                total_size = sum(len(seq_dates) for seq_dates in self.sequences_month[name[i]][point]['dates'])
                num_sequences = len(self.sequences_month[name[i]][point]['dates'])
                mean = round(total_size / num_sequences)
                if mean > 1:
                    self.sequences_month[name[i]][point]['mean_size'] = 2 * mean + 1
                else:
                    self.sequences_month[name[i]][point]['mean_size'] = 1

                logger.info(f'{point}, {name[i]} {df_point["departement"].unique()} : {mean} -> {self.sequences_month[name[i]][point]["mean_size"]}')

        ###################################### Applying convolution ##############################################

        self.historical_risk = pd.DataFrame(index=np.arange(len(dataframe)), columns=['graph_id', 'id', 'date', 'month', 'risk', 'past_risk'])
        self.historical_risk['date'] = dataframe['date'].values
        self.historical_risk['month'] = dataframe['month_non_encoder'].values
        self.historical_risk['graph_id'] = dataframe['graph_id'].values
        self.historical_risk['risk'] = 0.0
        self.historical_risk['past_risk'] = 0.0
        self.historical_risk = self.historical_risk.drop_duplicates(keep='first')

        # For each point, compute risk values using convolution
        for point in points:

            logger.info(f'############################ {point} #############################')

            # Get the data for the current point
            node_df_values = dataframe[dataframe['graph_id'] == point].reset_index(drop=True)
            uids = node_df_values['id'].unique()
            node_df_values = node_df_values[node_df_values['id'] == uids[0]]

            historical = 0
            historical_past = 0
            num_historical = 0
            num_historical_past = 0
            # Initialize historical data
            for index, date_int in enumerate(node_df_values['date']):
                # Assuming 'date' is of type datetime or convertible
                date = allDates[int(date_int)]
                date = dt.datetime.strptime(date, '%Y-%m-%d')
                month = date.month
                day = date.day
                # Iterate over years to create historical dates
                for y in years:
                    try:
                        newd = pd.Timestamp(year=int(y), month=month, day=day).strftime('%Y-%m-%d')
                    except:
                        continue
                    if newd not in allDates:
                        continue
                    newindex = allDates.index(newd)
                    if newindex not in node_df_values['date'].unique():
                        continue
                    historical += node_df_values[node_df_values['date'] == newindex][nbsinister_col].values[0]
                    num_historical += 1
                    if newindex < allDates.index(maxdate):
                        historical_past += node_df_values[node_df_values['date'] == newindex][nbsinister_col].values[0]
                        num_historical_past += 1

                # Average over the number of years
                if num_historical != 0:
                    historical /= num_historical
                if num_historical_past != 0:
                    historical_past /= num_historical_past

                self.historical_risk.loc[self.historical_risk[(self.historical_risk['date'] == date_int) & \
                                                                (self.historical_risk['graph_id'] == point)].index, 'risk'] = historical
                self.historical_risk.loc[self.historical_risk[(self.historical_risk['date'] == date_int) & \
                                                            (self.historical_risk['graph_id'] == point)].index, 'past_risk'] = historical_past

            # Apply convolution for each season
            for i, months in enumerate(group_month):
                # Get the indices of the current point for the current season
                node_values = dataframe[(dataframe['graph_id'] == point) & (dataframe['month_non_encoder'].isin(months))]
                uids = node_values['id'].unique()
                node_index = node_values[node_values['id'] == uids[0]].index

                # Get the 'nbsinister' values for these indices
                node_values = dataframe.loc[node_index, nbsinister_col].values

                # Get the kernel size based on the average size of sequences for the current point and season
                kernel_size = self.sequences_month[name[i]][point]['mean_size']

                # Create the daily kernel
                if kernel_size == 1:
                    node_values_daily = np.copy(node_values)
                    node_values_season = np.zeros(node_values.shape)
                else:
                    kernel_daily = np.abs(np.arange(-(kernel_size//2), kernel_size // 2 + 1))
                    kernel_daily += 1
                    kernel_daily = 1 / kernel_daily
                    kernel_daily = MinMaxScaler((0, 1)).fit_transform(kernel_daily.reshape(-1,1)).reshape(-1)

                    # Create the seasonal kernel
                    kernel_season = np.ones(kernel_size, dtype=float)
                    kernel_season /= kernel_size  # Normalize the kernel
                    kernel_season[kernel_size // 2] = 0

                    # Apply convolution with the daily kernel
                    node_values_daily = convolve_fft(
                        node_values, kernel_daily, normalize_kernel=False)
                    
                    # Apply convolution with the seasonal kernel
                    node_values_season = convolve_fft(
                        node_values, kernel_season, normalize_kernel=False)

                past_kernel_size = 7
                kernel_daily_past = np.abs(np.arange(-(past_kernel_size//2), past_kernel_size // 2 + 1))
                kernel_daily_past += 1
                kernel_daily_past = 1 / kernel_daily_past
                kernel_season_past = np.ones(past_kernel_size, dtype=float)
                kernel_season_past /= kernel_size  # Normalize the kernel
                kernel_daily_past = MinMaxScaler((0, 1)).fit_transform(kernel_daily_past.reshape(-1,1)).reshape(-1)
                kernel_daily_past[:past_kernel_size // 2 + 1] = 0.0
                kernel_season_past[:past_kernel_size // 2 + 1] = 0.0
                
                # Get the historical values for the current months
                node_values_historical = self.historical_risk[(self.historical_risk['month'].isin(months)) & (self.historical_risk['graph_id'] == point)]['risk']
                #node_values_historical_past = self.historical_risk[(self.historical_risk['month'].isin(months)) & (self.historical_risk['graph_id'] == point)]['past_risk']

                # Sum the different components to obtain the risk values
                nodes_values = node_values_daily + node_values_season + node_values_historical

                for id in uids:
                    node_index_total = dataframe[(dataframe['id'] == id) & (dataframe['month_non_encoder'].isin(months))].index
                    node_values_id = dataframe[(dataframe['id'] == id) & (dataframe['month_non_encoder'].isin(months))]['nbsinister_id'].values

                    # Apply convolution with the daily kernel
                    node_values_daily_past = convolve_fft(
                        node_values_id, kernel_daily_past, normalize_kernel=False)

                    nodes_values_past = node_values_daily_past

                    # Update the dataframe with the calculated risk values
                    dataframe.loc[node_index_total, 'pastinfluence_risk'] = nodes_values_past
                    dataframe.loc[node_index_total, f'risk_{target_spe}'] = nodes_values.values

                #print(np.unique(dataframe.loc[node_index, 'risk'].isna()))
                #print(dataframe.loc[node_index, 'risk'])
                #print(nodes_values)

        # Incorporate multi step historic values
        """for point in points:
            for var in auto_regression_variable_reg:
                ste = int(var.split('-')[1])
                df_point = dataframe[dataframe['graph_id'] == point]
                uids = df_point['id'].unique()
                for id in uids:
                    dataframe.loc[dataframe[dataframe['id'] == point].index, f'AutoRegressionReg-J-{ste}'] = dataframe[dataframe['id'] == id]['risk'].shift(ste).values"""

            #dataframe.loc[dataframe[dataframe['id'] == point].index, f'pastinfluence'] = dataframe[dataframe['id'] == point]['pastinfluence_risk'].shift(ste).values

        dataframe['historical'] = self.historical_risk['past_risk']
        dataframe[f'pastinfluence_{target_spe}'] = dataframe[f'pastinfluence_risk']

        dataframe[f'risk_{target_spe}'] = np.round(dataframe[f'risk_{target_spe}'].values, 3)
        dataframe[f'pastinfluence_{target_spe}'] = np.round(dataframe[f'pastinfluence_{target_spe}'].values, 3)

        return dataframe

    def compute_window_class(self, dataframe, window_sizes=[7], n_clusters=3, column='nbsinister', aggregate_funcs=['sum'], mode='train', target_name='', dir_output=Path('./')):
        # Compute the average of `nbsinister` for each unique (graph_id, date) pair

        logger.info(f'Compute window class for {column} with {n_clusters} class accross {window_sizes}')

        if target_name == '':
            target_name = column

        df = dataframe.copy(deep=True)

        df_unique = df.groupby(['graph_id', 'date'], as_index=False).agg({column: 'mean'})
        check_and_create_path(dir_output)

        # For each specified window size
        for window_size in window_sizes:
            # Loop through each aggregation function provided in aggregate_funcs
            for aggregate_func in aggregate_funcs:
                # Initialize a column to store the rolling aggregation
                column_name = f'window_{column}_{aggregate_func}_{window_size}'
                df_unique[column_name] = 0.0

                # Calculate the offset around the current day
                half_window = (window_size - 1) // 2

                # Case when window_size is 1
                if window_size == 1:
                    # Directly assign the class without using a rolling window
                    df_unique[column_name] = df_unique[column]  # Use the column's value directly for 'nbsinister'
                else:
                    # For each unique graph_id
                    for graph_id in df_unique['graph_id'].unique():
                        # Filter data for the current graph_id
                        df_graph = df_unique[df_unique['graph_id'] == graph_id]

                        # Iterate through each row in df_graph
                        for idx, row in df_graph.iterrows():
                            # Define the window bounds
                            date_min = row['date'] - half_window
                            date_max = row['date'] + window_size
                            
                            # Filter rows within the date window
                            #window_df = df_graph[(df_graph['date'] >= date_min) & (df_graph['date'] <= date_max)]
                            window_df = df_graph[(df_graph['date'] >= row['date']) & (df_graph['date'] <= date_max)]
                            
                            # Apply the aggregation function
                            if aggregate_func == 'sum':
                                df_unique.at[idx, column_name] = window_df[column].sum()
                            elif aggregate_func == 'mean':
                                df_unique.at[idx, column_name] = window_df[column].mean()
                            elif aggregate_func == 'grad':
                                # Determine if the trend is increasing, decreasing, or constant
                                first_value = window_df[column].iloc[0]
                                last_value = window_df[column].iloc[-1]
                                if last_value > first_value:
                                    df_unique.at[idx, column_name] = 2
                                elif last_value < first_value:
                                    df_unique.at[idx, column_name] = 0
                                else:
                                    df_unique.at[idx, column_name] = 1
                            elif aggregate_func == 'max':
                                df_unique.at[idx, column_name] = window_df[column].max()
                            elif aggregate_func == 'min':
                                df_unique.at[idx, column_name] = window_df[column].min()
                            elif aggregate_func == 'std':
                                df_unique.at[idx, column_name] = window_df[column].std()

                # Apply KMeans to the rolling aggregation column for all functions except 'grad' and 'std'
                if aggregate_func not in ['grad', 'std']:
                    type_clustering = 'kmeans'
                    if target_name == 'binary':
                        type_clustering = 'fix'
                    cluster_column_name = f'class_window_{target_name}_{aggregate_func}_{window_size}'
                    if mode == 'train' or not (dir_output / f'{cluster_column_name}.pkl').is_file(): 
                        values = df_unique[column_name].unique().reshape(-1,1)
                        kmeans = Predictor(n_clusters=n_clusters, type=type_clustering, binary = target_name == 'binary')
                        if values.shape[0] < n_clusters and target_name != 'binary':
                            kmeans = Predictor(n_clusters=values.shape[0], type=type_clustering, binary = target_name == 'binary')
                        kmeans.fit(values)
                        df_unique[cluster_column_name] = kmeans.predict(df_unique[column_name].values.reshape(-1,1))
                        save_object(kmeans, f'{cluster_column_name}.pkl', dir_output)
                        logger.info(f'{column_name} -> {kmeans.cluster_centers_}')
                    else:
                        kmeans = read_object(f'{cluster_column_name}.pkl', dir_output)
                        assert kmeans is not None
                        df_unique[cluster_column_name] = kmeans.predict(df_unique[column_name].values.reshape(-1,1))

                    # Sort the clusters based on the centroid values
                    if type_clustering == 'kmeans':
                        centroids = kmeans.cluster_centers_
                        sorted_indices = np.argsort(centroids[:, 0])  # Sort by the first dimension (centroid value)
                        df_unique[cluster_column_name] = np.vectorize(lambda x: np.where(sorted_indices == x)[0][0])(df_unique[cluster_column_name])

                    fig, ax = plt.subplots(2, figsize=(15, 7))
                    df_unique[column_name].plot(ax=ax[0])

                    df_unique[cluster_column_name].plot(ax=ax[1])
                    plt.title(column_name)
                    check_and_create_path(dir_output / 'Image')
                    plt.savefig(dir_output / 'Image' / f'{cluster_column_name}.png')
                    #plt.show()
                    plt.close('all')

        # Merge the results with the original DataFrame to keep other columns
        cols_to_merge = ['graph_id', 'date'] + [f'window_{column}_{aggregate_func}_{ws}' for ws in window_sizes for aggregate_func in aggregate_funcs] + \
        [f'class_window_{target_name}_{aggregate_func}_{ws}' for ws in window_sizes for aggregate_func in aggregate_funcs if aggregate_func not in ['grad', 'std']]

        for col in [f'window_{column}_{aggregate_func}_{ws}' for ws in window_sizes for aggregate_func in aggregate_funcs] + \
            [f'class_window_{target_name}_{aggregate_func}_{ws}' for ws in window_sizes for aggregate_func in aggregate_funcs if aggregate_func not in ['grad', 'std']]:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
        
        df = df.merge(df_unique[cols_to_merge], on=['graph_id', 'date'], how='left')

        return df  
    
    def _predict_test_loader(self, X: DataLoader) -> torch.tensor:
            """
            Generates predictions using the model on the provided DataLoader, with optional autoregression.

            Note:
            - 'node_id' and 'date_id' are not included in features_name but can be found in labels[:, 0] and labels[:, 4].
            Parameters:
            - X_: DataLoader providing the test data.
            - features: Numpy array of feature indices to use.
            - device: Torch device to use for computations.
            - target_name: Name of the target variable.
            - autoRegression: Boolean indicating whether to use autoregression.
            - features_name: List mapping feature names to indices.
            - dataset: Pandas DataFrame containing 'node_id', 'date_id', and 'nbsinister' columns.

            Returns:
            - pred: Tensor containing the model's predictions.
            - y: Tensor containing the true labels.
            """
            assert self.model is not None

            pred, y = self.model._predict_test_loader(X)

            return pred, y
    
    def predict_model_api_sklearn(self, X : pd.DataFrame,
                                  features : list, target_name : bool,
                                  autoRegression : bool, quantile=False) -> np.array:
        
        isBin = target_name == 'binary'
        assert self.model is not None
        X.reset_index(inplace=True, drop=True)

        if hasattr(self.model, 'col_id_name'):
            col_id = self.model.col_id_name
            if col_id not in list(X.columns):
                X = add_aggregation_column_df(X, col_id)
                if X is None:
                    return np.zeros(X.shape[0])
        else:
            col_id = None

        res = np.empty((X.shape[0], 2))
        res_max = None
        res_min = None

        if not autoRegression:
            if not isBin:
                if self.model.post_process is not None and hasattr(self.model.post_process, 'col_id'):
                    post_process_ids = X[self.model.post_process.col_id]
                else:
                    post_process_ids = None

                if col_id is not None:
                    res[:, 0] = self.model.predict_nbsinister(X[features], post_process_ids)
                    res[:, 1] = self.model.predict_risk(X[features], post_process_ids)
                else:
                    res[:, 0] = self.model.predict_nbsinister(X[features], post_process_ids)
                    res[:, 1] = self.model.predict_risk(X[features], post_process_ids)
            else:
                raise ValueError(f'Binary model are not available yet')
                if col_id:
                    return self.model.predict_proba(X[features], X[col_id])[:,1]
                else:
                    return self.model.predict_proba(X[features])[:,1]
        else:
            ValueError('Not implemented')

        return res, res_max, res_min

    def _predict_perference_with_Y(self, Y : np.array, target_name : str) -> np.array:
        res = np.empty(Y.shape[0])
        res[0] = 0
        for i, node in enumerate(Y):
            if i == 0:
                continue
            index = np.argwhere((Y[:,date_index] == node[date_index] - 1) & (Y[:,id_index] == node[id_index]))
            if index.shape[0] == 0:
                res[i] = 0
                continue
            if target_name == 'risk': 
                res[i] = Y[index[0], -1]
            elif target_name == 'binary':
                res[i] = Y[index[0], -2] > 0
            elif target_name == 'nbsinister':
                res[i] = Y[index[0], -2]
            else:
                logger.info(f'Unknow target_name {target_name}')
                exit(1)
        return res
        
    def _predict_perference_with_X(self, X : np.array) -> np.array:
        return X['pastInfluence_max'].values

    def _info_on_graph(self, nodes : np.array, output : Path) -> None:
        """
        logger.info informatio on current global graph structure and nodes
        """
        check_and_create_path(output)
        graphIds = np.unique(nodes[:,graph_id_index])

        for graphid in graphIds:
            logger.info(f'size : {graphid, self.nodes[self.nodes[:,graph_id_index] == graphid].shape[0]}')

        logger.info('**************************************************')
        if self.nodes is None:
            logger.info('No nodes found in graph')
        else:
            logger.info(f'The main graph has {self.nodes.shape[0]} nodes')
        if self.edges is None:
            logger.info('No edges found in graph')
        else:
            logger.info(f'The main graph has {self.edges.shape[1]} spatial edges')
        if self.temporalEdges is None:
            logger.info('No temporal edges found in graph')
        else:
            logger.info(f'The main graph has {self.temporalEdges.shape[1]} temporal edges')
        if nodes is not None:
            logger.info(f'We found {np.unique(nodes[:,date_index]).shape[0]} different graphs in the training set for {nodes.shape[0]} nodes')

        """logger.info(f'Mean number of nodes in sub graph : {np.mean(size)}')
        logger.info(f'Max number of nodes in subgraph {np.max(size)}')
        logger.info(f'Minimum number of nodes in subgraph {np.min(size)}')"""