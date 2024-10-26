import numpy as np
from sklearn.cluster import KMeans
import math
import geopandas as gpd 
from shapely import unary_union
import warnings
from shapely import Point
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
from dataloader import *
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn import pipeline
import math
import geopandas as gpd 
from shapely import unary_union
from astropy.convolution import convolve, Box2DKernel
from shapely import Point
from forecasting_models.models import *
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
from skimage import io, color, filters, measure, morphology
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import scipy.ndimage as ndimage

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Create graph structure from corresponding geoDataframe
class GraphStructure():
    def __init__(self, scale : int,
                 geo : gpd.GeoDataFrame,
                 maxDist: float,
                 numNei : int):

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
        self.oriLatitues = geo.latitude # all original latitude of interest
        self.oriLongitude = geo.longitude # all original longitude of interest
        self.oriLen = self.oriLatitues.shape[0] # len of data
        self.oriGeometry = geo['geometry'].values # original geometry
        self.oriIds = geo['scale0'].values
        self.departements = geo.departement # original dept of each geometry
        self.maxDist = maxDist # max dist to link two nodes
        self.numNei = numNei # max connect neighbor a node can have
        self.clusterer = {}

    def _train_clustering(self, base : str, doRaster: bool, path : Path, sinister : str, dataset_name : str, resolution) -> None:
        udept = np.unique(self.departements)
        if base.find('-') == -1:
            vec_base = [base]
            self.base = base
        else:
            vec_base = base.split('-')
            self.base = base
            
        self.train_kmeans = True
        node_already_predicted = 0
        self.numCluster = 0
        
        logger.info(f'Create node via DBSCAN on {vec_base}')

        self.ids = np.full(self.oriLatitues.shape[0], fill_value=np.nan)
        for dept in udept:
            mask = self.departements == dept
            dir_raster = root_target / sinister / 'raster' / resolution
            dir_data = root / 'csv' / dept / 'raster' / resolution
            dir_target = root_target / dataset_name / sinister / 'log' / resolution
            dir_target_bin = root_target / sinister / 'bin' / resolution
            dir_encoder = path / 'Encoder'
            raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)[0]
            pred = np.full(raster.shape, fill_value=-1, dtype=int)
            self.clusterer[dept] = {}
            values = []
            for vb in vec_base:
                if vb == 'risk':
                    data = read_object(f'{dept}Influence.pkl', dir_target)
                    data = np.nansum(data, axis=2)
                    values.append(data[~np.isnan(raster)].reshape(-1, 1))
                elif vb == 'nbsinister':
                    data = read_object(f'{dept}binScale0.pkl', dir_target_bin)
                    data = np.nansum(data, axis=2)
                    values.append(data[~np.isnan(raster)].reshape(-1, 1))
                elif vb == 'geometry':
                    coords = np.indices((raster.shape[0], raster.shape[1]))[:, ~np.isnan(raster)].reshape(2, -1).T
                    values.append(coords)
                else:
                    data = read_object(f'{vb}.pkl', dir_data)

                    if base == 'foret_landcover':
                        encoder = read_object('encoder_foret.pkl', dir_encoder)
                    elif base == 'dynamic_world_landcover':
                        encoder = read_object('encoder_landcover.pkl', dir_encoder)
                    else:
                        encoder = None

                    if encoder is not None:
                        values = data[~np.isnan(raster)].reshape(-1)
                        data[~np.isnan(raster)] = encoder.transform(values).values.reshape(-1)

                    if len(data.shape) > 2:
                        data = np.sum(data, axis=2)

                    values.append(data[~np.isnan(raster)].reshape(-1, 1))

                if vb != 'geometry':
                    check_and_create_path(path / 'features_geometry' / dept)
                    plt.figure(figsize=(15,15))
                    plt.imshow(data)
                    plt.title(vb)
                    plt.savefig(path / 'features_geometry' / dept / f'{vb}.png')
                    plt.close('all')

            values = np.concatenate(values, axis=1)
            min_cluster_size = 6 * self.scale
            hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                                    cluster_selection_method='leaf', min_samples=6, cluster_selection_epsilon=1.0)
            self.clusterer[dept]['algo'] = hdbscan
            self.clusterer[dept]['node_already_predicted'] = node_already_predicted
            pred[~np.isnan(raster)] = hdbscan.fit_predict(values)

            # Avoid bad prediction
            if -1 in np.unique(pred):

                valid_mask = pred != -1
                invalid_mask = pred == -1

                valid_coords = np.column_stack(np.where(valid_mask))
                valid_values = pred[valid_mask]

                # Coordonnées des pixels à remplacer
                invalid_coords = np.column_stack(np.where(invalid_mask))

                # Utilisation de KNeighbors pour prédire les valeurs manquantes
                knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
                knn.fit(valid_coords, valid_values)

                predicted_values = knn.predict(invalid_coords)

                pred[invalid_mask] = predicted_values

            pred[~np.isnan(raster)] = mode_filter(pred, 9)[~np.isnan(raster)].astype(float)
            pred[~np.isnan(raster)] = merge_small_clusters(pred, min_cluster_size)[~np.isnan(raster)].astype(float)
            
            X = np.unique(raster)
            X = X[~np.isnan(X)]
            for node in X:
                mask_node = self.oriIds == node
                mask2 = raster == node
                self.ids[mask_node] = pred[mask2]

            new_ids = relabel_clusters(self.ids[mask][~np.isnan(self.ids[mask])], node_already_predicted)

            self.ids[mask & ~np.isnan(self.ids)] = new_ids

            # Incorporate nan hexagone
            if True in np.isnan(self.ids[mask]):

                valid_mask = ~np.isnan(self.ids[mask])
                invalid_mask = np.isnan(self.ids[mask])

                valid_coords = np.column_stack(np.where(valid_mask))
                valid_values = self.ids[mask][valid_mask]

                # Coordonnées des pixels à remplacer
                invalid_coords = np.column_stack(np.where(invalid_mask))

                # Utilisation de KNeighbors pour prédire les valeurs manquantes
                knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
                knn.fit(valid_coords, valid_values)

                predicted_values = knn.predict(invalid_coords)

                self.ids[np.where(mask)[0][invalid_mask]] = predicted_values.astype(int)

            current_cluster = np.unique(self.ids[mask][~np.isnan(self.ids[mask])]).shape[0]
            logger.info(f'Unique cluster : {np.unique(self.ids[mask])}, {current_cluster}')
            node_already_predicted += current_cluster
            self.numCluster += current_cluster
        
        self.oriIds = self.oriIds[~np.isnan(self.ids)]
        self.oriLatitues = self.oriLatitues[~np.isnan(self.ids)].reset_index(drop=True)
        self.oriGeometry = self.oriGeometry[~np.isnan(self.ids)]
        self.oriLongitude = self.oriLongitude[~np.isnan(self.ids)].reset_index(drop=True)
        self.departements = self.departements[~np.isnan(self.ids)].reset_index(drop=True)
        self.ids = self.ids[~np.isnan(self.ids)].astype(int)
        self.oriLen = self.oriLatitues.shape[0]

        if doRaster:
            self._raster(path=path, sinister=sinister, base=base, resolution=resolution)

    def _predict_node_with_position(self, array : list, depts = None) -> np.array:
        assert self.clusterer is not None
        if depts is None:
            array2 = np.empty((np.asarray(array).shape[0], 4))
            array2[:, 1:3] = np.asarray(array).reshape(-1,2)
            depts = self._assign_department(array2)[:, -1]

        array = np.asarray(array)
        udept = np.unique(depts[~np.isnan(depts)])
        res = np.full(len(array), fill_value=np.nan)
        for dept in udept:
            if dept is None:
                continue
            mask = np.argwhere(depts == dept)[:, 0]
            x = array[mask]
            if self.base == 'geometry':
                res[mask] = self.clusterer[int2name[dept]]['algo'].predict(x) + self.clusterer[int2name[dept]]['node_already_predicted']
            else:
                mask_ori = self.nodes[:, 3] == dept
                existing_data = list(zip(self.nodes[mask_ori, 1], self.nodes[mask_ori, 2]))
                existing_ids = self.nodes[mask_ori, 0]
                nbrs = NearestNeighbors(n_neighbors=1).fit(existing_data)
                ux = np.unique(x, axis=0)
                neighbours = nbrs.kneighbors(ux, return_distance=False)
                for i, xx in enumerate(ux):
                    neighbours_x = neighbours[i][0]
                    mask_x = np.argwhere((array[:, 0] == xx[0]) & (array[:, 1] == xx[1]))[:, 0]
                    m1 = self.nodes[mask_ori, 1] == existing_data[neighbours_x][0]
                    m2 = self.nodes[mask_ori, 2] == existing_data[neighbours_x][1]
                    m3 = m1 & m2
                    res[mask_x] = existing_ids[m3]
        return res


    def _predict_node_with_features(self, dept: str, path: Path, resolution: str,
                                    dir_raster : Path, dir_target : Path,
                                    dir_target_bin : Path, dir_data : Path, dir_encoder : Path,
                                    geo : gpd.GeoDataFrame):
        """
        Predicts clusters (or nodes) for a given department using the data base defined in self.base.
        """
        vec_base = [self.base] if self.base.find('-') == -1 else self.base.split('-')
        
        logger.info(f'Predicting clusters for department {dept} using bases {vec_base}')
        geo['scale0'] = geo.index + self.oriIds.max()
        geo['departement'] = dept

        # Load raster data
        #raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)[0]
        raster = None
        if raster is None:
            raster, _, _ = rasterization(geo, resolutions[resolution]['y'], resolutions[resolution]['x'], f'scale0', Path('log'), 'ori')
            raster = raster[0]
            save_object(raster, f'{dept}rasterScale0.pkl', dir_raster)
        pred = np.full(raster.shape, fill_value=np.nan)
        values = []

        # Loop through each base in vec_base
        for vb in vec_base:
            if vb == 'risk':
                # Load and process risk data
                data = read_object(f'{dept}Influence.pkl', dir_target)
                data = np.nansum(data, axis=2)
                values.append(data[~np.isnan(raster)].reshape(-1, 1))
            elif vb == 'nbsinister':
                # Load and process nbsinister data
                data = read_object(f'{dept}binScale0.pkl', dir_target_bin)
                data = np.nansum(data, axis=2)
                values.append(data[~np.isnan(raster)].reshape(-1, 1))
            elif vb == 'geometry':
                # Process geometry data
                coords = np.indices((raster.shape[0], raster.shape[1]))[:, ~np.isnan(raster)].reshape(2, -1).T
                values.append(coords)
            else:
                # Load and process other types of data
                data = read_object(f'{vb}.pkl', dir_data)

                # Apply encoder if needed
                if self.base == 'foret_landcover':
                    encoder = read_object('encoder_foret.pkl', dir_encoder)
                elif self.base == 'dynamic_world_landcover':
                    encoder = read_object('encoder_landcover.pkl', dir_encoder)
                else:
                    encoder = None

                if encoder is not None:
                    values_data = data[~np.isnan(raster)].reshape(-1)
                    data[~np.isnan(raster)] = encoder.transform(values_data).values.reshape(-1)

                if len(data.shape) > 2:
                    data = np.sum(data, axis=2)

                values.append(data[~np.isnan(raster)].reshape(-1, 1))

        # Concatenate all values
        values = np.concatenate(values, axis=1)

        # Apply PCA if the number of features is greater than 5
        if values.shape[1] > 5:
            values = MinMaxScaler().fit_transform(values)
            values = self.pca_geometries.git_trasnform(values)

        # Perform clustering using HDBSCAN
        min_cluster_size = 6 * self.scale
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                        cluster_selection_method='leaf', min_samples=6)
        pred[~np.isnan(raster)] = hdbscan.fit_predict(values)

        # Post-processing of the clusters to handle invalid predictions
        if -1 in np.unique(pred):
            valid_mask = (pred != -1) & (~np.isnan(pred))
            invalid_mask = pred == -1

            valid_coords = np.column_stack(np.where(valid_mask))
            valid_values = pred[valid_mask]

            invalid_coords = np.column_stack(np.where(invalid_mask))

            knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
            knn.fit(valid_coords, valid_values)

            predicted_values = knn.predict(invalid_coords)

            pred[invalid_mask] = predicted_values

        # Further processing to filter and merge small clusters
        pred[~np.isnan(raster)] = mode_filter(pred, 9)[~np.isnan(raster)].astype(float)
        pred[~np.isnan(raster)] = merge_small_clusters(pred, min_cluster_size)[~np.isnan(raster)].astype(float)

        res = np.full(len(geo), fill_value=np.nan)
        for index, row in geo.iterrows():
            id0 = row['scale0']
            idscale = pred[raster == id0]
            if len(idscale) == 0:
                continue
            res[index] = idscale[0]

        res[~np.isnan(res)] = relabel_clusters(res[~np.isnan(res)], self.numCluster)
        self.oriIds = np.concatenate([self.oriIds, geo.scale0.values])
        self.oriLatitues = pd.concat([self.oriLatitues, geo.latitude]).reset_index(drop=True)
        self.oriGeometry = np.concatenate([self.oriGeometry, geo.geometry.values])
        self.oriLongitude = pd.concat([self.oriLongitude, geo.longitude]).reset_index(drop=True)
        self.departements = pd.concat([self.departements, geo.departement]).reset_index(drop=True)
        self.ids = np.concatenate([self.ids, res])

        self.oriIds = self.oriIds[~np.isnan(self.ids)]
        self.oriLatitues = self.oriLatitues[~np.isnan(self.ids)].reset_index(drop=True)
        self.oriGeometry = self.oriGeometry[~np.isnan(self.ids)]
        self.oriLongitude = self.oriLongitude[~np.isnan(self.ids)].reset_index(drop=True)
        self.departements = self.departements[~np.isnan(self.ids)].reset_index(drop=True)
        self.ids = self.ids[~np.isnan(self.ids)].astype(int)
        self.oriLen = self.oriLatitues.shape[0]

        self.numCluster += np.unique(res[~np.isnan(res)]).shape[0]

        maskDept = np.argwhere(self.departements == dept)
        geo2 = gpd.GeoDataFrame(index=np.arange(maskDept.shape[0]), geometry=self.oriGeometry[maskDept[:,0]])
        geo2[f'scale{self.scale}'] = self.ids[maskDept]
        mask, _, _ = rasterization(geo2, resolutions[resolution]['y'], resolutions[resolution]['x'], f'scale{self.scale}', Path('log'), 'ori')
        mask = mask[0]
        logger.info(f'{dept, mask.shape, raster.shape}')
        if mask.shape != raster.shape:
            mask = resize_no_dim(mask, raster.shape[0], raster.shape[1])

        mask[np.isnan(raster)] = np.nan

        save_object(mask, f'{dept}rasterScale{self.scale}_{self.base}.pkl', path)

        self._create_nodes_list()
        self._create_edges_list()

        return res

    def _raster2(self, path, n_pixel_y, n_pixel_x, resStr, doBin, base, sinister, dataset_name):

        dir_bin = root_target / dataset_name / sinister / 'bin' / resStr
        dir_target = root_target / dataset_name / sinister / 'log' / resStr
        dir_raster = root_target / dataset_name / sinister / 'raster' / resStr

        for dept in np.unique(self.departements):

            maskDept = np.argwhere(self.departements == dept)
            geo = gpd.GeoDataFrame(index=np.arange(maskDept.shape[0]), geometry=self.oriGeometry[maskDept[:,0]])
            geo[f'scale{self.scale}'] = self.ids[maskDept]
            mask, _, _ = rasterization(geo, n_pixel_y, n_pixel_x, f'scale{self.scale}', Path('log'), 'ori')
            mask = mask[0]
            outputName = f'{dept}rasterScale0.pkl'
            raster = read_object(outputName, dir_raster)[0]
            logger.info(f'{dept, mask.shape, raster.shape}')
            mask[np.isnan(raster)] = np.nan

            save_object(mask, f'{dept}rasterScale{self.scale}_{base}.pkl', path / 'raster')

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
            plt.savefig(path / 'raster' / f'{dept}_{self.scale}_{self.base}.png')
            plt.close('all')

            if doBin:
                outputName = dept+'binScale0.pkl'
                bin = read_object(outputName, dir_bin)
                if bin is None:
                    continue
                outputName = dept+'Influence.pkl'
                influence = read_object(outputName, dir_target)

                binImageScale, influenceImageScale = create_larger_scale_bin(mask, bin, influence, raster)
                save_object(binImageScale, f'{dept}binScale{self.scale}_{base}.pkl', path / 'bin')
                save_object(influenceImageScale, f'{dept}InfluenceScale{self.scale}_{base}.pkl', path / 'influence')

    def _raster(self, path : Path,
                sinister : str,
                resolution : str,
                base: str) -> None:
        """"
        Create new raster mask
        """
        
        check_and_create_path(path)
        check_and_create_path(path / 'raster')
        check_and_create_path(path / 'bin')
        check_and_create_path(path / 'proba')

        # 1 pixel = 1 Hexagone (2km) use for 1D process

        self._raster2(path, resolutions[resolution]['y'], resolutions[resolution]['x'], resolution, True, base, sinister)

        """# 1 pixel = 1 km, use for 2D process Not suitable for now
        n_pixel_y = 0.009
        n_pixel_x = 0.012
        resStr = '1x1'
        self._raster2(path, n_pixel_y, n_pixel_x, resStr, False, sinister)"""

    def _read_kmeans(self, path : Path) -> None:
        assert self.scale != 0
        self.train_kmeans = False
        name = 'kmeans'+str(self.scale)+'.pkl'
        self.numCluster = self.oriLen // (self.scale * 6)
        X = list(zip(self.oriLongitude, self.oriLatitues))
        self.clusterer = pickle.load(open(path / name, 'rb'))
        self.ids = self.clusterer.predict(X)

    def _create_predictor(self, start, end, dir, sinister, dataset_name, resolution):
        dir_influence = root_graph / dir / 'influence'
        dir_predictor = root_graph / dir / 'influenceClustering'
        dir_bin = root_graph / dir / 'bin'

        # Scale Shape
        logger.info(f'######### {self.scale} scale ##########')
        for dep in np.unique(self.departements):
            influence = read_object(f'{dep}InfluenceScale{self.scale}_{self.base}.pkl', dir_influence)
            binValues = read_object(f'{dep}binScale{self.scale}_{self.base}.pkl', dir_bin)
            if influence is None:
                continue
            influence = influence[:,:,allDates.index(start):allDates.index(end)]
            binValues = binValues[:,:,allDates.index(start):allDates.index(end)]
            values = np.unique(influence[~np.isnan(influence)])
            if values.shape[0] >= 5:
                predictor = Predictor(5, name=dep)
                predictor.fit(np.asarray(values))
            else:
                predictor = Predictor(values.shape[0], name=dep)
                predictor.fit(np.asarray(values))
            predictor.log(logger)
            check_class(influence, binValues)
            save_object(predictor, f'{dep}Predictor{self.scale}_{self.base}.pkl', path=dir_predictor)

        logger.info('######### Departement Scale ##########')
        dir_target = root_target / dataset_name / sinister / 'log' / resolution
        dir_bin = root_target / dataset_name / sinister / 'bin' / resolution
        if (dir_predictor / f'{dep}PredictorDepartement.pkl').is_file():
            return
        # Departement
        for dep in np.unique(self.departements):
            binValues = read_object(dep+'binScale0.pkl', dir_bin)
            influence = read_object(dep+'Influence.pkl', dir_target)
            if influence is None:
                continue
            influence = influence[:,:,allDates.index(start):allDates.index(end)]
            binValues = binValues[:,:,allDates.index(start):allDates.index(end)]
            influence = influence.reshape(-1, influence.shape[2])
            binValues = binValues.reshape(-1, binValues.shape[2])
            influence = np.nansum(influence, axis=0)
            binValues = np.nansum(binValues, axis=0)
            #influence = influence / np.max(influence)
            if influence.shape[0] >= 5:
                predictor = Predictor(5, name=dep)
                predictor.fit(np.asarray(influence))
            else:
                predictor = Predictor(influence.shape[0], name=dep)
                predictor.fit(np.asarray(influence))
            check_class(influence, binValues)
            predictor.log(logger)
            save_object(predictor, f'{dep}PredictorDepartement.pkl', path=dir_predictor)
        
    def _create_nodes_list(self) -> None:
        """
        Create nodes list in the form (N, 4) where N is the number of Nodes [ID, longitude, latitude, departement of centroid]
        """
        logger.info('Creating node list')
        self.nodes = np.full((self.numCluster, 5), np.nan) # [id, longitude, latitude]
        self.uniqueIDS = np.unique(self.ids)
        for id in self.uniqueIDS:
            mask = np.argwhere(id == self.ids)
            if self.scale != 0:
                nodeGeometry = unary_union(self.oriGeometry[mask[:,0]]).centroid
            else:
                nodeGeometry = self.oriGeometry[mask[:,0]].centroid
            valuesDep, counts = np.unique(self.departements[mask[:,0]], return_counts=True)
            ind = np.argmax(counts)
            self.nodes[id] = np.asarray([id, float(nodeGeometry.x), float(nodeGeometry.y), name2int[valuesDep[ind]], -1])
        
    def _create_edges_list(self) -> None:
        """
        Create edges list
        """
        logger.info('Creating edges list')
        self.edges = None # (2, E) where E is the number of edges in the graph
        src_nodes = []
        target_nodes = []
        if self.uniqueIDS.shape[0] == 1:
            self.edges = []
        for id in self.uniqueIDS:
            closest = self._distances_from_node(self.nodes[id], self.nodes[self.nodes[:,0] != id])
            closest = closest[closest[:, 1].argsort()]
            for i in range(min(self.numNei, len(closest))):
                if closest[i, 1] > self.maxDist:
                    break
                src_nodes.append(id)
                target_nodes.append(closest[i,0])

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
        ignore = [node[0]]
        for index, (id, lon, lat, _, _) in enumerate(otherNodes):
            if id in ignore:
                continue
            distance = haversine((node[1], node[2]), [lon, lat])
            res[index, 0 ] = id
            res[index, 1] = distance
            ignore.append(id)
        return res
    
    def _assign_department(self, nodes : np.array) -> np.array:
        logger.info('Get node department')
        uniqueDept = np.unique(self.departements)
        dico = {}
        for key in uniqueDept:
            mask = np.argwhere(self.departements.values == key)
            geoDep = unary_union(self.oriGeometry[mask[:,0]])
            dico[key] = geoDep

        def find_dept(x, y, dico):
            for key, value in dico.items():
                if value.contains(Point(x, y)):
                    return name2int[key]
 
            return np.nan

        nodes[:,3] = np.asarray([find_dept(node[1], node[2], dico) for node in nodes])

        return nodes
    
    def _assign_latitude_longitude(self, nodes : np.array) -> np.array:
        logger.info('Assign latitude longitude')

        uniqueNode = np.unique(nodes[:,0])
        for uN in uniqueNode:
            maskNode = np.argwhere(nodes[:,0] == uN)
            if maskNode.shape[0] == 0:
                continue
            nodes[maskNode[:,0],1:3] = self.nodes[np.argwhere(self.nodes[:,0] == uN)[:,0]][0][1:3]
        return nodes

    def _plot(self, nodes : np.array, time=0, dir_output = None) -> None:
        """
        Plotting
        """
        dis = np.unique(nodes[:,4])[:time+1].astype(int)
        if time > 0:
            ax = plt.figure(figsize=(15,15)).add_subplot(projection='3d')
        else:
            _, ax = plt.subplots(figsize=(15,15))
        if self.edges is not None:
            uniqueIDS = np.unique(nodes[:,0]).astype(int)
            for id in uniqueIDS:
                connection = self.edges[1][np.argwhere((self.edges[0] == id) & (np.isin(self.edges[1], nodes[:,0])))[:,0]]
                for target in connection:
                    target = int(target)
                    if target in uniqueIDS:
                        if time > 0:
                            for ti in range(time + 1):
                                ax.plot([dis[ti], dis[ti]],
                                        [nodes[nodes[:,0] == id][0][1], nodes[nodes[:,0] == target][0][1]],
                                        [nodes[nodes[:,0] == id][0][2], nodes[nodes[:,0] == target][0][2]],
                                    color='black')
                        else:
                            ax.plot([nodes[nodes[:,0] == id][0][1], nodes[nodes[:,0] == target][0][1]],
                                    [nodes[nodes[:,0] == id][0][2], nodes[nodes[:,0] == target][0][2]], color='black')
        if time > 0:
            if self.temporalEdges is not None:
                nb = 0
                uniqueDate = np.unique(nodes[:,4]).astype(int)
                uniqueIDS = np.unique(nodes[:,0]).astype(int)
                for di in uniqueDate:
                    connection = self.temporalEdges[1][np.argwhere((self.temporalEdges[0] == di))[:,0]]
                    for target in connection:
                        target = int(target)
                        if target in uniqueDate:
                            for id in uniqueIDS:
                                ax.plot([di, target],
                                         [nodes[nodes[:,0] == id][0][1], nodes[nodes[:,0] == id][0][1]],
                                         [nodes[nodes[:,0] == id][0][2], nodes[nodes[:,0] == id][0][2]],
                                        color='red')
                    nb += 1
                    if nb == time:
                        break
        
        node_x = nodes[nodes[:,4] == dis[0]][:,1]
        node_y = nodes[nodes[:,4] == dis[0]][:,2]
        for ti in range(1, time+1):
            node_x = np.concatenate((node_x, nodes[nodes[:,4] == dis[ti]][:,1]))
            node_y = np.concatenate((node_y, nodes[nodes[:,4] == dis[ti]][:,2]))
   
        if time > 0:
            node_z = nodes[nodes[:,4] == dis[0]][:,4]
            for ti in range(1, time+1):
                node_z = np.concatenate((node_z, nodes[nodes[:,4] == dis[ti]][:,4]))

            ax.scatter(node_z, node_x, node_y)
            ax.set_xlabel('Date')
            ax.set_ylabel('Longitude')
            ax.set_zlabel('Latitude')
        else:
            node_x = nodes[:,1]
            node_y = nodes[:,2]
            for i, n in enumerate(nodes):
                ax.annotate(str(n[0]), (n[1] + 0.001, n[2] + 0.001))
            ax.scatter(x=node_x, y=node_y, s=60)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        if dir_output is not None:
            name = f'graph_{self.scale}_{self.base}.png'
            plt.savefig(dir_output / name)

    def _set_model(self, model) -> None:
        """
        model : the neural network model. Train or not
        """
        self.model = model

    def _load_model_from_path(self, path : Path, model : torch.nn.Module, device) -> None:
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
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

    def _predict_dataset(self, X : Dataset) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            test_loader = DataLoader(X, 1)
            for i, data in enumerate(test_loader, 0):
                inputs, _, edges = data
                output = self.model(inputs, edges)
                return output
        
    def _predict_test_loader(self, X : DataLoader, features : np.array, device, target_name : str, autoRegression : bool, features_name : dict) -> torch.tensor:
        assert self.model is not None
        self.model.eval()

        with torch.no_grad():
            pred = []
            y = []

            for i, data in enumerate(X, 0):
                inputs, labels, edges = data
                inputs = inputs[:, features]

                inputs = inputs.to(device)
                labels = labels.to(device)
                edges = edges.to(device)

                weights = labels[:,-4].to(torch.long)

                output = self.model(inputs, edges)

                ids = torch.masked_select(labels[:,0], weights.gt(0))
                target = torch.empty((ids.shape[0], labels.shape[1]), device=device)
                target[:,0] = ids
                for b in range(1, labels.shape[1]):
                    target[:,b] = torch.masked_select(labels[:,b], weights.gt(0))

                #target = labels[weights.gt(0)]
                #target = target[:, -1]

                if target_name == 'binary':
                    output = output[:, 1]

                output = torch.masked_select(output, weights.gt(0))
                weights = torch.masked_select(weights, weights.gt(0))  

                pred.append(output)
                y.append(target)
            
            y = torch.cat(y, 0)
            pred = torch.cat(pred, 0).view(y.shape[0])

            return pred, y
        
    def predict_model_api_sklearn(self, X : pd.DataFrame,
                                  features : list, isBin : bool,
                                  autoRegression : bool) -> np.array:
        assert self.model is not None
        if not autoRegression:
            if not isBin:
                return self.model.predict(X[features])
            else:
                return self.model.predict_proba(X[features])[:,1]
        else:
            pred = np.empty(X.shape[0])
            uid = np.unique(X[:, 0])
            for id in uid:
                udate = np.unique(X[:, 4])
                for d in udate:
                    mask = np.argwhere((X[:, 0] == id) & (X[:, 4] == d))[:,0]
                    if mask.shape[0] == 0:
                        continue
                    if not isBin:
                        preddd = self.model.predict(X[mask, features].reshape(-1,features.shape[0]))
                    else:
                        preddd = self.model.predict_proba(X[mask, features].reshape(-1, features.shape[0]))[:,1]
                    
                    pred[mask] = preddd
                    maskNext = np.argwhere((X[:, 0] == id) & (X[:, 4] == d + 1))
                    if maskNext.shape[0] == 0:
                        continue
                    else:
                        if not isBin:
                            X[maskNext, features_name.index('AutoRegressionReg')] = preddd
                        else:
                            X[maskNext, features_name.index('AutoRegressionBin')] = preddd
            return pred

    def _predict_perference_with_Y(self, Y : np.array, isBin : bool) -> np.array:
        res = np.empty(Y.shape[0])
        res[0] = 0
        for i, node in enumerate(Y):
            if i == 0:
                continue
            index = np.argwhere((Y[:,4] == node[4] - 1) & (Y[:,0] == node[0]))
            if index.shape[0] == 0:
                res[i] = 0
                continue
            if not isBin: 
                res[i] = Y[index[0], -1]
            else:
                res[i] = Y[index[0], -2] > 0
        return res
        
    def _predict_perference_with_X(self, X : np.array, features_name : dict) -> np.array:
        return X[:, features_name.index('Historical') + 2]

    def _info_on_graph(self, nodes : np.array, output : Path) -> None:
        """
        logger.info informatio on current global graph structure and nodes
        """
        check_and_create_path(output)
        graphIds = np.unique(nodes[:,3])

        for graphid in graphIds:
            logger.info(f'size : {graphid, self.nodes[self.nodes[:,3] == graphid].shape[0]}')

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
            logger.info(f'We found {np.unique(nodes[:,4]).shape[0]} different graphs in the training set for {nodes.shape[0]} nodes')

        """logger.info(f'Mean number of nodes in sub graph : {np.mean(size)}')
        logger.info(f'Max number of nodes in subgraph {np.max(size)}')
        logger.info(f'Minimum number of nodes in subgraph {np.min(size)}')"""