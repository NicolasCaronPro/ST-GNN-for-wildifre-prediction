
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging

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

from GNN.graph_structure import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging
import json
import geopandas as gpd

from GNN.graph_structure import GraphStructure
from GNN.construct import construct_graph, parse_string
from GNN.tools import allDates, save_object, read_object
from GNN.arborescence import root_graph, root_target, rootDisk
from category_encoders import TargetEncoder, CatBoostEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import GNN.array_fet as fet
from GNN.arborescence import root_target, rootDisk
from skimage.transform import resize
from tslearn.clustering import TimeSeriesKMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TargetEncoder:
    def __init__(self, cols, smooth=0):
        self.cols = cols
        self.smooth = smooth
        self.mapping = {}
        self.global_mean = 0

    def fit(self, X, y, cat_indices):
        # X: (N, C) numpy array
        # y: (N,) numpy array (binary target)
        # cat_indices: list of indices of categorical columns in X
        
        self.global_mean = np.mean(y)
        
        for col_idx in cat_indices:
            self.mapping[col_idx] = {}
            values = X[:, col_idx]
            unique_values = np.unique(values)
            
            for val in unique_values:
                mask = (values == val)
                mean_target = np.mean(y[mask])
                # Smooth
                # smoothed = (count * mean + weight * global) / (count + weight)
                # For now simple mean
                self.mapping[col_idx][val] = mean_target
                
    def transform(self, X, cat_indices):
        # X: (N, C) numpy array
        X_out = X.copy()
        
        for col_idx in cat_indices:
            if col_idx not in self.mapping:
                continue
                
            values = X[:, col_idx]
            mapping = self.mapping[col_idx]
            
            # Vectorized map
            # Use np.vectorize or a loop if values are not too many
            # Or use pandas map if we convert
            # Let's use a loop over unique values in input to minimize operations
            unique_vals = np.unique(values)
            for val in unique_vals:
                if val in mapping:
                    X_out[values == val, col_idx] = mapping[val]
                else:
                    X_out[values == val, col_idx] = self.global_mean
                    
        return X_out

class DataLoader:
    def __init__(self, config_parser):
        self.config = config_parser

        # Construct feature list from GNN.array_fet
        self.features_to_use = []
        # Exclude: varying_time_variables, calendar_variables, geo_variables, region_variables
        
        # Add lists
        self.features_to_use.extend(fet.cems_variables)
        self.features_to_use.extend(fet.air_variables)
        self.features_to_use.extend(fet.sentinel_variables)
        self.features_to_use.extend(fet.landcover_variables)
        self.features_to_use.extend(fet.cluster_encoder)
        
        # Map foret_variables using foretint2str to get datacube column names
        from GNN.config import foretint2str
        for key in fet.foret_variables:
            if key in foretint2str:
                self.features_to_use.append(foretint2str[key])
                
        self.features_to_use.extend(fet.bdroute_variables) 

        self.features_to_use.extend(fet.elevation_variables)
        self.features_to_use.extend(fet.population_variabes)
        self.features_to_use.extend(fet.vigicrues_variables)
        self.features_to_use.extend(fet.nappes_variables)
        
        # Check if they are already covered by 'foret_encoder' etc.
        # If not, add them.
        self.features_to_use.append('forest_landcover')
        self.features_to_use.append('corine_landcover')
            
        # Ensure unique
        self.features_to_use = list(dict.fromkeys(self.features_to_use))
        
        self.target_variable = self.config.get_target_variable()
        self.frequency = self.config.get_frequency()
        self.scaler = StandardScaler()
        
        # Target Encoding
        self.cat_cols = ['forest_landcover', 'corine_landcover'] 
        self.forest_encoder = TargetEncoder(['forest_landcover'])
        self.corine_encoder = TargetEncoder(['corine_landcover'])
        
        self.target_type = self.config.get_target_type()
        self.cluster_model = None
        self.cluster_mapping = None
        self.ordinal_encoder = None

    def launch_segmentation(self, train_depts):
        """
        Launch the segmentation process using construct_graph.
        """
        logger.info("Launching segmentation...")
        
        # Configuration parameters
        scale = self.config.get_scale()
        dataset_name = self.config.get_dataset_name()
        graph_construct = self.config.get_graph_construct()

        assert graph_construct is not None, "Graph construct must be specified in config"
        assert scale is not None, "Scale must be specified in config"
        assert dataset_name is not None, "Dataset name must be specified in config"
        
        # Arguments for construct_graph
        maxDist = 100
        sinister = 'firepoint'
        sinister_encoding = 'occurence'
        nmax = 1
        k_days = 0
    
        dico_config = parse_string(graph_construct)
        
        n_clusters_node = self.config.get_n_clusters_node()
 
        dir_output = Path.cwd() / Path('Experiments') / f'target_{self.target_variable}_{self.config.get_target_type()}_scale_{scale}_tol_{dico_config["tol"]}_attempt_{dico_config["attempt"]}_reduce_{dico_config["reduce"]}_ncluster_{n_clusters_node}'
        doRaster = True
        doEdgesFeatures = False
        resolution = '2x2'
        graph_method = 'node'
        
        # Departments
        departements = self.config.config.get('train_departements', []) + \
                       self.config.config.get('test_departements', []) + \
                       self.config.config.get('new_test_departements', [])
        departements = list(set(departements)) # Unique
        
        # Geo
        geo_path = f'{root_graph}/regions/{sinister}/{dataset_name}/regions.geojson'
        try:
            geo = gpd.read_file(geo_path)
        except Exception as e:
            logger.error(f"Failed to load geojson from {geo_path}: {e}")
            raise e

        geo = geo[geo['departement'].isin(departements)].reset_index(drop=True)

        if not (dir_output / 'gs.pkl').is_file() and self.config.get_graph_flag():
            # Call construct_graph
            self.gs = construct_graph(
                scale=scale,
                maxDist=maxDist,
                sinister=sinister,
                dataset_name=dataset_name,
                sinister_encoding=sinister_encoding,
                train_departements=train_depts,
                departements=departements,
                geo=geo,
                nmax=nmax,
                k_days=k_days,
                dir_output=dir_output,
                doRaster=doRaster,
                doEdgesFeatures=doEdgesFeatures,
                resolution=resolution,
                graph_construct=graph_construct,
                train_dates=allDates,
                val_date=None,
                graph_method=graph_method,
                test_departements=self.config.config.get('test_departements', []) + \
                       self.config.config.get('new_test_departements', []),
                n_clusters_node=self.config.get_n_clusters_node()
            )
            save_object(self.gs, 'gs.pkl', dir_output)
        else:
            self.gs = read_object('gs.pkl', dir_output)
            
        if (dir_output / 'label_encoder.pkl').exists() and \
           (dir_output / 'ordinal_encoder.pkl').exists() and \
           (dir_output / 'forest_encoder.pkl').exists() and \
           (dir_output / 'corine_encoder.pkl').exists():
        
            logger.info("Loading existing encoders...")
            with open(dir_output / 'label_encoder.pkl', 'rb') as f:
                self.encoder = pickle.load(f)
            with open(dir_output / 'ordinal_encoder.pkl', 'rb') as f:
                self.ordinal_encoder = pickle.load(f)
            with open(dir_output / 'forest_encoder.pkl', 'rb') as f:
                self.forest_encoder = pickle.load(f)
            with open(dir_output / 'corine_encoder.pkl', 'rb') as f:
                self.corine_encoder = pickle.load(f)
            logger.info("Encoders loaded.")
            return

        self.fit_encoders(train_depts, dir_output, scale, graph_construct, graph_method, n_clusters_node)

    def fit_encoders(self, train_depts, dir_output, scale, graph_construct, graph_method, n_clusters_node):
        logger.info("Segmentation completed. Fitting encoder on training data...")
        
        # Fit Encoder on Training Data
        # We need to load the datacubes for training departments and extract 'time_series_clustering'
        
        all_clusters = []
        all_targets = []
        
        for dept in train_depts:
            # Path: dir_output / 'datacube' / f'datacube_target_{dept}_{scale}_{graph_construct}_{graph_method}.pkl'
            datacube_path = dir_output / 'datacube' / f'datacube_target_{dept}_{scale}_{graph_construct}_{graph_method}.pkl'
            
            if not datacube_path.exists():
                logger.warning(f"Datacube not found for {dept} at {datacube_path}")
                continue
                
            with open(datacube_path, 'rb') as f:
                datacube = pickle.load(f)
                
            if 'time_series_clustering' in datacube:
                clusters = datacube['time_series_clustering'].values

                # Extract target variable
                if self.target_variable in datacube:
                    target_values = datacube["occurence"].values

                    # Handle dimensions
                    # If (dept, lat, lon, date), take [0]
                    if len(target_values.shape) == 4:
                        target_values = target_values[0]
                        
                    # Now (lat, lon, date)
                    # Sum over time (last axis)
                    # Handle NaNs
                    target_sum = np.nansum(target_values, axis=-1)
                    
                    # Flatten and remove NaNs (using cluster mask)
                    clusters_flat = clusters.flatten()
                    target_flat = target_sum.flatten()
                    
                    mask = ~np.isnan(clusters_flat)
                    
                    if len(clusters_flat) != len(target_flat):
                         logger.error(f"Shape mismatch: clusters {clusters.shape}, target {target_sum.shape}")
                         continue
                    
                    all_clusters.append(clusters_flat[mask])
                    all_targets.append(target_flat[mask])
                else:
                     logger.warning(f"Target variable {self.target_variable} not found in datacube for {dept}")
            else:
                logger.warning(f"'time_series_clustering' column not found in datacube for {dept}")
                
        if not all_clusters:
            logger.error("No cluster data found to fit encoder.")
            return

        all_clusters = np.concatenate(all_clusters)
        all_targets = np.concatenate(all_targets)

        # Fit CatBoostEncoder
        # We assume the clusters are discrete IDs.
        self.encoder = CatBoostEncoder(cols=[0])
        self.encoder.fit(all_clusters.reshape(-1, 1), all_targets)
        
        # Transform clusters to get continuous values
        transformed_clusters = self.encoder.transform(all_clusters.reshape(-1, 1))

        # Fit Ordinal Encoder (Discretizer)
        self.ordinal_encoder = KBinsDiscretizer(n_bins=n_clusters_node, encode='ordinal', strategy='kmeans')
        self.ordinal_encoder.fit(transformed_clusters)

        transformed_clusters = self.ordinal_encoder.transform(transformed_clusters)

        logger.info(f"Encoder and Ordinal Encoder fitted.")

        # Fit TargetEncoders for landcover features
        logger.info("Fitting TargetEncoders for landcover features...")
        all_forest_features = []
        all_corine_features = []
        all_landcover_targets = []
        
        for dept in train_depts:
            # Load Target Datacube
            datacube_target_path = dir_output / 'datacube' / f'datacube_target_{dept}_{scale}_{graph_construct}_{graph_method}.pkl'
            if not datacube_target_path.exists():
                continue
                
            with open(datacube_target_path, 'rb') as f:
                datacube_target = pickle.load(f)
                
            # Load Feature Datacube
            resolution = '2x2'
            dir_data = rootDisk / 'csv' / dept / 'raster' / resolution
            datacube_feature_path = dir_data / 'datacube.pkl'
            
            if not datacube_feature_path.exists():
                continue
                
            with open(datacube_feature_path, 'rb') as f:
                datacube_feature = pickle.load(f)
                
            # Extract Target (occurence summed)
            if "occurence" in datacube_target:
                target_values = datacube_target["occurence"].values
                if len(target_values.shape) == 4:
                    target_values = target_values[0]
                target_sum = np.nansum(target_values, axis=-1)
                
                # Check if features exist
                has_forest = 'forest_landcover' in datacube_feature
                has_corine = 'corine_landcover' in datacube_feature
                
                if has_forest and has_corine:
                     f_forest = datacube_feature['forest_landcover'].values
                     f_corine = datacube_feature['corine_landcover'].values
                     
                     # Handle time dimension if present (take mode or first)
                     if len(f_forest.shape) == 3:
                         f_forest = f_forest[:, :, 0]
                     if len(f_corine.shape) == 3:
                         f_corine = f_corine[:, :, 0]
                         
                     # Flatten
                     f_forest_flat = f_forest.flatten()
                     f_corine_flat = f_corine.flatten()
                     t_flat = target_sum.flatten()
                     
                     # Mask NaNs
                     mask = ~np.isnan(t_flat)
                     
                     all_forest_features.append(f_forest_flat[mask])
                     all_corine_features.append(f_corine_flat[mask])
                     all_landcover_targets.append(t_flat[mask])
                     
        if all_landcover_targets:
            y_landcover = np.concatenate(all_landcover_targets, axis=0)
            
            if all_forest_features:
                X_forest = np.concatenate(all_forest_features, axis=0).reshape(-1, 1)
                self.forest_encoder.fit(X_forest, y_landcover, cat_indices=[0])
                logger.info("Forest Encoder fitted.")
                
            if all_corine_features:
                X_corine = np.concatenate(all_corine_features, axis=0).reshape(-1, 1)
                self.corine_encoder.fit(X_corine, y_landcover, cat_indices=[0])
                logger.info("Corine Encoder fitted.")
        else:
            logger.warning("No data found to fit TargetEncoders.")
            
        with open(dir_output / 'forest_encoder.pkl', 'wb') as f:
            pickle.dump(self.forest_encoder, f)
        with open(dir_output / 'corine_encoder.pkl', 'wb') as f:
            pickle.dump(self.corine_encoder, f)
            
        with open(dir_output / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.encoder, f)
        
        if hasattr(self, 'ordinal_encoder') and self.ordinal_encoder is not None:
            with open(dir_output / 'ordinal_encoder.pkl', 'wb') as f:
                pickle.dump(self.ordinal_encoder, f)

    def load_data_for_dept(self, dept, require_target=True):
        logger.info(f"Loading data for department {dept}...")

        graph_construct = self.config.get_graph_construct()
        assert graph_construct is not None, "Graph construct must be specified in config"
        
        dico_config = parse_string(graph_construct)
        
        # Configuration
        scale = self.config.config.get('scale', 0.3)
        graph_construct = self.config.config.get('graph_construct', self.config.config.get('graphConstruct', "risk-size-watershed-degree-a3-r4-t0.3"))
        graph_method = 'node'
        
        n_clusters_node = self.config.get_n_clusters_node()
        
        # Load Datacube Target (which contains target)
        dir_output = Path.cwd() / Path('Experiments') / f'target_{self.target_variable}_{self.config.get_target_type()}_scale_{scale}_tol_{dico_config["tol"]}_attempt_{dico_config["attempt"]}_reduce_{dico_config["reduce"]}_ncluster_{n_clusters_node}'
        datacube_target_path = dir_output / 'datacube' / f'datacube_target_{dept}_{scale}_{graph_construct}_{graph_method}.pkl'
        
        if not datacube_target_path.exists():
            logger.error(f"Target Datacube not found: {datacube_target_path}")
            return None, None, None, None

        with open(datacube_target_path, 'rb') as f:
            datacube_target = pickle.load(f)

        # Load Datacube Feature
        resolution = '2x2' # Hardcoded as per previous logic
        dir_data = rootDisk / 'csv' / dept / 'raster' / resolution
        datacube_feature_path = dir_data / 'datacube.pkl'
        
        if not datacube_feature_path.exists():
            logger.error(f"Feature Datacube not found: {datacube_feature_path}")
            return None, None, None, None
            
        with open(datacube_feature_path, 'rb') as f:
            datacube_feature = pickle.load(f)

        # Extract Features from Feature Datacube
        # Check available features
        loaded_features = [feat for feat in self.features_to_use if feat in datacube_feature]
        if not loaded_features:
            logger.error("No features found in feature datacube.")
            return None, None, None, None
            
        # Extract Data
        # We assume data is (H, W, T) or (H, W)
        # We need to aggregate over time if needed, or return time series?
        # The previous logic aggregated over time based on 'frequency'.
        # Let's replicate the aggregation logic but using this datacube.
        
        frequency = self.config.get_frequency()
        is_full = (frequency == 'full')
        freq_int = 1
        if not is_full:
            try:
                freq_int = int(frequency)
            except ValueError:
                is_full = True
                
        # Get dates
        dates = None
        if hasattr(datacube_feature, 'coords') and 'date' in datacube_feature.coords:
            dates = datacube_feature.coords['date'].values
            import pandas as pd
            dates = pd.to_datetime(dates)
            years = dates.year
            unique_years = np.unique(years)
        else:
            if not is_full:
                logger.warning("Datacube does not have 'date' coordinate. Defaulting to full.")
                is_full = True
                
        if is_full:
            periods = [None]
        else:
            unique_years = np.sort(unique_years)
            periods = []
            for i in range(0, len(unique_years), freq_int):
                chunk_years = unique_years[i:i+freq_int]
                indices = np.where(np.isin(years, chunk_years))[0]
                if len(indices) > 0:
                    periods.append(indices)
                    
        # Extract Target
        target_data = None
        if self.config.get_target_type() == 'cluster':
            if 'time_series_clustering' in datacube_target:
                target_data = datacube_target['time_series_clustering']

                if hasattr(target_data, 'values'):
                    target_data = target_data.values
                    
                # Apply encoder
                if hasattr(self, 'encoder'):
                    original_shape = target_data.shape
                    target_flat = target_data.flatten()
                    # Handle NaNs (background)
                    mask_valid = ~np.isnan(target_flat)
                    target_encoded = np.zeros_like(target_flat, dtype=float) - 1 # Initialize with -1.0
                    
                    if np.any(mask_valid):
                        try:
                            # CatBoostEncoder expects 2D input
                            transformed = self.encoder.transform(target_flat[mask_valid].reshape(-1, 1))
                            print(np.unique(transformed))
                            # Apply Ordinal Encoder
                            if self.ordinal_encoder is not None:
                                transformed = self.ordinal_encoder.transform(transformed)
                                # Shift by +1 so classes are 1, 2, 3, 4 (0 is background)
                                transformed += 1
                                
                            if hasattr(transformed, 'values'):
                                transformed = transformed.values
                            target_encoded[mask_valid] = transformed.flatten()
                        except ValueError:
                            target_encoded[mask_valid] = -1
                            
                    # target_encoded[mask_valid] += 1 # Removed for regression
                    target_encoded[~mask_valid] = 0
                    target_data = target_encoded.reshape(original_shape)

                    if len(target_data.shape) == 3:
                        target_data = target_data.squeeze(0)

        elif self.config.get_target_type() == 'risk':
            if 'influence' in datacube_target:
                target_data = datacube_target['influence']

                if hasattr(target_data, 'values'):
                    target_data = np.nansum(target_data.values, axis=-1)[0]

        elif self.config.get_target_type() == 'occurence':
            if 'occurence' in datacube_target:
                target_data = datacube_target['occurence']
                if hasattr(target_data, 'values'):
                    target_data = np.nansum(target_data.values, axis=-1)[0]

        elif self.config.get_target_type() == 'frontier':
            if 'area' in datacube_target:
                area_data = datacube_target['area']
                if hasattr(area_data, 'values'):
                    area_data = area_data.values
                
                # Assuming area_data is (H, W) or (1, H, W) or (H, W, 1)
                # We need (H, W) for ids_to_boundary_mask
                if len(area_data.shape) == 3:
                     area_data = area_data.squeeze()
                
                # Handle NaNs in area (convert to specific ID or handle in mask)
                # Here we assume area IDs are integers. NaNs might be present outside the department.
                # We can fill NaNs with a unique ID to treat them as a separate zone (or background)
                # But ids_to_boundary_mask expects int array.
                
                area_int = np.nan_to_num(area_data, nan=-1).astype(int)
                
                from LearnedSegmentation.frontier_utils import ids_to_boundary_mask
                from scipy.ndimage import distance_transform_edt
                
                boundary_mask = ids_to_boundary_mask(area_int)
                
                # Compute distance to nearest boundary (where boundary_mask == 1)
                # distance_transform_edt computes distance to nearest zero.
                # So we invert the mask: 0 on boundary, 1 elsewhere.
                inverted_mask = (boundary_mask == 0).astype(int)
                distance_map = distance_transform_edt(inverted_mask)
                
                # Stack to (H, W, 2)
                target_data = np.stack([boundary_mask, distance_map], axis=-1)

                # target_data is now (H, W) uint8 (0 or 1)
                # We might need to add channel dim if expected by downstream
                # But load_data usually returns (H, W) for target, and it gets reshaped later if needed.
            else:
                logger.warning("Area not found in target datacube for frontier target.")
        
        # Extract Mask
        mask_outside = None
        area = None
        if 'area' in datacube_feature:
            area = datacube_feature['area']
        elif 'area' in datacube_target:
            area = datacube_target['area']
            
        if area is not None:
            if hasattr(area, 'values'):
                area = area.values
            mask_outside = np.isnan(area)
            
        # Plot Segmentation vs Clustering
        if target_data is not None and area is not None:
             self.plot_segmentation_vs_clustering(dept, area, target_data, dir_output)
            
        # Process per period
        X_list_period = []
        y_list_period = []
        w_list_period = []
        
        for indices in periods:
            features_map = []
            for feat in loaded_features:
                data = datacube_feature[feat]
                if hasattr(data, 'values'):
                    data = data.values
                    
                if len(data.shape) == 3:
                    if dates is not None and data.shape[2] == len(dates):
                        if indices is not None:
                            data_slice = data[:, :, indices]
                        else:
                            data_slice = data
                        feat_mean = np.nanmean(data_slice, axis=2)
                    else:
                        feat_mean = np.nanmean(data, axis=2)
                else:
                    feat_mean = data
                features_map.append(feat_mean)
                
            # Extract y_image
            y_image = None
            if target_data is not None:
                if self.config.get_target_type() == 'frontier':
                     y_image = target_data # (H, W, 2)
                elif len(target_data.shape) == 3:
                     if indices is not None:
                         y_slice = target_data[:, :, indices]
                         # Mode along time axis
                         from scipy.stats import mode
                         # mode returns ModeResult(mode=..., count=...)
                         # We want mode.
                         # nan_policy='omit' is good.
                         m = mode(y_slice, axis=2, nan_policy='omit')
                         if hasattr(m, 'mode'):
                             y_image = m.mode
                         else:
                             y_image = m[0]
                             
                         if len(y_image.shape) > 2:
                             y_image = y_image.squeeze(axis=2)
                     else:
                         y_image = target_data
                else:
                     y_image = target_data
                
            X_image = np.stack(features_map, axis=-1) # (H, W, C)
            
            # Resize to 64x64
            target_shape = (64, 64)
            
            # Load raster for masking if not already loaded (it is loaded as 'area' above)
            # Resize raster to target shape to create mask
            if area is not None:
                if len(area.shape) == 3:
                    area_for_mask = area.squeeze(0)
                else:
                    area_for_mask = area
                area_resized = resize(area_for_mask, target_shape, anti_aliasing=False, preserve_range=True, order=0)
                mask_outside = np.isnan(area_resized)
            else:
                mask_outside = None

            if X_image is not None:
                # resize expects (H, W, C)
                X_image = resize(X_image, target_shape, anti_aliasing=True, preserve_range=True)
                
            if y_image is not None:
                # resize expects (H, W) or (H, W, C)
                y_image = resize(y_image, target_shape, anti_aliasing=False, preserve_range=True, order=0) # order=0 for nearest neighbor (labels)
            
            # Apply Masking
            if mask_outside is not None:
                if X_image is not None:
                    X_image[mask_outside] = 0
                if y_image is not None:
                    y_image[mask_outside] = 0
            
            # Generate weights
            # User wants weights=0 where y=0, and presumably 1 otherwise.
            # y_image is (H, W)
            if y_image is not None:
                if self.config.get_target_type() == 'frontier':
                    # For frontier, we want to weight all valid pixels (inside department)
                    if mask_outside is not None:
                        weights = (~mask_outside).astype(np.float32)
                    else:
                        weights = np.ones(y_image.shape[:2], dtype=np.float32)
                else:
                    weights = (y_image > 0).astype(np.float32)
            else:
                weights = None
                
            # Prepare for U-Net (B, C, H, W)
            # X_image is (H, W, C) -> (C, H, W)
            X_tensor = np.transpose(X_image, (2, 0, 1))
            
            # Handle NaNs in Features (already done partly, but ensure safety)
            X_tensor = np.nan_to_num(X_tensor, nan=0.0)
            
            # Add Batch Dimension
            X_tensor = np.expand_dims(X_tensor, axis=0) # (1, C, H, W)
            
            # Weights (1, 1, H, W)
            if weights is not None:
                weights = np.expand_dims(weights, axis=0) # (1, H, W)
                weights = np.expand_dims(weights, axis=0) # (1, 1, H, W)
            
            # Prepare Target (B, C, H, W)
            if y_image is not None:
                if len(y_image.shape) == 3: # (H, W, C)
                    y_tensor = np.transpose(y_image, (2, 0, 1)) # (C, H, W)
                    y_tensor = np.expand_dims(y_tensor, axis=0) # (1, C, H, W)
                else: # (H, W)
                    y_tensor = np.expand_dims(y_image, axis=0) # (1, H, W)
                    y_tensor = np.expand_dims(y_tensor, axis=0) # (1, 1, H, W)
            else:
                y_tensor = None
            
            return X_tensor, y_tensor, weights, loaded_features

    def load_all_data(self, departements, fit_scaler=True, require_target=True, load=True):
        X_list = []
        y_list = []
        weights_list = []

        # Configuration parameters
        scale = self.config.get_scale()
        dataset_name = self.config.get_dataset_name()
        graph_construct = self.config.get_graph_construct()

        assert graph_construct is not None, "Graph construct must be specified in config"
        assert scale is not None, "Scale must be specified in config"
        assert dataset_name is not None, "Dataset name must be specified in config"

        dico_config = parse_string(graph_construct)
        
        # Configuration
        scale = self.config.config.get('scale', 0.3)
        graph_construct = self.config.config.get('graph_construct', self.config.config.get('graphConstruct', "risk-size-watershed-degree-a3-r4-t0.3"))
        graph_method = 'node'
        
        n_clusters_node = self.config.get_n_clusters_node()
        
        # Load Datacube Target (which contains target)
        dir_output = Path.cwd() / Path('Experiments') / f'target_{self.target_variable}_{self.config.get_target_type()}_scale_{scale}_tol_{dico_config["tol"]}_attempt_{dico_config["attempt"]}_reduce_{dico_config["reduce"]}_ncluster_{n_clusters_node}'

        if load:
            if self.config.get_load_flag():
                X, y, weights = self.load_preprocessed_data(dir_output)
                return X, y, weights

        final_features = None # To store the consistent list of loaded features
        for dept in departements:
            X, y, w, loaded_features = self.load_data_for_dept(dept, require_target=require_target)
            if X is None: # If load_data_for_dept returned None for X
                continue
            
            if final_features is None:
                final_features = loaded_features
            else:
                if final_features != loaded_features:
                    logger.error(f"Inconsistent features across departments. Expected {len(final_features)} features: {final_features}, got {len(loaded_features)} features: {loaded_features}. Skipping department {dept}.")
                    continue # Skip this department due to feature mismatch
            
            X_list.append(X)
            if y is not None:
                y_list.append(y)
            if w is not None:
                weights_list.append(w)
            
        if not X_list:
            return None, None, None

        X_all = np.concatenate(X_list, axis=0) # (B, C, H, W)
        
        if y_list and any(y is not None for y in y_list):
             y_all = np.concatenate([y for y in y_list if y is not None], axis=0)
        else:
             y_all = None
             
        if weights_list and any(w is not None for w in weights_list):
             weights_all = np.concatenate([w for w in weights_list if w is not None], axis=0)
        else:
             weights_all = None

        # X_all is (B, C, H, W)
        B, C, H, W = X_all.shape
        
        # Reshape to (N, C) where N = B*H*W
        X_permuted = X_all.transpose(0, 2, 3, 1)
        X_reshaped = X_permuted.reshape(-1, C)
        
        # Flatten y for target encoding
        if y_all is not None:
            y_reshaped = y_all.reshape(-1) # (N,)
        else:
            y_reshaped = None
        
        # Identify indices of categorical columns
        cat_indices = []
            
        for col in self.cat_cols:
            if final_features is not None and col in final_features: # Use final_features here
                cat_indices.append(final_features.index(col))
        
        # Target Encoding
        logger.info("Applying target encoding...")
        
        # Forest Encoder
        if final_features is not None and 'forest_landcover' in final_features:
            idx = final_features.index('forest_landcover')
            logger.info("Applying Forest Encoder...")
            # Extract column, reshape to (N, 1)
            col_data = X_reshaped[:, idx].reshape(-1, 1)
            # Transform
            col_transformed = self.forest_encoder.transform(col_data, cat_indices=[0])
            # Put back
            X_reshaped[:, idx] = col_transformed.flatten()
            
        # Corine Encoder
        if final_features is not None and 'corine_landcover' in final_features:
            idx = final_features.index('corine_landcover')
            logger.info("Applying Corine Encoder...")
            # Extract column, reshape to (N, 1)
            col_data = X_reshaped[:, idx].reshape(-1, 1)
            # Transform
            col_transformed = self.corine_encoder.transform(col_data, cat_indices=[0])
            # Put back
            X_reshaped[:, idx] = col_transformed.flatten()
        
        # Normalization
        if fit_scaler:
            logger.info("Fitting scaler on training data...")
            self.scaler.fit(X_reshaped)
            
        logger.info("Transforming data with scaler...")
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to (B, C, H, W)
        X_scaled = X_scaled.reshape(B, H, W, C)
        X_scaled = X_scaled.transpose(0, 3, 1, 2)
        
        return X_scaled, y_all, weights_all

    def save_preprocessed_data(self, path, X, y, weights):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(path / 'X.pkl', 'wb') as f:
            pickle.dump(X, f)

        if y is not None:
            with open(path / 'y.pkl', 'wb') as f:
                pickle.dump(y, f)

        if weights is not None:
            with open(path / 'weights.pkl', 'wb') as f:
                pickle.dump(weights, f)

        logger.info(f"Preprocessed data saved to {path}")

    def load_preprocessed_data(self, path):
        path = Path(path)

        if not (path / 'X.pkl').exists():
            logger.error(f"Preprocessed data not found in {path}")
            return None, None, None
            
        with open(path / 'X.pkl', 'rb') as f:
            X = pickle.load(f)
            
        y = None
        if (path / 'y.pkl').exists():
            with open(path / 'y.pkl', 'rb') as f:
                y = pickle.load(f)
                
        weights = None
        if (path / 'weights.pkl').exists():
            with open(path / 'weights.pkl', 'rb') as f:
                weights = pickle.load(f)
                
        # Load scaler and encoder
        if (path / 'scaler.pkl').exists():
            with open(path / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
                
        if (path / 'forest_encoder.pkl').exists():
            with open(path / 'forest_encoder.pkl', 'rb') as f:
                self.forest_encoder = pickle.load(f)
        if (path / 'corine_encoder.pkl').exists():
            with open(path / 'corine_encoder.pkl', 'rb') as f:
                self.corine_encoder = pickle.load(f)
        
            with open(path / 'label_encoder.pkl', 'rb') as f:
                self.encoder = pickle.load(f)
                
        if (path / 'ordinal_encoder.pkl').exists():
            with open(path / 'ordinal_encoder.pkl', 'rb') as f:
                self.ordinal_encoder = pickle.load(f)
                
        logger.info(f"Preprocessed data loaded from {path}")
        return X, y, weights

    def plot_segmentation_vs_clustering(self, dept, area, clustering, dir_output):
        import matplotlib.pyplot as plt

        if clustering.ndim == 3:        
            fig, axes = plt.subplots(1, clustering.shape[2] + 1, figsize=(12, 6))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            clustering = np.expand_dims(clustering, 2)

        # Plot Area (Segmentation)
        if len(area.shape) == 3:
            area = area.squeeze(0)
        im1 = axes[0].imshow(area, cmap='jet')
        axes[0].set_title(f'{dept} - Segmentation (Area)')
        plt.colorbar(im1, ax=axes[0])

        for i in range(clustering.shape[2]):        
            # Plot Clustering
            clustering_plot = clustering[:, :, i]
                
            im2 = axes[i + 1].imshow(clustering_plot, cmap='jet')
            axes[i + 1].set_title(f'{dept} - Time Series Clustering')
            plt.colorbar(im2, ax=axes[i + 1])
        
        plt.tight_layout()
        plot_path = dir_output / f'{dept}_segmentation_vs_clustering.png'
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved segmentation vs clustering plot to {plot_path}")