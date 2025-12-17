
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

from GNN.graph_structure import GraphStructure, merge_adjacent_clusters, to_binary_mask, iou_binary
from GNN.dataloader import read_object, save_object
from GNN.tools import count_pixels_in_france_deg_square
from sklearn.preprocessing import StandardScaler
import GNN.array_fet as fet
from GNN.arborescence import root_target, rootDisk

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
        self.pipeline_params = self.config.get_pipeline_params()
        self.scaler = StandardScaler()
        
        # Target Encoding
        self.cat_cols = ['forest_landcover', 'corine_landcover'] 
        self.target_encoder = TargetEncoder(self.cat_cols)

    def load_data_for_dept(self, dept, require_target=True):
        logger.info(f"Loading data for department {dept}...")
        
        # Load datacube
        resolution = '2x2' # Hardcoded for now
        dir_data = rootDisk / 'csv' / dept / 'raster' / resolution
        features_path = dir_data / 'datacube.pkl'

        print(features_path)
        
        if not features_path.exists():
            logger.error(f"Features file not found: {features_path}")
            return None, None

        with open(features_path, 'rb') as f:
            datacube = pickle.load(f)
        
        sinister = 'firepoint'
        dataset_name = self.config.config.get('dataset_name', 'dataset') # Get from config or default
        sinister_encoding = 'occurence'
        resolution = '2x2' # Hardcoded for now based on user context, or derive from config?
        # User said "dir_target ... / resolution". And "features ... /raster/2x2/".
        # So resolution is likely '2x2'.
        
        dir_target_bin = root_target / sinister / dataset_name / sinister_encoding / 'bin' / resolution
        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
        dir_target = root_target / sinister / dataset_name / sinister_encoding / 'log' / resolution
        
        # Load Target Data
        target_data = None
        if self.target_variable == 'risk':
            target_data = read_object(f'{dept}Influence.pkl', dir_target)
        elif self.target_variable == 'nbsinister':
            target_data = read_object(f'{dept}binScale0.pkl', dir_target_bin)
            
        if target_data is None:
            if require_target:
                logger.error(f"Target data ({self.target_variable}) not found in {dir_target} or {dir_target_bin}")
                return None, None
            else:
                logger.warning(f"Target data ({self.target_variable}) not found. Proceeding without target.")

        # 3. Aggregate Data
        frequency = self.config.get_frequency()
        
        X_list = []
        y_list = []
        
        # Check if frequency is 'full' or an integer
        is_full = (frequency == 'full')
        freq_int = 1
        if not is_full:
            try:
                freq_int = int(frequency)
            except ValueError:
                logger.warning(f"Invalid frequency '{frequency}'. Defaulting to 'full'.")
                is_full = True
        
        # Get dates from datacube if available
        dates = None
        if hasattr(datacube, 'coords') and 'date' in datacube.coords:
            dates = datacube.coords['date'].values
            # Convert to pandas datetime if needed
            import pandas as pd
            dates = pd.to_datetime(dates)
            years = dates.year
            unique_years = np.unique(years)
        else:
            if not is_full:
                logger.error("Datacube does not have 'date' coordinate. Cannot apply frequency splitting. Defaulting to full.")
                is_full = True
                
        if is_full:
            # ... (Existing 'full' logic)
            # Define periods as a single list containing all indices
            periods = [None] # None means "use all"
        else:
            # Define periods based on years
            # unique_years sorted
            unique_years = np.sort(unique_years)
            periods = []
            # Create chunks of years
            for i in range(0, len(unique_years), freq_int):
                chunk_years = unique_years[i:i+freq_int]
                # Find indices corresponding to these years
                indices = np.where(np.isin(years, chunk_years))[0]
                if len(indices) > 0:
                    periods.append(indices)
                    
        # Identify available features once
        loaded_features = [feat for feat in self.features_to_use if feat in datacube]
        if not loaded_features:
            logger.error("No features found in datacube.")
            return None, None, None

        for indices in periods:
            # Extract features for this period
            features_map = []
            for feat in loaded_features:
                data = datacube[feat]
                # Check dimensions. If (H, W, T), mean over T.
                if hasattr(data, 'values'): # xarray
                    data = data.values
                
                if len(data.shape) == 3:
                    # Check if 3rd dim matches dates length
                    if dates is not None and data.shape[2] == len(dates):
                        # Slice by time if indices provided
                        if indices is not None:
                            data_slice = data[:, :, indices]
                        else:
                            data_slice = data
                        feat_mean = np.nanmean(data_slice, axis=2)
                    else:
                        # Assume static or incompatible time dim, take mean over whatever 3rd dim is (e.g. 1)
                        feat_mean = np.nanmean(data, axis=2)
                else:
                    feat_mean = data # Already 2D?
                
                features_map.append(feat_mean)
            
            if not features_map:
                continue
            
            X_image = np.stack(features_map, axis=-1) # (H, W, C)
            
            y_image = None
            if target_data is not None:
                if hasattr(target_data, 'values'):
                    target_data = target_data.values
                    
                # Slice target if it has time dimension
                if len(target_data.shape) == 3:
                    if indices is not None:
                        # Assume target aligns with datacube time
                        if target_data.shape[2] == len(dates):
                             target_slice = target_data[:, :, indices]
                        else:
                             # Fallback: cannot slice if dimensions don't match
                             logger.warning("Target time dimension does not match datacube dates. Using full target mean (might be wrong).")
                             target_slice = target_data
                    else:
                        target_slice = target_data
                        
                    target_sum = np.nansum(target_slice, axis=2)
                else:
                    target_sum = target_data

                valid_mask = ~np.isnan(target_sum) # Or from raster
                
                raster = read_object(f'{dept}rasterScale0.pkl', dir_raster)
                assert raster is not None
                if len(raster.shape) == 3:
                    raster = raster.squeeze()

                reduce_param = self.pipeline_params['reduce']
                attempt_param = self.pipeline_params['attempt']

                scale = self.pipeline_params.get('scale', 0)
                tol = self.pipeline_params.get('tol', 0)
                
                if reduce_param == 'search': reduce_param = 100 # Default fallback
                if attempt_param == 'search': attempt_param = 5 # Default fallback
                
                tmp_path = Path('/tmp/learned_segmentation')
                tmp_path.mkdir(exist_ok=True, parents=True)
                
                # Let's create a temporary GraphStructure to run the pipeline
                gs = GraphStructure(
                    scale=scale,
                    geo=None,
                    maxDist=10, # Default
                    numNei=5,   # Default
                    resolution='10m', # Default
                    graph_construct='watershed-size', # Default
                    sinister='fire',
                    sinister_encoding='utf-8',
                    dataset_name='dataset',
                    train_departements=[dept],
                    attempt=attempt_param,
                    reduce=reduce_param,
                    tol=tol,
                )
                
                pred_ws = gs.my_watershed(dept, target_sum, valid_mask, raster, tmp_path, 'risk', 'target_gen', reduce=reduce_param)
                
                # merge_adjacent_clusters
                # Need min/max cluster size
                # Logic from graph_structure.py
                size = count_pixels_in_france_deg_square(deg_size=scale)[-1]
                max_cluster_size = int(size + (tol * size))
                min_cluster_size = int(size - (tol * size))
                
                S_raw = merge_adjacent_clusters(pred_ws, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
                                                features=None, mode='size', exclude_label=0, background=-1,
                                                nb_attempt=attempt_param)
                
                # S_raw is the segmentation (Cluster IDs).
                # We want to predict this?
                # Or the binary mask?
                # "prédire la segmentation".
                # Let's assume we want to predict the Binary Mask (0/1) for now.
                y_image = to_binary_mask(np.asarray(S_raw))
            
            # Prepare for U-Net (B, C, H, W)
            # X_image is (H, W, C) -> (C, H, W)
            X_tensor = np.transpose(X_image, (2, 0, 1))
            
            # Handle NaNs in Features
            X_tensor = np.nan_to_num(X_tensor, nan=0.0)
            
            # Add Batch Dimension
            X_tensor = np.expand_dims(X_tensor, axis=0) # (1, C, H, W)
            
            X_list.append(X_tensor)
            
            # Prepare Target (B, 1, H, W)
            if y_image is not None:
                y_tensor = np.expand_dims(y_image, axis=0) # (1, H, W)
                y_tensor = np.expand_dims(y_tensor, axis=0) # (1, 1, H, W)
                y_list.append(y_tensor)
            else:
                # If target is missing, we append None? Or we handle it later.
                # If we return None for y, we can't concatenate.
                # We should handle X_list and y_list carefully.
                y_list.append(None)

        if not X_list:
            return None, None, None
            
        return X_list, y_list, loaded_features

    def load_all_data(self, departements, fit_scaler=True, fit_encoder=True, require_target=True):
        X_list = []
        y_list = []
        
        final_features = None # To store the consistent list of loaded features
        
        for dept in departements:
            X, y, loaded_features = self.load_data_for_dept(dept, require_target=require_target)
            if X is None: # If load_data_for_dept returned None for X_list
                continue
            
            if final_features is None:
                final_features = loaded_features
            else:
                if final_features != loaded_features:
                    logger.error(f"Inconsistent features across departments. Expected {len(final_features)} features: {final_features}, got {len(loaded_features)} features: {loaded_features}. Skipping department {dept}.")
                    continue # Skip this department due to feature mismatch
            
            X_list.extend(X) # Extend, as X is now a list of tensors
            y_list.extend(y) # Extend, as y is now a list of tensors
            
        if not X_list:
            return None, None
            
        X_all = np.concatenate(X_list, axis=0) # (B, C, H, W)
        
        if any(y is None for y in y_list):
             # If any target is missing, we assume we are in inference mode without targets?
             # Or we should handle mixed cases?
             # For now, if require_target=False, we might return y=None.
             y_all = None
             
             # If y is None, we can't fit encoder on y.
             if fit_encoder:
                 logger.warning("Cannot fit target encoder because targets are missing. Disabling encoder fitting.")
                 fit_encoder = False
        else:
            y_all = np.concatenate(y_list, axis=0)

        
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
        if cat_indices:
            if fit_encoder:
                if y_reshaped is None:
                    logger.warning("Cannot fit target encoder: y_reshaped is None.")
                else:
                    logger.info("Fitting target encoder...")
                    self.target_encoder.fit(X_reshaped, y_reshaped, cat_indices)
            else:
                if not self.target_encoder.mapping:
                    logger.warning("TargetEncoder is being applied but has empty mapping. Ensure it was fitted or loaded correctly.")
            
            logger.info("Applying target encoding...")
            X_reshaped = self.target_encoder.transform(X_reshaped, cat_indices)
        
        # Normalization
        if fit_scaler:
            logger.info("Fitting scaler on training data...")
            self.scaler.fit(X_reshaped)
            
        logger.info("Transforming data with scaler...")
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to (B, C, H, W)
        X_scaled = X_scaled.reshape(B, H, W, C)
        X_scaled = X_scaled.transpose(0, 3, 1, 2)
        
        return X_scaled, y_all

    def save_preprocessed_data(self, path, X, y):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'X.pkl', 'wb') as f:
            pickle.dump(X, f)
        if y is not None:
            with open(path / 'y.pkl', 'wb') as f:
                pickle.dump(y, f)
        
        # Save scaler and encoder if needed
        with open(path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(path / 'encoder.pkl', 'wb') as f:
            pickle.dump(self.target_encoder, f)
            
        logger.info(f"Preprocessed data saved to {path}")

    def load_preprocessed_data(self, path):
        path = Path(path)
        if not (path / 'X.pkl').exists():
            logger.error(f"Preprocessed data not found in {path}")
            return None, None
            
        with open(path / 'X.pkl', 'rb') as f:
            X = pickle.load(f)
            
        y = None
        if (path / 'y.pkl').exists():
            with open(path / 'y.pkl', 'rb') as f:
                y = pickle.load(f)
                
        # Load scaler and encoder
        if (path / 'scaler.pkl').exists():
            with open(path / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        if (path / 'encoder.pkl').exists():
            with open(path / 'encoder.pkl', 'rb') as f:
                self.target_encoder = pickle.load(f)
                
        logger.info(f"Preprocessed data loaded from {path}")
        return X, y
