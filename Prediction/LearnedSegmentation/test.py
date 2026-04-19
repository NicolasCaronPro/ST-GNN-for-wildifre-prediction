
import sys
import numpy as np
import logging
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

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

import torch
from GNN.forecasting_models.pytorch.models_2D import UNet
from GNN.graph_structure import iou_binary

from skimage.transform import resize

import pandas as pd

def iou_score(y_true, y_pred):
    """Calcule l'indice IoU entre deux signaux continus.

    Parameters
    ----------
    y_true : np.ndarray | DMatrix
        Signal de référence (vérité terrain).
    y_pred : np.ndarray | DMatrix
        Signal prédit (à comparer au signal de référence).

    Returns
    -------
    float
        Valeur IoU (aire d'intersection divisée par aire d'union).
    """

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    return intersection / union if union > 0 else 0

class Tester:
    def __init__(self, config, model_path, model_params):
        self.model_path = Path(model_path)
        self.model_params = model_params
        self.model_type = model_params.get('type', 'XGBRegressor')
        self.params = model_params.get('params', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._load_model()
        self.config = config
        self.score = pd.DataFrame()

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.model_type == 'UNet':
            model = UNet(
                n_channels=self.params['n_channels'],
                out_channels=self.params['out_channels'],
                conv_channels=self.params['conv_channels'],
                bilinear=self.params.get('bilinear', False),
                task_type=self.params.get('task_type')
            )
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model
        else:
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)

    def test(self, X, y, weights=None, dept_name="unknown", compute_metrics=True, output_dir=None, testset='test'):
        
        B, F, H, W = X.shape

        print("Testing model...")
        
        if self.model_type == 'UNet' or self.model_type == 'SwinUNet' or self.model_type == 'tbconvnet':
            # Convert to tensors
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                 if self.params.get('task_type') == 'classification':
                    y = torch.tensor(y, dtype=torch.long)
                    if len(y.shape) == 4 and y.shape[1] == 1:
                        y = y.squeeze(1)
                 else:
                    y = torch.tensor(y, dtype=torch.float32)
            
            if self.model_type == 'UNet':
                X = X.to(self.device)
                y = y.to(self.device)
            
            self.model.eval()
            
            with torch.no_grad():
                outputs, logits, hidden = self.model(X)
                if self.params.get('task_type') == 'regression':
                    preds = outputs[:, :, :, 0]
                elif self.params.get('task_type') == 'binary+regression':
                    preds = torch.argmax(outputs[:, :, :, 0:2], dim=-1)
                    y = y[:, 0, ...]
                elif self.params.get('task_type') == 'binary':
                    preds = outputs[:, :, :, 1]
                else:
                    preds = torch.argmax(outputs, dim=-1)

            metrics = {'department': dept_name}

            if weights is not None:
                # Ensure weights match preds shape (B, H, W)
                if len(weights.shape) == 4 and weights.shape[1] == 1:
                    weights = weights.squeeze(1)
                if len(y.shape) == 4 and y.shape[1] == 1:
                    y = y.squeeze(1)
                
            if compute_metrics:
                if self.params.get('task_type') == 'regression':
                    # Calculate MSE/MAE
                    y_np = y.cpu().numpy().squeeze()
                    preds_np = preds.cpu().numpy().squeeze()
                    
                    mse = np.nanmean((y_np - preds_np) ** 2)
                    mae = np.nanmean(np.abs(y_np - preds_np))
                    print(f"MSE: {mse}")
                    print(f"MAE: {mae}")
                    metrics['MSE'] = mse
                    metrics['MAE'] = mae
                else:
                    # Calculate IoU
                    y_np = y.cpu().numpy().squeeze()
                    preds_np = preds.cpu().numpy().squeeze()
                    
                    # Flatten
                    y_flat = y_np.flatten()
                    preds_flat = preds_np.flatten()

                    # Filter NaNs
                    mask = ~np.isnan(y_flat) & ~np.isnan(preds_flat)
                    
                    from sklearn.metrics import jaccard_score
                    if np.any(mask):
                        iou = iou_score(y_flat[mask], preds_flat[mask])
                    else:
                        iou = np.nan
                    print(f"Mean IoU: {iou}")
                    metrics['IoU'] = iou
                    
            # Visualize first sample
            pred_vis = preds[0].cpu().numpy().astype(np.float32)
            target_vis = y[0].cpu().numpy().astype(np.float32)

            pred_vis[weights[0] == 0] = np.nan
            target_vis[weights[0] == 0] = np.nan

            if weights is not None:
                if isinstance(weights, torch.Tensor):
                     weights_vis = weights[0].cpu().numpy()
                else:
                     weights_vis = weights[0]
                # Handle shape (1, H, W) -> (H, W)
                if len(weights_vis.shape) == 3:
                    weights_vis = weights_vis[0]
            else:
                weights_vis = None

            if self.params.get('task_type') == 'regression' or (self.params.get('task_type') == 'binary' and self.config.get_target_type() != 'frontier'):
                 clustering_metrics = self.process_regression_risk(pred_vis, y, dept_name, output_dir)
                 if clustering_metrics:
                     metrics.update(clustering_metrics)
            elif self.config.get_target_type() == 'frontier':
                 clustering_metrics = self.process_binary_frontier(pred_vis, y, dept_name, output_dir)
                 if clustering_metrics:
                     metrics.update(clustering_metrics)
            
            if output_dir:
                vis_path = output_dir / f'prediction_vis_{dept_name}.png'
            else:
                vis_path = self.model_path.parent / f'prediction_vis_{dept_name}.png'
                
            self.visualize_prediction(pred_vis, target_vis, vis_path, weights_vis)
            
            metrics['test_set'] = testset
            if compute_metrics:
                self.score = pd.concat([self.score, pd.DataFrame([metrics])], ignore_index=True)
            
            return preds.cpu().numpy()

        else:
            predictions = self.model.predict(X)
            metrics = {'department': dept_name}

            if compute_metrics:
                # Calculate metrics
                mse = np.nanmean((y - predictions) ** 2)
                mae = np.nanmean(np.abs(y - predictions))
                print(f"MSE: {mse}")
                print(f"MAE: {mae}")
                metrics['MSE'] = mse
                metrics['MAE'] = mae
            
            if self.params.get('task_type') == 'regression' or (self.params.get('task_type') == 'binary' and self.config.get_target_type() != 'frontier'):
                 clustering_metrics = self.process_regression_risk(predictions, y, dept_name, output_dir)
                 if clustering_metrics:
                     metrics.update(clustering_metrics)
            elif self.config.get_target_type() == 'frontier':
                 clustering_metrics = self.process_binary_frontier(predictions, y, dept_name, output_dir)
                 if clustering_metrics:
                     metrics.update(clustering_metrics)

            metrics['test_set'] = testset
            if compute_metrics:
                self.score = pd.concat([self.score, pd.DataFrame([metrics])], ignore_index=True)

            return predictions

    def process_regression_risk(self, predictions, y, dept_name, output_dir=None):

        print(f"Processing regression risk...")
        from LearnedSegmentation.segmentation import Segmentation
        from GNN.tools import read_object, save_object
        from GNN.arborescence import rootDisk, root_target

        metrics = {}

        dataset_name = self.config.get_dataset_name()

        # 4. Load Ground Truth from Datacube
        sinister = 'firepoint'
        sinister_encoding = 'occurence'
        
        # 1. Create Geometry with Watershed
        resolution = '2x2'
        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
        raster_obj = read_object(f'{dept_name}rasterScale0.pkl', dir_raster)
        
        if raster_obj is None:
            raster = np.copy(y)
            raster[y == -1] = np.nan
            #print(f"Could not load raster for {dept_name}")
            #return
        else:
            if len(y.shape) == 3:
                raster = resize(raster_obj[0], (y.shape[1], y.shape[2]), anti_aliasing=False, preserve_range=True, order=0)
            else:
                raster = resize(raster_obj[0], (y.shape[0], y.shape[1]), anti_aliasing=False, preserve_range=True, order=0)

        H, W = raster.shape
        
        valid_mask = (raster != -1) & (~np.isnan(raster))

        pred_map = np.full((H, W), np.nan)
        pred_map[valid_mask] = predictions[valid_mask]

        # Instantiate Segmentation
        scale = self.config.get_scale()
        graph_construct = self.config.get_graph_construct()
        
        # Parse graph_construct to get parameters
        from GNN.construct import parse_string
        dico_config = parse_string(graph_construct)
        
        attempt = int(dico_config.get('attempt'))
        reduce = int(dico_config.get('reduce'))
        tol = float(dico_config.get('tol'))
        base = graph_construct

        assert attempt is not None, "Attempt is not defined"
        assert reduce is not None, "Reduce is not defined"
        assert tol is not None, "Tol is not defined"
        
        seg = Segmentation(scale=scale, base=base, attempt=attempt, reduce=reduce, tol=tol, dataset_name=dataset_name)
        
        # Create Geometry
        if output_dir:
             dir_output = output_dir / f'segmentation_test_{dept_name}'
        else:
             dir_output = self.model_path.parent / f'segmentation_test_{dept_name}'
             
        dir_output.mkdir(parents=True, exist_ok=True)
        
        vec_base = ['watershed', 'size']
        mask = np.ones_like(raster, dtype=bool)
        train_date = None

        print("Running segmentation...")
        pred_seg, pred_seg_fz = seg.create_geometry_with_watershed(
            dept=dept_name,
            vec_base=vec_base,
            path=dir_output,
            sinister=sinister,
            dataset_name=dataset_name,
            sinister_encoding=sinister_encoding,
            resolution=resolution,
            node_already_predicted=0,
            train_date=train_date,
            data=pred_map
        )

        _, true_fz = seg.create_geometry_with_watershed(
            dept=dept_name,
            vec_base=vec_base,
            path=dir_output,
            sinister=sinister,
            dataset_name=dataset_name,
            sinister_encoding=sinister_encoding,
            resolution=resolution,
            node_already_predicted=0,
            train_date=train_date,
            data='true'
        )
        
        pred_seg_fz = (pred_seg_fz > 0).astype(float)
        true_fz = (true_fz > 0).astype(float)

        n_clusters_node = self.config.get_n_clusters_node()
        
        dir_encoders = Path.cwd() / Path('Experiments') / f'target_{self.config.get_target_variable()}_{self.config.get_target_type()}_scale_{scale}_tol_{dico_config["tol"]}_attempt_{dico_config["attempt"]}_reduce_{dico_config["reduce"]}_ncluster_{n_clusters_node}'

        gs_path = dir_encoders / 'gs.pkl'
        if not gs_path.exists():
             print(f"gs.pkl not found at {gs_path}")
             return metrics
             
        with open(gs_path, 'rb') as f:
            gs = pickle.load(f)
            
        from GNN.arborescence import rootDisk
        
        # Load config to check test_departements
        import json
        config_path = Path(__file__).resolve().parent / 'config_regression.json'
        test_departements = []
        train_departements = []
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                test_departements = config.get('test_departements', [])
                train_departements = config.get('train_departements', [])

        valid_mask = (raster_obj[0] != -1) & (~np.isnan(raster_obj[0]))
        pred_seg[~valid_mask] = np.nan
        pred_seg_fz[~valid_mask] = np.nan
        true_fz[~valid_mask] = np.nan

        print("Finding closest clusters...")
        if dept_name in test_departements or dept_name in train_departements:
            dir_target = root_target / sinister / dataset_name / 'datacube'
            gs.find_closest_cluster_with_time_series([dept_name], train_date=None, path=dir_output, root_data=dir_target, raster=pred_seg)
        else:
            dir_target = rootDisk / 'csv'
            gs.find_closest_cluster([dept_name], train_date=None, path=dir_output, root_data=dir_target, raster=pred_seg)

        # 3. Encode clusters
        encoder_path = dir_encoders / 'label_encoder.pkl'
        ordinal_encoder_path = dir_encoders / 'ordinal_encoder.pkl'
        
        if not encoder_path.exists() or not ordinal_encoder_path.exists():
            print("Encoders not found.")
            return metrics
            
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        with open(ordinal_encoder_path, 'rb') as f:
            ordinal_encoder = pickle.load(f)
            
        # Get the clustering result (Predicted)
        ts_cluster_path = dir_output / 'time_series_clustering' / f'{dept_name}_{gs.scale}_{gs.base}_{gs.graph_method}.pkl'
        if ts_cluster_path.exists():
            pred_clusters = read_object(ts_cluster_path.name, ts_cluster_path.parent)
            if isinstance(pred_clusters, list):
                pred_clusters = pred_clusters[0]
        else:
            print("Time series clustering image not found.")
            return metrics
            
        print("Encoding predicted clusters...")
        encoded_pred = self.encode_map(pred_clusters, encoder, ordinal_encoder)
        
        dir_target = dir_encoders / 'datacube'
        datacube_path = dir_target / f'datacube_target_{dept_name}_{scale}_{graph_construct}_node.pkl'
        
        gt_clusters = None
        if datacube_path.exists():
            print(f"Loading datacube from {datacube_path}")
            datacube_target = read_object(datacube_path.name, dir_target)
            if isinstance(datacube_target, list):
                datacube_target = datacube_target[0]
                
            if 'time_series_clustering' in datacube_target:
                gt_clusters = datacube_target['time_series_clustering']
                if hasattr(gt_clusters, 'values'):
                    gt_clusters = gt_clusters.values
            else:
                print("time_series_clustering not found in datacube.")
        else:
            print(f"Datacube not found at {datacube_path}")
        

        # 4. Compare with Ground Truth
        if gt_clusters is not None:
            if isinstance(gt_clusters, list):
                gt_clusters = gt_clusters[0]
            
            print("Encoding ground truth clusters...")
            encoded_gt = self.encode_map(gt_clusters, encoder, ordinal_encoder)
            
            # 5. Calculate IoU
            from sklearn.metrics import jaccard_score
            
            if encoded_pred.shape != encoded_gt.shape:
                print(f"Shape mismatch: Pred {encoded_pred.shape}, GT {encoded_gt.shape}")
            else:
                # Flatten clusters
                gt_flat = gt_clusters.flatten()
                pred_flat = pred_clusters.flatten()
                
                # Mask for valid clusters (not NaN)
                mask = valid_mask.flatten()
                
                if np.any(mask):
                    # Calculate IoU on encoded values (continuous) with mask
                    iou = iou_score(encoded_gt.flatten()[mask], encoded_pred.flatten()[mask])
                    print(f"IoU (Macro): {iou}")
                    metrics['IoU'] = iou

                    # Calculate F1, MSE, MAE
                    from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
                    
                    # F1 Score on discrete cluster labels
                    f1 = f1_score(gt_flat[mask], pred_flat[mask], average='macro')
                    metrics['Clustering_F1_Macro'] = f1
                    print(f"Clustering F1 (Macro): {f1}")
                    
                    # MSE/MAE on encoded values (risk)
                    encoded_gt_flat = encoded_gt.flatten()
                    encoded_pred_flat = encoded_pred.flatten()
                    
                    mse_clus = mean_squared_error(encoded_gt_flat[mask], encoded_pred_flat[mask])
                    mae_clus = mean_absolute_error(encoded_gt_flat[mask], encoded_pred_flat[mask])
                    
                    metrics['Clustering_MSE'] = mse_clus
                    metrics['Clustering_MAE'] = mae_clus
                    print(f"Clustering MSE: {mse_clus}")
                    print(f"Clustering MAE: {mae_clus}")


        else:
            print("Ground truth time_series_clustering not found. Skipping comparison.")
            self.visualize_prediction(encoded_pred, None, dir_output / f'encoded_clustering_{dept_name}.png')
            print(f"Encoded clustering saved to {dir_output / 'encoded_clustering.png'}")
        
        print(pred_seg_fz, true_fz)
        # Segmentation Metrics (pred_seg_fz vs true_fz)
        if pred_seg_fz is not None and true_fz is not None:
            print("Calculating Segmentation Metrics...")
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
            
            # Flatten
            pred_seg_flat = pred_seg_fz.flatten()
            true_seg_flat = true_fz.flatten()

            mask = valid_mask.flatten()
            
            # Metrics with 'seg_' prefix
            metrics['seg_accuracy'] = accuracy_score(true_seg_flat[mask], pred_seg_flat[mask])
            metrics['seg_precision'] = precision_score(true_seg_flat[mask], pred_seg_flat[mask], zero_division=0)
            metrics['seg_recall'] = recall_score(true_seg_flat[mask], pred_seg_flat[mask], zero_division=0)
            metrics['seg_f1'] = f1_score(true_seg_flat[mask], pred_seg_flat[mask], zero_division=0)
            metrics['seg_iou'] = jaccard_score(true_seg_flat[mask], pred_seg_flat[mask], zero_division=0)
            
            print(f"Segmentation IoU: {metrics['seg_iou']}")
            
            # Plot comparison
            vis_path_seg = dir_output / 'segmentation_comparison.png'
            self.visualize_prediction(pred_seg_fz, true_fz, vis_path_seg)
            print(f"Segmentation comparison saved to {vis_path_seg}")

        vis_path = dir_output / 'comparison_vis.png'
        encoded_pred[~valid_mask] = np.nan
        encoded_gt[~valid_mask] = np.nan
        self.visualize_prediction(encoded_pred, encoded_gt, vis_path)
        print(f"Comparison visualization saved to {vis_path}")
            
        return metrics

    def encode_map(self, map_data, encoder, ordinal_encoder):
        # Flatten
        original_shape = map_data.shape
        map_flat = map_data.flatten()
        
        # Handle NaNs (background)
        mask_valid = ~np.isnan(map_flat)
        encoded_map = np.zeros_like(map_flat, dtype=float) # Initialize with 0
        
        if np.any(mask_valid):
            try:
                # CatBoostEncoder expects 2D input
                # We need to reshape to (-1, 1)
                values_to_transform = map_flat[mask_valid].reshape(-1, 1)
                
                # Transform
                transformed = encoder.transform(values_to_transform)
                
                # Ordinal Encoder
                if ordinal_encoder is not None:
                    transformed = ordinal_encoder.transform(transformed)
                    # Shift by +1 so classes are 1, 2, 3, 4 (0 is background)
                    transformed += 1
                    
                if hasattr(transformed, 'values'):
                    transformed = transformed.values
                    
                encoded_map[mask_valid] = transformed.flatten()
            except ValueError as e:
                print(f"Error during encoding: {e}")
                # Keep 0
                pass
                
        return encoded_map.reshape(original_shape)

    def process_binary_frontier(self, predictions, y, dept_name, output_dir=None, pred_seg_fz=None, true_fz=None):
        print(f"Processing binary frontier for {dept_name}...")
        from LearnedSegmentation.frontier_utils import connected_components_8
        from LearnedSegmentation.segmentation import Segmentation
        from GNN.tools import read_object
        from GNN.arborescence import root_target, rootDisk
        import pickle
        import json

        metrics = {}
        
        # 1. Threshold predictions to get binary boundary
        boundary_mask = predictions.squeeze()
        
        # 2. Get zones from boundary mask
        interior_mask = (boundary_mask == 0).astype(int)
        pred_seg = connected_components_8(interior_mask)
        
        # 3. Clustering (similar to process_regression_risk)
        scale = self.config.get_scale()
        graph_construct = self.config.get_graph_construct()
        from GNN.construct import parse_string
        dico_config = parse_string(graph_construct)
        n_clusters_node = self.config.get_n_clusters_node()
        
        attempt = int(dico_config.get('attempt'))
        reduce = int(dico_config.get('reduce'))
        tol = float(dico_config.get('tol'))
        base = graph_construct
        
        dataset_name = self.config.get_dataset_name()
        
        seg = Segmentation(scale=scale, base=base, attempt=attempt, reduce=reduce, tol=tol, dataset_name=dataset_name)

        dir_encoders = Path.cwd() / Path('Experiments') / f'target_{self.config.get_target_variable()}_{self.config.get_target_type()}_scale_{scale}_tol_{dico_config["tol"]}_attempt_{dico_config["attempt"]}_reduce_{dico_config["reduce"]}_ncluster_{n_clusters_node}'

        gs_path = dir_encoders / 'gs.pkl'
        if not gs_path.exists():
             print(f"gs.pkl not found at {gs_path}")
             return metrics
             
        with open(gs_path, 'rb') as f:
            gs = pickle.load(f)
            
        dataset_name = self.config.get_dataset_name()
        sinister = 'firepoint' # TODO: make configurable?
        
        dir_output = self.model_path.parent / f'segmentation_test_{dept_name}'
        if output_dir:
             dir_output = output_dir / f'segmentation_test_{dept_name}'
        
        dir_output.mkdir(parents=True, exist_ok=True)
        
        # Load config to check test_departements
        config_path = Path(__file__).resolve().parent / 'config_regression.json'
        test_departements = []
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                test_departements = config.get('test_departements', [])

        metrics = {}

        dataset_name = self.config.get_dataset_name()

        # 4. Load Ground Truth from Datacube
        sinister = 'firepoint'
        sinister_encoding = 'occurence'
        
        vec_base = ['watershed', 'size']
        train_date = None
        
        # Calculate true_fz if not passed
        if true_fz is None:
            _, true_fz = seg.create_geometry_with_watershed(
                dept=dept_name,
                vec_base=vec_base,
                path=dir_output,
                sinister=sinister,
                dataset_name=dataset_name,
                sinister_encoding=sinister_encoding,
                resolution='2x2',
                node_already_predicted=0,
                train_date=train_date,
                data='true'
            )
            true_fz = (true_fz > 0).astype(int)

        # 1. Create Geometry with Watershed
        resolution = '2x2'
        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
        raster_obj = read_object(f'{dept_name}rasterScale0.pkl', dir_raster)

        assert raster_obj is not None, f"rasterScale0.pkl not found at {dir_raster}"

        pred_seg = resize(pred_seg, (raster_obj.shape[1], raster_obj.shape[2]), order=0)

        # Use pred_seg as pred_seg_fz if not passed (after resize)
        if pred_seg_fz is None:
             pred_seg_fz = pred_seg

        pred_seg = pred_seg.astype(float)
        pred_seg[np.isnan(raster_obj[0])] = np.nan # Background/Boundary

        print("Finding closest clusters...")
        if dept_name in test_departements:
            dir_target = root_target / sinister / dataset_name / 'datacube'
            gs.find_closest_cluster_with_time_series([dept_name], train_date=None, path=dir_output, root_data=dir_target, raster=pred_seg)
        else:
            dir_target = rootDisk / 'csv'
            gs.find_closest_cluster([dept_name], train_date=None, path=dir_output, root_data=dir_target, raster=pred_seg)

        # 4. Encode and Compare (reuse logic from process_regression_risk?)
        # For now, duplicate logic or refactor. Duplication is safer for now.
        
        encoder_path = dir_encoders / 'label_encoder.pkl'
        ordinal_encoder_path = dir_encoders / 'ordinal_encoder.pkl'
        
        if not encoder_path.exists() or not ordinal_encoder_path.exists():
            print("Encoders not found.")
            return metrics
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        with open(ordinal_encoder_path, 'rb') as f:
            ordinal_encoder = pickle.load(f)
            
        ts_cluster_path = dir_output / 'time_series_clustering' / f'{dept_name}_{gs.scale}_{gs.base}_{gs.graph_method}.pkl'
        if ts_cluster_path.exists():
            pred_clusters = read_object(ts_cluster_path.name, ts_cluster_path.parent)
            if isinstance(pred_clusters, list):
                pred_clusters = pred_clusters[0]
        else:
            print("Time series clustering image not found.")
            return metrics

        print("Encoding predicted clusters...")
        encoded_pred = self.encode_map(pred_clusters, encoder, ordinal_encoder)
        
        # Load Ground Truth for comparison
        dir_target_gt = dir_encoders / 'datacube'
        datacube_path = dir_target_gt / f'datacube_target_{dept_name}_{scale}_{graph_construct}_node.pkl'
        
        gt_clusters = None
        if datacube_path.exists():
            print(f"Loading datacube from {datacube_path}")
            datacube_target = read_object(datacube_path.name, dir_target_gt)
            if isinstance(datacube_target, list):
                datacube_target = datacube_target[0]
            if 'time_series_clustering' in datacube_target:
                gt_clusters = datacube_target['time_series_clustering']
                if hasattr(gt_clusters, 'values'):
                    gt_clusters = gt_clusters.values
        
        if gt_clusters is not None:
            if isinstance(gt_clusters, list):
                gt_clusters = gt_clusters[0]
            print("Encoding ground truth clusters...")
            encoded_gt = self.encode_map(gt_clusters, encoder, ordinal_encoder)
            
            if encoded_pred.shape != encoded_gt.shape:
                print(f"Shape mismatch: Pred {encoded_pred.shape}, GT {encoded_gt.shape}")
            else:
                # Flatten clusters
                gt_flat = gt_clusters.flatten()
                pred_flat = pred_clusters.flatten()
                
                # Mask for valid clusters (not NaN)
                mask = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
                
                if np.any(mask):
                    # Calculate IoU on encoded values (continuous) with mask
                    iou = iou_score(encoded_gt.flatten()[mask], encoded_pred.flatten()[mask])
                    print(f"Clustering IoU (Macro): {iou}")
                    metrics['IoU'] = iou

                    # Calculate F1, MSE, MAE
                    from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
                    
                    # F1 Score on discrete cluster labels
                    # We assume clusters are categorical labels
                    f1 = f1_score(gt_flat[mask], pred_flat[mask], average='macro')
                    metrics['Clustering_F1_Macro'] = f1
                    print(f"Clustering F1 (Macro): {f1}")
                    
                    # MSE/MAE on encoded values (risk)
                    # Use the same mask to compare only valid regions
                    encoded_gt_flat = encoded_gt.flatten()
                    encoded_pred_flat = encoded_pred.flatten()
                    
                    mse_clus = mean_squared_error(encoded_gt_flat[mask], encoded_pred_flat[mask])
                    mae_clus = mean_absolute_error(encoded_gt_flat[mask], encoded_pred_flat[mask])
                    
                    metrics['Clustering_MSE'] = mse_clus
                    metrics['Clustering_MAE'] = mae_clus
                    print(f"Clustering MSE: {mse_clus}")
                    print(f"Clustering MAE: {mae_clus}")
                
            # Segmentation Metrics (pred_seg_fz vs true_fz)
            if pred_seg_fz is not None and true_fz is not None:
                print("Calculating Segmentation Metrics...")
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
                
                # Flatten
                pred_seg_flat = pred_seg_fz.flatten()
                true_seg_flat = true_fz.flatten()
                
                print(np.unique(pred_seg_fz), np.unique(true_seg_flat))
                
                metrics['seg_accuracy'] = accuracy_score(true_seg_flat, pred_seg_flat)
                metrics['seg_precision'] = precision_score(true_seg_flat, pred_seg_flat, zero_division=0)
                metrics['seg_recall'] = recall_score(true_seg_flat, pred_seg_flat, zero_division=0)
                metrics['seg_f1'] = f1_score(true_seg_flat, pred_seg_flat, zero_division=0)
                metrics['seg_iou'] = jaccard_score(true_seg_flat, pred_seg_flat, zero_division=0)
                
                print(f"Segmentation IoU: {metrics['seg_iou']}")
                
                # Plot comparison
                vis_path_seg = dir_output / 'segmentation_comparison.png'
                self.visualize_prediction(pred_seg_fz, true_fz, vis_path_seg)
                print(f"Segmentation comparison saved to {vis_path_seg}")

            vis_path = dir_output / 'comparison_vis_frontier.png'
            self.visualize_prediction(encoded_pred, encoded_gt, vis_path)
        else:
            self.visualize_prediction(encoded_pred, None, dir_output / f'encoded_clustering_frontier_{dept_name}.png')
            
        return metrics

    def visualize_prediction(self, pred_vis, target_vis, save_path, weights_vis=None):
        print(f"Visualizing prediction. Type: {type(pred_vis)}, Shape: {pred_vis.shape}")
        if target_vis is not None:
            print(f"Target Type: {type(target_vis)}, Shape: {target_vis.shape}")
            
        # Handle shapes (remove channel dim if present)
        if len(pred_vis.shape) == 4:
            pred_vis = pred_vis[0, 0]
        elif len(pred_vis.shape) == 3:
            pred_vis = pred_vis[0]
            
        if target_vis is not None:
            if len(target_vis.shape) == 4:
                target_vis = target_vis[0, 0]
            elif len(target_vis.shape) == 3:
                target_vis = target_vis[0]
            else:
                target_vis = target_vis
        else:
            target_vis = None
            
        plt.figure(figsize=(15, 5))
        
        # Determine vmin/vmax based on task type
        vmin = None
        vmax = None
        vmin = 0
        vmax = np.nanmax(target_vis)
            
        if target_vis is not None:
            plt.subplot(1, 3, 1)
            plt.title("Prediction")
            # Use 'jet' or 'tab10' for classification
            cmap = 'jet'
            plt.imshow(pred_vis, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.title("Target")
            cmap_target = 'jet'
            plt.imshow(target_vis, cmap=cmap_target, vmin=vmin, vmax=vmax)
            plt.colorbar()
            
            if weights_vis is not None:
                plt.subplot(1, 3, 3)
                plt.title("Sample Weights")
                plt.imshow(weights_vis, cmap='gray')
                plt.colorbar()
                
        else:
            plt.subplot(1, 1, 1)
            plt.title("Prediction")
            cmap = 'jet'
            plt.imshow(pred_vis, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.close()

    def save_dataframe(self, path):
        self.score.to_csv(path, index=False)
        print(f"Score dataframe saved to {path}")
