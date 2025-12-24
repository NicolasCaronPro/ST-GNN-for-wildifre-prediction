
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

    def test(self, X, y, weights=None, dept_name="unknown", compute_metrics=True, output_dir=None):
        
        B, F, H, W = X.shape

        print("Testing model...")
        
        if self.model_type == 'UNet':
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
                else:
                    preds = torch.argmax(outputs, dim=-1)

            metrics = {'department': dept_name}

            if weights is not None:
                preds[weights == 0] = np.nan
                y[weights == 0] = np.nan

            if compute_metrics:
                if self.params.get('task_type') == 'regression':
                    # Calculate MSE/MAE
                    y_np = y.cpu().numpy().squeeze()
                    preds_np = preds.cpu().numpy().squeeze()
                    
                    mse = np.mean((y_np - preds_np) ** 2)
                    mae = np.mean(np.abs(y_np - preds_np))
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

                    from sklearn.metrics import jaccard_score
                    iou = jaccard_score(y_flat, preds_flat, average='macro')
                    print(f"Mean IoU: {iou}")
                    metrics['IoU'] = iou
                    
            # Visualize first sample
            pred_vis = preds[0].cpu().numpy()
            target_vis = y[0].cpu().numpy()
            
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

            if self.params.get('task_type') == 'regression':
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
            
            if compute_metrics:
                self.score = pd.concat([self.score, pd.DataFrame([metrics])], ignore_index=True)
            
            return preds.cpu().numpy()

        else:
            predictions = self.model.predict(X)
            metrics = {'department': dept_name}

            if compute_metrics:
                # Calculate metrics
                mse = np.mean((y - predictions) ** 2)
                mae = np.mean(np.abs(y - predictions))
                print(f"MSE: {mse}")
                print(f"MAE: {mae}")
                metrics['MSE'] = mse
                metrics['MAE'] = mae
            
            if self.params.get('task_type') == 'regression':
                 clustering_metrics = self.process_regression_risk(predictions, y, dept_name, output_dir)
                 if clustering_metrics:
                     metrics.update(clustering_metrics)
            elif self.params.get('task_type') == 'binary' and self.config.get_target_type() == 'frontier':
                 clustering_metrics = self.process_binary_frontier(predictions, y, dept_name, output_dir)
                 if clustering_metrics:
                     metrics.update(clustering_metrics)

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
            raster = resize(raster_obj[0], (y.shape[2], y.shape[3]), anti_aliasing=False, preserve_range=True, order=0)

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
        pred_seg = seg.create_geometry_with_watershed(
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

        pred_seg[pred_seg == -1] = np.nan

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
                iou = jaccard_score(encoded_gt.flatten(), encoded_pred.flatten(), average='macro')
                print(f"IoU (Macro): {iou}")
                metrics['IoU'] = iou
                
            vis_path = dir_output / 'comparison_vis.png'
            self.visualize_prediction(encoded_pred, encoded_gt, vis_path)
            print(f"Comparison visualization saved to {vis_path}")
        else:
            print("Ground truth time_series_clustering not found. Skipping comparison.")
            self.visualize_prediction(encoded_pred, None, dir_output / f'encoded_clustering_{dept_name}.png')
            print(f"Encoded clustering saved to {dir_output / 'encoded_clustering.png'}")
            
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

    def process_binary_frontier(self, predictions, y, dept_name, output_dir=None):
        print(f"Processing binary frontier for {dept_name}...")
        from LearnedSegmentation.frontier_utils import connected_components_8
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
        
        # 1. Create Geometry with Watershed
        resolution = '2x2'
        dir_raster = root_target / sinister / dataset_name / sinister_encoding / 'raster' / resolution
        raster_obj = read_object(f'{dept_name}rasterScale0.pkl', dir_raster)

        assert raster_obj is not None, f"rasterScale0.pkl not found at {dir_raster}"

        pred_seg = resize(pred_seg, (raster_obj.shape[1], raster_obj.shape[2]), order=0)

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
            
            from sklearn.metrics import jaccard_score
            if encoded_pred.shape != encoded_gt.shape:
                print(f"Shape mismatch: Pred {encoded_pred.shape}, GT {encoded_gt.shape}")
            else:
                iou = jaccard_score(encoded_gt.flatten(), encoded_pred.flatten(), average='macro')
                print(f"Clustering IoU (Macro): {iou}")
                metrics['IoU'] = iou
                
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
        vmax = np.max(target_vis)
            
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
