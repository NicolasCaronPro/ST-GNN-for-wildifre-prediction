
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

class Tester:
    def __init__(self, model_path, model_params):
        self.model_path = Path(model_path)
        self.model_params = model_params
        self.model_type = model_params.get('type', 'XGBRegressor')
        self.params = model_params.get('params', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._load_model()

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.model_type == 'UNet':
            model = UNet(
                n_channels=self.params['n_channels'],
                out_channels=self.params['out_channels'],
                conv_channels=self.params['conv_channels'],
                bilinear=self.params.get('bilinear', False)
            )
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model
        else:
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)

    def test(self, X, y, weights=None, dept_name="unknown", compute_metrics=True):
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
                probs = outputs
                if self.params.get('task_type') == 'regression':
                    preds = outputs
                else:
                    preds = torch.argmax(outputs, dim=-1)
                            
            if compute_metrics:
                if self.params.get('task_type') == 'regression':
                    # Calculate MSE/MAE
                    y_np = y.cpu().numpy().squeeze()
                    preds_np = preds.cpu().numpy().squeeze()
                    
                    mse = np.mean((y_np - preds_np) ** 2)
                    mae = np.mean(np.abs(y_np - preds_np))
                    print(f"MSE: {mse}")
                    print(f"MAE: {mae}")
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
                
            vis_path = self.model_path.parent / f'prediction_vis_{dept_name}.png'
            self.visualize_prediction(pred_vis, target_vis, vis_path, weights_vis)
            
            return preds.cpu().numpy()

        else:
            predictions = self.model.predict(X)
            if compute_metrics:
                # Calculate metrics
                mse = np.mean((y - predictions) ** 2)
                mae = np.mean(np.abs(y - predictions))
                print(f"MSE: {mse}")
                print(f"MAE: {mae}")
            return predictions

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
        print(self.params.get('task_type'))
        if self.params.get('task_type') == 'classification':
            vmin = 0
            vmax = np.max(target_vis)
            
        if target_vis is not None:
            plt.subplot(1, 3, 1)
            plt.title("Prediction")
            # Use 'jet' or 'tab10' for classification
            if self.params.get('task_type') == 'classification':
                cmap = 'jet'
            else:
                cmap = 'jet' if len(np.unique(pred_vis)) > 2 else 'gray'
            plt.imshow(pred_vis, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.title("Target")
            if self.params.get('task_type') == 'classification':
                cmap_target = 'jet'
            else:
                cmap_target = 'jet' if len(np.unique(target_vis)) > 2 else 'gray'
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
            if self.params.get('task_type') == 'classification':
                cmap = 'jet'
            else:
                cmap = 'jet' if len(np.unique(pred_vis)) > 2 else 'gray'
            plt.imshow(pred_vis, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.close()
