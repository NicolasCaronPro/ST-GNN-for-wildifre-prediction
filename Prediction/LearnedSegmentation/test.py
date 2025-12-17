
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

    def test(self, X, y, compute_metrics=True):
        print("Testing model...")
        
        if self.model_type == 'UNet':
            return self._test_unet(X, y, compute_metrics)
        else:
            predictions = self.model.predict(X)
            if compute_metrics:
                # Calculate metrics
                mse = np.mean((y - predictions) ** 2)
                mae = np.mean(np.abs(y - predictions))
                print(f"MSE: {mse}")
                print(f"MAE: {mae}")
            return predictions

    def _test_unet(self, X, y, compute_metrics=True):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            outputs, logits, hidden = self.model(X)
            # Apply sigmoid for binary classification
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
        if compute_metrics:
            # Calculate IoU
            # y is (B, 1, H, W), preds is (B, 1, H, W)
            y_np = y.cpu().numpy().squeeze()
            preds_np = preds.cpu().numpy().squeeze()
            
            # Calculate IoU per sample
            ious = []
            if len(y_np.shape) == 2: # Single sample (H, W)
                iou = iou_binary(preds_np, y_np)
                ious.append(iou)
            else:
                for i in range(y_np.shape[0]):
                    iou = iou_binary(preds_np[i], y_np[i])
                    ious.append(iou)
            
            mean_iou = np.mean(ious)
            print(f"Mean IoU: {mean_iou}")
        
        return preds.cpu().numpy().squeeze()

    def visualize_prediction(self, prediction, target, save_path):
        # Prediction and target are 2D arrays (H, W)
        # Or 3D (N, H, W)
        
        print(f"Visualizing prediction. Type: {type(prediction)}, Shape: {prediction.shape}")
        if target is not None:
             print(f"Target Type: {type(target)}, Shape: {target.shape}")
             
        # Handle prediction shape
        if len(prediction.shape) == 4:
            pred_vis = prediction[0, 0]
        elif len(prediction.shape) == 3:
            pred_vis = prediction[0]
        else:
            pred_vis = prediction
            
        # Handle target shape
        if target is not None:
            if len(target.shape) == 4:
                target_vis = target[0, 0]
            elif len(target.shape) == 3:
                target_vis = target[0]
            else:
                target_vis = target
        else:
            target_vis = None
            
        plt.figure(figsize=(10, 5))
        
        if target_vis is not None:
            plt.subplot(1, 2, 1)
            plt.title("Prediction")
            plt.imshow(pred_vis, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("Target")
            plt.imshow(target_vis, cmap='gray')
        else:
            plt.subplot(1, 1, 1)
            plt.title("Prediction")
            plt.imshow(pred_vis, cmap='gray')
            
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.close()
