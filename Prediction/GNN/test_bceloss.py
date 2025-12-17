import sys
from pathlib import Path
import torch
from unittest.mock import MagicMock
import types

# Mock dgl
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

# Mock blitz
blitz_mock = types.ModuleType('blitz')
sys.modules['blitz'] = blitz_mock
sys.modules['blitz.modules'] = MagicMock()
sys.modules['blitz.utils'] = MagicMock()
sys.modules['blitz.losses'] = MagicMock()

# Mock other potential dependencies
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

# Add GNN path
sys.path.append('/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN')

try:
    from forecasting_models.pytorch.classification_loss import BCELoss
    print("Successfully imported BCELoss")
    
    criterion = BCELoss()
    print("Successfully instantiated BCELoss")
    
    y_pred = torch.tensor([0.1, 0.9], dtype=torch.float32)
    y_true = torch.tensor([0.0, 1.0], dtype=torch.float32)
    
    loss = criterion(y_pred, y_true)
    print(f"Loss: {loss.item()}")
    
    # Test with weights
    weights = torch.tensor([0.5, 2.0], dtype=torch.float32)
    loss_weighted = criterion(y_pred, y_true, sample_weight=weights)
    print(f"Weighted Loss: {loss_weighted.item()}")
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
