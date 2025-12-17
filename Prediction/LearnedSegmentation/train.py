
import sys
import numpy as np
import logging
from pathlib import Path
import pickle

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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from GNN.forecasting_models.sklearn.models import MyXGBRegressor, MyXGBClassifier
from GNN.forecasting_models.pytorch.models_2D import UNet
from GNN.train import get_loss_function
# Import other models if needed

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model_params):
        self.model_params = model_params
        self.model_type = model_params.get('type', 'XGBRegressor')
        self.params = model_params.get('params', {})
        self.epochs = model_params.get('epochs', 10)
        self.batch_size = model_params.get('batch_size', 1)
        self.learning_rate = model_params.get('learning_rate', 0.001)
        self.loss_name = model_params.get('loss', 'bceloss')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self._initialize_model()

    def _initialize_model(self):
        logger.info(f"Initializing model: {self.model_type}")
        if self.model_type == 'UNet':
            model = UNet(
                n_channels=self.params['n_channels'],
                out_channels=self.params['out_channels'],
                conv_channels=self.params['conv_channels'],
                bilinear=self.params.get('bilinear', False),
                task_type=self.params.get('task_type', 'classification')
            )
            model = model.to(self.device)
            return model
        elif self.model_type == 'XGBRegressor':
            return MyXGBRegressor(**self.params)
        elif self.model_type == 'XGBClassifier':
            return MyXGBClassifier(**self.params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X, y):
        logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'UNet':
            self._train_unet(X, y)
        else:
            # Scikit-learn style models
            self.model.fit(X, y)
            
        logger.info("Training completed.")

    def _train_unet(self, X, y):
        # Convert to tensors if they are numpy arrays
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        # Create DataLoader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and Optimizer
        # Use get_loss_function from GNN.train
        # Note: get_loss_function might return a class instance or class?
        # Based on GNN/train.py: return loss_factories[loss_name]() -> Instance
        if self.loss_name == 'bceloss':
            criterion = nn.BCELoss()
        else:
            criterion = get_loss_function(self.loss_name)
        criterion = criterion.to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                # UNet forward returns: output, logits, hidden
                outputs, logits, hidden = self.model(inputs)
                
                # Use outputs (probabilities) for BCELoss, or logits for BCEWithLogitsLoss
                # Assuming BCELoss as per config
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss / len(dataloader)}")

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'UNet':
            torch.save(self.model.state_dict(), path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")

    def evaluate(self, X, y):
        # Basic evaluation
        if self.model_type == 'UNet':
            # Not implemented for now in Trainer, usually done in Tester
            pass
        else:
            score = self.model.score(X, y)
            logger.info(f"Model Score: {score}")
            return score
