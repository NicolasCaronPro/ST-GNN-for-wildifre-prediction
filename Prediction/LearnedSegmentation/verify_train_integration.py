import sys
import torch
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'GNN'))

# Mock modules that might cause import errors
import types
sys.modules['dgl'] = MagicMock()
sys.modules['dgl.nn'] = MagicMock()
sys.modules['dgl.nn.pytorch'] = MagicMock()
sys.modules['blitz'] = MagicMock()
sys.modules['blitz.modules'] = MagicMock()
sys.modules['blitz.utils'] = MagicMock()
sys.modules['blitz.losses'] = MagicMock()
sys.modules['pygam'] = MagicMock()
sys.modules['ngboost'] = MagicMock()
sys.modules['ngboost.distns'] = MagicMock()
sys.modules['ngboost.scores'] = MagicMock()
sys.modules['skopt'] = MagicMock()
sys.modules['skopt.space'] = MagicMock()
sys.modules['osmnx'] = MagicMock()
sys.modules['dtaidistance'] = MagicMock()
sys.modules['dtwParallel'] = MagicMock()

from train import Trainer

def test_train_integration():
    print("Testing Trainer integration with DataAugmentor...")
    
    # Config with augmentation enabled
    model_params = {
        "type": "UNet",
        "params": {
            "n_channels": 3,
            "out_channels": 2,
            "conv_channels": [16, 32],
            "task_type": "classification",
            "augmentation": True,
            "rotation_range": 30
        },
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "loss": "weightedcrossentropy" # Use a simple loss
    }
    
    # Dummy data
    B, C, H, W = 4, 3, 16, 16
    X = torch.randn(B, C, H, W)
    y = torch.randint(0, 2, (B, H, W)).long() # Classification target
    weights = torch.ones(B, 1, H, W)
    
    # Mock DataAugmentor
    with patch('train.DataAugmentor') as MockAugmentor:
        # Setup mock instance
        mock_instance = MockAugmentor.return_value
        mock_instance.return_value = (X, y, weights) # Return same data
        
        trainer = Trainer(model_params)
        
        # Run training for 1 epoch
        trainer.train(X, y, weights)
        
        # Check if DataAugmentor was initialized with correct params
        MockAugmentor.assert_called_with(rotation_range=30)
        print("DataAugmentor initialized correctly.")
        
        # Check if augmentor was called
        # It should be called for each batch.
        # Batch size 2, total 4 samples -> 2 batches.
        assert mock_instance.call_count >= 2
        print(f"DataAugmentor called {mock_instance.call_count} times (expected >= 2).")
        
    print("Integration test passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_train_integration()
