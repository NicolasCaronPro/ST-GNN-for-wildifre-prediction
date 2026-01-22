
import sys
from unittest.mock import MagicMock

# Mock the tools_2 module to avoid importing heavy dependencies like dgl
sys.modules['forecasting_models.pytorch.tools_2'] = MagicMock()

import torch
# Now import the losses
sys.path.insert(0,'/home/caron/Bureau/ST-GNN-for-wildifre-prediction/Prediction/GNN/')
from forecasting_models.pytorch.ordinal_loss import MCEAndWKLoss, DiceAndWKLoss, ForegroundDiceLossAndWKLoss

def test_loss_returns():
    num_classes = 5
    batch_size = 2
    # Create dummy predictions (logits) and targets
    y_pred = torch.randn(batch_size, num_classes, requires_grad=True)
    y_true = torch.randint(0, num_classes, (batch_size,))

    losses_to_test = [
        (MCEAndWKLoss(num_classes=num_classes), ['total_loss', 'mce', 'wk']),
        (DiceAndWKLoss(num_classes=num_classes), ['total_loss', 'dice', 'wk']),
        (ForegroundDiceLossAndWKLoss(num_classes=num_classes), ['total_loss', 'dice', 'wk']),
    ]

    for loss_fn, expected_keys in losses_to_test:
        print(f"Testing {type(loss_fn).__name__}...")
        try:
            result = loss_fn(y_pred, y_true)
            if not isinstance(result, dict):
                print(f"FAILED: {type(loss_fn).__name__} did not return a dict. Got {type(result)}")
                continue
            
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                print(f"FAILED: {type(loss_fn).__name__} missing keys: {missing_keys}")
                continue
            
            print(f"PASSED: {type(loss_fn).__name__} returned expected keys.")
            # print(f"  Result: {result}")

        except Exception as e:
            print(f"ERROR testing {type(loss_fn).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_loss_returns()
