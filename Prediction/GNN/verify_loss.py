
import torch
import sys
import os

# Add the project root to sys.path so imports work
sys.path.insert(0, '/Home/Users/ncaron/WORK/GNN')

from forecasting_models.pytorch.ordinal_loss import FocalLoss, FocalLossAndWKLoss

def test_focal_loss():
    print("Testing FocalLoss...")
    num_classes = 5
    batch_size = 4
    
    # Random logits
    y_pred = torch.randn(batch_size, num_classes, requires_grad=True)
    # Random targets
    y_true = torch.randint(0, num_classes, (batch_size,))
    
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    loss = criterion(y_pred, y_true)
    
    print(f"Focal Loss value: {loss.item()}")
    loss.backward()
    print("Focal Loss backward pass successful.")

def test_focal_and_wk_loss():
    print("\nTesting FocalLossAndWKLoss...")
    num_classes = 5
    batch_size = 4
    
    # Random logits
    y_pred = torch.randn(batch_size, num_classes, requires_grad=True)
    # Random targets
    y_true = torch.randint(0, num_classes, (batch_size,))
    
    criterion = FocalLossAndWKLoss(num_classes=num_classes, C=0.5, learned=True)
    loss = criterion(y_pred, y_true)
    
    print(f"FocalLossAndWKLoss value: {loss.item()}")
    loss.backward()
    print("FocalLossAndWKLoss backward pass successful.")
    
    # Check learnable parameter
    params = criterion.get_learnable_parameters()
    print(f"Learnable parameters: {params}")

if __name__ == "__main__":
    try:
        test_focal_loss()
        test_focal_and_wk_loss()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
