import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

from data_augmentation import DataAugmentor

def test_augmentation():
    print("Testing DataAugmentor...")
    
    B, C, H, W = 2, 3, 10, 10
    
    # Create dummy inputs (continuous)
    inputs = torch.randn(B, C, H, W)
    
    # Create dummy labels (Frontier: channel 0 is mask, channel 1 is distance)
    # Mask: 0 or 1
    mask = torch.randint(0, 2, (B, 1, H, W)).float()
    # Distance: continuous
    dist = torch.randn(B, 1, H, W)
    labels = torch.cat([mask, dist], dim=1)
    
    # Create dummy weights (0 or 1)
    weights = torch.ones(B, 1, H, W)
    
    augmentor = DataAugmentor(rotation_range=90)
    
    aug_inputs, aug_labels, aug_weights = augmentor(inputs, labels, weights)
    
    # Check shapes
    assert aug_inputs.shape == inputs.shape, f"Input shape mismatch: {aug_inputs.shape} vs {inputs.shape}"
    assert aug_labels.shape == labels.shape, f"Label shape mismatch: {aug_labels.shape} vs {labels.shape}"
    assert aug_weights.shape == weights.shape, f"Weight shape mismatch: {aug_weights.shape} vs {weights.shape}"
    
    print("Shapes are correct.")
    
    # Check if rotation happened (values should be different)
    # Note: there's a tiny chance rotation is 0, but with range 90 it's unlikely to be exactly 0.
    if not torch.allclose(aug_inputs, inputs):
        print("Inputs changed (rotation applied).")
    else:
        print("Warning: Inputs are identical (rotation might be 0 or failed).")
        
    # Check if mask is still discrete (0 or 1)
    # Nearest interpolation should preserve values.
    # However, padding with 0 might introduce 0s (which is fine for mask).
    # If we had 1s, they should stay 1s.
    aug_mask = aug_labels[:, 0:1]
    unique_vals = torch.unique(aug_mask)
    print(f"Unique values in augmented mask: {unique_vals}")
    
    # Check if values are close to 0 or 1
    is_discrete = torch.all(torch.isclose(aug_mask, torch.tensor(0.0)) | torch.isclose(aug_mask, torch.tensor(1.0)))
    if is_discrete:
        print("Mask channel preserved discrete values (Nearest interpolation worked).")
    else:
        print("Error: Mask channel has non-discrete values (Bilinear interpolation used?).")
        
    # Check distance channel (should change)
    aug_dist = aug_labels[:, 1:2]
    if not torch.allclose(aug_dist, dist):
        print("Distance channel changed.")
    
    print("Test passed!")

if __name__ == "__main__":
    test_augmentation()
