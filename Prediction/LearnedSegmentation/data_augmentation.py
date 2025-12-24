import torch
import torchvision.transforms.functional as TF
import random
import logging

logger = logging.getLogger(__name__)

class DataAugmentor:
    def __init__(self, rotation_range=15):
        self.rotation_range = rotation_range
        logger.info(f"DataAugmentor initialized with rotation_range={rotation_range}")

    def __call__(self, inputs, labels, weights=None):
        """
        Apply random rotation to a batch of data.
        
        Args:
            inputs: (B, C, H, W) tensor
            labels: (B, C_out, H, W) tensor
            weights: (B, 1, H, W) tensor or None
            
        Returns:
            inputs, labels, weights (all rotated)
        """
        B = inputs.shape[0]
        
        augmented_inputs = []
        augmented_labels = []
        augmented_weights = []
        
        for i in range(B):
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            
            img = inputs[i]
            lbl = labels[i]
            w = weights[i] if weights is not None else None
            
            # Rotate inputs (Bilinear)
            # Assuming inputs are continuous features
            img_rot = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            
            # Rotate labels
            # Check if we have mixed types (e.g. frontier: mask + distance)
            if lbl.shape[0] == 2: # Frontier: channel 0 is mask (discrete), channel 1 is distance (continuous)
                mask = lbl[0:1]
                dist = lbl[1:2]
                mask_rot = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
                dist_rot = TF.rotate(dist, angle, interpolation=TF.InterpolationMode.BILINEAR)
                lbl_rot = torch.cat([mask_rot, dist_rot], dim=0)
            else:
                # Default to Nearest for labels (segmentation masks)
                lbl_rot = TF.rotate(lbl, angle, interpolation=TF.InterpolationMode.NEAREST)
                
            # Rotate weights
            if w is not None:
                # Weights are usually 0 or 1 (mask of valid area)
                # Use Nearest to keep them binary
                w_rot = TF.rotate(w, angle, interpolation=TF.InterpolationMode.NEAREST)
            else:
                w_rot = None
                
            augmented_inputs.append(img_rot)
            augmented_labels.append(lbl_rot)
            if w_rot is not None:
                augmented_weights.append(w_rot)
                
        inputs = torch.stack(augmented_inputs)
        labels = torch.stack(augmented_labels)
        if weights is not None:
            weights = torch.stack(augmented_weights)
            
        return inputs, labels, weights
