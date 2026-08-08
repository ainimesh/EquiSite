import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List
import math
from torch_geometric.nn.inits import glorot_orthogonal
from torch.nn.init import constant_, xavier_uniform_
from functools import partial
import inspect


##########################################################################################################################################################
#!################################################     Different Loss Functions to be used          ######################################################
##########################################################################################################################################################

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        """Dice loss.

        Args:
            smooth (int, optional): The smoothing factor for dice loss. Defaults to 1.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, targets):
        # Ensure the input are in [0, 1]
        probs = torch.sigmoid(input)

        # Flatten the predictions and targets
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate the Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice_coef = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        # Return the loss, which is 1 minus the Dice coefficient
        return 1 - dice_coef




class FocalContrastiveLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, margin=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.contrastive = ContrastiveLoss(margin)
        
    def forward(self, logits, y, embeddings):
        return self.focal(logits, y) + self.contrastive(embeddings, y)

class FocalLoss(nn.Module):
    """Multi-class focal loss"""
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings, labels):
        pos_mask = labels == 1
        neg_mask = ~pos_mask
        
        # Positive pairs
        pos_dist = torch.cdist(embeddings[pos_mask], embeddings[pos_mask])**2
        pos_loss = pos_dist.mean()
        
        # Negative pairs
        neg_dist = torch.cdist(embeddings[pos_mask], embeddings[neg_mask])
        neg_loss = F.relu(self.margin - neg_dist).mean()
        
        return pos_loss + neg_loss


class MultiScaleFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, scales=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.scales = scales
        
    def forward(self, preds, y):
        loss = 0.0
        for s in range(self.scales):
            p = preds[s]
            ce_loss = F.binary_cross_entropy(p, y, reduction='none')
            pt = torch.where(y == 1, p, 1-p)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            loss += focal_loss.mean() * (1/(s+1))  # Weight coarser scales higher
        return loss



class FocalLoss1(nn.Module):
    """
    Focal Loss for addressing class imbalance in binding site prediction.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            predictions: [N] logits or probabilities
            targets: [N] binary targets (0 or 1)
        """
        # Convert to probabilities if logits
        if predictions.dim() > 1:
            predictions = F.softmax(predictions, dim=-1)[..., 1]  # Get positive class prob
        else:
            predictions = torch.sigmoid(predictions)
        
        # Compute focal loss
        pt = predictions * targets + (1 - predictions) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = -focal_weight * torch.log(pt + 1e-8)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class FocalLoss2(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    
    Args:
        alpha: Weight for positive class (use 0.25 for 1:3 imbalance, higher for more imbalance)
        gamma: Focusing parameter (2.0 is standard)
        logits: If True, expects raw logits; if False, expects probabilities
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: [N] raw logits or probabilities  
            target: [N] binary targets (0 or 1)
        """
        if self.logits:
            # Use numerically stable BCE with logits
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pred_prob = torch.sigmoid(pred)
        else:
            # Direct BCE for probabilities
            bce = F.binary_cross_entropy(pred, target, reduction='none')
            pred_prob = pred
        
        # Probability of correct class
        pt = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        # Alpha weighting (higher alpha = more weight on positive class)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal term: (1 - pt)^gamma reduces loss for well-classified samples
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        loss = focal_weight * bce
        
        return loss.mean()


    
class ConfidenceLoss(nn.Module):
    def __init__(self, gamma: float = 4.0, c0: float = 0.01):
        super().__init__()
        self.gamma = gamma
        self.c0 = c0
        self.loss_fn = nn.MSELoss()

    def forward(self, predicted_confidence: torch.Tensor, dcc_distances: torch.Tensor) -> torch.Tensor:
        dists = dcc_distances.detach()
        target_confidence = torch.where(
            dists <= self.gamma,
            1.0 - (dists / (2 * self.gamma)),
            torch.tensor(self.c0, device=dists.device)
        )
        return self.loss_fn(predicted_confidence.squeeze(), target_confidence)
    

class GeometricEnhancedLoss(nn.Module):
    """
    Your current Focal+Dice loss enhanced with geometric information
    """
    def __init__(self, 
                focal_alpha=0.25, 
                focal_gamma=2.0,
                dice_smooth=1.0,
                surface_weight=0.1):  # New geometric component
        super().__init__()
        self.focal_loss = FocalLoss2(alpha=focal_alpha, gamma=focal_gamma, logits=True)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.surface_weight = surface_weight
    
    def forward(self, logits, targets, X_steerable):
        # Your existing losses
        focal = self.focal_loss(logits, targets.float())
        dice = self.dice_loss(logits, targets.float())
        
        # New: Surface preference using l=2 harmonics
        surface = self._surface_curvature_loss(X_steerable, targets.float())
        
        total_loss = focal + dice + self.surface_weight * surface
        
        return total_loss
    
    def _surface_curvature_loss(self, X_steerable, targets):
        """Penalize binding predictions in non-surface regions"""
        if X_steerable.size(1) < 8:  # Need l=2 harmonics
            return torch.tensor(0.0, device=X_steerable.device)
        
        # l=2 harmonics (surface curvature): X[:, 3:8, :]
        l2_harmonics = X_steerable[:, 3:8, :]
        surface_score = torch.norm(l2_harmonics, dim=(1,2))
        
        # Binding sites should have high surface activity
        # Penalize predictions where surface activity is low
        surface_penalty = targets *torch.relu(0.5 - surface_score)
        
        return torch.mean(surface_penalty)
    

# In utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalLoss(nn.Module):
    """
    Computes the cosine similarity based loss for direction prediction.
    This version computes the loss over ALL atoms, not just binding site atoms.
    """
    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon (float): A small value to prevent division by zero during normalization.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, 
                predicted_directions: torch.Tensor, 
                true_directions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_directions (torch.Tensor): The predicted direction vectors from the model.
                                                Shape: [num_nodes, 3]
            true_directions (torch.Tensor): The pre-calculated ground truth direction vectors.
                                            Shape: [num_nodes, 3]
        Returns:
            torch.Tensor: A scalar tensor representing the mean directional loss for the batch.
        """
        # Normalize both sets of vectors for a stable cosine similarity calculation
        pred_dirs_norm = F.normalize(predicted_directions, p=2, dim=-1, eps=self.epsilon)
        true_dirs_norm = F.normalize(true_directions, p=2, dim=-1, eps=self.epsilon)

        # Cosine loss is defined as 1 - cosine_similarity
        cosine_loss = 1 - F.cosine_similarity(pred_dirs_norm, true_dirs_norm, dim=-1)

        # Return the mean of the loss across all atoms in the batch
        return cosine_loss.mean()