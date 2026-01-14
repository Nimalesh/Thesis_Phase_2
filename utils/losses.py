import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class DiceFocalLoss(nn.Module):
    def __init__(self, smooth=1e-6, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.smooth = smooth
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits, targets):
        # Dice Loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        # Focal Loss
        focal_loss = self.focal(logits, targets)
        return dice_loss + focal_loss