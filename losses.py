import torch
import torch.nn as nn
import torch.nn.functional as F

class RiskWeightedFocalLoss(nn.Module):
    """
    Combines Focal Loss with a dataset-driven risk taxonomy prior.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(RiskWeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, risk_weights):
        # Numerically stable BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute probabilities for the focal parameter
        pt = torch.exp(-bce_loss) 
        
        # Standard focal scaling
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        # Apply the structural risk prior from HNH40K
        # Ensure risk_weights matches the batch dimension
        weighted_focal_loss = focal_loss * risk_weights
        
        return torch.mean(weighted_focal_loss)