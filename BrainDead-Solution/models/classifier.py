import torch
import torch.nn as nn
import torch.nn.functional as F


class MixMLPClassifier(nn.Module):
    def __init__(self, input_dim=256, num_labels=14):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_dim = 512
        
        # (A) Residual Path: 256 → 256 → 14
        self.residual_path = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, num_labels)
        )
        
        # (B) Expansion Path: 256 → 512 → 256 → 14
        self.expansion_path = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_labels)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, organ_feature, region_features=None):
        """
        Forward pass through MIX-MLP classifier.
        
        Args:
            organ_feature: Tensor (B, 256) - global organ-level features
            region_features: Tensor (B, R, 256) or None - optional region features
        
        Returns:
            dict containing:
                - disease_logits: Tensor (B, 14) - raw classification logits
                - disease_probs: Tensor (B, 14) - sigmoid probabilities
        """
        # Integrate region features if provided
        if region_features is not None:
            # Mean pool across regions: (B, R, 256) → (B, 256)
            region_pooled = region_features.mean(dim=1)  # (B, 256)
            # Add to organ feature
            combined_feature = organ_feature + region_pooled  # (B, 256)
        else:
            combined_feature = organ_feature  # (B, 256)
        
        # (A) Residual path
        residual_logits = self.residual_path(combined_feature)  # (B, 14)
        
        # (B) Expansion path
        expansion_logits = self.expansion_path(combined_feature)  # (B, 14)
        
        # Combine both paths by summing logits
        final_logits = residual_logits + expansion_logits  # (B, 14)
        
        # Compute probabilities
        disease_probs = torch.sigmoid(final_logits)  # (B, 14)
        
        return {
            'disease_logits': final_logits,
            'disease_probs': disease_probs
        }


def mixmlp_loss(disease_logits, targets):
    """
    Multi-label BCE loss for CheXpert labels.
    
    Args:
        disease_logits: Tensor (B, 14) - raw logits from classifier
        targets: Tensor (B, 14) - binary ground truth labels
                 Values should be in {0, 1} or {-1, 0, 1} for CheXpert
    
    Returns:
        loss: Scalar tensor - BCE loss with logits
    
    Note:
        BCEWithLogitsLoss combines sigmoid + BCE for numerical stability.
        For CheXpert uncertain labels (-1), you may want to mask them out.
    """
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    # Handle CheXpert uncertain labels (-1) by masking
    # Only compute loss on certain labels (0 or 1)
    mask = (targets >= 0).float()  # Mask: 1 where label is certain, 0 where uncertain
    
    if mask.sum() > 0:
        # Apply mask to both logits and targets
        masked_logits = disease_logits * mask
        masked_targets = targets * mask
        
        # Compute loss only on certain labels
        loss = criterion(masked_logits, masked_targets.float())
        
        # Scale by the proportion of certain labels
        loss = loss * (mask.numel() / mask.sum())
    else:
        # All labels are uncertain - return zero loss
        loss = torch.tensor(0.0, device=disease_logits.device, requires_grad=True)
    
    return loss