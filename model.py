import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class CascadedRiskAwareFilter(nn.Module):
    def __init__(self, exit_threshold=0.1):
        super(CascadedRiskAwareFilter, self).__init__()
        self.exit_threshold = exit_threshold
        
        # --- Stage 1: Fast Convolutional Stem ---
        # Load lightweight pretrained base
        base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Extract the first 7 blocks (downsamples spatial res, extracts morphology)
        self.stem = nn.Sequential(*list(base_model.features.children())[:7])
        stem_out_channels = 40 # Channels at block 7 for mobilenet_v3_small
        
        # Early exit classifier
        self.early_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.early_classifier = nn.Linear(stem_out_channels, 1)
        
        # --- Stage 2: Structural Attention Scrutinizer ---
        # A single transformer encoder layer to capture global structural dependencies
        self.attention_norm = nn.LayerNorm(stem_out_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=stem_out_channels, 
            nhead=4, 
            dim_feedforward=128, 
            dropout=0.1, 
            batch_first=True
        )
        self.structural_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Final heavy classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(stem_out_channels, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Stage 1: Feature Extraction
        features = self.stem(x) # Shape: [B, C, H, W]
        
        # Early Classifier Path
        pooled_early = self.early_pool(features).flatten(1)
        early_logits = self.early_classifier(pooled_early)
        
        # Training Mode: Process both paths to calculate dual loss
        if self.training:
            final_logits = self._process_stage_two(features)
            return final_logits, early_logits
            
        # Inference Mode: Early Exit Logic for Deployment
        else:
            early_prob = torch.sigmoid(early_logits)
            
            # If batch size is 1 (standard edge deployment), exit dynamically
            if x.size(0) == 1:
                if early_prob.item() < self.exit_threshold:
                    return early_logits # Exit without running attention
                else:
                    return self._process_stage_two(features)
            
            # If batch size > 1, process all through stage 2 for vectorization safety
            return self._process_stage_two(features)

    def _process_stage_two(self, features):
        B, C, H, W = features.shape
        # Flatten spatial dimensions for Transformer sequence [B, SeqLen, Channels]
        seq = features.flatten(2).transpose(1, 2)
        seq = self.attention_norm(seq)
        
        # Apply Self-Attention
        attended_seq = self.structural_attention(seq)
        
        # Pool across the sequence dimension and classify
        attended_pooled = attended_seq.mean(dim=1)
        return self.final_classifier(attended_pooled)