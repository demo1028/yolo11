# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Prior-Guided Attention Module for Dental YOLO.

Design Philosophy:
- RGB image goes through standard backbone (edge/texture extraction)
- Prior channels (M_virtual, Distance, M_metal) act as spatial attention
- Prior modulates features at semantic-appropriate depths (P3-P5)
- Preserves prior's semantic meaning instead of treating it as texture

Architecture:
    Input: 4-channel [I_raw, M_virtual, Distance, M_metal]
           ↓
    ┌──────┴──────┐
    │             │
    I_raw      Prior (3ch)
    │             │
    Backbone    PriorEncoder (lightweight)
    │             │
    P2,P3,P4,P5   Attention Maps
    │             │
    └──────┬──────┘
           ↓
    Prior-Guided Features (attention fusion at P3,P4,P5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorEncoder(nn.Module):
    """
    Lightweight encoder for prior channels.
    Preserves spatial resolution while extracting multi-scale attention maps.

    Input: (B, 3, H, W) - [M_virtual, Distance, M_metal]
    Output: Dict of attention maps at different scales
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # Shallow convolutions to process prior (preserve semantics)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )

        # Multi-scale attention heads
        # P3: 1/8 scale
        self.to_p3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )

        # P4: 1/16 scale
        self.to_p4 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )

        # P5: 1/32 scale
        self.to_p5 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, prior):
        """
        Args:
            prior: (B, 3, H, W) prior channels

        Returns:
            dict with keys 'p3', 'p4', 'p5', each (B, C, H/s, W/s)
        """
        x = self.conv1(prior)  # (B, 32, H, W)

        p3 = self.to_p3(x)     # (B, 32, H/8, W/8)
        p4 = self.to_p4(p3)    # (B, 32, H/16, W/16)
        p5 = self.to_p5(p4)    # (B, 32, H/32, W/32)

        return {'p3': p3, 'p4': p4, 'p5': p5}


class PriorGuidedAttention(nn.Module):
    """
    Fuses backbone features with prior attention.

    Prior features are projected to channel attention weights,
    then applied to backbone features via element-wise multiplication.
    """

    def __init__(self, backbone_channels, prior_channels=32):
        super().__init__()

        # Project prior features to attention weights
        self.attention_proj = nn.Sequential(
            nn.Conv2d(prior_channels, backbone_channels, 1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.Sigmoid(),  # Attention weights in [0, 1]
        )

        # Optional: learnable residual weight
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, backbone_feat, prior_feat):
        """
        Args:
            backbone_feat: (B, C, H, W) features from backbone
            prior_feat: (B, 32, H, W) features from prior encoder

        Returns:
            (B, C, H, W) attention-modulated features
        """
        # Ensure spatial alignment
        if backbone_feat.shape[2:] != prior_feat.shape[2:]:
            prior_feat = F.interpolate(
                prior_feat,
                size=backbone_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Compute attention weights
        attn = self.attention_proj(prior_feat)  # (B, C, H, W), values in [0, 1]

        # Apply attention with residual connection
        # out = feat * (1 + gamma * attn) allows gradual learning
        out = backbone_feat * (1 + self.gamma * attn)

        return out


class PriorSplitter(nn.Module):
    """
    Splits 4-channel input into RGB (1ch expanded to 3ch) and Prior (3ch).
    First layer of the modified backbone.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W) input tensor
               Ch0: I_raw
               Ch1: M_virtual
               Ch2: Distance
               Ch3: M_metal

        Returns:
            rgb: (B, 3, H, W) - I_raw repeated 3 times for standard backbone
            prior: (B, 3, H, W) - [M_virtual, Distance, M_metal]
        """
        rgb = x[:, 0:1, :, :].repeat(1, 3, 1, 1)  # (B, 3, H, W)
        prior = x[:, 1:4, :, :]  # (B, 3, H, W)
        return rgb, prior


class DentalPriorBackbone(nn.Module):
    """
    Wrapper that adds prior-guided attention to any backbone.

    Usage:
        backbone = DentalPriorBackbone(original_backbone)
        features = backbone(x_4ch)  # x_4ch: (B, 4, H, W)
    """

    def __init__(self, backbone, inject_layers=('p3', 'p4', 'p5')):
        """
        Args:
            backbone: Original YOLO backbone (expects 3-channel input)
            inject_layers: Which layers to inject prior attention
        """
        super().__init__()

        self.splitter = PriorSplitter()
        self.backbone = backbone
        self.prior_encoder = PriorEncoder(in_channels=3, base_channels=32)
        self.inject_layers = inject_layers

        # Create attention modules for each injection point
        # Channel sizes for YOLO11s: P3=256, P4=512, P5=512
        self.prior_attention = nn.ModuleDict()

    def _get_backbone_channels(self, layer_name):
        """Get channel count for a backbone layer."""
        # Default YOLO11s channel sizes
        channels = {'p2': 128, 'p3': 256, 'p4': 512, 'p5': 512}
        return channels.get(layer_name, 256)

    def _ensure_attention_module(self, layer_name, channels):
        """Lazily create attention module if needed."""
        if layer_name not in self.prior_attention:
            self.prior_attention[layer_name] = PriorGuidedAttention(
                backbone_channels=channels,
                prior_channels=32
            ).to(next(self.parameters()).device)

    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W) 4-channel input

        Returns:
            List of feature maps [P3, P4, P5] or similar
        """
        # Split input
        rgb, prior = self.splitter(x)

        # Encode prior
        prior_feats = self.prior_encoder(prior)

        # Run backbone and inject attention at specified layers
        # This requires modifying how backbone forward works
        # For now, return backbone features directly
        # (Full integration requires backbone-specific modifications)

        features = self.backbone(rgb)

        # Apply prior attention to output features
        if isinstance(features, (list, tuple)):
            out_features = []
            layer_names = ['p3', 'p4', 'p5'][:len(features)]

            for i, (feat, layer_name) in enumerate(zip(features, layer_names)):
                if layer_name in self.inject_layers and layer_name in prior_feats:
                    channels = feat.shape[1]
                    self._ensure_attention_module(layer_name, channels)
                    feat = self.prior_attention[layer_name](feat, prior_feats[layer_name])
                out_features.append(feat)

            return out_features
        else:
            return features


# ============================================================================
# Alternative: Simpler Prior Injection via Input Modification
# ============================================================================

class PriorAsAttentionInput(nn.Module):
    """
    Simpler approach: Convert 4ch input to 3ch + attention mask.

    Instead of modifying backbone, we:
    1. Use I_raw as the main input (expanded to 3ch)
    2. Pre-compute a "prior attention mask" from M_virtual, Distance, M_metal
    3. Apply this mask to I_raw before feeding to backbone

    This preserves standard YOLO architecture while incorporating prior guidance.
    """

    def __init__(self, method='multiply'):
        """
        Args:
            method: How to apply prior attention
                - 'multiply': rgb * (1 + prior_attn)
                - 'add': rgb + prior_attn * scale
                - 'concat_reduce': concat then 1x1 conv to 3ch
        """
        super().__init__()
        self.method = method

        # Learn how to combine 3 prior channels into 1 attention map
        self.prior_to_attn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        if method == 'concat_reduce':
            # 4ch -> 3ch learned projection
            self.reduce = nn.Sequential(
                nn.Conv2d(4, 3, 1, bias=False),
                nn.BatchNorm2d(3),
            )

        # Learnable scale for attention strength
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W) input

        Returns:
            (B, 3, H, W) RGB with prior guidance baked in
        """
        rgb = x[:, 0:1, :, :]      # (B, 1, H, W)
        prior = x[:, 1:4, :, :]    # (B, 3, H, W)

        if self.method == 'multiply':
            attn = self.prior_to_attn(prior)  # (B, 1, H, W)
            rgb_attended = rgb * (1 + self.scale * attn)
            return rgb_attended.repeat(1, 3, 1, 1)

        elif self.method == 'add':
            attn = self.prior_to_attn(prior)
            rgb_attended = rgb + self.scale * attn
            return rgb_attended.repeat(1, 3, 1, 1)

        elif self.method == 'concat_reduce':
            return self.reduce(x)

        else:
            raise ValueError(f"Unknown method: {self.method}")


# ============================================================================
# Registration for YOLO
# ============================================================================

def register_prior_attention_modules():
    """Register prior attention modules for YOLO model building."""
    from ultralytics.nn.tasks import YOLO

    # Add to module registry if using custom YAML
    import ultralytics.nn.modules as modules

    modules.PriorEncoder = PriorEncoder
    modules.PriorGuidedAttention = PriorGuidedAttention
    modules.PriorSplitter = PriorSplitter
    modules.DentalPriorBackbone = DentalPriorBackbone
    modules.PriorAsAttentionInput = PriorAsAttentionInput

    print("Registered Prior Attention modules: PriorEncoder, PriorGuidedAttention, "
          "PriorSplitter, DentalPriorBackbone, PriorAsAttentionInput")
