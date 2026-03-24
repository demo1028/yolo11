# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Small Object Detection Enhancement Modules.

Specialized attention and feature enhancement modules designed to improve
detection of small dental lesions (periapical, periodontal lesions).

Key strategies:
1. High-resolution feature preservation
2. Context-aware attention for small regions
3. Prior-guided spatial attention using Stage1 masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = (
    "SmallObjectAttention",
    "HighResolutionFusion",
    "PriorGuidedSpatialAttention",
    "ScaleAwareAttention",
    "LesionFocusModule",
)


class SmallObjectAttention(nn.Module):
    """
    Attention module specifically designed for small object detection.

    Uses dilated convolutions to capture multi-scale context while
    preserving spatial resolution critical for small objects.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize Small Object Attention.

        Args:
            channels: Number of input/output channels
            reduction: Channel reduction ratio
        """
        super().__init__()

        # Multi-scale dilated convolutions for context
        self.dilated_conv1 = nn.Conv2d(channels, channels // 4, 3, padding=1, dilation=1)
        self.dilated_conv2 = nn.Conv2d(channels, channels // 4, 3, padding=2, dilation=2)
        self.dilated_conv3 = nn.Conv2d(channels, channels // 4, 3, padding=4, dilation=4)
        self.dilated_conv4 = nn.Conv2d(channels, channels // 4, 3, padding=8, dilation=8)

        # Fusion and attention
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Spatial attention for small regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale context aggregation."""
        # Multi-scale feature extraction
        d1 = self.dilated_conv1(x)
        d2 = self.dilated_conv2(x)
        d3 = self.dilated_conv3(x)
        d4 = self.dilated_conv4(x)

        # Concatenate and fuse
        multi_scale = torch.cat([d1, d2, d3, d4], dim=1)
        fused = self.fusion(multi_scale)

        # Generate spatial attention
        attention = self.spatial_attention(fused)

        # Apply attention with residual
        return x + x * attention


class HighResolutionFusion(nn.Module):
    """
    Fuses high-resolution features with deep features for small object detection.

    Addresses the problem of small objects being lost in deep feature maps
    by explicitly bringing in high-resolution information.
    """

    def __init__(self, high_res_channels: int, deep_channels: int, out_channels: int):
        """
        Initialize High Resolution Fusion.

        Args:
            high_res_channels: Channels from high-resolution feature map
            deep_channels: Channels from deep feature map
            out_channels: Output channels after fusion
        """
        super().__init__()

        # Process high-res features
        self.high_res_conv = nn.Sequential(
            nn.Conv2d(high_res_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Process deep features
        self.deep_conv = nn.Sequential(
            nn.Conv2d(deep_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Attention-based fusion
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_res: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            high_res: High-resolution feature map
            deep: Deep feature map (will be upsampled)

        Returns:
            Fused feature map at high resolution
        """
        # Process features
        h = self.high_res_conv(high_res)

        # Upsample deep features to match high_res size
        d = self.deep_conv(deep)
        d = F.interpolate(d, size=h.shape[2:], mode='bilinear', align_corners=False)

        # Compute attention weights
        combined = torch.cat([h, d], dim=1)
        weights = self.attention(combined)  # (B, 2, 1, 1)

        # Weighted fusion
        fused = h * weights[:, 0:1] + d * weights[:, 1:2]

        return self.refine(fused)


class PriorGuidedSpatialAttention(nn.Module):
    """
    Uses Stage1 segmentation priors to guide attention toward lesion-prone regions.

    The key insight is that lesions typically occur:
    - Near tooth boundaries (captured by distance field)
    - Around metal restorations (captured by metal prior)
    - In specific anatomical regions
    """

    def __init__(self, feature_channels: int, prior_channels: int = 3):
        """
        Initialize Prior-Guided Spatial Attention.

        Args:
            feature_channels: Number of feature channels
            prior_channels: Number of prior channels (virtual, distance, metal)
        """
        super().__init__()

        self.prior_channels = prior_channels

        # Encode priors into attention weights
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Generate spatial attention from priors
        self.attention_head = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Feature modulation
        self.feature_transform = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.BatchNorm2d(feature_channels)
        )

    def forward(self, features: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Feature map from backbone/neck
            priors: Prior channels (virtual_mask, distance_field, metal_prior)

        Returns:
            Attention-enhanced features
        """
        # Resize priors to match feature size if needed
        if priors.shape[2:] != features.shape[2:]:
            priors = F.interpolate(priors, size=features.shape[2:], mode='bilinear', align_corners=False)

        # Encode priors
        prior_features = self.prior_encoder(priors)

        # Generate spatial attention
        attention = self.attention_head(prior_features)

        # Apply attention
        modulated = self.feature_transform(features)

        return features + modulated * attention


class ScaleAwareAttention(nn.Module):
    """
    Scale-aware attention that emphasizes features at scales relevant for small objects.

    Uses learnable scale weights to balance features from different receptive fields.
    """

    def __init__(self, channels: int, num_scales: int = 4):
        """
        Initialize Scale-Aware Attention.

        Args:
            channels: Number of input channels
            num_scales: Number of scales to consider
        """
        super().__init__()

        self.num_scales = num_scales

        # Multi-scale feature extractors
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // num_scales, 3, padding=2**i, dilation=2**i),
                nn.BatchNorm2d(channels // num_scales),
                nn.ReLU(inplace=True)
            ) for i in range(num_scales)
        ])

        # Scale weight predictor
        self.scale_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, num_scales, 1),
            nn.Softmax(dim=1)
        )

        # Output projection
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scale-aware attention."""
        # Extract multi-scale features
        scale_features = [conv(x) for conv in self.scale_convs]

        # Predict scale weights
        weights = self.scale_weights(x)  # (B, num_scales, 1, 1)

        # Weighted combination
        out = torch.zeros_like(scale_features[0])
        for i, feat in enumerate(scale_features):
            out = out + feat * weights[:, i:i+1]

        # Expand back to original channels
        out = out.repeat(1, self.num_scales, 1, 1)

        return x + self.proj(out)


class LesionFocusModule(nn.Module):
    """
    Specialized module for focusing on small lesion regions.

    Combines:
    1. Edge-aware attention (lesions often have distinct boundaries)
    2. Local contrast enhancement (lesions differ from surrounding tissue)
    3. Size-adaptive processing
    """

    def __init__(self, channels: int):
        """
        Initialize Lesion Focus Module.

        Args:
            channels: Number of input/output channels
        """
        super().__init__()

        # Edge detection branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )

        # Sobel-like edge kernels (learnable)
        self.edge_x = nn.Conv2d(channels // 4, channels // 4, 3, padding=1, groups=channels // 4, bias=False)
        self.edge_y = nn.Conv2d(channels // 4, channels // 4, 3, padding=1, groups=channels // 4, bias=False)

        # Initialize with Sobel-like weights
        self._init_edge_weights()

        # Local contrast branch
        self.local_contrast = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels // 4 * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def _init_edge_weights(self):
        """Initialize edge detection weights."""
        # Sobel X
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        # Sobel Y
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        with torch.no_grad():
            for i in range(self.edge_x.weight.shape[0]):
                self.edge_x.weight[i, 0] = sobel_x
                self.edge_y.weight[i, 0] = sobel_y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with lesion-focused attention."""
        # Edge detection
        edge_feat = self.edge_conv(x)
        edge_x = self.edge_x(edge_feat)
        edge_y = self.edge_y(edge_feat)
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        # Local contrast (difference from local mean)
        local_mean = F.avg_pool2d(x, 5, stride=1, padding=2)
        contrast = self.local_contrast(torch.abs(x - local_mean))

        # Fuse edge and contrast information
        combined = torch.cat([edge_magnitude, contrast], dim=1)
        attention = self.fusion(combined)

        return x * (1 + attention)


# Utility function to add small object modules to existing model
def enhance_model_for_small_objects(model, feature_channels=256):
    """
    Utility to add small object detection enhancements to a model.

    Args:
        model: YOLO model to enhance
        feature_channels: Number of channels in feature maps

    Returns:
        Enhanced model
    """
    # This would require modifying the model architecture
    # Implementation depends on specific model structure
    pass
