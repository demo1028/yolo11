# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Dental-specific Attention Modules for Multi-channel YOLO11.

These modules are designed to:
1. Adaptively weight different input channels (raw, shape, distance, metal priors)
2. Enhance lesion-relevant features in multi-scale feature maps
3. Guide the network to focus on high-risk regions (near metal/boundaries)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "DentalChannelAttention",
    "DentalSpatialAttention",
    "DentalCBAM",
    "MultiScalePriorFusion",
    "PriorGuidedAttention",
    "C3k2_Dental",
)


class DentalChannelAttention(nn.Module):
    """
    Channel Attention Module for 4-channel dental input.

    Learns to weight different prior channels:
        - Ch0: Raw image intensity
        - Ch1: Virtual fusion (tooth shape)
        - Ch2: Distance field (spatial position)
        - Ch3: Metal prior (risk regions)

    Uses both max-pool and avg-pool for richer statistics.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize Dental Channel Attention.

        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio for FC layers
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (B, C, H, W)

        Returns:
            Attention-weighted tensor, same shape as input
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class DentalSpatialAttention(nn.Module):
    """
    Spatial Attention Module for dental lesion detection.

    Focuses on spatial regions that are likely to contain lesions,
    guided by the combined channel statistics.
    """

    def __init__(self, kernel_size: int = 7):
        """
        Initialize Dental Spatial Attention.

        Args:
            kernel_size: Convolution kernel size (must be 3 or 7)
        """
        super().__init__()

        assert kernel_size in {3, 7}, "kernel_size must be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (B, C, H, W)

        Returns:
            Spatially attended tensor, same shape as input
        """
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and compute spatial attention
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))

        return x * attention


class DentalCBAM(nn.Module):
    """
    Convolutional Block Attention Module adapted for dental imaging.

    Combines channel and spatial attention in sequence:
    1. Channel attention identifies which input channels are most informative
    2. Spatial attention focuses on lesion-prone regions
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Initialize Dental CBAM.

        Args:
            channels: Number of input channels
            reduction: Channel attention reduction ratio
            kernel_size: Spatial attention kernel size
        """
        super().__init__()

        self.channel_attention = DentalChannelAttention(channels, reduction)
        self.spatial_attention = DentalSpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through channel and spatial attention."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiScalePriorFusion(nn.Module):
    """
    Multi-scale fusion module for prior information.

    Fuses features from different scales while preserving
    prior channel semantics. Useful in FPN/PANet neck.
    """

    def __init__(self, in_channels: int, out_channels: int, num_scales: int = 3):
        """
        Initialize Multi-scale Prior Fusion.

        Args:
            in_channels: Input channels per scale
            out_channels: Output channels after fusion
            num_scales: Number of feature scales to fuse
        """
        super().__init__()

        self.num_scales = num_scales

        # Per-scale attention
        self.scale_attentions = nn.ModuleList([
            DentalChannelAttention(in_channels, reduction=8)
            for _ in range(num_scales)
        ])

        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        # Output attention
        self.output_attention = DentalCBAM(out_channels)

    def forward(self, features: list) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: List of feature tensors from different scales

        Returns:
            Fused feature tensor
        """
        assert len(features) == self.num_scales

        # Get target size (largest scale)
        target_size = features[0].shape[2:]

        # Apply per-scale attention and resize
        attended_features = []
        for i, (feat, attn) in enumerate(zip(features, self.scale_attentions)):
            attended = attn(feat)
            if attended.shape[2:] != target_size:
                attended = F.interpolate(attended, size=target_size, mode='bilinear', align_corners=False)
            attended_features.append(attended)

        # Concatenate and fuse
        fused = torch.cat(attended_features, dim=1)
        fused = self.fusion_conv(fused)

        # Apply output attention
        return self.output_attention(fused)


class PriorGuidedAttention(nn.Module):
    """
    Prior-Guided Attention for leveraging Stage1 priors.

    This module explicitly uses the prior channels (virtual mask, distance, metal)
    to guide attention on the raw image features.

    Architecture:
        1. Extract prior features from channels 1-3
        2. Generate spatial attention map from priors
        3. Modulate raw image features with prior attention
    """

    def __init__(self, channels: int, prior_channels: int = 3):
        """
        Initialize Prior-Guided Attention.

        Args:
            channels: Total input channels (including priors)
            prior_channels: Number of prior channels (default 3: shape, distance, metal)
        """
        super().__init__()

        self.prior_channels = prior_channels
        self.feature_channels = channels - prior_channels

        # Prior encoder
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(prior_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Attention generator
        self.attention_gen = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # Feature modulation
        self.feature_conv = nn.Sequential(
            nn.Conv2d(self.feature_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor with raw + prior channels, shape (B, C, H, W)

        Returns:
            Prior-guided feature tensor
        """
        # Split into raw features and priors
        raw_features = x[:, :self.feature_channels]
        priors = x[:, self.feature_channels:]

        # Encode priors and generate attention
        prior_features = self.prior_encoder(priors)
        spatial_attention = self.attention_gen(prior_features)

        # Modulate raw features
        modulated = raw_features * spatial_attention

        # Process through feature conv
        output = self.feature_conv(modulated)

        return output


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) for dental features.

    Uses 1D convolution instead of fully connected layers,
    making it more parameter-efficient while maintaining effectiveness.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        """
        Initialize ECA module.

        Args:
            channels: Number of input channels
            gamma: Kernel size calculation parameter
            b: Kernel size calculation parameter
        """
        super().__init__()

        # Adaptive kernel size
        import math
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with efficient channel attention."""
        # Global average pooling
        y = self.avg_pool(x)  # (B, C, 1, 1)

        # 1D convolution along channel dimension
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)

        # Attention weights
        attention = self.sigmoid(y)

        return x * attention


class C3k2_Dental(nn.Module):
    """
    C3k2 block enhanced with Dental attention.

    Integrates ECA attention into the standard C3k2 architecture
    for better lesion feature extraction.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        """
        Initialize C3k2_Dental block.

        Args:
            c1: Input channels
            c2: Output channels
            n: Number of bottleneck blocks
            shortcut: Whether to use residual connection
            e: Expansion ratio
        """
        super().__init__()

        from .conv import Conv

        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        # Bottleneck blocks
        self.m = nn.ModuleList([
            nn.Sequential(
                Conv(self.c, self.c, 3, 1),
                Conv(self.c, self.c, 3, 1)
            ) for _ in range(n)
        ])

        # Dental attention
        self.attention = EfficientChannelAttention(c2)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dental attention."""
        # Split and process
        y = list(self.cv1(x).chunk(2, 1))

        # Apply bottleneck blocks
        y.extend(m(y[-1]) for m in self.m)

        # Concatenate and final conv
        out = self.cv2(torch.cat(y, 1))

        # Apply attention
        out = self.attention(out)

        # Residual
        return out + x if self.shortcut else out


# Registration function to add modules to YOLO
def register_dental_modules():
    """
    Register dental attention modules so they can be used in YAML model definitions.

    parse_model() in ultralytics/nn/tasks.py resolves module names via globals(),
    so we must inject custom modules into that file's global namespace.
    """
    import ultralytics.nn.tasks as tasks_module

    # Core dental attention modules
    tasks_module.DentalChannelAttention = DentalChannelAttention
    tasks_module.DentalSpatialAttention = DentalSpatialAttention
    tasks_module.DentalCBAM = DentalCBAM
    tasks_module.MultiScalePriorFusion = MultiScalePriorFusion
    tasks_module.PriorGuidedAttention = PriorGuidedAttention
    tasks_module.EfficientChannelAttention = EfficientChannelAttention
    tasks_module.C3k2_Dental = C3k2_Dental

    # Prior-as-Attention modules (alternative architecture)
    from .prior_attention import (
        PriorAsAttentionInput,
        PriorEncoder,
        PriorGuidedAttention as PriorGuidedAttentionV2,
        PriorSplitter,
        DentalPriorBackbone,
    )
    tasks_module.PriorAsAttentionInput = PriorAsAttentionInput
    tasks_module.PriorEncoder = PriorEncoder
    tasks_module.PriorGuidedAttentionV2 = PriorGuidedAttentionV2
    tasks_module.PriorSplitter = PriorSplitter
    tasks_module.DentalPriorBackbone = DentalPriorBackbone

    print("Dental attention modules registered successfully.")
