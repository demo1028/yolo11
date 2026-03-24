# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Small Object Optimized Loss Functions for Dental Lesion Detection.

Problems with standard IoU loss for small objects:
1. IoU is sensitive to small position shifts (1px shift = huge IoU drop for small boxes)
2. Gradient vanishes when boxes don't overlap
3. Equal penalty for all samples regardless of quality

Solutions implemented:
1. WIoU (Wise-IoU): Dynamic non-monotonic focusing, reduces impact of low-quality samples
2. Inner-IoU: Uses auxiliary inner box for more stable gradients on small objects
3. NWD (Normalized Wasserstein Distance): Distribution-based, scale-invariant metric
4. Focal-EIoU: Combines focal mechanism with EIoU for hard sample mining
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False,
             SIoU=False, EIoU=False, WIoU=False, Inner=False,
             Focal=False, alpha=1, gamma=0.5, scale=False, eps=1e-7):
    """
    Calculate IoU and its variants between two sets of boxes.

    Enhanced with small-object-friendly options:
    - WIoU: Wise-IoU with dynamic focusing
    - Inner: Inner-IoU using auxiliary boxes
    - Focal: Focal modifier for hard samples

    Args:
        box1: (N, 4) tensor of boxes
        box2: (N, 4) tensor of boxes
        xywh: If True, boxes are (x, y, w, h), else (x1, y1, x2, y2)
        GIoU/DIoU/CIoU/SIoU/EIoU/WIoU: IoU variant flags
        Inner: Use Inner-IoU for small objects
        Focal: Apply focal weighting
        alpha: Focal alpha parameter
        gamma: Focal gamma parameter
        scale: Scale-aware weighting
        eps: Small constant for numerical stability

    Returns:
        IoU values (N,)
    """
    # Convert to x1y1x2y2 format
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Inner-IoU: Use auxiliary inner boxes for small object stability
    if Inner:
        ratio = 0.75  # Inner box ratio (can be tuned: 0.5-0.9)
        w1_inner, h1_inner = w1 * ratio, h1 * ratio
        w2_inner, h2_inner = w2 * ratio, h2 * ratio

        # Recalculate boxes with inner dimensions
        b1_x1_in = x1 - w1_inner / 2 if xywh else b1_x1 + w1 * (1 - ratio) / 2
        b1_x2_in = x1 + w1_inner / 2 if xywh else b1_x2 - w1 * (1 - ratio) / 2
        b1_y1_in = y1 - h1_inner / 2 if xywh else b1_y1 + h1 * (1 - ratio) / 2
        b1_y2_in = y1 + h1_inner / 2 if xywh else b1_y2 - h1 * (1 - ratio) / 2

        b2_x1_in = x2 - w2_inner / 2 if xywh else b2_x1 + w2 * (1 - ratio) / 2
        b2_x2_in = x2 + w2_inner / 2 if xywh else b2_x2 - w2 * (1 - ratio) / 2
        b2_y1_in = y2 - h2_inner / 2 if xywh else b2_y1 + h2 * (1 - ratio) / 2
        b2_y2_in = y2 + h2_inner / 2 if xywh else b2_y2 - h2 * (1 - ratio) / 2

        # Use inner boxes for intersection
        inter_x1 = torch.max(b1_x1_in, b2_x1_in)
        inter_y1 = torch.max(b1_y1_in, b2_y1_in)
        inter_x2 = torch.min(b1_x2_in, b2_x2_in)
        inter_y2 = torch.min(b1_y2_in, b2_y2_in)
    else:
        # Standard intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps

    # Basic IoU
    iou = inter / union

    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        # Convex (enclosing) box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        if CIoU or DIoU or EIoU or SIoU or WIoU:
            # Distance between centers
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 +
                    (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4

            if CIoU or WIoU:
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                )
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))

                if WIoU:
                    # Wise-IoU: Dynamic non-monotonic focusing
                    # Reduces gradient for low-quality (outlier) samples
                    with torch.no_grad():
                        # Focusing coefficient based on IoU
                        wise_scale = torch.exp((1 - iou) * scale) if scale else 1
                    iou = (iou - rho2 / c2 - v * alpha_ciou) * wise_scale
                else:
                    iou = iou - rho2 / c2 - v * alpha_ciou  # CIoU

            elif EIoU:
                rho_w2 = ((b1_x2 - b1_x1) - (b2_x2 - b2_x1)) ** 2
                rho_h2 = ((b1_y2 - b1_y1) - (b2_y2 - b2_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                iou = iou - rho2 / c2 - rho_w2 / cw2 - rho_h2 / ch2

            elif SIoU:
                # Angle cost
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha = torch.abs(s_ch) / sigma
                sin_beta = torch.abs(s_cw) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha > threshold, sin_beta, sin_alpha)
                angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma_siou = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma_siou * rho_x) - torch.exp(gamma_siou * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                iou = iou - (distance_cost + shape_cost) * 0.5

            else:  # DIoU
                iou = iou - rho2 / c2

        else:  # GIoU
            c_area = cw * ch + eps
            iou = iou - (c_area - union) / c_area

    # Focal modifier for hard samples
    if Focal:
        iou = iou * ((1 - iou) ** gamma) * alpha

    return iou.squeeze(-1)


def wasserstein_distance(box1, box2, xywh=True, eps=1e-7):
    """
    Calculate Normalized Wasserstein Distance (NWD) between boxes.

    NWD treats boxes as 2D Gaussian distributions and measures their distance.
    Key advantage: More stable for small objects compared to IoU.

    For small objects:
    - 1px shift causes huge IoU drop but small NWD change
    - Gradient doesn't vanish when boxes don't overlap

    Args:
        box1, box2: Boxes in xywh or xyxy format
        xywh: Format flag
        eps: Numerical stability

    Returns:
        NWD values (N,), in [0, 1] where 0 = identical, 1 = far apart
    """
    if xywh:
        cx1, cy1, w1, h1 = box1.chunk(4, -1)
        cx2, cy2, w2, h2 = box2.chunk(4, -1)
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Wasserstein distance components
    # Center distance
    center_dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    # Size difference (Wasserstein for 1D Gaussians)
    w_dist = ((w1 / 2) ** 0.5 - (w2 / 2) ** 0.5) ** 2
    h_dist = ((h1 / 2) ** 0.5 - (h2 / 2) ** 0.5) ** 2

    # Total Wasserstein distance
    wd = center_dist + w_dist + h_dist

    # Normalize by box size for scale invariance
    # Use geometric mean of box sizes as normalization factor
    norm_factor = ((w1 * h1 + w2 * h2) / 2 + eps) ** 0.5

    # Normalized Wasserstein Distance
    nwd = torch.exp(-wd.sqrt() / norm_factor)

    return nwd.squeeze(-1)


class SmallObjectBboxLoss(nn.Module):
    """
    Bounding Box Loss optimized for small dental lesions.

    Combines multiple strategies:
    1. WIoU/Inner-IoU for better gradient on small objects
    2. NWD as auxiliary loss for scale-invariant supervision
    3. Size-aware weighting (smaller boxes get more attention)
    """

    def __init__(self,
                 iou_type='wiou',      # 'ciou', 'diou', 'wiou', 'eiou'
                 use_inner=True,       # Use Inner-IoU
                 use_nwd=True,         # Add NWD auxiliary loss
                 nwd_weight=0.5,       # Weight for NWD loss
                 size_aware=True,      # Size-aware weighting
                 small_threshold=32,   # Threshold for "small" objects (pixels)
                 small_weight=2.0):    # Extra weight for small objects
        """
        Initialize small object bbox loss.

        Args:
            iou_type: IoU variant to use
            use_inner: Whether to use Inner-IoU
            use_nwd: Whether to add NWD auxiliary loss
            nwd_weight: Weight for NWD loss component
            size_aware: Apply size-aware weighting
            small_threshold: Box size threshold (geometric mean of w,h)
            small_weight: Extra weight multiplier for small boxes
        """
        super().__init__()
        self.iou_type = iou_type.lower()
        self.use_inner = use_inner
        self.use_nwd = use_nwd
        self.nwd_weight = nwd_weight
        self.size_aware = size_aware
        self.small_threshold = small_threshold
        self.small_weight = small_weight

    def forward(self, pred_boxes, target_boxes, xywh=True):
        """
        Calculate loss.

        Args:
            pred_boxes: (N, 4) predicted boxes
            target_boxes: (N, 4) target boxes
            xywh: Box format flag

        Returns:
            loss: Scalar loss value
        """
        # Get IoU flags based on type
        flags = {
            'giou': dict(GIoU=True),
            'diou': dict(DIoU=True),
            'ciou': dict(CIoU=True),
            'eiou': dict(EIoU=True),
            'siou': dict(SIoU=True),
            'wiou': dict(WIoU=True, scale=True),
        }.get(self.iou_type, dict(CIoU=True))

        # Calculate IoU
        iou = bbox_iou(pred_boxes, target_boxes, xywh=xywh,
                       Inner=self.use_inner, **flags)

        # IoU loss
        loss_iou = 1 - iou

        # Add NWD loss
        if self.use_nwd:
            nwd = wasserstein_distance(pred_boxes, target_boxes, xywh=xywh)
            loss_nwd = 1 - nwd
            loss = loss_iou + self.nwd_weight * loss_nwd
        else:
            loss = loss_iou

        # Size-aware weighting
        if self.size_aware:
            if xywh:
                w, h = target_boxes[:, 2], target_boxes[:, 3]
            else:
                w = target_boxes[:, 2] - target_boxes[:, 0]
                h = target_boxes[:, 3] - target_boxes[:, 1]

            # Geometric mean of width and height
            size = (w * h).sqrt()

            # Weight: higher for small objects
            # w = small_weight for size < threshold, 1.0 for large objects
            weight = torch.where(
                size < self.small_threshold,
                torch.full_like(size, self.small_weight),
                torch.ones_like(size)
            )

            loss = loss * weight

        return loss.mean(), iou.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for classification, addressing class imbalance.

    FL(p) = -alpha * (1-p)^gamma * log(p)

    For dental lesions:
    - Background dominates (easy negatives)
    - Lesions are rare (hard positives)
    - Focal loss down-weights easy samples, focuses on hard ones
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C) predicted logits
            target: (N,) target class indices
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss for objectness/classification.

    Modification of Focal Loss that uses soft labels based on IoU quality.
    Better for dense object detection where IoU quality varies.
    """

    def __init__(self, beta=2.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: (N,) predicted scores (after sigmoid)
            target: (N,) target quality scores (0-1, e.g., IoU)
            weight: (N,) optional sample weights
        """
        # Focal weight
        focal_weight = (pred - target).abs().pow(self.beta)

        # Binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        loss = focal_weight * bce

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Convenience function to get loss
def get_dental_bbox_loss(loss_type='wiou_nwd'):
    """
    Get pre-configured bbox loss for dental lesion detection.

    Args:
        loss_type: One of:
            - 'ciou': Standard CIoU
            - 'wiou': Wise-IoU (dynamic focusing)
            - 'inner_ciou': Inner-CIoU (small object friendly)
            - 'wiou_nwd': WIoU + NWD (recommended for small lesions)
            - 'inner_wiou_nwd': All optimizations (best for very small lesions)

    Returns:
        SmallObjectBboxLoss instance
    """
    configs = {
        'ciou': dict(iou_type='ciou', use_inner=False, use_nwd=False),
        'wiou': dict(iou_type='wiou', use_inner=False, use_nwd=False),
        'inner_ciou': dict(iou_type='ciou', use_inner=True, use_nwd=False),
        'wiou_nwd': dict(iou_type='wiou', use_inner=False, use_nwd=True, nwd_weight=0.5),
        'inner_wiou_nwd': dict(iou_type='wiou', use_inner=True, use_nwd=True, nwd_weight=0.5),
    }

    config = configs.get(loss_type, configs['wiou_nwd'])
    return SmallObjectBboxLoss(**config)
