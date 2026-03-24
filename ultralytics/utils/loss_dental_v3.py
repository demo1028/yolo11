# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Dental Detection Loss v3 - Integrated with YOLO training pipeline.

This module provides a drop-in replacement for the standard YOLO detection loss,
optimized for small dental lesion detection.

Usage in DentalTrainer:
    from ultralytics.utils.loss_dental_v3 import DentalDetectionLoss

    class DentalTrainer(DetectionTrainer):
        def get_loss(self):
            return DentalDetectionLoss(self.model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultralytics.utils.loss_dental import (
    bbox_iou,
    wasserstein_distance,
    SmallObjectBboxLoss,
    FocalLoss,
    QualityFocalLoss,
)
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors


class DentalDetectionLoss(nn.Module):
    """
    Detection Loss optimized for small dental lesions.

    Modifications from standard YOLO loss:
    1. Box loss: WIoU + NWD instead of CIoU
    2. Classification: Quality Focal Loss
    3. Size-aware weighting for small lesions
    4. Optional auxiliary small object loss

    Compatible with YOLO v8/v11 training pipeline.
    """

    def __init__(self, model,
                 box_loss_type='wiou_nwd',
                 use_size_aware=True,
                 small_threshold=32,
                 small_weight=2.0,
                 tal_topk=10):
        """
        Initialize dental detection loss.

        Args:
            model: YOLO model instance
            box_loss_type: 'ciou', 'wiou', 'wiou_nwd', 'inner_wiou_nwd'
            use_size_aware: Apply size-aware weighting
            small_threshold: Size threshold for "small" objects
            small_weight: Extra weight for small objects
            tal_topk: Top-k for Task-Aligned Assigner
        """
        super().__init__()

        # Get model parameters
        m = model.model[-1] if hasattr(model, 'model') else model[-1]  # Detect head
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4  # number of outputs per anchor
        self.reg_max = m.reg_max
        self.device = next(model.parameters()).device
        self.stride = m.stride

        # Loss parameters
        self.box_loss_type = box_loss_type
        self.use_size_aware = use_size_aware
        self.small_threshold = small_threshold
        self.small_weight = small_weight

        # Initialize assigner
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)

        # Initialize box loss
        self.init_box_loss()

        # Classification loss (use BCE with logits for stability)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # DFL loss for distribution focal loss
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

        # Loss weights (can be tuned)
        self.box_gain = 7.5      # Box loss weight
        self.cls_gain = 0.5      # Classification loss weight
        self.dfl_gain = 1.5      # DFL loss weight

    def init_box_loss(self):
        """Initialize box loss based on type."""
        # Parse loss type
        use_inner = 'inner' in self.box_loss_type
        use_nwd = 'nwd' in self.box_loss_type

        if 'wiou' in self.box_loss_type:
            iou_type = 'wiou'
        elif 'eiou' in self.box_loss_type:
            iou_type = 'eiou'
        elif 'siou' in self.box_loss_type:
            iou_type = 'siou'
        else:
            iou_type = 'ciou'

        self.iou_type = iou_type
        self.use_inner = use_inner
        self.use_nwd = use_nwd
        self.nwd_weight = 0.5 if use_nwd else 0.0

    def forward(self, preds, batch):
        """
        Calculate detection loss.

        Args:
            preds: Model predictions (tuple of tensors)
            batch: Batch dict with 'cls', 'bboxes', 'batch_idx'

        Returns:
            loss: Total loss scalar
            loss_items: Tensor of individual losses for logging
        """
        # Initialize losses
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        # Get predictions
        if isinstance(preds, tuple):
            feats = preds[1] if len(preds) > 1 else preds[0]
            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (self.reg_max * 4, self.nc), 1
            )
        else:
            pred_distri, pred_scores = preds.split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # Get dimensions
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Generate anchors
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Process targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Decode predictions
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Task-aligned assignment
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum * self.cls_gain

        # Box and DFL losses for foreground samples
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.box_gain
        loss[2] *= self.dfl_gain

        return loss.sum(), loss.detach()

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets for loss calculation."""
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)

        # Get batch indices
        i = targets[:, 0].long()
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)

        # Allocate output tensor
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]

        # Scale bboxes
        out[..., 1:5] = out[..., 1:5] * scale_tensor

        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted distribution to bounding boxes."""
        if self.reg_max > 1:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def bbox_loss(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                  target_scores, target_scores_sum, fg_mask):
        """
        Calculate bbox loss with small object optimizations.

        Returns:
            box_loss: Box regression loss
            dfl_loss: Distribution focal loss
        """
        # Select foreground predictions and targets
        pred_bboxes_pos = pred_bboxes[fg_mask]
        target_bboxes_pos = target_bboxes[fg_mask]
        target_scores_pos = target_scores[fg_mask].sum(-1)

        # Calculate IoU
        iou_flags = {
            'GIoU': self.iou_type == 'giou',
            'DIoU': self.iou_type == 'diou',
            'CIoU': self.iou_type == 'ciou',
            'EIoU': self.iou_type == 'eiou',
            'SIoU': self.iou_type == 'siou',
            'WIoU': self.iou_type == 'wiou',
            'Inner': self.use_inner,
            'scale': self.iou_type == 'wiou',
        }

        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, **iou_flags)
        loss_iou = 1 - iou

        # Add NWD loss if enabled
        if self.use_nwd:
            nwd = wasserstein_distance(pred_bboxes_pos, target_bboxes_pos, xywh=False)
            loss_nwd = 1 - nwd
            loss_box = loss_iou + self.nwd_weight * loss_nwd
        else:
            loss_box = loss_iou

        # Size-aware weighting
        if self.use_size_aware:
            w = target_bboxes_pos[:, 2] - target_bboxes_pos[:, 0]
            h = target_bboxes_pos[:, 3] - target_bboxes_pos[:, 1]
            size = (w * h).sqrt()

            # Weight: higher for small objects
            size_weight = torch.where(
                size < self.small_threshold,
                torch.full_like(size, self.small_weight),
                torch.ones_like(size)
            )
            loss_box = loss_box * size_weight

        # Weight by target scores
        loss_box = (loss_box * target_scores_pos).sum() / target_scores_sum

        # DFL loss
        pred_dist_pos = pred_dist[fg_mask]
        target_ltrb = self.bbox2dist(anchor_points[fg_mask], target_bboxes_pos, self.reg_max - 1)

        loss_dfl = self.df_loss(pred_dist_pos.view(-1, self.reg_max), target_ltrb.view(-1))
        loss_dfl = (loss_dfl * target_scores_pos.unsqueeze(-1).expand(-1, 4).reshape(-1)).sum() / target_scores_sum

        return loss_box, loss_dfl

    @staticmethod
    def bbox2dist(anchor_points, bbox, reg_max):
        """Convert bbox to distance representation."""
        x1y1, x2y2 = bbox.chunk(2, -1)
        return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)

    @staticmethod
    def df_loss(pred_dist, target):
        """Distribution Focal Loss."""
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl

        loss_left = F.cross_entropy(pred_dist, tl, reduction='none') * wl
        loss_right = F.cross_entropy(pred_dist, tr.clamp_(0, pred_dist.shape[-1] - 1), reduction='none') * wr

        return loss_left + loss_right


def get_dental_loss(model, loss_type='wiou_nwd', **kwargs):
    """
    Factory function to get dental detection loss.

    Args:
        model: YOLO model
        loss_type: Loss configuration
            - 'default': Standard CIoU
            - 'wiou': Wise-IoU
            - 'wiou_nwd': WIoU + NWD (recommended)
            - 'inner_wiou_nwd': All optimizations

    Returns:
        DentalDetectionLoss instance
    """
    return DentalDetectionLoss(model, box_loss_type=loss_type, **kwargs)
