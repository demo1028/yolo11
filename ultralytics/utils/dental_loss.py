# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Dental-specific Loss Functions for Small Lesion Detection.

These loss functions address common challenges in dental lesion detection:
1. Small object detection (lesions are often < 30 pixels)
2. Class imbalance (lesions are rare compared to background)
3. Hard negative mining (similar-looking healthy tissue)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SmallObjectFocalLoss(nn.Module):
    """
    Modified Focal Loss with size-aware weighting for small objects.

    Standard Focal Loss: FL(p) = -α(1-p)^γ * log(p)

    Size-aware modification:
    - Applies higher weight to small objects
    - Uses IoU-based difficulty estimation
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        small_object_threshold: float = 0.02,  # Objects smaller than 2% of image
        small_object_weight: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize Small Object Focal Loss.

        Args:
            alpha: Balancing factor for positive/negative samples
            gamma: Focusing parameter
            small_object_threshold: Relative size threshold for "small" objects
            small_object_weight: Additional weight for small objects
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.small_object_threshold = small_object_threshold
        self.small_object_weight = small_object_weight
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bbox_sizes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute size-aware focal loss.

        Args:
            pred: Predicted logits (B, num_classes, H, W) or (N, num_classes)
            target: Target labels (B, H, W) or (N,)
            bbox_sizes: Relative sizes of bounding boxes (N,), optional

        Returns:
            Loss value
        """
        # Standard focal loss computation
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p = torch.softmax(pred, dim=1 if pred.dim() > 2 else -1)

        # Get probability of correct class
        if pred.dim() > 2:
            p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha balancing
        alpha_weight = torch.where(target > 0, self.alpha, 1 - self.alpha)

        # Size-aware weighting
        if bbox_sizes is not None:
            size_weight = torch.where(
                bbox_sizes < self.small_object_threshold,
                torch.full_like(bbox_sizes, self.small_object_weight),
                torch.ones_like(bbox_sizes)
            )
        else:
            size_weight = 1.0

        # Combined loss
        loss = alpha_weight * focal_weight * size_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SizeAwareIoULoss(nn.Module):
    """
    IoU Loss with size-dependent weighting for bounding box regression.

    Applies stronger penalty for IoU errors on small objects, where
    even small pixel errors can significantly affect IoU.
    """

    def __init__(
        self,
        iou_type: str = "ciou",
        small_threshold: float = 32,  # pixels
        small_weight: float = 1.5,
        reduction: str = "mean"
    ):
        """
        Initialize Size-Aware IoU Loss.

        Args:
            iou_type: Type of IoU ('iou', 'giou', 'diou', 'ciou')
            small_threshold: Size threshold in pixels for small objects
            small_weight: Additional weight for small objects
            reduction: Reduction method
        """
        super().__init__()
        self.iou_type = iou_type
        self.small_threshold = small_threshold
        self.small_weight = small_weight
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute size-aware IoU loss.

        Args:
            pred_boxes: Predicted boxes (N, 4) in xyxy format
            target_boxes: Target boxes (N, 4) in xyxy format
            eps: Small constant for numerical stability

        Returns:
            Loss value
        """
        # Compute intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        # Compute union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                      (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + eps)

        if self.iou_type == "iou":
            loss = 1 - iou

        elif self.iou_type == "giou":
            # Enclosing box
            enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
            enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
            enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
            enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
            enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

            giou = iou - (enc_area - union_area) / (enc_area + eps)
            loss = 1 - giou

        elif self.iou_type in ("diou", "ciou"):
            # Center distance
            pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
            pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
            target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
            target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

            center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

            # Enclosing box diagonal
            enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
            enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
            enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
            enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
            diag_dist = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

            diou = iou - center_dist / (diag_dist + eps)

            if self.iou_type == "diou":
                loss = 1 - diou
            else:  # ciou
                # Aspect ratio consistency
                pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
                pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
                target_w = target_boxes[:, 2] - target_boxes[:, 0]
                target_h = target_boxes[:, 3] - target_boxes[:, 1]

                v = (4 / (3.14159 ** 2)) * torch.pow(
                    torch.atan(target_w / (target_h + eps)) -
                    torch.atan(pred_w / (pred_h + eps)), 2
                )
                alpha = v / (1 - iou + v + eps)

                ciou = diou - alpha * v
                loss = 1 - ciou
        else:
            raise ValueError(f"Unknown IoU type: {self.iou_type}")

        # Size-aware weighting
        target_size = torch.sqrt(target_area)
        size_weight = torch.where(
            target_size < self.small_threshold,
            torch.full_like(target_size, self.small_weight),
            torch.ones_like(target_size)
        )

        loss = loss * size_weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DentalDetectionLoss(nn.Module):
    """
    Combined detection loss for dental lesion detection.

    Combines:
    1. Size-aware classification loss (Focal Loss)
    2. Size-aware bounding box regression loss (CIoU)
    3. Optional objectness loss
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        box_weight: float = 2.0,
        obj_weight: float = 1.0,
        small_object_weight: float = 2.0
    ):
        """
        Initialize Dental Detection Loss.

        Args:
            cls_weight: Weight for classification loss
            box_weight: Weight for box regression loss
            obj_weight: Weight for objectness loss
            small_object_weight: Additional weight for small objects
        """
        super().__init__()

        self.cls_loss = SmallObjectFocalLoss(
            alpha=0.25,
            gamma=2.0,
            small_object_weight=small_object_weight
        )

        self.box_loss = SizeAwareIoULoss(
            iou_type="ciou",
            small_weight=small_object_weight
        )

        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.obj_weight = obj_weight

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined detection loss.

        Args:
            predictions: Model predictions (boxes, scores, classes)
            targets: Ground truth targets

        Returns:
            Total loss and loss components dict
        """
        # This would integrate with YOLO's existing loss structure
        # Implementation depends on specific prediction format

        loss_dict = {
            "cls_loss": 0,
            "box_loss": 0,
            "obj_loss": 0,
        }

        total_loss = (
            self.cls_weight * loss_dict["cls_loss"] +
            self.box_weight * loss_dict["box_loss"] +
            self.obj_weight * loss_dict["obj_loss"]
        )

        return total_loss, loss_dict


class AnatomyConstrainedAugmentation:
    """
    Anatomy-Constrained Augmentation for dental lesion detection.

    Unlike naive Copy-Paste, this module respects anatomical constraints:
    - Periapical lesions (根尖周病变): can ONLY appear at tooth root apices
    - Furcation lesions (根分叉病变): can ONLY appear at root furcation areas
    - Combined lesions (联合病变): constrained to overlapping regions

    The valid paste regions are derived from Stage1 segmentation priors:
    - Root apex zone: bottom portion of each tooth instance
    - Furcation zone: mid-root region where roots diverge

    This ensures every augmented sample is anatomically plausible.
    """

    # Lesion class definitions
    CLS_PERIAPICAL = 0   # 根尖周病变
    CLS_PERIODONTAL = 1  # 根分叉病变
    CLS_COMBINED = 2     # 联合病变

    def __init__(
        self,
        paste_prob: float = 0.5,
        max_paste_per_image: int = 2,
        apex_ratio: float = 0.3,
        furcation_ratio: Tuple[float, float] = (0.3, 0.7),
        blend_mode: str = "poisson",
        buffer_size: int = 200
    ):
        """
        Initialize Anatomy-Constrained Augmentation.

        Args:
            paste_prob: Probability of applying augmentation per image
            max_paste_per_image: Maximum number of pastes per image
            apex_ratio: Bottom proportion of tooth considered as root apex zone
                        (e.g., 0.3 means bottom 30% of each tooth)
            furcation_ratio: (start, end) proportions defining the furcation zone
                             (e.g., (0.3, 0.7) means middle 30%-70% of tooth height)
            blend_mode: How to blend pasted lesions ("direct", "alpha", "poisson")
            buffer_size: Maximum number of lesion crops to store
        """
        self.paste_prob = paste_prob
        self.max_paste_per_image = max_paste_per_image
        self.apex_ratio = apex_ratio
        self.furcation_ratio = furcation_ratio
        self.blend_mode = blend_mode
        self.buffer_size = buffer_size

        # Per-class lesion buffers
        # Each entry: {"crop": ndarray, "crop_mask": ndarray, "cls": int, "size": (h,w)}
        self.lesion_buffers = {
            self.CLS_PERIAPICAL: [],
            self.CLS_PERIODONTAL: [],
            self.CLS_COMBINED: [],
        }

    def collect_lesion(
        self,
        image: "np.ndarray",
        bbox: "np.ndarray",
        cls: int
    ):
        """
        Collect a lesion crop from training data into the buffer.

        Call this during data loading to build up the lesion bank.

        Args:
            image: Full image (H, W, C) or (H, W)
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
            cls: Lesion class (0=periapical, 1=periodontal, 2=combined)
        """
        import numpy as np

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return

        crop = image[y1:y2, x1:x2].copy()

        # Create elliptical alpha mask for smooth blending
        ch, cw = crop.shape[:2]
        yy, xx = np.mgrid[:ch, :cw]
        cy, cx = ch / 2, cw / 2
        dist = ((xx - cx) / (cw / 2)) ** 2 + ((yy - cy) / (ch / 2)) ** 2
        crop_mask = np.clip(1.0 - dist, 0, 1).astype(np.float32)

        buf = self.lesion_buffers.get(cls)
        if buf is not None:
            if len(buf) >= self.buffer_size:
                buf.pop(0)
            buf.append({
                "crop": crop,
                "crop_mask": crop_mask,
                "cls": cls,
                "size": (ch, cw)
            })

    def compute_valid_zones(
        self,
        stage1_mask: "np.ndarray"
    ) -> dict:
        """
        Compute anatomically valid paste zones from Stage1 segmentation.

        For each tooth instance, determines:
        - Apex zone: bottom region where periapical lesions can occur
        - Furcation zone: mid region where furcation lesions can occur

        Args:
            stage1_mask: Stage1 segmentation mask (H, W) with classes 0-4

        Returns:
            Dictionary with keys "apex_mask" and "furcation_mask", each (H, W) binary
        """
        import numpy as np
        import cv2
        from scipy import ndimage

        h, w = stage1_mask.shape

        # Teeth regions (classes 1-4, all tooth types)
        teeth_mask = (stage1_mask >= 1).astype(np.uint8)

        # Separate tooth instances via connected components
        num_labels, labels = cv2.connectedComponents(teeth_mask)

        apex_mask = np.zeros((h, w), dtype=np.uint8)
        furcation_mask = np.zeros((h, w), dtype=np.uint8)

        for label_id in range(1, num_labels):
            instance = (labels == label_id).astype(np.uint8)

            # Find bounding box of this tooth
            ys, xs = np.where(instance > 0)
            if len(ys) == 0:
                continue

            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            tooth_h = y_max - y_min + 1

            if tooth_h < 10:  # Skip tiny noise
                continue

            # Apex zone: bottom portion of the tooth
            # (In dental X-rays, root apex is typically at the bottom of each tooth)
            apex_y_start = y_max - int(tooth_h * self.apex_ratio)
            apex_zone = instance.copy()
            apex_zone[:apex_y_start, :] = 0
            apex_mask = np.maximum(apex_mask, apex_zone)

            # Furcation zone: mid portion where multi-root teeth diverge
            furc_y_start = y_min + int(tooth_h * self.furcation_ratio[0])
            furc_y_end = y_min + int(tooth_h * self.furcation_ratio[1])
            furc_zone = instance.copy()
            furc_zone[:furc_y_start, :] = 0
            furc_zone[furc_y_end:, :] = 0
            furcation_mask = np.maximum(furcation_mask, furc_zone)

        # Slight dilation to allow lesions to extend slightly beyond tooth boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        apex_mask = cv2.dilate(apex_mask, kernel, iterations=1)
        furcation_mask = cv2.dilate(furcation_mask, kernel, iterations=1)

        return {
            "apex_mask": apex_mask,
            "furcation_mask": furcation_mask,
        }

    def _get_valid_mask_for_class(self, cls: int, zones: dict) -> "np.ndarray":
        """
        Get the anatomically valid placement mask for a given lesion class.

        Args:
            cls: Lesion class
            zones: Output of compute_valid_zones()

        Returns:
            Binary mask (H, W) indicating valid paste locations
        """
        import numpy as np

        if cls == self.CLS_PERIAPICAL:
            return zones["apex_mask"]
        elif cls == self.CLS_PERIODONTAL:
            return zones["furcation_mask"]
        elif cls == self.CLS_COMBINED:
            # Combined lesions can appear where periapical and periodontal overlap
            # or in the broader root region
            return np.maximum(zones["apex_mask"], zones["furcation_mask"])
        else:
            return np.zeros_like(zones["apex_mask"])

    def _find_paste_location(
        self,
        valid_mask: "np.ndarray",
        crop_h: int,
        crop_w: int,
        existing_bboxes: "np.ndarray",
        max_attempts: int = 50
    ) -> Optional[Tuple[int, int]]:
        """
        Find a valid paste location that doesn't overlap with existing boxes.

        Args:
            valid_mask: Binary mask of valid locations
            crop_h: Height of crop to paste
            crop_w: Width of crop to paste
            existing_bboxes: Existing bounding boxes [N, 4] in xyxy pixel coords
            max_attempts: Maximum random attempts

        Returns:
            (y, x) top-left corner of paste location, or None if no valid spot found
        """
        import numpy as np

        h, w = valid_mask.shape

        # Erode valid mask by crop size to ensure crop fits entirely inside
        if crop_h >= h or crop_w >= w:
            return None

        # Find candidate positions where the center of the crop can be placed
        half_h, half_w = crop_h // 2, crop_w // 2
        candidate_mask = valid_mask[half_h:h - half_h, half_w:w - half_w].copy()

        if candidate_mask.sum() == 0:
            return None

        ys, xs = np.where(candidate_mask > 0)

        for _ in range(max_attempts):
            idx = np.random.randint(len(ys))
            cy, cx = ys[idx] + half_h, xs[idx] + half_w

            # Compute paste box
            py1 = cy - half_h
            px1 = cx - half_w
            py2 = py1 + crop_h
            px2 = px1 + crop_w

            if py2 > h or px2 > w:
                continue

            # Check overlap with existing boxes (IoU > 0.3 → reject)
            if len(existing_bboxes) > 0:
                overlaps = self._compute_overlap(
                    np.array([[px1, py1, px2, py2]]),
                    existing_bboxes
                )
                if overlaps.max() > 0.3:
                    continue

            return (py1, px1)

        return None

    @staticmethod
    def _compute_overlap(box: "np.ndarray", boxes: "np.ndarray") -> "np.ndarray":
        """Compute IoU between one box and multiple boxes."""
        import numpy as np

        x1 = np.maximum(box[0, 0], boxes[:, 0])
        y1 = np.maximum(box[0, 1], boxes[:, 1])
        x2 = np.minimum(box[0, 2], boxes[:, 2])
        y2 = np.minimum(box[0, 3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[0, 2] - box[0, 0]) * (box[0, 3] - box[0, 1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter

        return inter / (union + 1e-6)

    def _blend_crop(
        self,
        image: "np.ndarray",
        crop: "np.ndarray",
        crop_mask: "np.ndarray",
        y: int,
        x: int
    ) -> "np.ndarray":
        """
        Blend a lesion crop into the image at position (y, x).

        Args:
            image: Target image (H, W, C) or (H, W)
            crop: Lesion crop
            crop_mask: Alpha mask for blending
            y, x: Top-left paste position

        Returns:
            Image with blended lesion
        """
        import numpy as np

        ch, cw = crop.shape[:2]
        result = image.copy()

        if self.blend_mode == "direct":
            if image.ndim == 3:
                result[y:y+ch, x:x+cw] = crop
            else:
                result[y:y+ch, x:x+cw] = crop

        elif self.blend_mode == "alpha":
            alpha = crop_mask
            if image.ndim == 3:
                alpha = alpha[:, :, np.newaxis]
            region = result[y:y+ch, x:x+cw].astype(np.float32)
            blended = region * (1 - alpha) + crop.astype(np.float32) * alpha
            result[y:y+ch, x:x+cw] = blended.astype(image.dtype)

        elif self.blend_mode == "poisson":
            # Poisson blending for seamless integration
            import cv2
            try:
                center = (x + cw // 2, y + ch // 2)
                mask_u8 = (crop_mask * 255).astype(np.uint8)

                if image.ndim == 2:
                    img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    crop_3ch = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
                else:
                    img_3ch = image
                    crop_3ch = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

                blended = cv2.seamlessClone(crop_3ch, img_3ch, mask_u8, center, cv2.NORMAL_CLONE)

                if image.ndim == 2:
                    result = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
                else:
                    result = blended
            except cv2.error:
                # Fallback to alpha blending if Poisson fails
                alpha = crop_mask
                if image.ndim == 3:
                    alpha = alpha[:, :, np.newaxis]
                region = result[y:y+ch, x:x+cw].astype(np.float32)
                blended = region * (1 - alpha) + crop.astype(np.float32) * alpha
                result[y:y+ch, x:x+cw] = blended.astype(image.dtype)

        return result

    def __call__(
        self,
        image: "np.ndarray",
        bboxes: "np.ndarray",
        classes: "np.ndarray",
        stage1_mask: "np.ndarray"
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """
        Apply anatomy-constrained augmentation.

        Args:
            image: Input image (H, W, C) or (H, W)
            bboxes: Bounding boxes (N, 4) in xyxy pixel coordinates
            classes: Class labels (N,) — 0=periapical, 1=periodontal, 2=combined
            stage1_mask: Stage1 segmentation mask (H, W) with class indices 0-4

        Returns:
            Augmented (image, bboxes, classes)
        """
        import numpy as np
        import random

        if random.random() > self.paste_prob:
            return image, bboxes, classes

        # Compute anatomically valid zones from Stage1 output
        zones = self.compute_valid_zones(stage1_mask)

        result_image = image.copy()
        result_bboxes = list(bboxes) if len(bboxes) > 0 else []
        result_classes = list(classes) if len(classes) > 0 else []

        num_pasted = 0

        # Try to paste lesions from each class buffer
        paste_order = [self.CLS_PERIAPICAL, self.CLS_PERIODONTAL, self.CLS_COMBINED]
        random.shuffle(paste_order)

        for cls in paste_order:
            if num_pasted >= self.max_paste_per_image:
                break

            buf = self.lesion_buffers[cls]
            if len(buf) == 0:
                continue

            # Get valid mask for this class
            valid_mask = self._get_valid_mask_for_class(cls, zones)
            if valid_mask.sum() == 0:
                continue

            # Pick a random lesion from buffer
            entry = random.choice(buf)
            crop = entry["crop"]
            crop_mask = entry["crop_mask"]
            ch, cw = entry["size"]

            # Random scale variation (0.8x - 1.2x)
            scale = random.uniform(0.8, 1.2)
            new_h, new_w = int(ch * scale), int(cw * scale)
            if new_h < 4 or new_w < 4:
                continue

            import cv2
            crop = cv2.resize(crop, (new_w, new_h))
            crop_mask = cv2.resize(crop_mask, (new_w, new_h))

            # Find valid placement
            existing_arr = np.array(result_bboxes) if result_bboxes else np.zeros((0, 4))
            location = self._find_paste_location(
                valid_mask, new_h, new_w, existing_arr
            )

            if location is None:
                continue

            py, px = location

            # Blend lesion into image
            result_image = self._blend_crop(result_image, crop, crop_mask, py, px)

            # Add new bbox and class
            result_bboxes.append([px, py, px + new_w, py + new_h])
            result_classes.append(cls)
            num_pasted += 1

        result_bboxes = np.array(result_bboxes, dtype=np.float32).reshape(-1, 4) \
            if result_bboxes else np.zeros((0, 4), dtype=np.float32)
        result_classes = np.array(result_classes, dtype=np.float32) \
            if result_classes else np.zeros((0,), dtype=np.float32)

        return result_image, result_bboxes, result_classes
