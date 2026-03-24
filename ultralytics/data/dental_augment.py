# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Comprehensive data augmentation for 4-channel dental images.

Channel semantics:
- Ch0: I_raw (raw grayscale image) - supports all augmentations
- Ch1: M_virtual (virtual fusion mask, binary 0/1) - geometry only
- Ch2: M_metal (metal prior mask, binary 0/1) - geometry only
- Ch3: F_distance (distance transform field) - geometry only

Augmentation design principles:
- GEOMETRIC transforms (affine, flip, mosaic): Applied to ALL 4 channels
  synchronously to maintain spatial correspondence.
- PHOTOMETRIC transforms (brightness, contrast, CLAHE, blur, noise):
  Applied ONLY to Ch0. Prior channels (Ch1-3) have semantic meaning
  that would be corrupted by intensity changes.
- MIXUP: Only blends Ch0; prior channels keep primary image's values.
- RANDOM ERASING: Only erases Ch0; prior channels preserved.

Augmentation types:
- Mosaic: 4-image combination (geometry sync across all channels)
- MixUp: Image blending (Ch0 only, priors unchanged)
- Random Affine: Rotation, scale, translate, shear (all channels)
- Photometric: CLAHE, blur, noise, brightness/contrast (Ch0 only)
- Flip: Horizontal + vertical (all channels)
- Random Erasing: Rectangular dropout (Ch0 only)
"""

import math
from typing import Tuple

import cv2
import numpy as np


class Dental4chAugmentation:
    """
    Full augmentation pipeline for 4-channel dental tensors.

    Designed for dental X-ray images where:
    - Ch0: Raw grayscale (supports photometric augmentation)
    - Ch1-3: Prior channels (geometric transforms only)
    """

    def __init__(
        self,
        dataset,
        imgsz: int = 640,
        # Mosaic / MixUp
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.1,
        mixup_alpha: float = 0.5,
        # Affine (adjusted for dental periapical X-rays)
        degrees: float = 10.0,
        translate: float = 0.2,
        scale: Tuple[float, float] = (0.5, 1.5),
        shear: float = 0.0,
        perspective: float = 0.0,
        # Flip (both OK since data contains upper and lower jaw)
        fliplr: float = 0.5,
        flipud: float = 0.5,
        # Photometric (Ch0 only)
        brightness: float = 0.3,
        contrast: float = 0.3,
        clahe_prob: float = 0.3,
        blur_prob: float = 0.2,
        noise_prob: float = 0.2,
        # Erasing (reduced to avoid erasing small lesions)
        erasing_prob: float = 0.1,
        erasing_ratio: Tuple[float, float] = (0.02, 0.1),
    ):
        self.dataset = dataset
        self.imgsz = imgsz

        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

        self.fliplr = fliplr
        self.flipud = flipud

        self.brightness = brightness
        self.contrast = contrast
        self.clahe_prob = clahe_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob

        self.erasing_prob = erasing_prob
        self.erasing_ratio = erasing_ratio

    def __call__(self, tensor_4ch, labels, index):
        """
        Apply full augmentation pipeline.

        Args:
            tensor_4ch: (H, W, 4) float32 in [0, 1]
            labels: (N, 5) array of [cls, cx, cy, w, h] normalized
            index: Current index in dataset

        Returns:
            Augmented tensor (imgsz, imgsz, 4), augmented labels (M, 5) normalized
        """
        s = self.imgsz
        h, w = tensor_4ch.shape[:2]

        # Convert labels to pixel xyxy for internal processing
        boxes, cls = self._labels_to_pixel_xyxy(labels, w, h)

        # === Mosaic (4-image combination) ===
        if np.random.random() < self.mosaic_prob:
            tensor_4ch, boxes, cls = self._mosaic4(tensor_4ch, boxes, cls, index)
        else:
            tensor_4ch, boxes = self._resize_4ch(tensor_4ch, boxes, s, s)

        # === Random Affine (maps 2s→s for mosaic, or s→s otherwise) ===
        h_in, w_in = tensor_4ch.shape[:2]
        tensor_4ch, boxes = self._random_affine(
            tensor_4ch, boxes, w_in, h_in, s, s
        )

        # === MixUp (image blending) ===
        if np.random.random() < self.mixup_prob:
            tensor_4ch, boxes, cls = self._mixup(tensor_4ch, boxes, cls, index)

        # === Random Flips ===
        tensor_4ch, boxes = self._flips(tensor_4ch, boxes)

        # === Photometric on Ch0 ===
        tensor_4ch = self._photometric_ch0(tensor_4ch)

        # === Random Erasing ===
        tensor_4ch = self._random_erasing(tensor_4ch)

        # Clip boxes and filter invalid ones
        h_out, w_out = tensor_4ch.shape[:2]
        boxes, cls = self._clip_and_filter(boxes, cls, w_out, h_out)

        # Convert back to normalized xywh
        labels_out = self._pixel_xyxy_to_labels(boxes, cls, w_out, h_out)
        return tensor_4ch, labels_out

    def close_mosaic(self):
        """Disable mosaic and mixup (call during last training epochs)."""
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0

    # ================================================================
    # Label conversion helpers
    # ================================================================

    @staticmethod
    def _labels_to_pixel_xyxy(labels, w, h):
        """[cls, cx, cy, w, h] normalized → pixel xyxy boxes + cls."""
        if len(labels) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
            )
        cls = labels[:, 0:1].copy()
        cx = labels[:, 1] * w
        cy = labels[:, 2] * h
        bw = labels[:, 3] * w
        bh = labels[:, 4] * h
        boxes = np.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1
        )
        return boxes.astype(np.float32), cls.astype(np.float32)

    @staticmethod
    def _pixel_xyxy_to_labels(boxes, cls, w, h):
        """Pixel xyxy boxes + cls → [cls, cx, cy, w, h] normalized."""
        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2) / w
        cy = ((boxes[:, 1] + boxes[:, 3]) / 2) / h
        bw = (boxes[:, 2] - boxes[:, 0]) / w
        bh = (boxes[:, 3] - boxes[:, 1]) / h
        return np.hstack(
            [cls, np.stack([cx, cy, bw, bh], axis=1)]
        ).astype(np.float32)

    # ================================================================
    # Mosaic (4-image)
    # ================================================================

    def _mosaic4(self, tensor_4ch, boxes, cls, index):
        """
        Combine current image with 3 random images in a 2x2 mosaic grid.
        Returns a 2s x 2s canvas (reduced to s x s by subsequent affine).
        """
        s = self.imgsz
        canvas = np.zeros((2 * s, 2 * s, 4), dtype=np.float32)

        # Random mosaic center
        yc = int(np.random.uniform(s * 0.5, s * 1.5))
        xc = int(np.random.uniform(s * 0.5, s * 1.5))

        n_dataset = len(self.dataset)
        indices = [index] + [np.random.randint(0, n_dataset) for _ in range(3)]

        all_boxes = []
        all_cls = []

        for i, idx in enumerate(indices):
            if i == 0:
                img = tensor_4ch
                bx = boxes.copy() if len(boxes) else np.zeros((0, 4), dtype=np.float32)
                cl = cls.copy() if len(cls) else np.zeros((0, 1), dtype=np.float32)
            else:
                img, bx, cl = self._load_other_item(idx)

            # Resize tile to s x s
            img, bx = self._resize_4ch(img, bx, s, s)

            # Compute placement regions
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - s, 0), max(yc - s, 0), xc, yc
                x1b, y1b = s - (x2a - x1a), s - (y2a - y1a)
                x2b, y2b = s, s
            elif i == 1:  # top-right
                x1a, y1a = xc, max(yc - s, 0)
                x2a, y2a = min(xc + s, 2 * s), yc
                x1b, y1b = 0, s - (y2a - y1a)
                x2b, y2b = min(s, x2a - x1a), s
            elif i == 2:  # bottom-left
                x1a, y1a = max(xc - s, 0), yc
                x2a, y2a = xc, min(yc + s, 2 * s)
                x1b, y1b = s - (x2a - x1a), 0
                x2b, y2b = s, min(s, y2a - y1a)
            else:  # bottom-right
                x1a, y1a = xc, yc
                x2a = min(xc + s, 2 * s)
                y2a = min(yc + s, 2 * s)
                x1b, y1b = 0, 0
                x2b, y2b = min(s, x2a - x1a), min(s, y2a - y1a)

            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # Offset boxes from tile space to canvas space
            if len(bx):
                bx_adj = bx.copy()
                pad_x = x1a - x1b
                pad_y = y1a - y1b
                bx_adj[:, [0, 2]] += pad_x
                bx_adj[:, [1, 3]] += pad_y
                all_boxes.append(bx_adj)
                all_cls.append(cl)

        if all_boxes:
            boxes_out = np.concatenate(all_boxes, axis=0)
            cls_out = np.concatenate(all_cls, axis=0)
        else:
            boxes_out = np.zeros((0, 4), dtype=np.float32)
            cls_out = np.zeros((0, 1), dtype=np.float32)

        return canvas, boxes_out, cls_out

    def _load_other_item(self, idx):
        """Load and preprocess another dataset item for mosaic/mixup."""
        tensor, labels = self.dataset._get_base_item(idx)
        h, w = tensor.shape[:2]
        boxes, cls = self._labels_to_pixel_xyxy(labels, w, h)
        return tensor, boxes, cls

    # ================================================================
    # MixUp
    # ================================================================

    def _mixup(self, tensor_4ch, boxes, cls, index):
        """
        Blend current image with a random image.

        IMPORTANT: Only Ch0 (raw image) is blended. Ch1-3 (prior channels) keep
        the primary image's values to preserve semantic meaning of priors.
        - M_virtual, M_metal are binary masks (0/1)
        - F_distance is a distance field
        Blending these would create invalid intermediate values.
        """
        s = self.imgsz
        n_dataset = len(self.dataset)
        idx2 = np.random.randint(0, n_dataset)

        img2, bx2, cl2 = self._load_other_item(idx2)
        img2, bx2 = self._resize_4ch(img2, bx2, s, s)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1 - lam)

        h, w = tensor_4ch.shape[:2]
        if (h, w) != (s, s):
            tensor_4ch, boxes = self._resize_4ch(tensor_4ch, boxes, s, s)

        # Only blend Ch0 (raw image), keep Ch1-3 (priors) from primary image
        tensor_4ch = tensor_4ch.copy()
        tensor_4ch[:, :, 0] = tensor_4ch[:, :, 0] * lam + img2[:, :, 0] * (1 - lam)
        # Ch1-3 remain unchanged (priors from primary image)

        if len(bx2):
            boxes = np.concatenate([boxes, bx2], 0) if len(boxes) else bx2.copy()
            cls = np.concatenate([cls, cl2], 0) if len(cls) else cl2.copy()

        return tensor_4ch, boxes, cls

    # ================================================================
    # Random Affine Transform
    # ================================================================

    def _random_affine(self, img, boxes, w_in, h_in, w_out, h_out):
        """
        Apply random affine/perspective transform.
        Maps from (w_in, h_in) input to (w_out, h_out) output.
        """
        # 1. Center input at origin
        C = np.eye(3, dtype=np.float64)
        C[0, 2] = -w_in / 2
        C[1, 2] = -h_in / 2

        # 2. Perspective
        P = np.eye(3, dtype=np.float64)
        P[2, 0] = np.random.uniform(-self.perspective, self.perspective)
        P[2, 1] = np.random.uniform(-self.perspective, self.perspective)

        # 3. Rotation + Scale
        R = np.eye(3, dtype=np.float64)
        angle = np.random.uniform(-self.degrees, self.degrees)
        scale = np.random.uniform(self.scale[0], self.scale[1])
        R[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)

        # 4. Shear
        S = np.eye(3, dtype=np.float64)
        S[0, 1] = math.tan(math.radians(np.random.uniform(-self.shear, self.shear)))
        S[1, 0] = math.tan(math.radians(np.random.uniform(-self.shear, self.shear)))

        # 5. Translation
        T = np.eye(3, dtype=np.float64)
        T[0, 2] = np.random.uniform(-self.translate, self.translate) * w_out
        T[1, 2] = np.random.uniform(-self.translate, self.translate) * h_out

        # 6. Center to output
        Cb = np.eye(3, dtype=np.float64)
        Cb[0, 2] = w_out / 2
        Cb[1, 2] = h_out / 2

        # Combined transform
        M = Cb @ T @ S @ R @ P @ C

        # Apply to all 4 channels
        result = np.zeros((h_out, w_out, 4), dtype=np.float32)
        if self.perspective != 0:
            for c in range(4):
                result[:, :, c] = cv2.warpPerspective(
                    img[:, :, c], M, (w_out, h_out),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
        else:
            M_affine = M[:2]
            for c in range(4):
                result[:, :, c] = cv2.warpAffine(
                    img[:, :, c], M_affine, (w_out, h_out),
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )

        # Transform bounding boxes via 4-corner projection
        if len(boxes):
            n = len(boxes)
            corners = np.zeros((n, 4, 2), dtype=np.float64)
            corners[:, 0] = boxes[:, [0, 1]]  # top-left
            corners[:, 1] = boxes[:, [2, 1]]  # top-right
            corners[:, 2] = boxes[:, [2, 3]]  # bottom-right
            corners[:, 3] = boxes[:, [0, 3]]  # bottom-left

            pts = corners.reshape(-1, 2)
            pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
            pts_t = (M @ pts_h.T).T
            pts_t = pts_t[:, :2] / pts_t[:, 2:3]
            pts_t = pts_t.reshape(n, 4, 2)

            x_min = pts_t[:, :, 0].min(axis=1)
            y_min = pts_t[:, :, 1].min(axis=1)
            x_max = pts_t[:, :, 0].max(axis=1)
            y_max = pts_t[:, :, 1].max(axis=1)
            boxes = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.float32)

        return result, boxes

    # ================================================================
    # Resize helper
    # ================================================================

    @staticmethod
    def _resize_4ch(img, boxes, tw, th):
        """Resize 4-channel image and scale boxes accordingly."""
        h, w = img.shape[:2]
        if (h, w) == (th, tw):
            return img, boxes

        result = np.zeros((th, tw, 4), dtype=np.float32)
        for c in range(4):
            result[:, :, c] = cv2.resize(
                img[:, :, c], (tw, th), interpolation=cv2.INTER_LINEAR
            )

        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= tw / w
            boxes[:, [1, 3]] *= th / h

        return result, boxes

    # ================================================================
    # Flips
    # ================================================================

    def _flips(self, img, boxes):
        """Apply random horizontal and vertical flips."""
        h, w = img.shape[:2]

        if np.random.random() < self.fliplr:
            img = np.ascontiguousarray(np.flip(img, axis=1))
            if len(boxes):
                boxes = boxes.copy()
                x1_new = w - boxes[:, 2]
                x2_new = w - boxes[:, 0]
                boxes[:, 0] = x1_new
                boxes[:, 2] = x2_new

        if np.random.random() < self.flipud:
            img = np.ascontiguousarray(np.flip(img, axis=0))
            if len(boxes):
                boxes = boxes.copy()
                y1_new = h - boxes[:, 3]
                y2_new = h - boxes[:, 1]
                boxes[:, 1] = y1_new
                boxes[:, 3] = y2_new

        return img, boxes

    # ================================================================
    # Photometric (Ch0 only)
    # ================================================================

    def _photometric_ch0(self, img):
        """Apply photometric augmentations to channel 0 only."""
        ch0 = img[:, :, 0].copy()

        # Brightness & Contrast
        if self.brightness > 0 or self.contrast > 0:
            alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            beta = np.random.uniform(-self.brightness, self.brightness)
            ch0 = np.clip(ch0 * alpha + beta, 0, 1)

        # CLAHE
        if np.random.random() < self.clahe_prob:
            ch0_u8 = (ch0 * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ch0_u8 = clahe.apply(ch0_u8)
            ch0 = ch0_u8.astype(np.float32) / 255.0

        # Gaussian blur
        if np.random.random() < self.blur_prob:
            ksize = np.random.choice([3, 5, 7])
            ch0 = cv2.GaussianBlur(ch0, (ksize, ksize), 0)

        # Gaussian noise
        if np.random.random() < self.noise_prob:
            sigma = np.random.uniform(0.005, 0.03)
            noise = np.random.normal(0, sigma, ch0.shape).astype(np.float32)
            ch0 = np.clip(ch0 + noise, 0, 1)

        img = img.copy()
        img[:, :, 0] = ch0
        return img

    # ================================================================
    # Random Erasing
    # ================================================================

    def _random_erasing(self, img):
        """
        Randomly erase a rectangular region in Ch0 only.

        Only Ch0 (raw image) is erased. Ch1-3 (prior channels) are preserved
        to maintain semantic consistency of priors (masks, distance fields).
        """
        if np.random.random() >= self.erasing_prob:
            return img

        h, w = img.shape[:2]
        area = h * w

        for _ in range(10):
            target_area = np.random.uniform(*self.erasing_ratio) * area
            aspect = np.random.uniform(0.3, 3.3)
            eh = int(round(math.sqrt(target_area * aspect)))
            ew = int(round(math.sqrt(target_area / aspect)))

            if eh < h and ew < w:
                y = np.random.randint(0, h - eh)
                x = np.random.randint(0, w - ew)
                img = img.copy()
                img[y:y + eh, x:x + ew, 0] = 0  # Only erase Ch0
                break

        return img

    # ================================================================
    # Box filtering
    # ================================================================

    @staticmethod
    def _clip_and_filter(boxes, cls, w, h, min_area_ratio=0.001, min_wh=2):
        """Clip boxes to image bounds and remove tiny/invalid ones."""
        if len(boxes) == 0:
            return boxes, cls

        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        area = bw * bh

        valid = (bw >= min_wh) & (bh >= min_wh) & (area >= min_area_ratio * w * h)
        return boxes[valid], cls[valid]
