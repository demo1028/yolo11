# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Dental Dataset for Stage2 YOLO11 Detection with 4-channel input.

This dataset class handles:
1. Loading raw images and corresponding Stage1 masks
2. Preprocessing into 4-channel tensors
3. Loading YOLO format detection labels (for lesion boxes)
4. Comprehensive data augmentation compatible with multi-channel input
   (Mosaic, MixUp, RandomAffine, Photometric, Flip, Erasing)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.dental_loss import AnatomyConstrainedAugmentation
from .dental_preprocess import DentalStage1Processor
from .dental_augment import Dental4chAugmentation


class DentalYOLODataset(Dataset):
    """
    Dataset for Stage2 dental lesion detection with 4-channel input.

    Expected directory structure:
        data_root/
        ├── images/
        │   ├── train/
        │   │   ├── image1.png
        │   │   └── ...
        │   └── val/
        ├── stage1_masks/
        │   ├── train/
        │   │   ├── image1.png  (class indices 0-4)
        │   │   └── ...
        │   └── val/
        └── labels/
            ├── train/
            │   ├── image1.txt  (YOLO format: cls x_center y_center width height)
            │   └── ...
            └── val/

    Label classes (Stage2):
        0: periapical (根尖周病变)
        1: periodontal (根分叉病变)
        2: combined (联合病变)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        split_file: Optional[str] = None,
        imgsz: int = 640,
        augment: bool = True,
        processor_kwargs: Optional[Dict] = None,
        augment_kwargs: Optional[Dict] = None,
        cache: bool = False,
        prefix: str = ""
    ):
        """
        Initialize Dental YOLO Dataset.

        Args:
            data_root: Root directory containing images/, stage1_masks/, labels/
            split: Dataset split ("train", "val", or "test") - used when split_file is None
            split_file: Path to txt file containing image paths (one per line).
                        If provided, images are loaded from this file instead of
                        scanning data_root/images/{split}/ directory.
            imgsz: Target image size
            augment: Whether to apply augmentation
            processor_kwargs: Arguments for DentalStage1Processor
            augment_kwargs: Arguments for Dental4chAugmentation (e.g. mosaic_prob,
                            mixup_prob, degrees, scale, etc.)
            cache: Whether to cache processed tensors in memory
            prefix: Prefix for logging
        """
        self.data_root = Path(data_root)
        self.split = split
        self.split_file = Path(split_file) if split_file else None
        self.imgsz = imgsz
        self.augment = augment
        self.cache = cache
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""

        # Paths - flat structure (no train/val subdirs) when using split_file
        if self.split_file is not None:
            self.img_dir = self.data_root / "images"
            self.mask_dir = self.data_root / "stage1_masks"
            self.label_dir = self.data_root / "labels"
        else:
            # Legacy mode: use train/val subdirectories
            self.img_dir = self.data_root / "images" / split
            self.mask_dir = self.data_root / "stage1_masks" / split
            self.label_dir = self.data_root / "labels" / split

        # Validate directories
        self._validate_dirs()

        # Get image list
        self.im_files = self._get_image_files()
        LOGGER.info(f"{self.prefix}Found {len(self.im_files)} images")

        # Initialize preprocessor
        processor_kwargs = processor_kwargs or {}
        self.processor = DentalStage1Processor(**processor_kwargs)

        # Anatomy-constrained augmentation is DISABLED by default.
        # Reason: Periapical lesions have strict positional correspondence with
        # specific tooth apices. Random copy-paste (even within "correct" zones)
        # would break this spatial relationship and teach incorrect position patterns.
        # Set use_anatomy_aug=True only if you have a reliable way to match
        # pasted lesions to specific tooth instances.
        self.anatomy_aug = None  # Disabled for dental detection tasks

        # Initialize comprehensive augmentation pipeline (only for training)
        augment_kwargs = augment_kwargs or {}
        self.augmentor = Dental4chAugmentation(
            dataset=self,
            imgsz=imgsz,
            **augment_kwargs
        ) if augment else None

        # Pre-collect lesion crops on first epoch (lazy init)
        self._lesion_bank_built = False

        # Cache storage
        self.cached_tensors = {} if cache else None

        # Lazy-loaded labels list for compatibility with plot_training_labels()
        self._labels = None

    def _validate_dirs(self):
        """Validate that required directories exist."""
        for dir_path, name in [
            (self.img_dir, "images"),
            (self.mask_dir, "stage1_masks"),
            (self.label_dir, "labels")
        ]:
            if not dir_path.exists():
                raise FileNotFoundError(f"{self.prefix}{name} directory not found: {dir_path}")

    def _get_image_files(self) -> List[Path]:
        """Get list of image files from split_file or by scanning directory."""
        if self.split_file is not None:
            # Load from txt file
            if not self.split_file.exists():
                raise FileNotFoundError(f"{self.prefix}Split file not found: {self.split_file}")

            files = []
            with open(self.split_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Handle different path formats in txt file
                    img_path = Path(line)

                    if img_path.is_absolute() and img_path.exists():
                        # Absolute path that exists
                        files.append(img_path)
                    elif (self.img_dir / img_path.name).exists():
                        # Just filename, look in img_dir
                        files.append(self.img_dir / img_path.name)
                    elif (self.data_root / line).exists():
                        # Relative to data_root
                        files.append(self.data_root / line)
                    else:
                        LOGGER.warning(f"{self.prefix}Image not found: {line}")

            return files
        else:
            # Scan directory for image files
            extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            files = []
            for ext in extensions:
                files.extend(self.img_dir.glob(f"*{ext}"))
                files.extend(self.img_dir.glob(f"*{ext.upper()}"))
            return sorted(files)

    def _get_mask_path(self, img_path: Path) -> Path:
        """Get corresponding Stage1 mask path for an image."""
        # Try same extension first
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            return mask_path

        # Try .png extension
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            return mask_path

        raise FileNotFoundError(f"Stage1 mask not found for {img_path.name}")

    def _get_label_path(self, img_path: Path) -> Path:
        """Get corresponding label file path."""
        return self.label_dir / f"{img_path.stem}.txt"

    @property
    def labels(self):
        """Return labels in ultralytics format for plot_training_labels() compatibility.

        Returns list of dicts, each with:
            - "bboxes": np.ndarray (N, 4) normalized xywh
            - "cls": np.ndarray (N,)
        """
        if self._labels is None:
            self._labels = []
            for img_path in self.im_files:
                raw = self._load_labels(img_path)
                if len(raw):
                    self._labels.append({
                        "bboxes": raw[:, 1:5],
                        "cls": raw[:, 0],
                    })
                else:
                    self._labels.append({
                        "bboxes": np.zeros((0, 4), dtype=np.float32),
                        "cls": np.zeros((0,), dtype=np.float32),
                    })
        return self._labels

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.im_files)

    def build_lesion_bank(self):
        """
        Scan the training set to collect lesion crops into the anatomy-constrained
        augmentation buffer. Should be called once before training starts.
        """
        if self.anatomy_aug is None or self._lesion_bank_built:
            return

        LOGGER.info(f"{self.prefix}Building lesion bank for anatomy-constrained augmentation...")
        collected = 0

        for img_path in self.im_files:
            labels = self._load_labels(img_path)
            if len(labels) == 0:
                continue

            raw_image = cv2.imread(str(img_path))
            if raw_image is None:
                continue

            h, w = raw_image.shape[:2]

            for row in labels:
                cls = int(row[0])
                # Convert normalized xywh → pixel xyxy
                cx, cy, bw, bh = row[1], row[2], row[3], row[4]
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                self.anatomy_aug.collect_lesion(
                    raw_image, np.array([x1, y1, x2, y2]), cls
                )
                collected += 1

        self._lesion_bank_built = True
        for cls_id, buf in self.anatomy_aug.lesion_buffers.items():
            cls_names = {0: "periapical", 1: "periodontal", 2: "combined"}
            LOGGER.info(f"{self.prefix}  {cls_names.get(cls_id, cls_id)}: {len(buf)} crops")
        LOGGER.info(f"{self.prefix}Lesion bank built: {collected} total crops")

    # ------------------------------------------------------------------
    # Core item loading
    # ------------------------------------------------------------------

    def _get_base_item(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess a single image WITHOUT augmentation.
        Used internally by mosaic/mixup to load additional tiles.

        Returns:
            tensor_4ch: (H, W, 4) float32 in [0, 1]
            labels: (N, 5) [cls, cx, cy, w, h] normalized
        """
        img_path = self.im_files[index]

        # Check cache
        if self.cache and index in self.cached_tensors:
            tensor_4ch = self.cached_tensors[index].copy()
        else:
            raw_image, stage1_mask = self._load_raw_and_mask(img_path)
            tensor_4ch = self.processor(raw_image, stage1_mask)
            if self.cache:
                self.cached_tensors[index] = tensor_4ch.copy()

        labels = self._load_labels(img_path)
        return tensor_4ch, labels

    def __getitem__(self, index: int) -> Dict:
        """
        Get a single sample.

        Pipeline:
            1. Load image + mask → 4ch tensor
            2. (Train) Anatomy-constrained copy-paste augmentation
            3. (Train) Full augmentation pipeline (Mosaic/MixUp/Affine/Flip/Photometric/Erasing)
               → outputs imgsz x imgsz
            4. (Val)   Letterbox resize to imgsz x imgsz
            5. Convert to torch tensors

        Returns:
            Dictionary with keys:
                - img: 4-channel tensor, shape (4, H, W)
                - bboxes: Bounding boxes, shape (N, 4) in normalized xywh
                - cls: Class labels, shape (N, 1)
                - batch_idx: Batch index tensor
                - im_file: Image file path
        """
        img_path = self.im_files[index]

        # Lazy build lesion bank on first access
        if self.augment and not self._lesion_bank_built:
            self.build_lesion_bank()

        # Load raw image and Stage1 mask
        raw_image, stage1_mask = self._load_raw_and_mask(img_path)

        # Check cache for 4ch tensor
        if self.cache and index in self.cached_tensors:
            tensor_4ch = self.cached_tensors[index].copy()
        else:
            tensor_4ch = self.processor(raw_image, stage1_mask)
            if self.cache:
                self.cached_tensors[index] = tensor_4ch.copy()

        # Load labels
        labels = self._load_labels(img_path)

        if self.augment:
            # Step 1: Anatomy-constrained copy-paste (on primary image only)
            if self.anatomy_aug is not None:
                tensor_4ch, labels = self._apply_anatomy_aug(
                    tensor_4ch, labels, raw_image, stage1_mask
                )

            # Step 2: Full augmentation (Mosaic/MixUp/Affine/Flip/Photometric/Erasing)
            # The augmentor outputs imgsz x imgsz, so no separate resize needed
            tensor_4ch, labels = self.augmentor(tensor_4ch, labels, index)
        else:
            # Val/test: letterbox resize only
            tensor_4ch, labels = self._resize(tensor_4ch, labels)

        # Convert to tensors — HWC -> CHW
        img_tensor = torch.from_numpy(tensor_4ch.transpose(2, 0, 1)).float()

        nl = len(labels)
        if nl:
            bboxes = torch.from_numpy(labels[:, 1:5]).float()
            cls = torch.from_numpy(labels[:, 0:1]).float()
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            cls = torch.zeros((0, 1), dtype=torch.float32)

        return {
            "img": img_tensor,
            "bboxes": bboxes,
            "cls": cls,
            "batch_idx": torch.zeros(nl, dtype=torch.float32),
            "im_file": str(img_path),
            "ori_shape": (raw_image.shape[0], raw_image.shape[1]),
            "resized_shape": (self.imgsz, self.imgsz),
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_raw_and_mask(self, img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load the raw image and Stage1 mask for a given path."""
        raw_image = cv2.imread(str(img_path))
        if raw_image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        mask_path = self._get_mask_path(img_path)
        stage1_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if stage1_mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        return raw_image, stage1_mask

    def _apply_anatomy_aug(
        self,
        tensor_4ch: np.ndarray,
        labels: np.ndarray,
        raw_image: np.ndarray,
        stage1_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply anatomy-constrained copy-paste augmentation.

        The augmentation operates on the raw image (Ch0), then the
        4-channel tensor is re-generated from the augmented image.
        """
        h, w = raw_image.shape[:2]

        if len(labels) == 0:
            classes_px = np.zeros((0,), dtype=np.float32)
            bboxes_px = np.zeros((0, 4), dtype=np.float32)
        else:
            classes_px = labels[:, 0]
            # Convert normalized xywh → pixel xyxy for overlap checking
            cx = labels[:, 1] * w
            cy = labels[:, 2] * h
            bw = labels[:, 3] * w
            bh = labels[:, 4] * h
            bboxes_px = np.stack([
                cx - bw / 2, cy - bh / 2,
                cx + bw / 2, cy + bh / 2
            ], axis=1)

        # Run anatomy-constrained augmentation on raw image
        aug_image, aug_bboxes_px, aug_classes = self.anatomy_aug(
            raw_image, bboxes_px, classes_px, stage1_mask
        )

        # If something was pasted, re-generate 4ch tensor from augmented image
        if len(aug_bboxes_px) != len(bboxes_px):
            tensor_4ch = self.processor(aug_image, stage1_mask)

            # Convert augmented pixel xyxy → normalized xywh
            new_labels = []
            for i in range(len(aug_classes)):
                x1, y1, x2, y2 = aug_bboxes_px[i]
                cx_n = ((x1 + x2) / 2) / w
                cy_n = ((y1 + y2) / 2) / h
                bw_n = (x2 - x1) / w
                bh_n = (y2 - y1) / h
                new_labels.append([aug_classes[i], cx_n, cy_n, bw_n, bh_n])
            labels = np.array(new_labels, dtype=np.float32) if new_labels else np.zeros((0, 5), dtype=np.float32)

        return tensor_4ch, labels

    def _load_labels(self, img_path: Path) -> np.ndarray:
        """
        Load YOLO format labels.

        Format: class x_center y_center width height (normalized)

        Returns:
            Array of shape (N, 5) with [class, x_center, y_center, width, height]
        """
        label_path = self._get_label_path(img_path)

        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        try:
            with open(label_path, "r") as f:
                lines = f.read().strip().splitlines()

            labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    labels.append([cls, x_center, y_center, width, height])

            return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

        except Exception as e:
            LOGGER.warning(f"Error loading labels from {label_path}: {e}")
            return np.zeros((0, 5), dtype=np.float32)

    def _resize(
        self,
        tensor_4ch: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize tensor to target size with letterbox padding (used for val/test)."""
        h, w = tensor_4ch.shape[:2]

        # Calculate scale and padding
        scale = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(h * scale), int(w * scale)
        pad_h = (self.imgsz - new_h) // 2
        pad_w = (self.imgsz - new_w) // 2

        # Resize
        resized = np.zeros((self.imgsz, self.imgsz, 4), dtype=np.float32)
        for c in range(4):
            channel_resized = cv2.resize(tensor_4ch[:, :, c], (new_w, new_h))
            resized[pad_h:pad_h + new_h, pad_w:pad_w + new_w, c] = channel_resized

        # Adjust labels for letterbox
        if len(labels):
            labels = labels.copy()
            labels[:, 1] = (labels[:, 1] * new_w + pad_w) / self.imgsz  # x_center
            labels[:, 2] = (labels[:, 2] * new_h + pad_h) / self.imgsz  # y_center
            labels[:, 3] = labels[:, 3] * new_w / self.imgsz  # width
            labels[:, 4] = labels[:, 4] * new_h / self.imgsz  # height

        return resized, labels

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for DataLoader."""
        new_batch = {}
        keys = batch[0].keys()

        for key in keys:
            values = [b[key] for b in batch]

            if key == "img":
                new_batch[key] = torch.stack(values, 0)
            elif key in {"bboxes", "cls"}:
                new_batch[key] = torch.cat(values, 0)
            elif key == "batch_idx":
                # Add batch index offset
                batch_idx = []
                for i, v in enumerate(values):
                    batch_idx.append(v + i)
                new_batch[key] = torch.cat(batch_idx, 0)
            elif key in {"im_file", "ori_shape", "resized_shape"}:
                new_batch[key] = values
            else:
                new_batch[key] = values

        return new_batch


def create_dental_dataloader(
    data_root: str,
    split: str = "train",
    split_file: Optional[str] = None,
    batch_size: int = 16,
    imgsz: int = 640,
    augment: bool = True,
    workers: int = 4,
    shuffle: bool = True,
    augment_kwargs: Optional[Dict] = None,
    **kwargs
):
    """
    Create DataLoader for Dental YOLO training.

    Args:
        data_root: Path to dataset root (containing images/, stage1_masks/, labels/)
        split: Dataset split ("train", "val", "test") - used when split_file is None
        split_file: Path to txt file with image paths. When provided:
                    - Images are loaded from this file (one path per line)
                    - data_root should have flat structure: images/, stage1_masks/, labels/
                    - Example: fold_1_train.txt, fold_1_val.txt for cross-validation
        batch_size: Batch size
        imgsz: Image size
        augment: Enable augmentation (only applies when this is True)
        workers: Number of workers
        shuffle: Shuffle data
        augment_kwargs: Dict of augmentation parameters passed to
                        Dental4chAugmentation, e.g.:
                        {"mosaic_prob": 0.5, "mixup_prob": 0.1,
                         "degrees": 10.0, "scale": (0.5, 1.5), ...}
        **kwargs: Additional arguments for DentalYOLODataset

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader

    # Determine if this is a training set
    is_train = augment and (
        split == "train" or
        (split_file and "train" in str(split_file).lower())
    )

    dataset = DentalYOLODataset(
        data_root=data_root,
        split=split,
        split_file=split_file,
        imgsz=imgsz,
        augment=is_train,
        augment_kwargs=augment_kwargs,
        **kwargs
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and is_train,
        num_workers=workers,
        collate_fn=DentalYOLODataset.collate_fn,
        pin_memory=True,
        drop_last=is_train
    )

    return loader
