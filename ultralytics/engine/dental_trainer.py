# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Custom Trainer/Validator for 4-channel dental YOLO detection.

Inherits all official training optimizations from DetectionTrainer:
- EMA (Exponential Moving Average)
- Warmup scheduler
- AMP (Automatic Mixed Precision)
- Cosine LR schedule
- Close-mosaic for final epochs
- Smart logging, checkpointing, early stopping

Enhancements for dental lesion detection:
- 4-channel input support (raw + prior channels)
- Small object optimized loss (WIoU + NWD)
- Size-aware weighting for small lesions

Usage:
    from ultralytics.engine.dental_trainer import DentalTrainer

    trainer = DentalTrainer(overrides={
        "model": "yolo11-dental.yaml",
        "data": "dental3cls.yaml",
        "epochs": 200,
        "imgsz": 640,
        "batch": 16,
        "device": "0",
        "optimizer": "AdamW",
        "lr0": 0.001,
        "cos_lr": True,
        "close_mosaic": 10,
        "dental_loss": "wiou_nwd",  # Small object optimized loss
    })
    trainer.train()
"""

from copy import copy

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, RANK


class DentalValidator(DetectionValidator):
    """
    Validator for 4-channel dental YOLO detection.
    Overrides data loading and preprocessing to handle 4ch input.
    """

    def get_dataloader(self, dataset_path, batch_size):
        """Return 4-channel dental validation dataloader."""
        from torch.utils.data import DataLoader
        from ultralytics.data.dental_dataset import DentalYOLODataset

        data_root = self.data.get("path", "") if isinstance(self.data, dict) else ""

        dataset = DentalYOLODataset(
            data_root=data_root,
            split_file=str(dataset_path),
            imgsz=self.args.imgsz,
            augment=False,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=DentalYOLODataset.collate_fn,
            pin_memory=True,
        )

    def preprocess(self, batch):
        """Dental data is already float32 [0, 1] — skip /255 normalization."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        for k in ["batch_idx", "cls", "bboxes"]:
            if k in batch:
                batch[k] = batch[k].to(self.device)
        self.lb = []
        return batch


class DentalTrainer(DetectionTrainer):
    """
    Trainer for 4-channel dental YOLO detection.

    Inherits ALL official training optimizations from DetectionTrainer.
    Overrides:
    - get_dataloader: use DentalYOLODataset (4-channel)
    - preprocess_batch: skip /255 (already normalized)
    - get_model: register C3k2_Dental, ECA, etc.
    - get_validator: use DentalValidator
    - get_loss: use small object optimized loss (WIoU + NWD)
    - _close_dataloader_mosaic: call augmentor.close_mosaic()
    - plot_training_samples: visualize Ch0 only

    Custom args:
    - dental_loss: Loss type ('ciou', 'wiou', 'wiou_nwd', 'inner_wiou_nwd')
    - small_threshold: Size threshold for small object weighting (default: 32)
    - small_weight: Extra weight for small objects (default: 2.0)
    """

    # Keys that are dental-specific and not recognized by ultralytics get_cfg()
    _DENTAL_KEYS = {"dental_loss", "small_threshold", "small_weight"}

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize with custom loss settings."""
        # Extract dental-specific keys before parent validates overrides
        overrides = dict(overrides) if overrides else {}
        self.dental_loss_type = overrides.pop("dental_loss", "wiou_nwd")
        self.small_threshold = overrides.pop("small_threshold", 32)
        self.small_weight = overrides.pop("small_weight", 2.0)

        super().__init__(cfg, overrides, _callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Return 4-channel dental dataloader."""
        from torch.utils.data import DataLoader
        from ultralytics.data.dental_dataset import DentalYOLODataset

        is_train = mode == "train"
        data_root = self.data.get("path", "")

        dataset = DentalYOLODataset(
            data_root=data_root,
            split_file=str(dataset_path),
            imgsz=self.args.imgsz,
            augment=is_train,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train and rank in {-1, 0},
            num_workers=self.args.workers,
            collate_fn=DentalYOLODataset.collate_fn,
            pin_memory=True,
            drop_last=is_train,
        )

    def preprocess_batch(self, batch):
        """Dental data is already float32 [0, 1] — skip /255 normalization."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build detection model with custom dental modules registered."""
        from ultralytics.nn.modules.dental_attention import register_dental_modules
        register_dental_modules()
        return super().get_model(cfg, weights, verbose)

    def get_validator(self):
        """Return dental-specific validator."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        val = DentalValidator(
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        val.data = self.data
        return val

    def get_loss(self, model=None):
        """
        Return small-object optimized loss for dental lesion detection.

        Loss options (set via 'dental_loss' in overrides):
        - 'ciou': Standard CIoU loss
        - 'wiou': Wise-IoU with dynamic focusing
        - 'wiou_nwd': WIoU + Normalized Wasserstein Distance (recommended)
        - 'inner_wiou_nwd': Inner-IoU + WIoU + NWD (for very small lesions)
        """
        try:
            from ultralytics.utils.loss_dental_v3 import DentalDetectionLoss

            if model is None:
                model = self.model

            loss = DentalDetectionLoss(
                model,
                box_loss_type=self.dental_loss_type,
                use_size_aware=True,
                small_threshold=self.small_threshold,
                small_weight=self.small_weight,
            )
            LOGGER.info(f"Using dental loss: {self.dental_loss_type} "
                        f"(small_threshold={self.small_threshold}, small_weight={self.small_weight})")
            return loss

        except Exception as e:
            LOGGER.warning(f"Failed to load dental loss: {e}. Using default loss.")
            return super().get_loss(model)

    def _close_dataloader_mosaic(self):
        """Disable mosaic/mixup in dental augmentor for final epochs."""
        dataset = self.train_loader.dataset
        if hasattr(dataset, "augmentor") and dataset.augmentor is not None:
            dataset.augmentor.close_mosaic()
            LOGGER.info("Dental mosaic/mixup disabled for final epochs")

    def plot_training_samples(self, batch, ni):
        """Use Ch0 (grayscale) for visualization since images are 4-channel."""
        try:
            vis_batch = {k: v for k, v in batch.items()}
            ch0 = batch["img"][:, 0:1, :, :]
            vis_batch["img"] = (ch0.repeat(1, 3, 1, 1) * 255).clamp(0, 255).to(batch["img"].dtype)
            super().plot_training_samples(vis_batch, ni)
        except Exception:
            pass
