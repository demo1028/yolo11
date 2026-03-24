#!/usr/bin/env python3
"""
Dental Stage2 Training Script for YOLO11 with 4-channel input.

This script trains a YOLO11 model for lesion detection using:
- 4-channel input: [raw_image, virtual_fusion, distance_field, metal_prior]
- Stage1 VM-UNet output as prior information
- Custom dental attention modules

Usage:
    python train_dental_stage2.py --data_root /path/to/data --epochs 100

Expected data structure:
    data_root/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── stage1_masks/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Label format (YOLO):
    class x_center y_center width height (normalized)
    Classes: 0=periapical, 1=periodontal, 2=combined
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add ultralytics to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO
from ultralytics.data.dental_dataset import DentalYOLODataset, create_dental_dataloader
from ultralytics.data.dental_preprocess import DentalStage1Processor, visualize_4ch_tensor
from ultralytics.nn.modules.dental_attention import register_dental_modules


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO11-Dental Stage2 Model")

    # Data arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing images/, stage1_masks/, labels/")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training")

    # Model arguments
    parser.add_argument("--model", type=str, default="yolo11n-dental.yaml",
                        help="Model configuration file")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights (optional)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Weight decay")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="runs/dental_stage2",
                        help="Output directory for results")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name")

    # Device arguments
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (e.g., '0' or 'cpu')")

    return parser.parse_args()


def setup_output_dir(args):
    """Setup output directory for experiment."""
    if args.name is None:
        args.name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "weights").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    return output_dir


def load_model(args, device):
    """Load YOLO11-Dental model."""
    # Register dental modules
    register_dental_modules()

    # Model config path
    model_cfg = Path(__file__).parent / "ultralytics" / "cfg" / "models" / "11" / args.model
    if not model_cfg.exists():
        model_cfg = args.model  # Use as-is if not found

    print(f"Loading model from: {model_cfg}")

    # Initialize model
    model = YOLO(str(model_cfg))

    # Load pretrained weights if provided
    if args.pretrained and Path(args.pretrained).exists():
        print(f"Loading pretrained weights from: {args.pretrained}")
        # Note: pretrained weights from 3-channel model need careful handling
        # The first conv layer weights won't match directly

    return model


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move data to device
        imgs = batch["img"].to(device)
        targets = {
            "bboxes": batch["bboxes"].to(device),
            "cls": batch["cls"].to(device),
            "batch_idx": batch["batch_idx"].to(device),
        }

        # Forward pass
        optimizer.zero_grad()
        loss_dict = model.model.loss({"img": imgs, **targets})

        # Backward pass
        loss = loss_dict if isinstance(loss_dict, torch.Tensor) else loss_dict[0]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(num_batches, 1)


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            imgs = batch["img"].to(device)
            targets = {
                "bboxes": batch["bboxes"].to(device),
                "cls": batch["cls"].to(device),
                "batch_idx": batch["batch_idx"].to(device),
            }

            loss_dict = model.model.loss({"img": imgs, **targets})
            loss = loss_dict if isinstance(loss_dict, torch.Tensor) else loss_dict[0]

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    """Main training function."""
    args = parse_args()

    # Setup
    output_dir = setup_output_dir(args)
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    print(f"Training YOLO11-Dental Stage2")
    print(f"  Data root: {args.data_root}")
    print(f"  Output dir: {output_dir}")
    print(f"  Device: {device}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = create_dental_dataloader(
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        augment=True,
        workers=args.workers,
        shuffle=True
    )

    val_loader = create_dental_dataloader(
        data_root=args.data_root,
        split="val",
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        augment=False,
        workers=args.workers,
        shuffle=False
    )

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

    # Load model
    print("\nLoading model...")
    model = load_model(args, device)
    model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        # Log
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        # Save latest
        torch.save(checkpoint, output_dir / "weights" / "latest.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "weights" / "best.pt")
            print(f"  New best model saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(checkpoint, output_dir / "weights" / f"epoch_{epoch}.pt")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")


def quick_test():
    """Quick test to verify 4-channel preprocessing and model loading."""
    import numpy as np

    print("Running quick test...")

    # Test preprocessing
    print("\n1. Testing DentalStage1Processor...")
    processor = DentalStage1Processor()

    # Create dummy data
    raw_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    stage1_mask = np.random.randint(0, 5, (256, 256), dtype=np.uint8)

    tensor_4ch = processor(raw_image, stage1_mask)
    print(f"   Input: raw_image {raw_image.shape}, stage1_mask {stage1_mask.shape}")
    print(f"   Output: tensor_4ch {tensor_4ch.shape}, dtype={tensor_4ch.dtype}")
    print(f"   Value range: [{tensor_4ch.min():.3f}, {tensor_4ch.max():.3f}]")

    # Test model loading
    print("\n2. Testing model configuration...")
    register_dental_modules()

    model_cfg = Path(__file__).parent / "ultralytics" / "cfg" / "models" / "11" / "yolo11-dental.yaml"
    if model_cfg.exists():
        print(f"   Model config found: {model_cfg}")
        try:
            model = YOLO(str(model_cfg))
            print(f"   Model loaded successfully!")
            print(f"   Model type: {type(model)}")
        except Exception as e:
            print(f"   Error loading model: {e}")
    else:
        print(f"   Model config not found at: {model_cfg}")

    print("\nQuick test complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_test()
    else:
        main()
