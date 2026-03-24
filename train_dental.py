#!/usr/bin/env python3
"""
Training script for YOLO11-Dental 4-channel lesion detection.

Usage:
    # Basic training
    python train_dental.py --data_root /path/to/data --train_txt train.txt --val_txt val.txt

    # With custom model config
    python train_dental.py --model yolo11-dental-full.yaml --epochs 200 --batch 16

    # Resume training
    python train_dental.py --resume runs/dental/train/weights/last.pt

Examples:
    # Train with default settings (yolo11-dental.yaml, 4-channel input)
    python train_dental.py \
        --data_root D:/data/dental \
        --train_txt D:/data/dental/fold_3_train.txt \
        --val_txt D:/data/dental/fold_3_val.txt \
        --epochs 200 \
        --batch 16 \
        --device 0

    # Train full model (4ch + P2 + ECA)
    python train_dental.py \
        --model yolo11-dental-full.yaml \
        --data_root D:/data/dental \
        --train_txt D:/data/dental/train.txt \
        --val_txt D:/data/dental/val.txt
"""

import argparse
import sys
from pathlib import Path

# Add script directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11-Dental 4-channel lesion detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model configs available:
  yolo11-dental.yaml      - 4ch input, P3-P5 heads (default)
  yolo11-dental-p2.yaml   - 4ch input, P2-P5 heads (better for small lesions)
  yolo11-dental-eca.yaml  - 4ch input, P3-P5 + ECA attention
  yolo11-dental-full.yaml - 4ch input, P2-P5 + ECA (full model)

Data structure required:
  data_root/
  ├── images/          # Original dental X-rays (.png/.jpg)
  ├── stage1_masks/    # Stage1 segmentation masks (class indices 0-4)
  └── labels/          # YOLO format labels (.txt)
        """
    )

    # Data arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing images/, stage1_masks/, labels/")
    parser.add_argument("--train_txt", type=str, required=True,
                        help="Path to txt file listing training image names (one per line)")
    parser.add_argument("--val_txt", type=str, required=True,
                        help="Path to txt file listing validation image names (one per line)")

    # Model arguments
    parser.add_argument("--model", type=str, default="yolo11-dental.yaml",
                        help="Model config YAML file (default: yolo11-dental.yaml)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Pretrained weights path (optional)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (default: 0, use 'cpu' for CPU)")

    # Optimizer arguments
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["SGD", "Adam", "AdamW"],
                        help="Optimizer (default: AdamW)")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate ratio (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="Weight decay (default: 0.0005)")

    # Training settings
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (default: 50)")
    parser.add_argument("--close_mosaic", type=int, default=10,
                        help="Disable mosaic for last N epochs (default: 10)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")

    # Loss settings (small object optimized)
    parser.add_argument("--loss_type", type=str, default="wiou_nwd",
                        choices=["ciou", "wiou", "wiou_nwd", "inner_wiou_nwd"],
                        help="Box loss type: ciou, wiou, wiou_nwd (default), inner_wiou_nwd")
    parser.add_argument("--small_threshold", type=int, default=32,
                        help="Size threshold (pixels) for small object weighting")
    parser.add_argument("--small_weight", type=float, default=2.0,
                        help="Extra weight multiplier for small objects")

    # Output arguments
    parser.add_argument("--project", type=str, default="runs/dental",
                        help="Project directory (default: runs/dental)")
    parser.add_argument("--name", type=str, default="train",
                        help="Experiment name (default: train)")
    parser.add_argument("--exist_ok", action="store_true",
                        help="Allow overwriting existing experiment")

    # Class names
    parser.add_argument("--nc", type=int, default=3,
                        help="Number of classes (default: 3)")
    parser.add_argument("--names", type=str, nargs="+",
                        default=["periapical", "periodontal", "combined"],
                        help="Class names")

    return parser.parse_args()


def main():
    args = parse_args()

    # Import here to avoid slow startup for --help
    import yaml
    from ultralytics.engine.dental_trainer import DentalTrainer

    # Resolve model path
    model_path = args.model
    if not Path(model_path).exists():
        # Try to find in default config directory
        default_path = SCRIPT_DIR / "ultralytics" / "cfg" / "models" / "11" / args.model
        if default_path.exists():
            model_path = str(default_path)
        else:
            print(f"ERROR: Model config not found: {args.model}")
            print(f"Searched: {args.model}, {default_path}")
            sys.exit(1)

    # Create output directory
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data config
    data_config = {
        "path": args.data_root,
        "train": args.train_txt,
        "val": args.val_txt,
        "nc": args.nc,
        "names": {i: name for i, name in enumerate(args.names)},
    }

    # Save data config
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print("=" * 60)
    print("  YOLO11-Dental Training")
    print("=" * 60)
    print(f"  Model:      {model_path}")
    print(f"  Data root:  {args.data_root}")
    print(f"  Train:      {args.train_txt}")
    print(f"  Val:        {args.val_txt}")
    print(f"  Output:     {output_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device:     {args.device}")
    print(f"  Classes:    {args.names}")
    print(f"  Loss:       {args.loss_type} (small_th={args.small_threshold}, weight={args.small_weight})")
    print("=" * 60)

    # Build trainer overrides
    overrides = {
        "model": model_path,
        "data": str(data_yaml_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "close_mosaic": args.close_mosaic,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "cos_lr": True,
        "save_period": 10,
        "val": True,
        "plots": True,
        "verbose": True,
        # Small object optimized loss
        "dental_loss": args.loss_type,
        "small_threshold": args.small_threshold,
        "small_weight": args.small_weight,
    }

    # Add resume/weights if specified
    if args.resume:
        overrides["resume"] = args.resume
    if args.weights:
        overrides["weights"] = args.weights

    # Create and run trainer
    trainer = DentalTrainer(overrides=overrides)
    results = trainer.train()

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best weights: {output_dir}/weights/best.pt")
    print(f"  Last weights: {output_dir}/weights/last.pt")
    print(f"  Results:      {output_dir}/results.csv")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
