#!/usr/bin/env python3
"""
Evaluation script for trained YOLO11-Dental model.

Usage:
    python evaluate_dental.py --weights runs/dental/train/weights/best.pt \
        --data_root /path/to/data --val_txt val.txt
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11-Dental model")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights (best.pt)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Data root directory")
    parser.add_argument("--val_txt", type=str, required=True,
                        help="Validation txt file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for evaluation")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU threshold for NMS")
    parser.add_argument("--save_dir", type=str, default="runs/dental/eval",
                        help="Save directory for results")
    return parser.parse_args()


def main():
    args = parse_args()

    import yaml
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.dental_attention import register_dental_modules

    # Register custom modules
    register_dental_modules()

    # Load model
    print(f"Loading model from {args.weights}")
    model = YOLO(args.weights)

    # Create data config
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_config = {
        "path": args.data_root,
        "val": args.val_txt,
        "nc": 3,
        "names": {0: "periapical", 1: "periodontal", 2: "combined"},
    }
    data_yaml = save_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(data_config, f)

    # Run validation
    print("\n" + "=" * 60)
    print("  Running Evaluation")
    print("=" * 60)

    results = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        plots=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)

    # Overall metrics
    print(f"\n  Overall Metrics:")
    print(f"    mAP@50:      {results.box.map50:.4f}")
    print(f"    mAP@50-95:   {results.box.map:.4f}")
    print(f"    Precision:   {results.box.mp:.4f}")
    print(f"    Recall:      {results.box.mr:.4f}")

    # Per-class metrics
    class_names = ["periapical", "periodontal", "combined"]
    print(f"\n  Per-Class AP@50:")
    for i, name in enumerate(class_names):
        if i < len(results.box.ap50):
            print(f"    {name:12s}: {results.box.ap50[i]:.4f}")

    print("\n" + "=" * 60)
    print(f"  Results saved to: {save_dir}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
