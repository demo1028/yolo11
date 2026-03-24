#!/usr/bin/env python3
"""
Ablation Study Runner for YOLO11-Dental Stage2 Lesion Detection.

Usage:
    # Run single experiment
    python run_ablation.py --exp 0 --data_root /path/to/data

    # Run all experiments sequentially
    python run_ablation.py --exp all --data_root /path/to/data

    # Run specific experiments
    python run_ablation.py --exp 0 1 3 5 --data_root /path/to/data

Experiments:
    --- Track A: Early Fusion (4ch直接拼接) ---
    Exp0: Baseline        - Original YOLO11, 3-ch RGB, P3-P5
    Exp1: +4ch Prior      - 4-ch input (raw+virtual+distance+metal), P3-P5
    Exp2: +P2 Head        - 4-ch input, P2-P5 detection
    Exp3: +ECA Attention  - 4-ch input, C3k2_Dental (ECA), P3-P5
    Exp4: Full Model      - 4-ch input, P2-P5, C3k2_Dental (all improvements combined)

    --- Track B: Prior-as-Attention (Prior调制RGB) ---
    Exp5: Prior-as-Attn       - Prior as spatial attention, P3-P5
    Exp6: +P2 Head (Attn)     - Prior-as-Attn + P2-P5 detection
    Exp7: +ECA (Attn)         - Prior-as-Attn + C3k2_Dental (ECA), P3-P5
    Exp8: Full Model (Attn)   - Prior-as-Attn + P2-P5 + C3k2_Dental
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics.nn.modules.dental_attention import register_dental_modules


# ============================================================================
#  Experiment Definitions
# ============================================================================

EXPERIMENTS = {
    0: {
        "name": "exp0_baseline",
        "desc": "Baseline: Original YOLO11, 3-ch RGB, P3-P5",
        "yaml": "yolo11-baseline3cls.yaml",
        "input_channels": 3,
        "use_4ch_preprocessing": False,
        "changes": "None (original YOLO11)",
    },
    1: {
        "name": "exp1_4ch_prior",
        "desc": "+4ch Prior Input (raw + virtual + distance + metal)",
        "yaml": "yolo11-dental.yaml",
        "input_channels": 4,
        "use_4ch_preprocessing": True,
        "changes": "Input channels: 3 → 4 (ch: 4 in yaml)",
    },
    2: {
        "name": "exp2_p2_head",
        "desc": "+P2 Detection Head (4x downsample layer)",
        "yaml": "yolo11-dental-p2.yaml",
        "input_channels": 4,
        "use_4ch_preprocessing": True,
        "changes": "Added P2 feature extraction + P2 detection head, Detect: P3-P5 → P2-P5",
    },
    3: {
        "name": "exp3_eca_attention",
        "desc": "+C3k2_Dental (ECA Attention)",
        "yaml": "yolo11-dental-eca.yaml",
        "input_channels": 4,
        "use_4ch_preprocessing": True,
        "changes": "Replaced all C3k2 with C3k2_Dental (C3k2 + EfficientChannelAttention)",
    },
    4: {
        "name": "exp4_full_model",
        "desc": "Full Model: 4ch + P2 + ECA",
        "yaml": "yolo11-dental-full.yaml",
        "input_channels": 4,
        "use_4ch_preprocessing": True,
        "changes": "All structural improvements combined (4ch input + P2 head + ECA attention)",
    },
    5: {
        "name": "exp5_prior_attn",
        "desc": "Prior-as-Attention: Prior guides RGB instead of direct concat",
        "yaml": "yolo11-dental-attn.yaml",
        "input_channels": 4,
        "use_4ch_preprocessing": True,
        "changes": "Prior channels converted to spatial attention, applied to RGB before backbone",
    },
    6: {
        "name": "exp6_attn_p2",
        "desc": "Prior-as-Attention + P2 Detection Head",
        "yaml": "yolo11-dental-attn-p2.yaml",
        "input_channels": 4,
        "use_4ch_preprocessing": True,
        "changes": "Prior-as-Attention + P2 feature block + 4-scale detection (P2-P5)",
    },
    7: {
        "name": "exp7_attn_eca",
        "desc": "Prior-as-Attention + C3k2_Dental (ECA)",
        "yaml": "yolo11-dental-attn-eca.yaml",
        "input_channels": 4,
        "use_4ch_preprocessing": True,
        "changes": "Prior-as-Attention + all C3k2 replaced with C3k2_Dental (ECA)",
    },
    8: {
        "name": "exp8_attn_full",
        "desc": "Prior-as-Attention Full: PriorAsAttn + P2 + ECA",
        "yaml": "yolo11-dental-attn-full.yaml",
        "input_channels": 4,
        "use_4ch_preprocessing": True,
        "changes": "Prior-as-Attention + P2 head + C3k2_Dental (all improvements on Attn track)",
    },
}

# Two ablation tracks:
#
# Track A (Early Fusion): Exp0 → Exp1(+4ch) → Exp2(+P2) / Exp3(+ECA) → Exp4(Full)
#   - 4ch channels directly concatenated and convolved together
#   - Simple, but Prior and RGB share the same low-level processing
#
# Track B (Prior-as-Attention): Exp0 → Exp5(+Attn) → Exp6(+P2) / Exp7(+ECA) → Exp8(Full)
#   - Prior converted to spatial attention, modulates RGB features
#   - Preserves semantic level separation between Prior and RGB
#
# Decision: Run Track A first. If Exp1 ≈ Exp0, switch to Track B.
#
# Note: AnatomyConstrainedAugmentation (lesion copy-paste) is DISABLED.
# Reason: Periapical lesions have strict positional correspondence with specific
# tooth apices. Random pasting would break this diagnostic spatial relationship.


# ============================================================================
#  Training Functions
# ============================================================================

def train_baseline_3ch(exp_cfg, args):
    """
    Exp0: Train standard YOLO11 with 3-channel RGB input.
    Uses the native ultralytics training API directly.
    """
    from ultralytics import YOLO

    yaml_path = SCRIPT_DIR / "ultralytics" / "cfg" / "models" / "11" / exp_cfg["yaml"]
    output_dir = Path(args.output_dir) / exp_cfg["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(yaml_path))

    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        workers=args.workers,
        patience=args.patience,
        optimizer="AdamW",
        lr0=args.lr,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        save_period=10,
        val=True,
        plots=True,
    )

    return results


def train_4ch_dental(exp_cfg, args):
    """
    Exp1-4: Train with 4-channel dental input using DentalTrainer.

    Uses the same training infrastructure as Exp0 (DetectionTrainer) including:
    - EMA (Exponential Moving Average)
    - Warmup scheduler
    - AMP (Automatic Mixed Precision)
    - Cosine LR schedule
    - Close-mosaic for final epochs
    - Smart logging, checkpointing, early stopping

    This ensures fair comparison between baseline and 4-channel experiments.
    """
    from ultralytics.engine.dental_trainer import DentalTrainer

    yaml_path = SCRIPT_DIR / "ultralytics" / "cfg" / "models" / "11" / exp_cfg["yaml"]
    output_dir = Path(args.output_dir) / exp_cfg["name"]

    # Build data config dict for DentalTrainer
    # DentalTrainer reads split files from data["train"] and data["val"]
    data_config = {
        "path": args.data_root,
        "train": args.train_txt,
        "val": args.val_txt,
        "nc": 3,  # number of classes
        "names": {0: "periapical", 1: "periodontal", 2: "combined"},
    }

    # Save temporary data.yaml for DentalTrainer
    data_yaml_path = output_dir / "data.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f)

    print(f"DEBUG: exp_cfg type is {type(exp_cfg)}, value is {exp_cfg}")
    print(f"错误是 yaml_path:{yaml_path}, data_yaml_path: {data_yaml_path}")

    trainer = DentalTrainer(overrides={
        "model": str(yaml_path),
        "data": str(data_yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": args.device,
        "project": str(output_dir),
        "name": "train",
        "exist_ok": True,
        "workers": args.workers,
        "patience": args.patience,
        "optimizer": "AdamW",
        "lr0": args.lr,
        "cos_lr": True,
        "close_mosaic": args.close_mosaic,
        "save_period": 10,
        "val": True,
        "plots": True,
        # Small object optimized loss
        "dental_loss": args.loss_type,
        "small_threshold": args.small_threshold,
        "small_weight": args.small_weight,
    })

    results = trainer.train()
    return results


# ============================================================================
#  Main Runner
# ============================================================================

def run_experiment(exp_id, args):
    """Run a single ablation experiment."""
    exp_cfg = EXPERIMENTS[exp_id]

    print("=" * 70)
    print(f"  Exp{exp_id}: {exp_cfg['desc']}")
    print(f"  YAML:    {exp_cfg['yaml']}")
    print(f"  Changes: {exp_cfg['changes']}")
    print(f"  4ch Input: {exp_cfg['use_4ch_preprocessing']}")
    print("=" * 70)

    start = time.time()

    if exp_cfg["input_channels"] == 3:
        result = train_baseline_3ch(exp_cfg, args)
    else:
        result = train_4ch_dental(exp_cfg, args)

    elapsed = time.time() - start
    print(f"  Exp{exp_id} finished in {elapsed/3600:.1f}h\n")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11-Dental Ablation Study")

    parser.add_argument("--exp", nargs="+", default=["all"],
                        help="Experiment IDs: 0-8, 'all' (Track A: 0-4), 'trackA' (0-4), 'trackB' (0,5-8)")
    parser.add_argument("--data_root", type=str, default="/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_from_mask",
                        help="Root dir with images/, stage1_masks/, labels/")
    parser.add_argument("--train_txt", type=str,
                        default='/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_from_mask/fold_3_train.txt',
                        help="Path to txt file listing training images (one per line)")
    parser.add_argument("--val_txt", type=str,
                        default='/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_from_mask/fold_3_val.txt',
                        help="Path to txt file listing validation images (one per line)")
    parser.add_argument("--data_yaml", type=str, default="/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_detection_kfold/dataset_fold_3.yaml",
                        help="data.yaml for Exp0 baseline (standard YOLO format)")
    parser.add_argument("--output_dir", type=str, default="runs/ablation",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--close_mosaic", type=int, default=10,
                        help="Disable mosaic/mixup for the last N epochs")
    parser.add_argument("--scale", type=str, default="s", choices=["n","s","m","l","x"],
                        help="Model scale (all experiments use the same scale for fair comparison)")

    # Small object loss settings
    parser.add_argument("--loss_type", type=str, default="wiou_nwd",
                        choices=["ciou", "wiou", "wiou_nwd", "inner_wiou_nwd"],
                        help="Box loss type (default: wiou_nwd for small lesions)")
    parser.add_argument("--small_threshold", type=int, default=32,
                        help="Size threshold for small object weighting (pixels)")
    parser.add_argument("--small_weight", type=float, default=2.0,
                        help="Extra weight for small objects")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which experiments to run
    if "all" in args.exp or "trackA" in args.exp:
        exp_ids = [0, 1, 2, 3, 4]
    elif "trackB" in args.exp:
        exp_ids = [0, 5, 6, 7, 8]
    else:
        exp_ids = [int(e) for e in args.exp]

    # Exp0 requires data_yaml for standard YOLO training
    if 0 in exp_ids and args.data_yaml is None:
        print("WARNING: Exp0 (baseline) requires --data_yaml for standard YOLO format.")
        print("         Provide a data.yaml pointing to 3-channel images + YOLO labels.")
        print("         Skipping Exp0.\n")
        exp_ids = [e for e in exp_ids if e != 0]

    print(f"\n{'='*70}")
    print(f"  YOLO11-Dental Ablation Study")
    print(f"  Experiments: {exp_ids}")
    print(f"  Data root:   {args.data_root}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Model scale: {args.scale}")
    print(f"{'='*70}\n")

    results_summary = {}
    for exp_id in exp_ids:
        result = run_experiment(exp_id, args)
        results_summary[f"exp{exp_id}"] = {
            "name": EXPERIMENTS[exp_id]["name"],
            "desc": EXPERIMENTS[exp_id]["desc"],
        }

    # Save summary
    summary_path = Path(args.output_dir) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll experiments complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
