#!/usr/bin/env python3
"""
Analyze training results and generate summary report.

Usage:
    python analyze_results.py --exp_dir runs/dental/train
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze training results")
    parser.add_argument("--exp_dir", type=str, default="runs/dental/train",
                        help="Experiment directory containing results.csv")
    parser.add_argument("--save", action="store_true",
                        help="Save analysis plots")
    return parser.parse_args()


def load_results(exp_dir):
    """Load results.csv and return DataFrame."""
    results_path = Path(exp_dir) / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)
    # Clean column names (remove leading spaces)
    df.columns = df.columns.str.strip()
    return df


def print_summary(df, exp_dir):
    """Print training summary."""
    print("\n" + "=" * 70)
    print(f"  Training Summary: {exp_dir}")
    print("=" * 70)

    # Find best epoch
    if "metrics/mAP50(B)" in df.columns:
        map50_col = "metrics/mAP50(B)"
        map_col = "metrics/mAP50-95(B)"
        prec_col = "metrics/precision(B)"
        rec_col = "metrics/recall(B)"
    else:
        # Try alternative column names
        map50_col = [c for c in df.columns if "mAP50" in c and "95" not in c][0]
        map_col = [c for c in df.columns if "mAP50-95" in c][0]
        prec_col = [c for c in df.columns if "precision" in c][0]
        rec_col = [c for c in df.columns if "recall" in c][0]

    best_epoch = df[map50_col].idxmax() + 1
    best_row = df.loc[df[map50_col].idxmax()]

    print(f"\n  Total Epochs:    {len(df)}")
    print(f"  Best Epoch:      {best_epoch}")
    print(f"\n  Best Metrics:")
    print(f"    mAP@50:        {best_row[map50_col]:.4f}")
    print(f"    mAP@50-95:     {best_row[map_col]:.4f}")
    print(f"    Precision:     {best_row[prec_col]:.4f}")
    print(f"    Recall:        {best_row[rec_col]:.4f}")

    # Final epoch metrics
    final_row = df.iloc[-1]
    print(f"\n  Final Epoch ({len(df)}) Metrics:")
    print(f"    mAP@50:        {final_row[map50_col]:.4f}")
    print(f"    mAP@50-95:     {final_row[map_col]:.4f}")
    print(f"    Precision:     {final_row[prec_col]:.4f}")
    print(f"    Recall:        {final_row[rec_col]:.4f}")

    # Loss values
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    if loss_cols:
        print(f"\n  Final Losses:")
        for col in loss_cols:
            print(f"    {col.split('/')[-1]:15s}: {final_row[col]:.4f}")

    # Check for overfitting
    if len(df) > 20:
        recent_map = df[map50_col].iloc[-10:].mean()
        peak_map = df[map50_col].max()
        if recent_map < peak_map * 0.95:
            print(f"\n  [!] Warning: Possible overfitting detected")
            print(f"      Peak mAP@50: {peak_map:.4f} at epoch {best_epoch}")
            print(f"      Recent avg:  {recent_map:.4f}")

    print("\n" + "=" * 70)


def plot_training_curves(df, exp_dir, save=False):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Detect column names
    map50_col = [c for c in df.columns if "mAP50" in c and "95" not in c]
    map_col = [c for c in df.columns if "mAP50-95" in c]
    prec_col = [c for c in df.columns if "precision" in c]
    rec_col = [c for c in df.columns if "recall" in c]
    box_loss = [c for c in df.columns if "box" in c.lower() and "loss" in c.lower()]
    cls_loss = [c for c in df.columns if "cls" in c.lower() and "loss" in c.lower()]

    epochs = range(1, len(df) + 1)

    # Plot 1: mAP curves
    ax1 = axes[0, 0]
    if map50_col:
        ax1.plot(epochs, df[map50_col[0]], 'b-', label='mAP@50', linewidth=2)
    if map_col:
        ax1.plot(epochs, df[map_col[0]], 'g-', label='mAP@50-95', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP')
    ax1.set_title('Mean Average Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Precision & Recall
    ax2 = axes[0, 1]
    if prec_col:
        ax2.plot(epochs, df[prec_col[0]], 'b-', label='Precision', linewidth=2)
    if rec_col:
        ax2.plot(epochs, df[rec_col[0]], 'r-', label='Recall', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.set_title('Precision & Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box Loss
    ax3 = axes[1, 0]
    train_box = [c for c in df.columns if "train" in c.lower() and "box" in c.lower()]
    val_box = [c for c in df.columns if "val" in c.lower() and "box" in c.lower()]
    if train_box:
        ax3.plot(epochs, df[train_box[0]], 'b-', label='Train', linewidth=2)
    if val_box:
        ax3.plot(epochs, df[val_box[0]], 'r-', label='Val', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Box Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Classification Loss
    ax4 = axes[1, 1]
    train_cls = [c for c in df.columns if "train" in c.lower() and "cls" in c.lower()]
    val_cls = [c for c in df.columns if "val" in c.lower() and "cls" in c.lower()]
    if train_cls:
        ax4.plot(epochs, df[train_cls[0]], 'b-', label='Train', linewidth=2)
    if val_cls:
        ax4.plot(epochs, df[val_cls[0]], 'r-', label='Val', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Classification Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        save_path = Path(exp_dir) / "training_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {save_path}")

    plt.show()


def main():
    args = parse_args()

    try:
        df = load_results(args.exp_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print_summary(df, args.exp_dir)
    plot_training_curves(df, args.exp_dir, save=args.save)


if __name__ == "__main__":
    main()
