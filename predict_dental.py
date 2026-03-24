#!/usr/bin/env python3
"""
Inference script for YOLO11-Dental model.

Usage:
    # Single image
    python predict_dental.py --weights best.pt --image test.png --mask test_mask.png

    # Directory of images
    python predict_dental.py --weights best.pt --data_root /path/to/data --image_list test.txt
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with YOLO11-Dental")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained weights")

    # Single image mode
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image")
    parser.add_argument("--mask", type=str, default=None,
                        help="Path to Stage1 mask for single image")

    # Batch mode
    parser.add_argument("--data_root", type=str, default=None,
                        help="Data root for batch inference")
    parser.add_argument("--image_list", type=str, default=None,
                        help="Text file listing images to process")

    # Inference settings
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="0")

    # Output
    parser.add_argument("--save_dir", type=str, default="runs/dental/predict",
                        help="Save directory")
    parser.add_argument("--save_txt", action="store_true",
                        help="Save results as txt")
    parser.add_argument("--show", action="store_true",
                        help="Show results")

    return parser.parse_args()


def load_and_preprocess(image_path, mask_path, imgsz):
    """Load image and mask, create 4-channel tensor."""
    from ultralytics.data.dental_preprocess import DentalStage1Processor

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {mask_path}")

    # Create 4-channel tensor
    processor = DentalStage1Processor()
    tensor_4ch = processor(image, mask)  # (H, W, 4), float32, [0, 1]

    # Resize to target size
    h, w = tensor_4ch.shape[:2]
    if h != imgsz or w != imgsz:
        result = np.zeros((imgsz, imgsz, 4), dtype=np.float32)
        for c in range(4):
            result[:, :, c] = cv2.resize(tensor_4ch[:, :, c], (imgsz, imgsz))
        tensor_4ch = result

    # Convert to torch tensor (B, C, H, W)
    tensor = torch.from_numpy(tensor_4ch).permute(2, 0, 1).unsqueeze(0)

    return tensor, image


def draw_boxes(image, boxes, scores, classes, class_names):
    """Draw detection boxes on image."""
    colors = {
        0: (0, 0, 255),    # periapical: red
        1: (0, 255, 0),    # periodontal: green
        2: (255, 0, 0),    # combined: blue
    }

    h, w = image.shape[:2]
    result = image.copy()

    for box, score, cls in zip(boxes, scores, classes):
        cls = int(cls)
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(cls, (255, 255, 255))

        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_names[cls]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def main():
    args = parse_args()

    from ultralytics import YOLO
    from ultralytics.nn.modules.dental_attention import register_dental_modules

    # Register custom modules
    register_dental_modules()

    # Load model
    print(f"Loading model from {args.weights}")
    model = YOLO(args.weights)
    model.to(args.device if args.device != "cpu" else "cpu")

    class_names = ["periapical", "periodontal", "combined"]

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Single image mode
    if args.image:
        if not args.mask:
            print("Error: --mask required for single image mode")
            return

        print(f"\nProcessing: {args.image}")
        tensor, orig_image = load_and_preprocess(args.image, args.mask, args.imgsz)
        tensor = tensor.to(model.device)

        # Run inference
        with torch.no_grad():
            results = model.model(tensor)

        # Process results
        if hasattr(results, '__len__') and len(results) > 0:
            pred = results[0] if isinstance(results, (list, tuple)) else results

            # Get boxes (assuming standard YOLO output format)
            if hasattr(pred, 'boxes'):
                boxes = pred.boxes.xyxy.cpu().numpy()
                scores = pred.boxes.conf.cpu().numpy()
                classes = pred.boxes.cls.cpu().numpy()
            else:
                print("No detections")
                boxes, scores, classes = [], [], []
        else:
            boxes, scores, classes = [], [], []

        # Filter by confidence
        if len(boxes) > 0:
            mask = scores >= args.conf
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

        # Draw results
        result_img = draw_boxes(orig_image, boxes, scores, classes, class_names)

        # Save
        save_path = save_dir / f"{Path(args.image).stem}_pred.jpg"
        cv2.imwrite(str(save_path), result_img)
        print(f"Saved: {save_path}")

        # Print detections
        print(f"\nDetections ({len(boxes)}):")
        for box, score, cls in zip(boxes, scores, classes):
            print(f"  {class_names[int(cls)]}: {score:.3f} at {box}")

        if args.show:
            cv2.imshow("Prediction", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Batch mode
    elif args.data_root and args.image_list:
        data_root = Path(args.data_root)

        with open(args.image_list, 'r') as f:
            image_names = [line.strip() for line in f if line.strip()]

        print(f"\nProcessing {len(image_names)} images...")

        for img_name in image_names:
            img_path = data_root / "images" / img_name
            mask_path = data_root / "stage1_masks" / img_name

            if not img_path.exists() or not mask_path.exists():
                print(f"  Skip (missing): {img_name}")
                continue

            try:
                tensor, orig_image = load_and_preprocess(img_path, mask_path, args.imgsz)
                tensor = tensor.to(model.device)

                with torch.no_grad():
                    results = model(tensor)

                # Save result
                result_img = results[0].plot() if hasattr(results[0], 'plot') else orig_image
                save_path = save_dir / f"{Path(img_name).stem}_pred.jpg"
                cv2.imwrite(str(save_path), result_img)
                print(f"  Saved: {save_path}")

            except Exception as e:
                print(f"  Error processing {img_name}: {e}")

        print(f"\nResults saved to: {save_dir}")

    else:
        print("Error: Specify --image and --mask, or --data_root and --image_list")


if __name__ == "__main__":
    main()
