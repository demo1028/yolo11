import os
import json
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 模型路径
MODEL_PATH = '/home/user/Han/dangpeipei/ultralytics-8.3.6/runs/train/fold1/weights/best.pt'

# 2. 数据配置
DATA_YAML = '/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_detection_kfold/dataset_fold_3.yaml'

# 3. 输出目录
OUTPUT_DIR = 'runs/detect_result'

# 4. 模式选择
TASK_MODE = 'val_and_predict'

# 5. 评估使用的数据集划分 ('val' 或 'test')
EVAL_SPLIT = 'val'

# 6. 置信度和 NMS 阈值
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# 7. 对比图中显示差距最大的 Top-N 张图
TOP_N_BAD = 20
# ===========================================

# 类别颜色配置 (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),    # periapical - 绿色
    1: (255, 165, 0),  # periodontal - 橙色
    2: (255, 0, 0),    # endo-periodontal - 红色
}


def get_image_paths_from_yaml(yaml_path, split='val'):
    """从 yaml -> txt 读取所有图片绝对路径"""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root = cfg.get('path', '')
    txt_rel = cfg.get(split, '')

    if not txt_rel:
        return []

    txt_path = os.path.join(root, txt_rel)
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]

    return paths


def load_gt_labels(img_path):
    """
    根据图片路径推断标签路径并加载 GT 标签

    YOLO 标签路径规则: images/xxx.jpg -> labels/xxx.txt
    """
    img_path = Path(img_path)
    # 尝试 images -> labels 替换
    label_path = Path(str(img_path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
    label_path = label_path.with_suffix('.txt')

    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                boxes.append((cls_id, x_center, y_center, w, h))

    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    """YOLO 归一化格式 -> 像素坐标 (x1, y1, x2, y2)"""
    cls_id, xc, yc, w, h = box
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return cls_id, max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def compute_iou(box1, box2):
    """计算两个 (x1, y1, x2, y2) 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def compute_image_score(gt_boxes, pred_boxes, img_w, img_h, iou_thresh=0.5):
    """
    计算单张图片的匹配得分

    得分越低 = 差距越大:
    - 漏检 (FN): 每个未匹配的 GT 扣分
    - 误检 (FP): 每个未匹配的 Pred 扣分
    - IoU 低: 匹配但 IoU 低扣分

    Returns:
        score: 0~1, 越高越好
        details: dict
    """
    gt_xyxy = [yolo_to_xyxy(b, img_w, img_h) for b in gt_boxes]
    pred_xyxy = pred_boxes  # 已经是 (cls, x1, y1, x2, y2)

    matched_gt = set()
    matched_pred = set()
    total_iou = 0.0
    matches = 0

    # 贪心匹配
    for pi, pred in enumerate(pred_xyxy):
        best_iou = 0
        best_gi = -1
        for gi, gt in enumerate(gt_xyxy):
            if gi in matched_gt:
                continue
            if pred[0] != gt[0]:  # 类别必须匹配
                continue
            iou = compute_iou(pred[1:], gt[1:])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            total_iou += best_iou
            matches += 1

    n_gt = len(gt_xyxy)
    n_pred = len(pred_xyxy)
    fn = n_gt - matches  # 漏检
    fp = n_pred - matches  # 误检
    avg_iou = total_iou / matches if matches > 0 else 0

    # 综合得分
    if n_gt == 0 and n_pred == 0:
        score = 1.0  # 都没有框，完美匹配
    elif n_gt == 0:
        score = 0.0  # 没有 GT 但有预测 = 全部误检
    else:
        precision = matches / (n_pred + 1e-6)
        recall = matches / n_gt
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        score = f1 * avg_iou if matches > 0 else 0.0

    return score, {
        'n_gt': n_gt, 'n_pred': n_pred,
        'matches': matches, 'fn': fn, 'fp': fp,
        'avg_iou': avg_iou, 'score': score,
    }


def draw_comparison(img, gt_boxes, pred_boxes, names, img_w, img_h):
    """
    在图片上同时画出 GT (实线) 和 Pred (虚线) 框

    GT: 实线框 + 标签带 [GT]
    Pred: 虚线框 + 标签带 [Pred] + 置信度
    """
    canvas = img.copy()

    # 画 GT 框 (实线, 粗)
    for box in gt_boxes:
        cls_id, xc, yc, w, h = box
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"[GT] {names.get(cls_id, cls_id)}"
        cv2.putText(canvas, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 画 Pred 框 (虚线效果, 细)
    for pred in pred_boxes:
        cls_id, x1, y1, x2, y2 = pred[:5]
        conf = pred[5] if len(pred) > 5 else 0
        color = CLASS_COLORS.get(cls_id, (0, 0, 255))
        # 用间断线模拟虚线
        for i in range(x1, x2, 8):
            cv2.line(canvas, (i, y1), (min(i + 4, x2), y1), color, 1)
            cv2.line(canvas, (i, y2), (min(i + 4, x2), y2), color, 1)
        for i in range(y1, y2, 8):
            cv2.line(canvas, (x1, i), (x1, min(i + 4, y2)), color, 1)
            cv2.line(canvas, (x2, i), (x2, min(i + 4, y2)), color, 1)
        label = f"[Pred] {names.get(cls_id, cls_id)} {conf:.2f}"
        cv2.putText(canvas, label, (x1, max(y2 + 15, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return canvas


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {MODEL_PATH} on {device}...")

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = model.names
    num_classes = len(names)

    # ====================================================
    # 阶段 1: 计算指标 (Precision, Recall, F1, mAP)
    # ====================================================
    if TASK_MODE == 'val_and_predict':
        print("\n" + "=" * 60)
        print(f"Stage 1: Calculating Metrics on '{EVAL_SPLIT}' split...")
        print("=" * 60)

        metrics = model.val(
            data=DATA_YAML,
            split=EVAL_SPLIT,
            project=OUTPUT_DIR,
            name='metrics',
            plots=True,
            save_json=False,
            exist_ok=True,
        )

        print("\n" + "=" * 70)
        print(f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
        print("-" * 70)

        ap50_per_class = metrics.box.ap50

        try:
            p_per_class = metrics.box.p
            r_per_class = metrics.box.r
        except Exception:
            p_per_class = [metrics.box.mp] * num_classes
            r_per_class = [metrics.box.mr] * num_classes

        class_f1_list = []
        for i in range(num_classes):
            p = p_per_class[i] if i < len(p_per_class) else 0
            r = r_per_class[i] if i < len(r_per_class) else 0
            f1 = 2 * p * r / (p + r + 1e-6)
            ap50 = ap50_per_class[i] if i < len(ap50_per_class) else 0
            map_i = metrics.box.maps[i] if i < len(metrics.box.maps) else 0
            class_f1_list.append(f1)
            name = names.get(i, f'class_{i}')
            print(f"{name:<25} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {ap50:<10.4f} | {map_i:<10.4f}")

        print("-" * 70)
        mean_f1 = sum(class_f1_list) / len(class_f1_list) if class_f1_list else 0
        print(f"{'ALL (Mean)':<25} | {metrics.box.mp:<10.4f} | {metrics.box.mr:<10.4f} | {mean_f1:<10.4f} | {metrics.box.map50:<10.4f} | {metrics.box.map:<10.4f}")
        print("=" * 70)

        save_txt_path = os.path.join(OUTPUT_DIR, 'final_metrics.txt')
        with open(save_txt_path, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"Model: {MODEL_PATH}\n")
            f.write(f"Data:  {DATA_YAML}\n")
            f.write(f"Split: {EVAL_SPLIT}\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Class':<25} | {'P':<10} | {'R':<10} | {'F1':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}\n")
            f.write("-" * 70 + "\n")
            for i in range(num_classes):
                p = p_per_class[i] if i < len(p_per_class) else 0
                r = r_per_class[i] if i < len(r_per_class) else 0
                f1 = class_f1_list[i]
                ap50 = ap50_per_class[i] if i < len(ap50_per_class) else 0
                map_i = metrics.box.maps[i] if i < len(metrics.box.maps) else 0
                name = names.get(i, f'class_{i}')
                f.write(f"{name:<25} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {ap50:<10.4f} | {map_i:<10.4f}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'ALL (Mean)':<25} | {metrics.box.mp:<10.4f} | {metrics.box.mr:<10.4f} | {mean_f1:<10.4f} | {metrics.box.map50:<10.4f} | {metrics.box.map:<10.4f}\n")
            f.write("=" * 70 + "\n")

        print(f"\n指标已保存到: {save_txt_path}")

    # ====================================================
    # 阶段 2: GT vs Pred 对比图 + 差距分析
    # ====================================================
    print("\n" + "=" * 60)
    print("Stage 2: Generating GT vs Pred Comparison...")
    print("=" * 60)

    img_paths = get_image_paths_from_yaml(DATA_YAML, split=EVAL_SPLIT)

    if not img_paths:
        print("Warning: 无法获取图片路径，跳过对比分析。")
        return

    compare_dir = os.path.join(OUTPUT_DIR, 'gt_vs_pred')
    bad_cases_dir = os.path.join(OUTPUT_DIR, 'bad_cases')
    os.makedirs(compare_dir, exist_ok=True)
    os.makedirs(bad_cases_dir, exist_ok=True)

    image_scores = []

    for img_path in tqdm(img_paths, desc="生成对比图"):
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        base_name = Path(img_path).stem

        # 加载 GT
        gt_boxes = load_gt_labels(img_path)

        # 模型预测
        results = model.predict(
            source=img_path,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            verbose=False,
        )

        # 解析预测结果
        pred_boxes = []
        if results and len(results) > 0:
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                pred_boxes.append((cls_id, x1, y1, x2, y2, conf))

        # 计算匹配得分
        score, details = compute_image_score(gt_boxes, pred_boxes, img_w, img_h)
        image_scores.append({
            'img_path': img_path,
            'base_name': base_name,
            'score': score,
            'details': details,
        })

        # 画对比图
        canvas = draw_comparison(img, gt_boxes, pred_boxes, names, img_w, img_h)

        # 在图片左上角标注得分
        info_text = f"Score: {score:.3f} | GT:{details['n_gt']} Pred:{details['n_pred']} Match:{details['matches']} FN:{details['fn']} FP:{details['fp']}"
        cv2.putText(canvas, info_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 保存对比图
        save_path = os.path.join(compare_dir, f"{base_name}_compare.jpg")
        cv2.imwrite(save_path, canvas)

    # ====================================================
    # 阶段 3: 分析差距最大的图片
    # ====================================================
    print("\n" + "=" * 60)
    print(f"Stage 3: Analyzing Top-{TOP_N_BAD} Worst Cases...")
    print("=" * 60)

    # 只对有 GT 的图片排序（无 GT 无 Pred 的不算 bad case）
    scored = [s for s in image_scores if s['details']['n_gt'] > 0 or s['details']['n_pred'] > 0]
    scored.sort(key=lambda x: x['score'])

    worst_cases = scored[:TOP_N_BAD]

    print(f"\n{'Rank':<5} | {'Score':<8} | {'GT':<4} | {'Pred':<5} | {'Match':<6} | {'FN':<4} | {'FP':<4} | {'AvgIoU':<8} | {'Image'}")
    print("-" * 90)

    for rank, case in enumerate(worst_cases, 1):
        d = case['details']
        print(f"{rank:<5} | {case['score']:<8.3f} | {d['n_gt']:<4} | {d['n_pred']:<5} | {d['matches']:<6} | {d['fn']:<4} | {d['fp']:<4} | {d['avg_iou']:<8.3f} | {case['base_name']}")

        # 复制 bad case 对比图到 bad_cases 目录
        src = os.path.join(compare_dir, f"{case['base_name']}_compare.jpg")
        dst = os.path.join(bad_cases_dir, f"rank{rank:02d}_{case['base_name']}_compare.jpg")
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)

    # 保存分析报告
    report_path = os.path.join(OUTPUT_DIR, 'bad_cases_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Bad Cases Analysis Report\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data:  {DATA_YAML}\n")
        f.write(f"Total images: {len(image_scores)}\n")
        f.write(f"Images with GT or Pred: {len(scored)}\n")
        avg_score = sum(s['score'] for s in scored) / len(scored) if scored else 0
        f.write(f"Average score: {avg_score:.4f}\n\n")

        f.write("=" * 90 + "\n")
        f.write(f"{'Rank':<5} | {'Score':<8} | {'GT':<4} | {'Pred':<5} | {'Match':<6} | {'FN':<4} | {'FP':<4} | {'AvgIoU':<8} | {'Image'}\n")
        f.write("-" * 90 + "\n")
        for rank, case in enumerate(worst_cases, 1):
            d = case['details']
            f.write(f"{rank:<5} | {case['score']:<8.3f} | {d['n_gt']:<4} | {d['n_pred']:<5} | {d['matches']:<6} | {d['fn']:<4} | {d['fp']:<4} | {d['avg_iou']:<8.3f} | {case['base_name']}\n")

        # 错误类型统计
        fn_total = sum(s['details']['fn'] for s in scored)
        fp_total = sum(s['details']['fp'] for s in scored)
        f.write(f"\n总漏检数 (FN): {fn_total}\n")
        f.write(f"总误检数 (FP): {fp_total}\n")

    # ====================================================
    # 完成
    # ====================================================
    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)
    print(f"1. 指标文件:       {os.path.join(OUTPUT_DIR, 'final_metrics.txt')}")
    print(f"2. 对比图目录:     {compare_dir} ({len(image_scores)} 张)")
    print(f"3. Bad Cases 目录: {bad_cases_dir} (Top-{TOP_N_BAD})")
    print(f"4. 分析报告:       {report_path}")


if __name__ == "__main__":
    main()
