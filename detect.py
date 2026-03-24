import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 更新后的模型权重路径
MODEL_WEIGHTS = '/home/user/Han/dangpeipei/ultralytics-8.3.6/runs/ablation/exp0_baseline/train/weights/best.pt'

# 2. 测试文件列表路径
TEST_TXT_PATH = '/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_from_mask/fold_1_val.txt'

# 3. 标签根目录
LABEL_DIR_ROOT = '/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_from_mask/labels'

# 4. 结果保存设置
OUTPUT_DIR = 'unified_results_blue'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCORE_FILE = os.path.join(OUTPUT_DIR, 'score_ranking.txt')

# 5. 拼图设置 (3行3列)
GRID_ROWS = 3
GRID_COLS = 3
BATCH_SIZE = GRID_ROWS * GRID_COLS

# ================= 样式与颜色配置 =================

# 统一颜色：OpenCV中为 (Blue, Green, Red)
# 这里设为纯蓝色 (255, 0, 0)，让 GT 和 Pred 看起来完全一样
UNIFIED_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)  # 白色文字

# 统一线条风格
LINE_THICKNESS = 2
FONT_SCALE = 0.8


# ================= 工具函数 =================

def get_label_path_from_img_path(img_path):
    """根据图片路径推导txt标签路径"""
    path_obj = Path(img_path.strip())
    label_name = path_obj.stem + '.txt'
    parts = list(path_obj.parts)
    try:
        # 尝试替换 images -> labels
        idx = parts.index('images')
        parts[idx] = 'labels'
    except ValueError:
        # 找不到 images 目录则拼接根目录
        return os.path.join(LABEL_DIR_ROOT, label_name)

    parts[-1] = label_name
    return str(Path(*parts))


def draw_box_with_label(img, x1, y1, x2, y2, label_text, color):
    """
    核心绘图函数：确保 GT 和 Pred 使用完全相同的绘制逻辑
    """
    # 1. 画框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_THICKNESS)

    # 2. 计算文字大小
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)

    # 3. 确定标签位置 (如果框在顶部，标签往下画，否则往上画)
    y1_label = max(y1, h)

    # 4. 画文字背景条 (实心)
    cv2.rectangle(img, (x1, y1_label - h - 5), (x1 + w, y1_label), color, -1)

    # 5. 画文字
    cv2.putText(img, label_text, (x1, y1_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 1)


def draw_ground_truth(img, label_path, class_names):
    """读取标签并绘制 (使用统一颜色)"""
    img_copy = img.copy()
    img_h, img_w = img.shape[:2]

    # 顶部标题 (为了区分左右，标题颜色保持不同：左绿右红或统一白色)
    # 这里为了清晰，标题还是用绿色区分一下是“真值”，但框是统一蓝色的
    cv2.putText(img_copy, "Ground Truth", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not os.path.exists(label_path):
        return img_copy

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        if len(data) < 5: continue

        cls_id = int(data[0])
        x_c, y_c, w, h = map(float, data[1:5])

        # 反归一化
        x1 = int((x_c - w / 2) * img_w)
        y1 = int((y_c - h / 2) * img_h)
        x2 = int((x_c + w / 2) * img_w)
        y2 = int((y_c + h / 2) * img_h)

        # 获取类别名
        class_name = class_names[cls_id] if cls_id in class_names else str(cls_id)

        # 【关键】使用统一颜色绘制
        draw_box_with_label(img_copy, x1, y1, x2, y2, class_name, UNIFIED_COLOR)

    return img_copy


def draw_prediction(img, results, class_names):
    """解析预测结果并绘制 (使用统一颜色)"""
    img_copy = img.copy()

    # 顶部标题
    cv2.putText(img_copy, "Prediction", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = class_names[cls_id] if cls_id in class_names else str(cls_id)
        label_text = f"{class_name} {conf:.2f}"

        # 【关键】使用统一颜色绘制
        draw_box_with_label(img_copy, x1, y1, x2, y2, label_text, UNIFIED_COLOR)

    return img_copy


def create_grid_image(image_list, rows, cols):
    """将图片列表拼成大图"""
    if not image_list: return None

    # 统一尺寸 (以列表第一张图为准)
    h_ref, w_ref = image_list[0].shape[:2]
    resized_imgs = []

    for img in image_list:
        if img.shape[:2] != (h_ref, w_ref):
            img = cv2.resize(img, (w_ref, h_ref))
        resized_imgs.append(img)

    # 补齐空位 (如果不够9张)
    total_slots = rows * cols
    while len(resized_imgs) < total_slots:
        resized_imgs.append(np.zeros((h_ref, w_ref, 3), dtype=np.uint8))

    # 拼接
    row_imgs = []
    for r in range(rows):
        row_subset = resized_imgs[r * cols: (r + 1) * cols]
        row_imgs.append(np.hstack(row_subset))

    return np.vstack(row_imgs)


# ================= 主程序 =================

def main():
    print(f"正在加载模型: {MODEL_WEIGHTS} ...")
    model = YOLO(MODEL_WEIGHTS)
    class_names = model.names  # 获取类别字典

    with open(TEST_TXT_PATH, 'r') as f:
        img_paths = [line.strip() for line in f.readlines() if line.strip()]

    results_data = []  # 存储 (文件名, 得分)
    batch_buffer = []  # 拼图缓存
    batch_count = 1

    print(f"开始处理 {len(img_paths)} 张图片...")

    for img_path in tqdm(img_paths):
        if not os.path.exists(img_path): continue

        original_img = cv2.imread(img_path)
        if original_img is None: continue

        # --- 1. 左图 (Ground Truth) ---
        label_path = get_label_path_from_img_path(img_path)
        gt_img = draw_ground_truth(original_img, label_path, class_names)

        # --- 2. 右图 (Prediction) ---
        results = model(original_img, verbose=False)[0]
        pred_img = draw_prediction(original_img, results, class_names)

        # --- 3. 记录分数 ---
        if len(results.boxes) > 0:
            score = np.min(results.boxes.conf.cpu().numpy())
        else:
            score = 0.0
        results_data.append((os.path.basename(img_path), score))

        # --- 4. 左右拼接单张样本 ---
        # 确保高度一致
        if gt_img.shape[0] != pred_img.shape[0]:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

        pair_img = np.hstack((gt_img, pred_img))

        # --- 5. 添加顶部文件名条 ---
        pad_h = 40
        h, w = pair_img.shape[:2]
        padded_img = np.zeros((h + pad_h, w, 3), dtype=np.uint8)
        padded_img[pad_h:, :] = pair_img

        # 写入文件名
        cv2.putText(padded_img, f"File: {os.path.basename(img_path)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        batch_buffer.append(padded_img)

        # --- 6. 攒够9张，保存大图 ---
        if len(batch_buffer) == BATCH_SIZE:
            grid_img = create_grid_image(batch_buffer, GRID_ROWS, GRID_COLS)
            save_name = os.path.join(OUTPUT_DIR, f"batch_{batch_count:03d}.jpg")
            cv2.imwrite(save_name, grid_img)
            batch_buffer = []  # 清空
            batch_count += 1

    # --- 7. 处理剩余图片 ---
    if batch_buffer:
        grid_img = create_grid_image(batch_buffer, GRID_ROWS, GRID_COLS)
        save_name = os.path.join(OUTPUT_DIR, f"batch_{batch_count:03d}_last.jpg")
        cv2.imwrite(save_name, grid_img)

    # --- 8. 保存排序结果 ---
    results_data.sort(key=lambda x: x[1])
    with open(SCORE_FILE, 'w') as f:
        f.write(f"{'Image Name':<40} | {'Min Confidence Score'}\n")
        f.write("-" * 65 + "\n")
        for name, score in results_data:
            f.write(f"{name:<40} | {score:.4f}\n")

    print(f"处理完成！结果保存在: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()