import os
import cv2
import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
import shutil

# ================= 配置区域 =================
# 1. 模型路径 (使用你训练好的 best.pt)
MODEL_PATH = 'runs/detect/train/weights/best.pt'

# 2. 验证集路径 (指向 images/val 文件夹)
VAL_IMG_DIR = 'datasets/dental_v1/images/val'
VAL_LABEL_DIR = 'datasets/dental_v1/labels/val'

# 3. 输出坏案例的文件夹
OUTPUT_DIR = 'debug_bad_cases'

# 4. 置信度阈值 (低于这个不算检测到)
CONF_THRES = 0.25


# ===========================================

def get_ground_truth_count(label_path):
    """读取 txt 标签文件，返回包含的病变数量"""
    if not os.path.exists(label_path):
        return 0
    with open(label_path, 'r') as f:
        lines = f.readlines()
        # 过滤空行
        lines = [l for l in lines if l.strip()]
    return len(lines)


def run_error_analysis():
    # 初始化
    model = YOLO(MODEL_PATH)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # 获取所有验证集图片
    img_paths = glob.glob(os.path.join(VAL_IMG_DIR, '*.jpg')) + \
                glob.glob(os.path.join(VAL_IMG_DIR, '*.png'))

    print(f"开始分析 {len(img_paths)} 张验证集图片...")

    missed_count = 0
    ghost_count = 0

    for img_path in img_paths:
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        label_path = os.path.join(VAL_LABEL_DIR, base_name + '.txt')

        # 1. 获取真实标签数量 (GT)
        gt_count = get_ground_truth_count(label_path)

        # 2. 模型预测
        results = model.predict(img_path, conf=CONF_THRES, verbose=False)
        result = results[0]
        pred_count = len(result.boxes)

        # 3. 判断是否为“坏案例”
        is_bad_case = False
        error_type = ""

        # 情况 A: 漏检 (GT有，预测没有，或者预测的少)
        if gt_count > 0 and pred_count == 0:
            is_bad_case = True
            error_type = "Missed_Detection"
            missed_count += 1

        # 情况 B: 误检 (GT没有，预测却有)
        elif gt_count == 0 and pred_count > 0:
            is_bad_case = True
            error_type = "False_Positive"
            ghost_count += 1

        # 情况 C: 数量对不上 (复杂情况，也算坏)
        elif gt_count != pred_count:
            is_bad_case = True
            error_type = f"Count_Mismatch(GT{gt_count}_Pred{pred_count})"

        # 4. 如果是坏案例，画图并保存
        if is_bad_case:
            # 绘制预测结果图
            plot_img = result.plot()  # BGR numpy array

            # 保存路径
            save_name = f"{error_type}_{base_name}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            # 可以在图上写上 GT 数量
            cv2.putText(plot_img, f"GT: {gt_count} vs Pred: {pred_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imwrite(save_path, plot_img)

    print(f"分析结束！结果保存在 {OUTPUT_DIR}")
    print(f"完全漏检图片数: {missed_count}")
    print(f"完全误检图片数: {ghost_count}")


if __name__ == '__main__':
    run_error_analysis()