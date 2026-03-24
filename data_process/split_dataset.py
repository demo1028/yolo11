# -*- coding: utf-8 -*-
"""
@Auth ：落花不写码
@File ：split_yolo_train_val_test.py
@IDE ：PyCharm
@Motto :学习新思想，争做新青年
"""
import os
import shutil
from collections import Counter
from sklearn.model_selection import train_test_split
from pathlib import Path

test_ratio = 0.2  # 测试集比例
val_ratio = 0.1   # 验证集比例
imgpath = r'D:\Desktop\资料\YOLO数据\img'  # 图片路径
txtpath = r'D:\Desktop\资料\YOLO数据\labels'      # 标签路径

BASE_DIR = Path(r'D:\Desktop\资料\YOLO数据\labels-split-分层抽样')  # 输出目录

SUPPORTED_IMG_EXTS = ['.jpg', '.jpeg', '.png']

folders = {
    'train': {'img': BASE_DIR / 'images' / 'train', 'lbl': BASE_DIR / 'labels' / 'train'},
    'val':   {'img': BASE_DIR / 'images' / 'val',   'lbl': BASE_DIR / 'labels' / 'val'},
    'test':  {'img': BASE_DIR / 'images' / 'test',  'lbl': BASE_DIR / 'labels' / 'test'},
}

for f in folders.values():
    f['img'].mkdir(parents=True, exist_ok=True)
    f['lbl'].mkdir(parents=True, exist_ok=True)

# 【修复 1】: 过滤掉标注软件自动生成的 classes.txt
txt_files = [f for f in os.listdir(txtpath) if f.endswith('.txt') and f != 'classes.txt']

def find_image_file(stem, img_dir, exts):
    for ext in exts:
        img_path = os.path.join(img_dir, stem + ext)
        if os.path.exists(img_path):
            return img_path, ext
    return None, None

paired_files = []
missing_imgs = []

for txt_file in txt_files:
    stem = txt_file[:-4]
    img_file, _ = find_image_file(stem, imgpath, SUPPORTED_IMG_EXTS)
    if img_file:
        paired_files.append((txt_file, img_file))
    else:
        missing_imgs.append(txt_file)

if missing_imgs:
    print(f"{len(missing_imgs)} 个标签文件找不到对应图像，例如：{missing_imgs[:3]}")

# ==========================================
# 多标签长尾分布的分层抽样逻辑
# ==========================================
print("正在统计多标签类别频率，准备进行分层抽样...")
img_classes = {}
global_class_counts = Counter()

for txt_file, _ in paired_files:
    txt_path = os.path.join(txtpath, txt_file)
    classes_in_img = set()
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    classes_in_img.add(cls_id)
                    global_class_counts[cls_id] += 1
    img_classes[txt_file] = classes_in_img

y_stratify = []
txt_list = [pair[0] for pair in paired_files]

for txt_file in txt_list:
    classes = img_classes[txt_file]
    if not classes:
        rep_cls = -1
    else:
        rep_cls = min(classes, key=lambda c: global_class_counts[c])
    y_stratify.append(rep_cls)

stratify_counts = Counter(y_stratify)
for i in range(len(y_stratify)):
    if stratify_counts[y_stratify[i]] < 3:
        y_stratify[i] = -2

# 第一次切分：分离出 Test
trainval_txt, test_txt, y_trainval, _ = train_test_split(
    txt_list, y_stratify, test_size=test_ratio, random_state=42, stratify=y_stratify
)

# 第二次切分：分离出 Train 和 Val
relative_val_ratio = val_ratio / (1 - test_ratio)
# 【修复 2】: 接收 4 个返回值，解决 ValueError
train_txt, val_txt, y_train, y_val = train_test_split(
    trainval_txt, y_trainval, test_size=relative_val_ratio, random_state=42, stratify=y_trainval
)

split_map = {}
for t in train_txt:
    split_map[t] = 'train'
for v in val_txt:
    split_map[v] = 'val'
for te in test_txt:
    split_map[te] = 'test'

def copy_file_pair(txt_file, img_file, split):
    dst_img = folders[split]['img'] / os.path.basename(img_file)
    dst_lbl = folders[split]['lbl'] / txt_file
    shutil.copy(img_file, dst_img)
    shutil.copy(os.path.join(txtpath, txt_file), dst_lbl)

print("\n开始移动文件，请稍候...")
for txt_file, img_file in paired_files:
    split = split_map[txt_file]
    copy_file_pair(txt_file, img_file, split)

print("-" * 40)
print(f"训练集: {len(train_txt)} 张")
print(f"验证集: {len(val_txt)} 张")
print(f"测试集: {len(test_txt)} 张")
print(f"数据集已利用长尾分层策略保存到: {BASE_DIR}")