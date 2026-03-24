# -*- coding: utf-8 -*-
import os
from collections import Counter
from pathlib import Path

# 指向你刚才生成的新数据集目录
BASE_DIR = Path(r'D:\Desktop\资料\YOLO数据\labels-split')

subsets = ['train', 'val', 'test']
class_counts = {subset: Counter() for subset in subsets}

print(f"正在扫描 {BASE_DIR} 下的标签文件...")

# 遍历三个子集
for subset in subsets:
    label_dir = BASE_DIR / 'labels' / subset
    if not label_dir.exists():
        print(f"警告：找不到目录 {label_dir}")
        continue

    txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt') and f != 'classes.txt']

    for txt_file in txt_files:
        txt_path = label_dir / txt_file
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[subset][cls_id] += 1

# 收集所有出现过的类别 ID
all_classes = set()
for subset in subsets:
    all_classes.update(class_counts[subset].keys())

sorted_classes = sorted(list(all_classes))

# 打印表格
print("\n" + "=" * 55)
print("📊 各类别标注框 (Bounding Boxes) 数量统计")
print("=" * 55)
print(f"{'类别 ID':<8} | {'Train (70%)':<12} | {'Val (10%)':<10} | {'Test (20%)':<10} | {'Total':<8}")
print("-" * 55)

for cls_id in sorted_classes:
    train_c = class_counts['train'].get(cls_id, 0)
    val_c = class_counts['val'].get(cls_id, 0)
    test_c = class_counts['test'].get(cls_id, 0)
    total_c = train_c + val_c + test_c

    print(f"{cls_id:<10} | {train_c:<12} | {val_c:<10} | {test_c:<10} | {total_c:<8}")

print("-" * 55)
print("提示：如果某些稀有类别的数量极其稀少，YOLO 训练时建议开启 Focal Loss。")