# YOLO11-Dental 消融实验指南 (v2.0)

## 一、消融实验设计

### 实验总览

| 实验编号 | 名称 | 4通道 | P2 | ECA | Loss | 对应YAML |
|---------|------|:-----:|:--:|:---:|------|----------|
| **Exp0** | Baseline | ✗ | ✗ | ✗ | CIoU | yolo11-baseline3cls.yaml |
| **Exp1** | +4ch Prior | ✓ | ✗ | ✗ | WIoU+NWD | yolo11-dental.yaml |
| **Exp2** | +P2 Head | ✓ | ✓ | ✗ | WIoU+NWD | yolo11-dental-p2.yaml |
| **Exp3** | +ECA Attention | ✓ | ✗ | ✓ | WIoU+NWD | yolo11-dental-eca.yaml |
| **Exp4** | Full Model | ✓ | ✓ | ✓ | WIoU+NWD | yolo11-dental-full.yaml |
| **Exp5** | Prior-as-Attn | ✓ | ✗ | ✗ | WIoU+NWD | yolo11-dental-attn.yaml |

> **Exp5说明**: Prior-as-Attention是备选方案，当Exp1效果不好时使用。
> Prior不直接参与卷积，而是转换为空间注意力调制RGB特征。

### 实验目的

| 对比 | 验证内容 |
|------|----------|
| Exp0 → Exp1 | 4通道先验输入的增益 |
| Exp1 → Exp2 | P2检测头对小目标的增益 |
| Exp1 → Exp3 | ECA通道注意力的增益 |
| Exp4 | 所有结构改进叠加效果 |
| Exp1 vs Exp5 | Early Fusion vs Prior-as-Attention |

### 禁用的改进

> **解剖约束增广(Anatomy-Constrained Augmentation)已禁用**
> 原因：根尖周病变与特定牙齿根尖有严格的空间对应关系，随机粘贴会破坏诊断性位置特征。

---

## 二、改进技术详解

### 2.1 4通道输入 (Exp1)

```
输入: 4通道 (H, W, 4)
  Ch0: I_raw      - 原始灰度图 [0,1]
  Ch1: M_virtual  - 虚拟融合掩码 {0,1}
  Ch2: Distance   - 距离变换场 [0,1] (实例内归一化)
  Ch3: M_metal    - 金属先验掩码 {0,1}

改动: Conv(3→64) → Conv(4→64)
```

### 2.2 P2检测头 (Exp2)

```
原始 (3尺度):  Detect [P3, P4, P5] → 8/16/32x 下采样
改进 (4尺度):  Detect [P2, P3, P4, P5] → 4/8/16/32x 下采样

效果: 20px病灶在P3只有2.5px特征，在P2有5px特征
```

### 2.3 ECA注意力 (Exp3)

```python
C3k2_Dental = C3k2 + EfficientChannelAttention

# ECA: 1D卷积实现的轻量通道注意力
x → AdaptiveAvgPool2d(1) → Conv1d(k=auto) → Sigmoid → x * attention
```

### 2.4 小目标Loss (所有4ch实验)

| Loss | 作用 |
|------|------|
| **WIoU** | 动态聚焦，降低outlier影响 |
| **NWD** | Wasserstein距离，对小目标梯度稳定 |
| **Size-aware** | 小目标(<32px)获得2倍权重 |

```bash
# Loss配置
--loss_type wiou_nwd      # 推荐
--small_threshold 32      # 小目标阈值
--small_weight 2.0        # 小目标权重
```

### 2.5 通道感知增强

| 增强 | Ch0 (Raw) | Ch1-3 (Prior) | 原因 |
|------|:---------:|:-------------:|------|
| Mosaic | ✓ | ✓ | 几何同步 |
| MixUp | ✓ | ✗ | 保护Prior语义 |
| Affine/Flip | ✓ | ✓ | 几何同步 |
| Photometric | ✓ | ✗ | Prior不做亮度变换 |
| Erasing | ✓ | ✗ | 保护Prior完整性 |

---

## 三、运行方法

### 数据目录结构

```
data_root/
├── images/           # 原始X光图像
│   ├── img001.png
│   └── ...
├── stage1_masks/     # Stage1分割掩码 (类别索引0-4)
│   ├── img001.png
│   └── ...
├── labels/           # YOLO格式标签
│   ├── img001.txt    # cls x_center y_center w h
│   └── ...
├── fold_3_train.txt  # 训练集图像列表
└── fold_3_val.txt    # 验证集图像列表
```

### txt文件格式

```
# fold_3_train.txt (每行一个图像名)
img001.png
img002.png
...
```

### 运行命令

```bash
cd D:\下载\ultralytics-8.3.6\ultralytics-8.3.6

# 运行 Exp0 vs Exp1 对比
python run_ablation.py --exp 0 1 \
    --data_root /path/to/data \
    --train_txt /path/to/train.txt \
    --val_txt /path/to/val.txt \
    --data_yaml /path/to/data.yaml \
    --loss_type wiou_nwd \
    --epochs 200 \
    --batch_size 16 \
    --device 0

# 运行所有实验 (不含Exp5)
python run_ablation.py --exp all \
    --data_root /path/to/data \
    ...

# 单独运行备选方案Exp5
python run_ablation.py --exp 5 \
    --data_root /path/to/data \
    ...
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp` | all | 实验编号: 0 1 2 3 4 5 或 all |
| `--data_root` | 必填 | 数据根目录 |
| `--train_txt` | 必填 | 训练集txt文件 |
| `--val_txt` | 必填 | 验证集txt文件 |
| `--data_yaml` | 必填(Exp0) | Exp0用的标准YOLO数据配置 |
| `--epochs` | 200 | 训练轮数 |
| `--batch_size` | 16 | 批大小 |
| `--imgsz` | 640 | 输入尺寸 |
| `--lr` | 0.001 | 初始学习率 |
| `--loss_type` | wiou_nwd | Loss类型 |
| `--small_threshold` | 32 | 小目标尺寸阈值 |
| `--small_weight` | 2.0 | 小目标权重 |
| `--close_mosaic` | 10 | 最后N epoch关闭Mosaic |

---

## 四、训练完成后

### 4.1 结果分析

```bash
# 分析训练曲线
python analyze_results.py --exp_dir runs/ablation/exp0_baseline
python analyze_results.py --exp_dir runs/ablation/exp1_4ch_prior

# 正式评估
python evaluate_dental.py \
    --weights runs/ablation/exp1_4ch_prior/train/weights/best.pt \
    --data_root /path/to/data \
    --val_txt val.txt
```

### 4.2 验证Prior有效性

如果 Exp1 ≈ Exp0 (无提升)，检查Prior是否被学习：

```python
import torch
ckpt = torch.load("runs/ablation/exp1_4ch_prior/train/weights/best.pt")
conv1_weight = ckpt["model"].model[0].conv.weight  # [64, 4, 3, 3]

for c in range(4):
    norm = conv1_weight[:, c].abs().mean()
    print(f"Ch{c} weight magnitude: {norm:.4f}")

# 期望: Ch1-Ch3权重非零
# 如果Ch1-Ch3接近0 → Prior被忽略 → 尝试Exp5
```

### 4.3 输出目录结构

```
runs/ablation/
├── exp0_baseline/
│   └── train/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.csv
│       ├── results.png
│       ├── confusion_matrix.png
│       └── PR_curve.png
├── exp1_4ch_prior/
│   └── train/
│       └── ...
├── exp2_p2_head/
├── exp3_eca_attention/
├── exp4_full_model/
├── exp5_prior_attn/
└── ablation_summary.json
```

---

## 五、论文表格模板

### Table 1: 消融实验结果

| Method | 4ch | P2 | ECA | Loss | mAP@50 | mAP@50-95 | Params(M) | GFLOPs |
|--------|:---:|:--:|:---:|------|:------:|:---------:|:---------:|:------:|
| Baseline (Exp0) | - | - | - | CIoU | — | — | — | — |
| +4ch Prior (Exp1) | ✓ | - | - | WIoU+NWD | — | — | — | — |
| +P2 Head (Exp2) | ✓ | ✓ | - | WIoU+NWD | — | — | — | — |
| +ECA (Exp3) | ✓ | - | ✓ | WIoU+NWD | — | — | — | — |
| **Full (Exp4)** | ✓ | ✓ | ✓ | WIoU+NWD | — | — | — | — |

### Table 2: 按类别AP

| Method | periapical | periodontal | combined |
|--------|:----------:|:-----------:|:--------:|
| Baseline | — | — | — |
| Full Model | — | — | — |

### Table 3: Loss对比 (可选)

| Loss Type | mAP@50 | mAP@50-95 | 小目标mAP |
|-----------|:------:|:---------:|:---------:|
| CIoU | — | — | — |
| WIoU | — | — | — |
| WIoU+NWD | — | — | — |

---

## 六、YAML文件清单

| 文件名 | 用途 | 改动 |
|--------|------|------|
| yolo11-baseline3cls.yaml | Exp0 Baseline | 标准YOLO11, 3ch |
| yolo11-dental.yaml | Exp1 +4ch | ch: 4 |
| yolo11-dental-p2.yaml | Exp2 +P2 | ch: 4, +P2 head |
| yolo11-dental-eca.yaml | Exp3 +ECA | ch: 4, C3k2→C3k2_Dental |
| yolo11-dental-full.yaml | Exp4 Full | ch: 4, +P2, +ECA |
| yolo11-dental-attn.yaml | Exp5 Attn | ch: 4, PriorAsAttention |

---

## 七、注意事项

1. **公平对比**：所有实验使用相同的训练基础设施
   - EMA (指数移动平均)
   - Warmup (学习率预热)
   - AMP (混合精度训练)
   - Cosine LR + AdamW
   - Early Stopping

2. **DentalTrainer**：Exp1-5使用自定义训练器
   - 继承自 `DetectionTrainer`
   - 支持4ch数据加载
   - 支持自定义Loss
   - 与Exp0使用相同的训练优化

3. **评估指标**
   - mAP@50: IoU=0.5时的mAP
   - mAP@50-95: IoU 0.5-0.95的平均mAP
   - Per-class AP: periapical, periodontal, combined
   - **Recall**: 医疗场景更重要

4. **统计显著性**
   - 建议每个实验跑3次
   - 报告 mean ± std
   - 使用相同的随机种子

5. **决策流程**
   ```
   Exp0 vs Exp1
        │
        ├── Exp1 > Exp0 (+3%+) → Prior有效 → 继续Exp2,3,4
        │
        └── Exp1 ≈ Exp0 → 检查权重
                │
                ├── 权重正常 → Prior本身没用
                └── Ch1-3权重≈0 → Prior被忽略 → 尝试Exp5
   ```

---

## 八、预期性能

| 改进 | 预期增益 | 主要受益场景 |
|------|---------|-------------|
| 4ch Prior | +3-8% mAP | 金属干扰区域 |
| P2 Head | +2-5% mAP | 小病灶 (<20px) |
| ECA | +1-3% mAP | 通道特征选择 |
| WIoU+NWD | +2-4% mAP | 小目标定位 |
| Size-aware | +1-2% mAP | 小目标召回 |

**综合预期**: 相比baseline可能有 **5-15% mAP提升**
