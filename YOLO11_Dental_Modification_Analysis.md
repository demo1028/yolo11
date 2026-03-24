# YOLO11-Dental 改进分析报告 (v2.0)

## 一、整体架构对比

```
原始 YOLO11                              YOLO11-Dental
───────────────────────────────────────────────────────────────────────────
输入: 3ch RGB                            输入: 4ch [I_raw, M_virtual, Dist, M_metal]
      │                                        │
      ▼                                        ▼
Conv(3→64) ──────────────────           Conv(4→64) 或 PriorAsAttention
      │                                        │
   Backbone                                 Backbone
   (C3k2)                                  (C3k2_Dental + ECA)
      │                                        │
   P3,P4,P5                                P2,P3,P4,P5 (可选P2)
      │                                        │
    Head                                     Head
      │                                        │
   Detect (3 scale)                        Detect (3-4 scale)
      │                                        │
   CIoU Loss                               WIoU + NWD Loss (size-aware)
───────────────────────────────────────────────────────────────────────────
```

## 二、改动分类汇总

### 2.1 文件结构

```
ultralytics-8.3.6/
├── 【网络结构】
│   ├── ultralytics/cfg/models/11/
│   │   ├── yolo11-baseline3cls.yaml      # Exp0: 原始3ch基线
│   │   ├── yolo11-dental.yaml            # Exp1: 4ch输入
│   │   ├── yolo11-dental-p2.yaml         # Exp2: +P2检测头
│   │   ├── yolo11-dental-eca.yaml        # Exp3: +ECA注意力
│   │   ├── yolo11-dental-full.yaml       # Exp4: 全部改进
│   │   └── yolo11-dental-attn.yaml       # Exp5: Prior-as-Attention
│   └── ultralytics/nn/modules/
│       ├── dental_attention.py           # ECA, C3k2_Dental等
│       └── prior_attention.py            # PriorAsAttentionInput等
│
├── 【数据处理】
│   └── ultralytics/data/
│       ├── dental_dataset.py             # 4ch数据集 + collate_fn
│       ├── dental_preprocess.py          # Stage1 mask → 4ch tensor
│       └── dental_augment.py             # 通道感知增强
│
├── 【训练流程】
│   ├── ultralytics/engine/
│   │   └── dental_trainer.py             # DentalTrainer + DentalValidator
│   └── ultralytics/utils/
│       ├── loss_dental.py                # WIoU, Inner-IoU, NWD实现
│       └── loss_dental_v3.py             # 集成Loss类
│
└── 【入口脚本】
    ├── train_dental.py                   # 单次训练
    ├── run_ablation.py                   # 消融实验
    ├── evaluate_dental.py                # 评估
    ├── predict_dental.py                 # 推理
    └── analyze_results.py                # 结果分析
```

### 2.2 改动对比表

| 模块 | 原始 YOLO11 | YOLO11-Dental | 改动原因 |
|------|-------------|---------------|----------|
| **输入通道** | 3 (RGB) | 4 (Raw+Prior) | 融合Stage1语义先验 |
| **第一层Conv** | Conv(3→64) | Conv(4→64) | 适配4通道输入 |
| **Backbone模块** | C3k2 | C3k2_Dental (含ECA) | 增强通道特征选择 |
| **检测尺度** | P3,P4,P5 (3层) | P2,P3,P4,P5 (4层，可选) | 提升小目标检测 |
| **归一化** | img/255 | 跳过(已归一化) | 4ch已在[0,1] |
| **增强-MixUp** | 全通道混合 | 仅Ch0混合 | 保护Prior语义 |
| **增强-Erasing** | 全通道擦除 | 仅Ch0擦除 | 保护Prior完整性 |
| **增强-Photometric** | 全通道 | 仅Ch0 | Prior不应做亮度变换 |
| **Box Loss** | CIoU | WIoU + NWD | 小目标梯度更稳定 |
| **小目标加权** | 无 | Size-aware weighting | 提升小目标贡献 |
| **训练器** | DetectionTrainer | DentalTrainer | 支持4ch + 自定义Loss |

---

## 三、网络结构改动详解

### 3.1 输入通道 3→4

**原始**：
```yaml
# yolo11.yaml (无ch字段，默认3)
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # Conv2d(3, 64, ...)
```

**修改**：
```yaml
# yolo11-dental.yaml
ch: 4  # 显式指定4通道
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 自动变为Conv2d(4, 64, ...)
```

**4通道语义**：
| 通道 | 名称 | 内容 | 值域 |
|------|------|------|------|
| Ch0 | I_raw | 原始灰度图 | [0, 1] 连续 |
| Ch1 | M_virtual | 虚拟融合掩码 | {0, 1} 二值 |
| Ch2 | Distance | 距离变换场(实例内归一化) | [0, 1] 连续 |
| Ch3 | M_metal | 金属先验掩码 | {0, 1} 二值 |

### 3.2 P2检测头

```
原始 (3尺度)                      改进 (4尺度)
─────────────────────────────────────────────────────
Backbone:                         Backbone:
  P2/4: Conv 128                    P2/4: Conv 128 + C3k2  ← 新增特征块
  P3/8: Conv 256 + C3k2             P3/8: Conv 256 + C3k2
  P4/16: Conv 512 + C3k2            P4/16: Conv 512 + C3k2
  P5/32: Conv 1024 + C3k2           P5/32: Conv 1024 + C3k2

Head (Top-down):                  Head (Top-down):
  P5→P4: Upsample+Concat            P5→P4: Upsample+Concat
  P4→P3: Upsample+Concat            P4→P3: Upsample+Concat
  (结束)                             P3→P2: Upsample+Concat  ← 新增

Head (Bottom-up):                 Head (Bottom-up):
  P3→P4: Conv+Concat                P2→P3: Conv+Concat      ← 新增
  P4→P5: Conv+Concat                P3→P4: Conv+Concat
                                    P4→P5: Conv+Concat

Detect:                           Detect:
  [P3, P4, P5] → 8/16/32x           [P2, P3, P4, P5] → 4/8/16/32x
```

**效果**：20px病灶在P3只有2.5px特征，在P2有5px特征。

### 3.3 ECA注意力 (C3k2_Dental)

```python
C3k2_Dental = C3k2 + EfficientChannelAttention

class EfficientChannelAttention(nn.Module):
    """1D卷积实现的轻量通道注意力"""
    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)                    # (B, C, 1, 1)
        y = self.conv1d(y.squeeze().transpose())  # 1D conv along C
        attention = self.sigmoid(y)              # (B, C, 1, 1)
        return x * attention
```

### 3.4 Prior-as-Attention (备选方案)

当Early Fusion效果不好时，可使用此方案：

```
原始 Early Fusion:                 Prior-as-Attention:
[I_raw, Prior] → Conv(4→64)        [I_raw, Prior]
                                         │
                                   ┌─────┴─────┐
                                   │           │
                                 I_raw      Prior(3ch)
                                   │           │
                                   │     PriorToAttn
                                   │      (3→1 conv)
                                   │           │
                                   └─── × ─────┘
                                         │
                                   RGB_attended
                                         │
                                   Conv(3→64)  ← 标准backbone
```

**优势**：Prior不被当作纹理处理，保持语义层级。

---

## 四、数据处理改动详解

### 4.1 4通道生成 (dental_preprocess.py)

```
输入:
  - raw_image: (H, W, 3) 原始X光
  - stage1_mask: (H, W) 类别索引0-4

处理流程:
  1. 灰度化 + 归一化 → Ch0
  2. 虚拟融合 (union所有牙齿类 + 形态学闭运算) → M_virtual
  3. 距离变换 + Watershed实例分离 + 实例内归一化 → Ch2
  4. 金属先验 (union种植体+修复体) → Ch3
  5. Stack → (H, W, 4)

Watershed改进:
  原始: threshold = 0.3 * distance.max()  # 不稳定
  改进:
    D_smooth = gaussian_filter(D, sigma=1.5)
    markers = (D_smooth > 8.0)  # 绝对阈值
    markers = remove_small_objects(markers, min_size=50)
    markers = markers & (h_maxima(D_smooth, h=3.0) > 0)
    labels = watershed(-D_smooth, label(markers), mask=M_virtual)
    # 每个实例独立归一化
```

### 4.2 通道感知增强 (dental_augment.py)

| 增强类型 | Ch0 (Raw) | Ch1-3 (Prior) | 原因 |
|---------|-----------|---------------|------|
| **几何变换** | ✓ | ✓ | 空间对应必须同步 |
| Mosaic | ✓ | ✓ | 拼接后Prior仍对应 |
| MixUp | ✓ | ✗ | 混合会破坏Prior语义 |
| Affine | ✓ | ✓ | 几何同步 |
| Flip | ✓ | ✓ | 几何同步 |
| **光度变换** | ✓ | ✗ | Prior不应做亮度变换 |
| Brightness | ✓ | ✗ | - |
| Contrast | ✓ | ✗ | - |
| CLAHE | ✓ | ✗ | - |
| Blur/Noise | ✓ | ✗ | - |
| **擦除** | ✓ | ✗ | 保持Prior完整性 |

### 4.3 数据加载 (dental_dataset.py)

```python
class DentalYOLODataset(Dataset):
    def __init__(self, data_root, split_file, imgsz, augment):
        # 从txt文件读取图像列表
        # 自动定位 images/, stage1_masks/, labels/

    def __getitem__(self, idx):
        # 1. 加载图像和Stage1 mask
        # 2. DentalStage1Processor → 4ch tensor
        # 3. 加载YOLO格式标签
        # 4. Dental4chAugmentation (if augment)
        # 5. 返回 {img, cls, bboxes, batch_idx, ...}
```

---

## 五、Loss函数改动详解

### 5.1 小目标Loss问题

```
原始CIoU问题:
  小目标 (10x10px) 平移1px → IoU下降 ~20%
  大目标 (100x100px) 平移1px → IoU下降 ~2%
  → 小目标梯度不稳定，难以优化
```

### 5.2 改进Loss

| Loss | 公式 | 作用 |
|------|------|------|
| **WIoU** | `(1-IoU+penalty) * exp((1-IoU)*scale)` | 动态聚焦，降低outlier影响 |
| **Inner-IoU** | 使用内部框(ratio=0.75)计算IoU | 小目标边界更稳定 |
| **NWD** | `exp(-√(center_dist+size_dist)/norm)` | Wasserstein距离，尺度不变 |

### 5.3 Size-aware Weighting

```python
size = sqrt(w * h)  # 目标尺寸
weight = 2.0 if size < small_threshold else 1.0
loss = loss * weight
```

### 5.4 Loss配置

| 配置 | 组合 | 适用场景 |
|------|------|---------|
| `ciou` | CIoU | 基线 |
| `wiou` | WIoU | 一般目标 |
| `wiou_nwd` | WIoU + NWD | **推荐：小病灶** |
| `inner_wiou_nwd` | Inner + WIoU + NWD | 极小病灶 (<16px) |

---

## 六、训练流程改动

### 6.1 DentalTrainer

```python
class DentalTrainer(DetectionTrainer):
    """
    继承所有官方优化:
    - EMA (指数移动平均)
    - Warmup (学习率预热)
    - AMP (混合精度训练)
    - Cosine LR
    - Close-mosaic
    - Early stopping

    新增:
    - 4ch数据加载 (get_dataloader)
    - 跳过/255归一化 (preprocess_batch)
    - 小目标Loss (get_loss)
    - 注册自定义模块 (get_model)
    """
```

### 6.2 训练命令

```bash
# 单次训练
python train_dental.py \
    --data_root /path/to/data \
    --train_txt train.txt \
    --val_txt val.txt \
    --model yolo11-dental.yaml \
    --loss_type wiou_nwd \
    --epochs 200

# 消融实验
python run_ablation.py --exp 0 1 2 3 4 \
    --data_root /path/to/data \
    --train_txt train.txt \
    --val_txt val.txt \
    --data_yaml data.yaml \
    --loss_type wiou_nwd
```

---

## 七、消融实验设计

| Exp | 名称 | 4ch | P2 | ECA | Loss | YAML |
|-----|------|:---:|:--:|:---:|------|------|
| 0 | Baseline | - | - | - | CIoU | yolo11-baseline3cls.yaml |
| 1 | +4ch Prior | ✓ | - | - | WIoU+NWD | yolo11-dental.yaml |
| 2 | +P2 Head | ✓ | ✓ | - | WIoU+NWD | yolo11-dental-p2.yaml |
| 3 | +ECA | ✓ | - | ✓ | WIoU+NWD | yolo11-dental-eca.yaml |
| 4 | Full | ✓ | ✓ | ✓ | WIoU+NWD | yolo11-dental-full.yaml |
| 5 | Prior-Attn | ✓ | - | - | WIoU+NWD | yolo11-dental-attn.yaml |

**验证Prior有效性**：
- Exp0 vs Exp1: 验证4ch输入增益
- 若Exp1 ≈ Exp0: 检查权重是否学习，考虑Exp5

---

## 八、预期性能提升

| 改进 | 预期增益 | 主要受益场景 |
|------|---------|-------------|
| 4ch Prior | +3-8% mAP | 金属干扰区域 |
| P2 Head | +2-5% mAP | 小病灶 (<20px) |
| ECA | +1-3% mAP | 通道特征选择 |
| WIoU+NWD | +2-4% mAP | 小目标定位 |
| Size-aware | +1-2% mAP | 小目标召回 |

**综合预期**：相比baseline可能有 **5-15% mAP提升**，具体取决于数据集特性。
