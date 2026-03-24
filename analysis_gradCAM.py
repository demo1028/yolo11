import warnings

warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# --- Wrapper 和 Target 类保持不变 (照搬上面的修复版) ---
class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if not torch.is_grad_enabled():
            torch.set_grad_enabled(True)
        results = self.model(x)
        if isinstance(results, (tuple, list)):
            return results[0]
        return results


class SpecificClassTarget:
    def __init__(self, target_class_idx):
        self.target_class_idx = target_class_idx

    def __call__(self, model_output):
        if model_output.ndim == 2:
            return model_output[4 + self.target_class_idx, :].max()
        else:
            return model_output[:, 4 + self.target_class_idx, :].max()


# --- 配置 ---
# 请替换为你自己的路径
MODEL_PATH = '/home/user/Han/dangpeipei/ultralytics-8.3.6/runs/train/fold3/weights/best.pt'
IMG_PATH = '/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_from_mask/images/1033.jpg'
TARGET_CLASS_ID = 1
OUTPUT_DIR = "runs/layer_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print("Loading model...")
yolo_model = YOLO(MODEL_PATH)
model = yolo_model.model
wrapped_model = YOLOWrapper(model)
wrapped_model.eval()

# 图像预处理
img_bgr = cv2.imread(IMG_PATH)
img_resized = cv2.resize(img_bgr, (640, 640))
rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_norm = np.float32(rgb_img) / 255.0
input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(next(model.parameters()).device)
input_tensor.requires_grad_(True)

# ==========================================
# 【核心修改】：对比不同深度的层
# ==========================================
# YOLOv8/11 典型结构层索引 (根据实际情况可能微调，但通常如下):
# Layer 10: Backbone P3 (高分辨率, 细节)
# Layer 15: Neck P4 (中分辨率)
# Layer 21: Neck P5 / C2PSA (低分辨率, 语义) - 你之前用的类似这个
layers_to_test = {
    "Layer_10_Backbone_P3": model.model.model[9],  # 浅层，看细节/纹理
    "Layer_15_Neck_P4": model.model.model[15],  # 中层
    "Layer_Head_Input": model.model.model[-2]  # 深层，看全局
}

print(f"Starting comparison on image: {os.path.basename(IMG_PATH)}")

plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis('off')

targets = [SpecificClassTarget(TARGET_CLASS_ID)]

for i, (layer_name, layer_module) in enumerate(layers_to_test.items()):
    print(f"Processing {layer_name}...")

    try:
        # 对每一层初始化 Grad-CAM
        cam = GradCAM(model=wrapped_model, target_layers=[layer_module])

        # 生成热力图
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # 可视化
        visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

        plt.subplot(1, 4, i + 2)
        plt.imshow(visualization)
        plt.title(f"{layer_name}\n(Focus: {'Texture' if i == 0 else 'Semantic'})")
        plt.axis('off')

    except Exception as e:
        print(f"Skipping {layer_name}: {e}")

save_path = os.path.join(OUTPUT_DIR, "layer_depth_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"对比图已保存: {save_path}")