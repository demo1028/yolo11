import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


# --- 1. YOLO 模型包装器 ---
class YOLOWrapper(torch.nn.Module):
    """包装 YOLO 模型，使 forward 返回单个 tensor（适配 pytorch_grad_cam）"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        # YOLO forward 返回 tuple，取第一个元素 (检测输出)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


# --- 2. 加载模型 ---
model = YOLO('/home/user/Han/dangpeipei/ultralytics-8.3.6/runs/train/fold3/weights/best.pt')
wrapped_model = YOLOWrapper(model.model)
wrapped_model.eval()

target_layers = [model.model.model[-2]]  # C2PSA 层 (倒数第二层)

# --- 3. 读取图片 ---
img_path = "/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_from_mask/images/1033.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm = np.float32(rgb_img) / 255.0

# --- 4. 初始化 CAM ---
cam = EigenCAM(
    model=wrapped_model,
    target_layers=target_layers,
)

# --- 5. 生成热力图 ---
tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(model.device)

grayscale_cam = cam(input_tensor=tensor, targets=None)
grayscale_cam = grayscale_cam[0, :]

# --- 6. 叠加显示 ---
visualization = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

# --- 7. 保存图片 ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title("YOLO11 EigenCAM Heatmap")
plt.axis('off')
plt.tight_layout()
save_path = img_path.rsplit('.', 1)[0] + '_eigencam.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存到: {save_path}")
