from ultralytics import YOLO

# 1. 加载官方预训练模型 (或你自己训练的RGB模型)
model = YOLO('/home/user/Han/dangpeipei/ultralytics-8.3.6/runs/train/fold3/weights/best.pt')  # 或者 'runs/detect/train/weights/best.pt'

# 2. 运行推理并开启可视化
# source 填你的测试图片路径
results = model.predict(source='/home/user/Han/dangpeipei/ultralytics-8.3.6/yolo_from_mask/images/10157.jpg', visualize=True)

print("特征图已保存到 runs/detect/predict/ 文件夹下")