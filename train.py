# -*- coding: utf-8 -*-
"""
@Auth ： 落花不写码
@File ：train.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'/home/user/Han/dangpeipei/ultralytics-8.3.6/ultralytics/cfg/models/11/yolo11m.yaml')
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=300,
                batch=16,
                workers=4,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
