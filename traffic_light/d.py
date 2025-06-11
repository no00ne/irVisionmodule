#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv3 Traffic Light Detection - Inference Script
Configuration items are hard-coded for simplicity.
"""
import numpy as np
import torch
import cv2
from pathlib import Path

# Utility imports from the original project
from models import Darknet
from utils.utils import non_max_suppression, scale_coords, plot_one_box
from utils.datasets import letterbox

# —— 配置项（写死在脚本同目录下） ——
MODEL_CFG     = 'cfg/yolov3-spp-6cls.cfg'     # 网络配置文件路径
MODEL_WEIGHTS = 'best_model_12.pt'            # 权重文件路径（同目录）
IMG_SIZE      = (512, 512)                    # 推理输入尺寸 (height, width)
CONF_THRES    = 0.3                           # 置信度阈值
IOU_THRES     = 0.6                           # NMS IOU 阈值
DEVICE        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# —— 模型加载（全局一次） ——
_net = Darknet(MODEL_CFG, IMG_SIZE)
_net.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE)['model'])
_net.to(DEVICE).eval()

MAP_6_TO_3 = {
    0: 'GREEN',
    1: 'RED',
    2: 'RED',
    3: 'GREEN',
    4: 'YELLOW',
    5: 'YELLOW',
    # (如需 YELLOW，可在此添加)
}
# labels = {
#     'go': 0,
#     'stop': 1,
#     'stopLeft': 2,
#     'goLeft': 3,
#     'warning': 4,
#     'warningLeft': 5
# }
def detect_image(img_bgr):
    """
    对一张 BGR 图像执行 YOLOv3 推理，返回标注图和检测列表。

    Args:
        img_bgr (np.ndarray): 原始 BGR 图像

    Returns:
        img_out (np.ndarray): 绘制了检测框和标签的 BGR 图像
        results (list of dict): 每个检测结果包含:
            - 'box': [x1, y1, x2, y2]
            - 'conf': float 置信度
            - 'cls':  int 类别索引
    """
    # 1. 预处理：letterbox 保持宽高比填充到 IMG_SIZE
    img0 = img_bgr.copy()
    img_resized = letterbox(img0, new_shape=IMG_SIZE)[0]
    # BGR->RGB, HWC->CHW, 转为 0~1 浮点张量
    img_np = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img_np = np.ascontiguousarray(img_np)
    img_tensor = torch.from_numpy(img_np).to(DEVICE).float() / 255.0

    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # 2. 推理
    with torch.no_grad():
        pred = _net(img_tensor)[0]

    # 3. NMS
    det = non_max_suppression(pred, CONF_THRES, IOU_THRES)[0]

    # 4. 解析 & 绘制
    results = []
    if det is not None and len(det):
        # 恢复到原始图像尺度
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in det:
            orig_cls = int(cls.item())
            merged = MAP_6_TO_3.get(orig_cls, None)
            if merged is None:
                continue  # 若有未映射的类别，可跳过或自定义
            box = [int(x.item()) for x in xyxy]
            results.append({
                'box': box,
                'conf': float(conf.item()),
                'orig_cls': orig_cls,
                'merged_cls': merged
            })
            # 绘制时直接用合并后的标签
            plot_one_box(box, img0, label=merged)

    return img0, results


# —— 单元测试入口：读取图片或摄像头 ——
if __name__ == '__main__':
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else '0'
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        # 当 source 不是摄像头，尝试作为单张图片处理
        img = cv2.imread(str(source))
        out, res = detect_image(img)
        cv2.imwrite('result.jpg', out)
        print(res)
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out, res = detect_image(frame)
            cv2.imshow('YOLOv3 Traffic Light', out)
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
