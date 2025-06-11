#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch, cv2, numpy as np, uuid
from pathlib import Path

from scipy.optimize import linear_sum_assignment

from traffic_light.models import Darknet
from traffic_light.utils.utils import non_max_suppression, scale_coords, plot_one_box
from traffic_light.utils.datasets import letterbox

# —— 硬编码配置 ——
MODEL_CFG     = 'irVisionmodule/traffic_light/cfg/yolov3-spp-6cls.cfg'
MODEL_WEIGHTS = 'irVisionmodule/traffic_light/best_model_12.pt'
IMG_SIZE      = (512, 512)
CONF_THRES    = 0.25
IOU_THRES     = 0.6
DEVICE        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 6->3 类映射
MAP_6_TO_3 = {
    0: 'GREEN',
    1: 'RED',
    2: 'RED',
    3: 'GREEN',
    4: 'YELLOW',
    5: 'YELLOW',
}
# labels = {
#     'go': 0,
#     'stop': 1,
#     'stopLeft': 2,
#     'goLeft': 3,
#     'warning': 4,
#     'warningLeft': 5
# }
# —- 跟踪状态 ——
tracked = {}  # id -> {'box': [x1,y1,x2,y2], 'cls': 'RED'/...}

# —— 距离过滤（基于框高，模拟距离） ——
MIN_BOX_H, MAX_BOX_H = 30, 200  # 高度在 30~200 像素范围内才算“在身边”
IOU_MATCH_THRES = 0.1   # 匈牙利匹配最低 IoU
# —— 模型初始化 ——
_net = Darknet(MODEL_CFG, IMG_SIZE)
_net.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE)['model'])
_net.to(DEVICE).eval()

def compute_iou(boxA, boxB):
    """计算两个框的 IoU"""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(boxAArea + boxBArea - inter + 1e-6)



def update_tracking(detections, max_lost=5):
    """
    用匈牙利算法全局最优地把 detections 分配给已有 tracks。
    """
    global tracked
    track_ids = list(tracked.keys())
    T = len(track_ids)
    D = len(detections)

    # 1. 构造 IoU 矩阵（只有同类才算分数，不同类为 0）
    iou_mat = np.zeros((T, D), dtype=np.float32)
    for i, tid in enumerate(track_ids):
        boxA, clsA = tracked[tid]['box'], tracked[tid]['cls']
        for j, det in enumerate(detections):
            if det['merged_cls'] != clsA:
                continue
            iou_mat[i, j] = compute_iou(boxA, det['box'])

    # 2. 用线性和分配（匈牙利）最大化总 IoU（cost = 1 - IoU）
    if T>0 and D>0:
        row_ind, col_ind = linear_sum_assignment(1 - iou_mat)
    else:
        row_ind, col_ind = np.array([]), np.array([])

    # 3. 更新匹配成功的 track
    new_tracked = {}
    matched_det = set()
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= IOU_MATCH_THRES:
            tid = track_ids[r]
            box = detections[c]['box']
            cls = detections[c]['merged_cls']
            new_tracked[tid] = {'box': box, 'cls': cls, 'lost': 0}
            matched_det.add(c)
        else:
            # 小于阈值视为未匹配
            tracked[track_ids[r]]['lost'] += 1

    # 4. 处理漏匹配的老 track（lost 计数 + 删除）
    for tid in track_ids:
        if tid not in new_tracked:
            tracked[tid]['lost'] += 1
            if tracked[tid]['lost'] <= max_lost:
                new_tracked[tid] = tracked[tid]

    # 5. 对剩余未匹配的 det 创建新 ID
    triggered = []
    for j, det in enumerate(detections):
        if j not in matched_det:
            nid = str(uuid.uuid4())
            new_tracked[nid] = {'box': det['box'], 'cls': det['merged_cls'], 'lost': 0}
            triggered.append(det['merged_cls'])

    tracked = new_tracked
    return triggered


def detect_image(img_bgr):
    img0 = img_bgr.copy()
    # 1. 预处理
    img_r = letterbox(img0, new_shape=IMG_SIZE)[0]
    arr = img_r[:,:,::-1].transpose(2,0,1)
    arr = np.ascontiguousarray(arr)
    tensor = torch.from_numpy(arr).to(DEVICE).float()/255.0
    if tensor.ndimension()==3: tensor = tensor.unsqueeze(0)

    # 2. 推理
    with torch.no_grad():
        pred = _net(tensor)[0]
    det = non_max_suppression(pred, CONF_THRES, IOU_THRES)[0]

    # 3. 解析 & 绘制
    detections = []
    if det is not None and len(det):
        det[:,:4] = scale_coords(tensor.shape[2:], det[:,:4], img0.shape).round()
        for *xyxy, conf, cls in det:
            merged = MAP_6_TO_3[int(cls.item())]
            box = [int(x.item()) for x in xyxy]
            detections.append({'box':box,'merged_cls':merged})
            plot_one_box(box, img0, label=merged)

    # 4. 跟踪，无距离过滤
    triggered = update_tracking(detections)

    # 5. 基于 triggered 列表和距离过滤计算动作
    actions = []
    for cls in triggered:
        # 找到对应 box 来判高度过滤
        # 假设 last tracked 与 triggered 顺序一致，可按 cls 匹配 box
        for tid, info in tracked.items():
            if info['cls'] == cls:
                box = info['box']
                h = box[3] - box[1]
                # 只有在合适距离（框高范围）内才触发动作
                if MIN_BOX_H <= h <= MAX_BOX_H:
                    action = {'RED':'stop','YELLOW':'slow','GREEN':'go'}[cls]
                    actions.append(action)
                break

    # 合并最终建议
    if 'stop' in actions: suggestion='stop'
    elif 'slow' in actions: suggestion='slow'
    else:                 suggestion='go'

    # 5. 绘制所有跟踪框 ID 和状态
    for tid, info in tracked.items():
        x1,y1,x2,y2 = info['box']
        in_range = (MIN_BOX_H <= (y2-y1) <= MAX_BOX_H)
        color = (0,255,0) if in_range else (0,0,255)
        txt = f"{tid[:4]}:{'IN' if in_range else 'OUT'}"
        cv2.putText(img0, txt, (x1, y1-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 6. 绘制全局动作建议
    cv2.putText(img0, f"Action:{suggestion}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return img0, suggestion

# —— 测试 ——
if __name__=='__main__':
    import sys
    src = sys.argv[1] if len(sys.argv)>1 else '0'
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    while True:
        ret,frame = cap.read()
        if not ret: break
        out,act = detect_image(frame)
        cv2.imshow('TL', out)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()
