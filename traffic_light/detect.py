#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic-light detector + Hungarian tracker + Kalman 运动预测
============================================================

改动要点
1. 每条轨迹挂 KalmanFilter → 漏检 ≤ max_lost 帧时仍能匹配成功，ID 不抖
2. `lost` 计数 + `max_lost` 清理轨迹
3. 动作去抖：stop/s​low 至少维持 HOLD 秒，防漏检闪烁
"""

import time, uuid, cv2, torch, numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter          # ← 新依赖：pip install filterpy

from traffic_light.models import Darknet
from traffic_light.utils.utils import non_max_suppression, scale_coords, plot_one_box
from traffic_light.utils.datasets import letterbox

# ───────── CONFIG ──────────────────────────────────────────
MODEL_CFG     = 'irVisionmodule/traffic_light/cfg/yolov3-spp-6cls.cfg'
MODEL_WEIGHTS = 'irVisionmodule/traffic_light/best_model_12.pt'
IMG_SIZE      = (512, 512)
CONF_THRES    = 0.25
IOU_THRES     = 0.6
DEVICE        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAP_6_TO_3 = {0:'GREEN',1:'RED',2:'RED',3:'GREEN',4:'YELLOW',5:'YELLOW'}

MIN_BOX_H, MAX_BOX_H = 30, 200          # 框高过滤
IOU_MATCH_THRES = 0.1
MAX_LOST        = 5                    # 最多允许漏检帧
HOLD            = dict(stop=0.3, slow=0.3)  # 动作维持时间

# ───────── GLOBAL STATE ───────────────────────────────────
tracked = {}                           # id -> {'box','cls','kf','lost'}
state, state_until = 'go', 0.0         # 去抖状态机

# ───────── YOLOv3 init ────────────────────────────────────
_net = Darknet(MODEL_CFG, IMG_SIZE)
_net.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE)['model'])
_net.to(DEVICE).eval()

# ───────── HELPERS ────────────────────────────────────────
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0],boxB[0]), max(boxA[1],boxB[1])
    xB, yB = min(boxA[2],boxB[2]), min(boxA[3],boxB[3])
    inter  = max(0,xB-xA)*max(0,yB-yA)
    areaA  = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB  = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA+areaB-inter+1e-6)

def _init_kf(box):
    """8D 常速 Kalman：x=[cx,cy,w,h,vx,vy,vw,vh]."""
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    w,  h  = box[2]-box[0], box[3]-box[1]
    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.eye(8)
    for i in range(4): kf.F[i, i+4] = 1
    kf.H = np.hstack([np.eye(4), np.zeros((4,4))])
    kf.P *= 100.
    kf.R *= 10.
    kf.Q *= 0.01
    kf.x[:4,0] = [cx,cy,w,h]
    return kf

def _update_action(frame_actions, now):
    """延迟去抖 —— stop > slow > go"""
    global state, state_until
    want = 'stop' if 'stop' in frame_actions else \
           'slow' if 'slow' in frame_actions else 'go'
    if want != state and now >= state_until:
        state = want
        state_until = now + HOLD.get(state, 0.0)
    return state

# ───────── TRACKING ───────────────────────────────────────
def update_tracking(detections):
    """
    detections: [{'box':[x1,y1,x2,y2],'merged_cls':str}]
    返回 frame 中触发的新 cls 列表
    """
    global tracked
    # 0) 所有旧轨迹 : 预测一步 & lost+1
    for tid in list(tracked):
        info = tracked[tid]
        info['kf'].predict()
        info['lost'] += 1
        # 用预测更新 box（供 IoU）
        cx,cy,w,h = info['kf'].x[:4].flatten()
        info['box'] = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

    # 1) IoU cost matrix
    track_ids = list(tracked.keys())
    T, D = len(track_ids), len(detections)
    iou_mat = np.zeros((T, D), dtype=np.float32)
    for i, tid in enumerate(track_ids):
        boxA, clsA = tracked[tid]['box'], tracked[tid]['cls']
        for j, det in enumerate(detections):
            if det['merged_cls'] == clsA:
                iou_mat[i,j] = compute_iou(boxA, det['box'])

    if T and D:
        rows, cols = linear_sum_assignment(1 - iou_mat)
    else:
        rows, cols = np.array([]), np.array([])

    new_tracked, matched_det = {}, set()
    # 2) 成功匹配
    for r, c in zip(rows, cols):
        if iou_mat[r, c] >= IOU_MATCH_THRES:
            tid  = track_ids[r]
            det  = detections[c]
            box  = det['box']; cls = det['merged_cls']
            kf   = tracked[tid]['kf']
            cx   = (box[0]+box[2])/2; cy = (box[1]+box[3])/2
            w, h = box[2]-box[0], box[3]-box[1]
            kf.update([cx,cy,w,h])           # ← KF 更新
            new_tracked[tid] = {'box':box,'cls':cls,'kf':kf,'lost':0}
            matched_det.add(c)

    # 3) 未匹配旧轨迹：若没超时保留
    for tid, info in tracked.items():
        if tid not in new_tracked and info['lost'] <= MAX_LOST:
            new_tracked[tid] = info

    # 4) 新建轨迹
    triggered = []
    for j, det in enumerate(detections):
        if j not in matched_det:
            nid = str(uuid.uuid4())
            new_tracked[nid] = {'box':det['box'], 'cls':det['merged_cls'],
                                'kf':_init_kf(det['box']), 'lost':0}
            triggered.append(det['merged_cls'])

    tracked = new_tracked
    return triggered

# ───────── MAIN INFERENCE FUNCTION ────────────────────────
def detect_image(img_bgr):
    global state
    img0 = img_bgr.copy()

    # 1. 预处理
    img_r = letterbox(img0, new_shape=IMG_SIZE)[0]
    arr   = img_r[:,:,::-1].transpose(2,0,1)
    tensor= torch.from_numpy(np.ascontiguousarray(arr)).to(DEVICE).float()/255.0
    if tensor.ndimension()==3: tensor = tensor.unsqueeze(0)

    # 2. 推理
    with torch.no_grad():
        pred = _net(tensor)[0]
    det = non_max_suppression(pred, CONF_THRES, IOU_THRES)[0]

    # 3. 解析检测
    detections = []
    if det is not None and len(det):
        det[:,:4] = scale_coords(tensor.shape[2:], det[:,:4], img0.shape).round()
        for *xyxy, conf, cls in det:
            merged = MAP_6_TO_3[int(cls)]
            box    = [int(x) for x in xyxy]
            detections.append({'box':box,'merged_cls':merged})
            plot_one_box(box, img0, label=merged)

    # 4. 跟踪更新
    triggered = update_tracking(detections)

    # 5. frame-level actions（距离过滤）
    actions = []
    for cls in triggered:
        for tid, info in tracked.items():
            if info['cls'] == cls:
                h = info['box'][3] - info['box'][1]
                if MIN_BOX_H <= h <= MAX_BOX_H:
                    actions.append({'RED':'stop','YELLOW':'slow','GREEN':'go'}[cls])
                break

    # 6. 去抖状态机
    suggestion = _update_action(actions, time.monotonic())

    # 7. 画跟踪框 + ID
    for tid, info in tracked.items():
        x1,y1,x2,y2 = map(int, info['box'])
        h = y2-y1
        in_range = MIN_BOX_H <= h <= MAX_BOX_H
        color = (0,255,0) if in_range else (0,0,255)
        cv2.rectangle(img0, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img0, f"{tid[:4]}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 8. 显示最终动作
    cv2.putText(img0, f"Action:{suggestion}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return img0, suggestion

# ───────── CLI TEST ───────────────────────────────────────
if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv)>1 else '0'
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    while True:
        ret,frame = cap.read()
        if not ret: break
        out,act = detect_image(frame)
        cv2.imshow('Traffic Light', out)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()
