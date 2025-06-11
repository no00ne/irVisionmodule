#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_zebra.py  ·  Zebra-crossing detector
=========================================================
新增：
  • 每条轨迹挂 8D 常速 KalmanFilter → 漏检 ≤ MAX_LOST 帧仍保持同 ID
  • IoU + 匈牙利全局匹配，匹配成功后使用测量更新 KF
  • 其余逻辑（stop_until 1 s、可视化）保持不变
依赖：
  pip install filterpy
"""
import argparse, time, uuid, cv2, torch, numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter                       # ← 新依赖

from ZebraCrossing.models.common import DetectMultiBackend
from ZebraCrossing.utils.datasets import LoadStreams, LoadImages
from ZebraCrossing.utils.general import (check_requirements, non_max_suppression,
                                         scale_coords)
from ZebraCrossing.utils.plots import Annotator
from ZebraCrossing.utils.torch_utils import select_device

# ───────── CONSTANTS ──────────────────────────────────────
ZEBRA_CLASS_ID  = 0
CONF_THRES      = 0.25
IOU_THRES       = 0.45
IOU_MATCH_THRES = 0.3
REGION_FRAC     = 0.10
STOP_DURATION   = 1.0
MAX_LOST        = 10                       # 漏检帧上限

# ───────── GLOBAL STATE ───────────────────────────────────
tracked    = {}                            # id -> {box, kf, stopped, lost}
stop_until = 0.0                           # 停车计时器

# ───────── KALMAN INITIALISER ─────────────────────────────
def _init_kf(box):
    """8D 常速 Kalman: x=[cx,cy,w,h,vx,vy,vw,vh]"""
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    w,  h  = box[2]-box[0], box[3]-box[1]
    kf = KalmanFilter(dim_x=8, dim_z=4)
    kf.F = np.eye(8);  kf.F[:4,4:] = np.eye(4)   # Δt=1
    kf.H = np.hstack([np.eye(4), np.zeros((4,4))])
    kf.P *= 100.;  kf.R *= 10.;  kf.Q *= 0.01
    kf.x[:4,0] = [cx,cy,w,h]
    return kf

def _predict_tracks():
    """对所有现存轨迹做一步 Kalman 预测并更新 box、lost"""
    for info in tracked.values():
        info['kf'].predict()
        info['lost'] += 1
        cx,cy,w,h = info['kf'].x[:4].flatten()
        info['box'] = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

# ───────── IoU ────────────────────────────────────────────
def compute_iou(b1, b2):
    xA, yA = max(b1[0],b2[0]), max(b1[1],b2[1])
    xB, yB = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter  = max(0,xB-xA)*max(0,yB-yA)
    area1  = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2  = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (area1+area2-inter+1e-6)

# ───────── TRACKER ────────────────────────────────────────
def update_tracking(detections, frame_w, frame_h):
    """
    detections: [{'box':[x1,y1,x2,y2]}]
    返回 triggered_ids, region_low, region_high, x_mid
    """
    global tracked
    region_low  = int(frame_h*(0.7-REGION_FRAC))
    region_high = int(frame_h*(0.7+REGION_FRAC))
    x_mid       = frame_w//2

    # 0) Kalman 预测 & lost+1
    _predict_tracks()

    # 1) IoU 矩阵
    tids = list(tracked.keys())
    T, D = len(tids), len(detections)
    iou_mat = np.zeros((T,D),dtype=np.float32)
    for i, tid in enumerate(tids):
        boxA = tracked[tid]['box']
        for j, det in enumerate(detections):
            iou_mat[i,j] = compute_iou(boxA, det['box'])

    rows, cols = linear_sum_assignment(1-iou_mat) if T and D else ([],[])
    matched_det = set();  new_tracked = {}

    # 2) 成功匹配
    for r,c in zip(rows,cols):
        if iou_mat[r,c] >= IOU_MATCH_THRES:
            tid  = tids[r]; det = detections[c]; box = det['box']
            kf   = tracked[tid]['kf']
            cx   = (box[0]+box[2])/2; cy=(box[1]+box[3])/2
            w,h  = box[2]-box[0], box[3]-box[1]
            kf.update([cx,cy,w,h])          # Kalman 更新
            new_tracked[tid] = {'box':box,'kf':kf,
                                'stopped':tracked[tid]['stopped'],'lost':0}
            matched_det.add(c)

    # 3) 未匹配旧轨迹：若 lost ≤ MAX_LOST 继续保留
    for tid,info in tracked.items():
        if tid not in new_tracked and info['lost']<=MAX_LOST:
            new_tracked[tid] = info

    # 4) 创建新轨迹
    for j,det in enumerate(detections):
        if j not in matched_det:
            nid = str(uuid.uuid4())
            new_tracked[nid] = {'box':det['box'],'kf':_init_kf(det['box']),
                                'stopped':False,'lost':0}

    tracked = new_tracked

    # 5) 判定新触发
    triggered = []
    for tid,info in tracked.items():
        if not info['stopped']:
            x1,y1,x2,_ = info['box']
            if (x1 < x_mid < x2) and (region_low <= y1 <= region_high):
                info['stopped'] = True
                triggered.append(tid)
    return triggered, region_low, region_high, x_mid

# ───────── FRAME PROCESS ─────────────────────────────────
@torch.no_grad()
def detect_zebra_frame(img_bgr, model, device):
    """
    返回: im0, suggestion('stop'/'go'), tracked_info, (region_low,region_high)
    """
    global stop_until
    now = time.monotonic()

    # --- 推理 ---
    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    img_t  = torch.from_numpy(img_np).to(device).permute(2,0,1).unsqueeze(0)
    pred   = non_max_suppression(model(img_t), CONF_THRES, IOU_THRES)[0]

    im0 = img_bgr.copy()
    h,w = im0.shape[:2]

    # --- 收集斑马线框 ---
    detections=[]
    if pred is not None and len(pred):
        pred[:,:4]=scale_coords(img_t.shape[2:],pred[:,:4],im0.shape).round()
        for *xyxy,conf,cls in pred:
            if int(cls)==ZEBRA_CLASS_ID:
                detections.append({'box':[int(x) for x in xyxy]})

    # --- 跟踪 ---
    triggered, region_low, region_high, x_mid = update_tracking(detections,w,h)
    if triggered: stop_until = now + STOP_DURATION
    suggestion   = 'stop' if now < stop_until else 'go'

    # --- 绘制 ---
    annot = Annotator(im0, line_width=2)
    cv2.line(im0,(x_mid,0),(x_mid,h),(255,255,0),2)
    cv2.rectangle(im0,(0,region_low),(w,region_high),(255,0,0),1)

    tracked_info=[]
    for tid,info in tracked.items():
        x1,y1,x2,y2 = map(int,info['box'])
        in_line = (x1<x_mid<x2) and (region_low<=y1<=region_high)
        color   = (0,0,255) if in_line and suggestion=='stop' else (0,255,0)
        cv2.rectangle(im0,(x1,y1),(x2,y2),color,2)
        if x1<x_mid<x2:
            cv2.line(im0,(x_mid,y1),(x_mid,min(y2,h-1)),(255,255,0),2)
        cv2.putText(im0,tid[:4],(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
        tracked_info.append({'id':tid,'box':info['box'],'in_range':in_line})

    if suggestion=='stop':
        remain = stop_until-now
        cv2.putText(im0,f"STOP {remain:.1f}s",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    else:
        cv2.putText(im0,"GO",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    return im0,suggestion,tracked_info,(region_low,region_high)

# ───────── CLI ───────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='runs/train/exp2/weights/best.pt')
    parser.add_argument('--source', type=str,default='0')
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img',action='store_true')
    opt=parser.parse_args()

    check_requirements()
    device = select_device(opt.device)
    model  = DetectMultiBackend(opt.weights,device=device); model.eval()

    ds = (LoadStreams if opt.source.isnumeric() else LoadImages)(
            opt.source,img_size=(640,640),stride=model.stride,auto=model.pt)

    for _,_,im0s,_,_ in ds:
        frame = im0s if isinstance(im0s,np.ndarray) else im0s[0]
        im0, sugg, _, _ = detect_zebra_frame(frame, model, device)
        cv2.imshow('Zebra', im0)
        if cv2.waitKey(1)&0xFF==27: break
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
