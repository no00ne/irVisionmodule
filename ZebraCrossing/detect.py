#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_zebra.py
Standalone zebra crossing detection with single-stop-per-line tracking.
Draws detection boxes, IDs, midline region, in-range status, and returns
processed frame, suggestion, tracked info, and region boundaries.
Environment: aarch64, ROS Noetic, Agilex WS
"""
import argparse
import sys
import time
import uuid
from pathlib import Path

import cv2
import torch
import numpy as np
from ZebraCrossing.models.common import DetectMultiBackend
from ZebraCrossing.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadImages
from ZebraCrossing.utils.general import (
    check_img_size, check_requirements,
    non_max_suppression, scale_coords, print_args
)
from ZebraCrossing.utils.plots import Annotator, colors
from ZebraCrossing.utils.torch_utils import select_device, time_sync

# ----- PARAMETERS -----
ZEBRA_CLASS_ID = 0    # class index for zebra crossing
IOU_THRESH = 0.3      # IoU threshold for tracking
REGION_FRAC = 0.1     # fraction around midline to trigger stop

# Global tracker state: id -> {'box': [x1,y1,x2,y2], 'stopped': bool}
tracked = {}

# ----- UTILS -----
def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

# match detections to tracked and detect midline crossing once
def update_tracking(detections, frame_h):
    global tracked
    new_tracked = {}
    region_low = int(frame_h * (0.5 - REGION_FRAC))
    region_high = int(frame_h * (0.5 + REGION_FRAC))
    triggered = []
    # match or create IDs
    for det in detections:
        box = det['box']
        best_id, best_iou = None, 0.0
        for tid, info in tracked.items():
            iou = compute_iou(info['box'], box)
            if iou > best_iou:
                best_id, best_iou = tid, iou
        if best_iou > IOU_THRESH:
            new_tracked[best_id] = {'box': box, 'stopped': tracked[best_id]['stopped']}
        else:
            nid = str(uuid.uuid4())
            new_tracked[nid] = {'box': box, 'stopped': False}
    tracked = new_tracked
    # detect first crossing per ID
    for tid, info in tracked.items():
        if not info['stopped']:
            y_top = info['box'][1]
            if region_low <= y_top <= region_high:
                tracked[tid]['stopped'] = True
                triggered.append(tid)
    return triggered, region_low, region_high

# ----- MAIN DETECTION FUNCTION -----
@torch.no_grad()
def detect_zebra_frame(img_bgr, model, device, conf_thres=0.25, iou_thres=0.45, view_img=False):
    """
    Process one frame, return:
      im0: annotated image
      suggestion: 'stop' or 'go'
      tracked_info: list of {'id', 'box', 'in_range'}
      region: (region_low, region_high)
    """
    # prepare input
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).to(device).permute(2, 0, 1).unsqueeze(0)

    # inference
    pred_raw = model(img_tensor, augment=False, visualize=False)
    preds = non_max_suppression(pred_raw, conf_thres, iou_thres)
    pred = preds[0]  # batch size 1

    # setup annotations
    im0 = img_bgr.copy()
    h, w = im0.shape[:2]
    triggered, region_low, region_high = update_tracking([], h)
    # draw midline region
    cv2.rectangle(im0, (0, region_low), (w, region_high), (255, 0, 0), 2)
    annotator = Annotator(im0, line_width=2, example='Zebra')

    # collect zebra detections
    detections = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], im0.shape).round()
        for *xyxy, conf, cls in pred:
            if int(cls) == ZEBRA_CLASS_ID:
                box = [int(x) for x in xyxy]
                detections.append({'box': box})
                annotator.box_label(xyxy, 'ZebraCrossing', color=colors(ZEBRA_CLASS_ID, True))

    # update tracking with real detections
    triggered, region_low, region_high = update_tracking(detections, h)
    suggestion = 'stop' if triggered else 'go'
    if suggestion == 'stop':
        time.sleep(1)

    # draw tracked boxes with ID and in-range flag
    tracked_info = []
    for tid, info in tracked.items():
        x1, y1, x2, y2 = info['box']
        in_range = (region_low <= y1 <= region_high)
        color = (0, 255, 0) if in_range else (0, 0, 255)
        cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
        txt = f"{tid[:4]}:{'IN' if in_range else 'OUT'}"
        cv2.putText(im0, txt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        tracked_info.append({'id': tid, 'box': info['box'], 'in_range': in_range})

    # show suggestion
    if view_img:
        cv2.putText(im0, f"SUGGESTION: {suggestion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Zebra Crossing Detection', annotator.result())
        cv2.waitKey(1)

    return im0, suggestion, tracked_info, (region_low, region_high)

# ----- SCRIPT ENTRYPOINT -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp2/weights/best.pt')
    parser.add_argument('--source', type=str, default='0', help='camera index or file path')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='')
    parser.add_argument('--view-img', action='store_true')
    opt = parser.parse_args()

    check_requirements()
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device)
    model.eval()

    # data loader
    is_stream = opt.source.isnumeric() or opt.source.endswith('.txt')
    if is_stream:
        dataset = LoadStreams(opt.source, img_size=(640,640), stride=model.stride, auto=model.pt)
    else:
        dataset = LoadImages(opt.source, img_size=(640,640), stride=model.stride, auto=model.pt)

    for path, img, im0s, vid_cap, s in dataset:
        im0 = im0s.copy() if not isinstance(im0s, list) else im0s[0]
        im0, suggestion, tracked_info, region = detect_zebra_frame(
            im0, model, device, opt.conf_thres, opt.iou_thres, opt.view_img
        )
        cv2.imshow('Zebra Crossing Detection', im0)
    if opt.view_img:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
