#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_zebra.py
Modified for CDNet-YOLOv5: zebra crossing detection with single-stop-per-line tracking
Environment: aarch64, ROS Noetic, Agilex WS
"""
import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadImages
from utils.general import (
    check_img_size, check_requirements, increment_path,
    non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh, print_args
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

# --- Class ID for zebra crossing in your model ---
ZEBRA_CLASS_ID = 0  # adjust to your model's zebra class index

# --- Global tracker state: id -> {box, stopped_flag} ---
tracked = {}

# --- IoU calculation for matching ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(boxA_area + boxB_area - inter + 1e-6)

# --- Tracking update: match detections to existing, detect midline crossing once ---
def update_tracking(detections, img_height, iou_thres=0.3, region_frac=0.1):
    global tracked
    new_tracked = {}
    region_low = img_height * (0.5 - region_frac)
    region_high = img_height * (0.5 + region_frac)
    triggered = []
    # Match or create IDs
    for det in detections:
        box = det['box']
        best_id, best_iou = None, 0.0
        for tid, info in tracked.items():
            iou = compute_iou(info['box'], box)
            if iou > best_iou:
                best_id, best_iou = tid, iou
        if best_iou > iou_thres:
            # existing track
            new_tracked[best_id] = {'box': box, 'stopped': tracked[best_id]['stopped']}
        else:
            # new zebra line track
            nid = str(uuid.uuid4())
            new_tracked[nid] = {'box': box, 'stopped': False}
    tracked = new_tracked
    # Check for midline crossing
    for tid, info in tracked.items():
        if not info['stopped']:
            y_top = info['box'][1]
            if region_low <= y_top <= region_high:
                tracked[tid]['stopped'] = True
                triggered.append(tid)
    return triggered

@torch.no_grad()
def run(weights, source, imgsz, conf_thres, iou_thres, device, view_img):
    # Initialize model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Data loader
    is_file = Path(source).suffix[1:].lower() in IMG_FORMATS + VID_FORMATS
    is_stream = source.isnumeric() or source.endswith('.txt')
    if is_stream:
        view_img = True
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Warmup
    model.warmup(imgsz=(1, 3, *imgsz))

    # Inference loop
    for path, img, im0s, vid_cap, s in dataset:
        # Preprocess
        t1 = time_sync()
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        t2 = time_sync()

        # Inference
        pred = model(img_tensor, augment=False, visualize=False)
        t3 = time_sync()

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        t4 = time_sync()

        # Prepare image for drawing
        im0 = im0s.copy() if not isinstance(im0s, list) else im0s[0]
        annotator = Annotator(im0, line_width=2, example=str(names))

        # Collect zebra detections
        detections = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    if int(cls) == ZEBRA_CLASS_ID:
                        box = [int(x) for x in xyxy]
                        detections.append({'box': box})
                        annotator.box_label(xyxy, 'ZebraCrossing', color=colors(ZEBRA_CLASS_ID, True))

        # Update tracking & get new crosses
        triggered = update_tracking(detections, im0.shape[0], iou_thres)

        # If any new zebra crosses midline, stop for 1s
        suggestion = 'go'
        if triggered:
            suggestion = 'stop'
            time.sleep(1)

        # Display suggestion
        if view_img:
            cv2.putText(im0, f"SUGGESTION: {suggestion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(str(path), annotator.result())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if view_img:
        cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp2/weights/best.pt', help='model path')
    parser.add_argument('--source', type=str, default='0', help='camera index or file path')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    opt.imgsz = tuple(opt.imgsz)
    print_args(FILE.stem, opt)
    return opt


def main():
    opt = parse_opt()
    check_requirements()
    run(
        weights=opt.weights,
        source=opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device,
        view_img=opt.view_img
    )

if __name__ == '__main__':
    main()
