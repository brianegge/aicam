"""Old YOLOv4 vs new YOLO11 on the same held-out images, same preprocessing."""
import glob, os, sys, time
import cv2, numpy as np
sys.path.insert(0, "/Users/claw/aicam")
from yolov4_detection import ONNXRuntimeYolov4ObjectDetection
from ultralytics_detection import ONNXRuntimeUltralyticsObjectDetection

LABELS = ["package", "vehicle"]
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
ROOT = "/Users/claw/train/packages-vehicles2-v11/%s" % SPLIT
# operational thresholds from aicam config: [thresholds] package=0.80, default 0.70
OPER = {"package": 0.80, "vehicle": 0.70}

def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0

def load_gt(label_path):
    gt = []
    if not os.path.exists(label_path):
        return gt
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(parts[0]); cx, cy, w, h = map(float, parts[1:5])
        gt.append((c, [cx - w / 2, cy - h / 2, w, h]))
    return gt

def evaluate(model, name, thresholds):
    tp = {c: 0 for c in LABELS}; fp = {c: 0 for c in LABELS}; fn = {c: 0 for c in LABELS}
    total_ms = 0.0; n = 0
    for img_path in sorted(glob.glob(os.path.join(ROOT, "images", "*.jpg"))):
        img = cv2.imread(img_path)
        if img is None:
            continue
        # exactly what camera.py hands a model
        resized = cv2.cvtColor(cv2.resize(img, (608, 608)), cv2.COLOR_BGR2RGB)
        t = time.perf_counter()
        preds = model.predict_image(resized)
        total_ms += (time.perf_counter() - t) * 1000; n += 1
        preds = [p for p in preds if p["probability"] >= thresholds[p["tagName"]]]
        base = os.path.basename(img_path).rsplit(".", 1)[0]
        gt = load_gt(os.path.join(ROOT, "labels", base + ".txt"))
        used = set()
        for p in sorted(preds, key=lambda x: -x["probability"]):
            bb = p["boundingBox"]; box = [bb["left"], bb["top"], bb["width"], bb["height"]]
            best, best_i = 0.0, None
            for i, (gc, gbox) in enumerate(gt):
                if i in used or LABELS[gc] != p["tagName"]:
                    continue
                v = iou(box, gbox)
                if v > best:
                    best, best_i = v, i
            if best >= 0.5:
                used.add(best_i); tp[p["tagName"]] += 1
            else:
                fp[p["tagName"]] += 1
        for i, (gc, _) in enumerate(gt):
            if i not in used:
                fn[LABELS[gc]] += 1
    print("\n== %s == (%d images, %.1f ms/frame)" % (name, n, total_ms / max(n, 1)))
    print("%-9s %6s %6s %6s %9s %8s %6s" % ("class", "TP", "FP", "FN", "precision", "recall", "F1"))
    for c in LABELS:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        print("%-9s %6d %6d %6d %9.3f %8.3f %6.3f" % (c, tp[c], fp[c], fn[c], p, r, f))
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    print("%-9s %6d %6d %6d %9.3f %8.3f %6.3f" % ("ALL", TP, FP, FN, p, r, f))
    return f

old_cfg = {"onnx": "/Users/claw/aicam-models/vehicles_yolov4.onnx", "width": "608",
           "height": "608", "channels": "3", "prob_threshold": "0.10"}
new_cfg = {"onnx": "/Users/claw/train/runs/pv11-yolo11s/weights/best.onnx", "width": "608",
           "height": "608", "channels": "3", "prob_threshold": "0.10"}

for label, thresholds in (("operational thresholds (package>=0.80, vehicle>=0.70)", OPER),
                          ("common threshold 0.40", {"package": 0.4, "vehicle": 0.4})):
    print("\n" + "=" * 68); print(label); print("=" * 68)
    evaluate(ONNXRuntimeYolov4ObjectDetection(old_cfg, LABELS), "OLD  yolov4-tiny (in production)", thresholds)
    evaluate(ONNXRuntimeUltralyticsObjectDetection(new_cfg, LABELS), "NEW  yolo11s (just trained)", thresholds)
