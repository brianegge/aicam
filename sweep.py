#!/usr/bin/env python3
"""Per-class acquire/hold threshold sweep for the ipcams detector.

aicam runs hysteresis already: a class-specific *acquire* threshold creates a
track, and a single global HOLD_PROBABILITY (0.15) lets an under-threshold
detection sustain one that already exists. This measures both ends per class
rather than inheriting values tuned for yolov4's confidence distribution.

Acquire is measurable directly from the split. Hold is not -- the split has no
temporal sequences, so "the same deer at 0.60 in the next frame" cannot be
observed. What the split *does* give is the low tail of genuine detections and
the score distribution of false positives, and the hold floor has to sit
between them: low enough to re-match a real object having a bad frame, high
enough that noise cannot sustain a track.
"""
import argparse, collections, glob, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser("~/aicam"))
MODEL_SIZE = 608


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def ground_truth(path, labels):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        f = line.split()
        if len(f) < 5:
            continue
        cx, cy, w, h = [float(v) for v in f[1:5]]
        out.append((labels[int(f[0])], [cx - w / 2, cy - h / 2, w, h]))
    return out


def collect(model, root, labels, floor=0.02):
    """Every prediction, tagged as matching a ground-truth box or not."""
    preds = collections.defaultdict(list)     # class -> [(score, is_tp)]
    truth_n = collections.Counter()
    for image_path in sorted(glob.glob(os.path.join(root, "images", "*.jpg"))):
        image = cv2.imread(image_path)
        if image is None:
            continue
        resized = cv2.cvtColor(cv2.resize(image, (MODEL_SIZE, MODEL_SIZE)),
                               cv2.COLOR_BGR2RGB)
        stem = os.path.basename(image_path).rsplit(".", 1)[0]
        truth = ground_truth(os.path.join(root, "labels", stem + ".txt"), labels)
        for c, _ in truth:
            truth_n[c] += 1
        matched = set()
        for p in sorted(model.predict_image(resized),
                        key=lambda x: -x["probability"]):
            if p["probability"] < floor:
                continue
            b = p["boundingBox"]
            box = [b["left"], b["top"], b["width"], b["height"]]
            best, best_i = 0.0, None
            for i, (c, gt) in enumerate(truth):
                if i in matched or c != p["tagName"]:
                    continue
                s = iou(box, gt)
                if s > best:
                    best, best_i = s, i
            if best >= 0.5:
                matched.add(best_i)
                preds[p["tagName"]].append((p["probability"], True))
            else:
                preds[p["tagName"]].append((p["probability"], False))
    return preds, truth_n


def sweep(preds, truth_n, classes, n_images):
    print("\n%-9s %5s %6s  %s" % ("class", "n_gt", "best", "acquire sweep (threshold: P/R/F1)"))
    chosen = {}
    for c in classes:
        rows = preds.get(c, [])
        n = truth_n.get(c, 0)
        if not n:
            print("%-9s %5d  %s" % (c, 0, "no ground truth in split"))
            continue
        best = None
        line = []
        for t in [x / 100.0 for x in range(5, 100, 5)]:
            tp = sum(1 for s, ok in rows if s >= t and ok)
            fp = sum(1 for s, ok in rows if s >= t and not ok)
            fn = n - tp
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / n
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if best is None or f1 > best[3]:
                best = (t, prec, rec, f1)
            if t in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                line.append("%.2f:%.2f/%.2f/%.2f" % (t, prec, rec, f1))
        chosen[c] = best
        print("%-9s %5d  %.2f  %s" % (c, n, best[0], "  ".join(line)))

    print("\n--- where genuine detections live (true positives) ---")
    print("%-9s %5s %6s %6s %6s %6s   %s" % (
        "class", "n_tp", "min", "p05", "p25", "median", "hold floor implied"))
    for c in classes:
        tps = sorted(s for s, ok in preds.get(c, []) if ok)
        if not tps:
            continue
        q = lambda p: tps[max(0, min(len(tps) - 1, int(len(tps) * p)))]
        print("%-9s %5d %6.2f %6.2f %6.2f %6.2f" % (
            c, len(tps), tps[0], q(0.05), q(0.25), q(0.50)))

    print("\n--- where false positives live ---")
    print("%-9s %5s %6s %6s %6s   %s" % (
        "class", "n_fp", "max", "p95", "median", "fp per image above 0.15 / 0.25 / 0.40"))
    for c in classes:
        fps = sorted((s for s, ok in preds.get(c, []) if not ok), reverse=True)
        if not fps:
            print("%-9s %5d %6s %6s %6s   %.3f / %.3f / %.3f" % (c, 0, "-", "-", "-", 0, 0, 0))
            continue
        q = lambda p: fps[max(0, min(len(fps) - 1, int(len(fps) * p)))]
        rates = [sum(1 for s in fps if s >= t) / float(n_images) for t in (0.15, 0.25, 0.40)]
        print("%-9s %5d %6.2f %6.2f %6.2f   %.3f / %.3f / %.3f" % (
            c, len(fps), fps[0], q(0.05), q(0.50), *rates))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--backend", default="ultralytics")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()

    labels = [x.strip() for x in a.labels.split(",") if x.strip()]
    cfg = {"onnx": a.onnx, "width": str(MODEL_SIZE), "height": str(MODEL_SIZE),
           "channels": "3", "prob_threshold": "0.02"}
    if a.backend == "ultralytics":
        from ultralytics_detection import ONNXRuntimeUltralyticsObjectDetection as M
    else:
        from yolov4_detection import ONNXRuntimeYolov4ObjectDetection as M
    model = M(cfg, labels)

    root = os.path.join(os.path.expanduser(a.dataset), a.split)
    n_images = len(glob.glob(os.path.join(root, "images", "*.jpg")))
    print("model: %s" % a.onnx)
    print("split: %s (%d images)" % (root, n_images))
    preds, truth_n = collect(model, root, labels)
    sweep(preds, truth_n, labels, n_images)


if __name__ == "__main__":
    main()
