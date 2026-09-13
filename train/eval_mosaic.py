"""Score the mosaic-on and mosaic-off models on both things that matter.

Two dimensions, because they can disagree and the whole question is whether
they do:

  1. the held-out test split -- ordinary precision/recall/F1 at the thresholds
     in aicam's [thresholds]. This is what mosaic is conventionally expected to
     be neutral-to-slightly-negative on for a fixed-camera dataset.

  2. the 31 driveway frames aicam actually flagged on 2026-09-04, every one of
     them dappled morning sunlight on empty asphalt with no vehicle present.
     These are ground-truth negatives, so any detection is a false positive.
     A model that scores well on (1) and badly on (2) is the failure mode we
     already shipped once.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/claw/aicam")
from ultralytics_detection import ONNXRuntimeUltralyticsObjectDetection  # noqa: E402

DEFAULT_THRESHOLD = 0.70
THRESHOLDS = {"package": 0.80, "vehicle": 0.70}


def build(onnx, labels, size):
    return ONNXRuntimeUltralyticsObjectDetection(
        {
            "onnx": onnx,
            "width": str(size[0]),
            "height": str(size[1]),
            "channels": "3",
            "prob_threshold": "0.10",
        },
        labels,
    )


def predict(model, img, size):
    resized = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
    return model.predict_image(resized)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def score_split(model, root, labels, size):
    tp = dict((c, 0) for c in labels)
    fp = dict((c, 0) for c in labels)
    fn = dict((c, 0) for c in labels)
    for path in sorted(glob.glob(os.path.join(root, "images", "*.jpg"))):
        img = cv2.imread(path)
        if img is None:
            continue
        dets = [
            p
            for p in predict(model, img, size)
            if p["probability"] >= THRESHOLDS.get(p["tagName"], DEFAULT_THRESHOLD)
        ]
        stem = os.path.basename(path).rsplit(".", 1)[0]
        lbl = os.path.join(root, "labels", stem + ".txt")
        truth = []
        if os.path.exists(lbl):
            for line in open(lbl):
                f = line.split()
                if len(f) >= 5:
                    cx, cy, w, h = [float(v) for v in f[1:5]]
                    truth.append((labels[int(f[0])], [cx - w / 2, cy - h / 2, w, h]))
        matched = set()
        for p in sorted(dets, key=lambda d: -d["probability"]):
            b = p["boundingBox"]
            box = [b["left"], b["top"], b["width"], b["height"]]
            best, best_i = 0.0, None
            for i, (cls, gt) in enumerate(truth):
                if i in matched or cls != p["tagName"]:
                    continue
                s = iou(box, gt)
                if s > best:
                    best, best_i = s, i
            if best >= 0.5:
                matched.add(best_i)
                tp[p["tagName"]] += 1
            else:
                fp[p["tagName"]] += 1
        for i, (cls, _) in enumerate(truth):
            if i not in matched:
                fn[cls] += 1
    return tp, fp, fn


def score_negatives(model, frames, size):
    """Every detection here is a false positive: no vehicle is present."""
    fired = 0
    scores = []
    for path in frames:
        img = cv2.imread(path)
        if img is None:
            continue
        best = 0.0
        for p in predict(model, img, size):
            if p["tagName"] == "vehicle":
                best = max(best, p["probability"])
        scores.append(best)
        if best >= THRESHOLDS["vehicle"]:
            fired += 1
    return fired, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--negatives", required=True, help="glob of known-empty frames")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--size", default="608x608")
    ap.add_argument("--models", nargs="+", required=True, help="tag=path ...")
    a = ap.parse_args()

    w, _, h = a.size.partition("x")
    size = (int(w), int(h))
    labels = [x.strip() for x in a.labels.split(",") if x.strip()]
    root = os.path.join(a.dataset, a.split)
    frames = [
        f
        for f in sorted(glob.glob(a.negatives))
        if "annotated" not in f and "prior" not in f
    ]
    print("test split: %s" % root)
    print("negatives:  %d known-empty driveway frames" % len(frames))
    print("thresholds: %s (default %.2f)\n" % (THRESHOLDS, DEFAULT_THRESHOLD))

    rows = []
    for spec in a.models:
        tag, _, path = spec.partition("=")
        model = build(path, labels, size)
        tp, fp, fn = score_split(model, root, labels, size)
        fired, scores = score_negatives(model, frames, size)
        rows.append((tag, tp, fp, fn, fired, scores))

    print("%-12s %-8s %8s %8s %8s" % ("model", "class", "P", "R", "F1"))
    for tag, tp, fp, fn, _, _ in rows:
        for c in labels + ["ALL"]:
            if c == "ALL":
                t, f, m = sum(tp.values()), sum(fp.values()), sum(fn.values())
            else:
                t, f, m = tp[c], fp[c], fn[c]
            p = t / float(t + f) if t + f else 0.0
            r = t / float(t + m) if t + m else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            print("%-12s %-8s %8.3f %8.3f %8.3f" % (tag, c, p, r, f1))
        print()

    print("=== driveway false positives (no vehicle present in any frame) ===")
    print("%-12s %10s %10s %10s %10s" % ("model", "fires>=0.70", "rate", "max", "mean"))
    for tag, _, _, _, fired, scores in rows:
        s = np.array(scores) if scores else np.array([0.0])
        print(
            "%-12s %10s %9.0f%% %10.2f %10.2f"
            % (tag, "%d/%d" % (fired, len(scores)), 100.0 * fired / len(scores), s.max(), s.mean())
        )


if __name__ == "__main__":
    main()
