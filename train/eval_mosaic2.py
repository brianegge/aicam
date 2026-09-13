"""Count only the false positives aicam would actually act on.

The driveway frames contain real cars on the street at the top of frame.
aicam discards those via the camera's `road_line`: a detection whose centre
is above the line is renamed vehicle_road and ignored. Counting every
detection therefore mixes genuine street traffic in with the shadow false
positives and measures the wrong thing.

driveway road_line = 0:0.5, 1.0:0.2
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/claw/aicam")
from ultralytics_detection import ONNXRuntimeUltralyticsObjectDetection  # noqa: E402

VEHICLE_THRESHOLD = 0.70
ROAD_LINE = [(0.0, 0.5), (1.0, 0.2)]


def road_y_at(x):
    for i in range(len(ROAD_LINE) - 1):
        x0, y0 = ROAD_LINE[i]
        x1, y1 = ROAD_LINE[i + 1]
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 != x0 else y0
    return ROAD_LINE[-1][1]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", required=True)
    ap.add_argument("--labels", default="package,vehicle")
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()

    labels = [x.strip() for x in a.labels.split(",") if x.strip()]
    frames = [
        f
        for f in sorted(glob.glob(a.negatives))
        if "annotated" not in f and "prior" not in f
    ]
    print("%d driveway frames, no vehicle on the property in any of them" % len(frames))
    print("counting only detections BELOW the road line (what aicam acts on)\n")
    print("%-12s %-6s %14s %8s %8s %8s" % ("model", "size", "frames-firing", "rate", "max", "mean"))

    for spec in a.models:
        tag, _, rest = spec.partition("=")
        path, _, sz = rest.partition("@")
        w, _, h = (sz or "608x608").partition("x")
        size = (int(w), int(h))
        model = build(path, labels, size)
        fired = 0
        best_scores = []
        for f in frames:
            img = cv2.imread(f)
            if img is None:
                continue
            resized = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2RGB)
            best = 0.0
            for p in model.predict_image(resized):
                if p["tagName"] != "vehicle":
                    continue
                b = p["boundingBox"]
                cx = b["left"] + b["width"] / 2.0
                cy = b["top"] + b["height"] / 2.0
                if cy < road_y_at(cx):
                    continue  # street traffic -- aicam ignores this
                best = max(best, p["probability"])
            best_scores.append(best)
            if best >= VEHICLE_THRESHOLD:
                fired += 1
        s = np.array(best_scores) if best_scores else np.array([0.0])
        print(
            "%-12s %-6s %14s %7.0f%% %8.2f %8.2f"
            % (tag, "%dx%d" % size, "%d/%d" % (fired, len(s)), 100.0 * fired / len(s), s.max(), s.mean())
        )


if __name__ == "__main__":
    main()
