"""Replay saved capture frames through two vehicle models, the way detect.py runs them.

Every raw frame in ~/aicam-data/capture from a vehicle_check camera goes through
camera.py's colour path (resize, BGR->RGB) and each model's vehicle boxes are
written out as CSV for scoring. The test split cannot answer the night question
-- it holds no IR glare false positives -- but the archive is full of them.

    replay_vehicle.py OLD.onnx NEW.onnx > replay.csv
"""
import csv
import glob
import json
import os
import re
import sys

import cv2

sys.path.insert(0, "/Users/claw/aicam")
from compare_models import build_model  # noqa: E402

CAMS = {"driveway", "front_entry", "front entry", "shed", "garage-r", "garage-l", "peach_tree", "peach tree"}
LABELS = ["package", "vehicle"]
NAME = re.compile(r"^(\d{6})-(.+?)-([a-z_]+?)(-prior)?\.jpg$")


def main():
    models = [build_model(p, "ultralytics", LABELS, (608, 608)) for p in sys.argv[1:3]]
    out = csv.writer(sys.stdout)
    out.writerow(["file", "cam", "model", "prob", "left", "top", "width", "height"])
    for path in sorted(glob.glob(os.path.expanduser("~/aicam-data/capture/2026*/*.jpg"))):
        m = NAME.match(os.path.basename(path))
        if not m or m.group(2) not in CAMS:
            continue
        image = cv2.imread(path)
        if image is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(image, (608, 608)), cv2.COLOR_BGR2RGB)
        rel = os.path.relpath(path, os.path.expanduser("~/aicam-data/capture"))
        for i, model in enumerate(models):
            hits = [p for p in model.predict_image(rgb) if p["tagName"] == "vehicle"]
            for p in hits or [{"probability": 0, "boundingBox": {}}]:
                b = p["boundingBox"]
                out.writerow([rel, m.group(2), "old" if i == 0 else "new", round(p["probability"], 4),
                              b.get("left"), b.get("top"), b.get("width"), b.get("height")])


if __name__ == "__main__":
    main()
