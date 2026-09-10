#!/usr/bin/env python3
"""Score two detection models against the same held-out split.

The mAP a trainer prints is measured at its own thresholds, not the ones in
`[thresholds]` that actually decide what aicam acts on, and it says nothing
about a model in a different architecture. This runs both models through the
same preprocessing camera.py uses and reports per-class precision/recall/F1 at
whatever thresholds you are really running.

    python compare_models.py \\
        --dataset ~/train/packages-vehicles2-v11 --split test \\
        --old vehicles_yolov4.onnx --old-backend yolov4 \\
        --new best.onnx --new-backend ultralytics \\
        --labels package,vehicle --threshold package=0.55 --threshold vehicle=0.70

Models with different input geometry are compared by pointing each at the
dataset version rendered for it. Both are stretches of the same source frames
and YOLO labels are normalised, so the ground truth is identical:

    ... --old-size 608x608 --old-dataset ~/train/packages-vehicles2-v11 \\
        --new-size 1088x608 --new-dataset ~/train/pv-v12-rect

The ipcams animal detector is not one model but two, routed by camera.py's
own rule (HSV hue sum == 0 -> the single-channel grey specialist, otherwise
the colour one). Pass --old-grey to score that pair as a single *system*
against a combined replacement, so both arms answer for the same images:

    ... --old ipcams_color_yolov4.onnx --old-grey ipcams_grey_yolov4.onnx \\
        --old-backend yolov4 --new ipcams_yolo11m.onnx

ms/frame covers preprocessing as well as inference, because the routed arm's
extra colourspace conversion is a real cost of running that arrangement.
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np

DEFAULT_SIZE = "608x608"


def parse_size(text):
    width, _, height = text.lower().partition("x")
    return int(width), int(height)


def build_model(onnx_path, backend, labels, size, channels=3):
    width, height = size
    config = {
        "onnx": onnx_path,
        "width": str(width),
        "height": str(height),
        "channels": str(channels),
        "prob_threshold": "0.10",
    }
    if backend == "ultralytics":
        from ultralytics_detection import ONNXRuntimeUltralyticsObjectDetection

        return ONNXRuntimeUltralyticsObjectDetection(config, labels)
    from yolov4_detection import ONNXRuntimeYolov4ObjectDetection

    return ONNXRuntimeYolov4ObjectDetection(config, labels)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2 = min(a[0] + a[2], b[0] + b[2])
    iy2 = min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        boxes.append((class_id, [cx - w / 2, cy - h / 2, w, h]))
    return boxes


def single_arm(model, size):
    """One model for every image -- camera.py's colour path."""

    def predict(image):
        return model.predict_image(
            cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB)
        )

    return predict


def routed_arm(color_model, grey_model, size, counts):
    """Two specialists routed exactly as camera.py:resize() routes them.

    The grey branch converts to single-channel *before* resizing and the
    colour branch converts *after*; that ordering is copied from camera.py so
    a model sees here what it sees live.
    """

    def predict(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if np.sum(hsv[:, :, 0]) == 0:
            counts["grey"] += 1
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return grey_model.predict_image(cv2.resize(grey, size))
        counts["colour"] += 1
        return color_model.predict_image(
            cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2RGB)
        )

    return predict


def evaluate(predict, name, root, labels, thresholds):
    tp = dict((c, 0) for c in labels)
    fp = dict((c, 0) for c in labels)
    fn = dict((c, 0) for c in labels)
    total_ms = 0.0
    count = 0

    for image_path in sorted(glob.glob(os.path.join(root, "images", "*.jpg"))):
        image = cv2.imread(image_path)
        if image is None:
            continue
        started = time.perf_counter()
        predictions = predict(image)
        total_ms += (time.perf_counter() - started) * 1000
        count += 1

        predictions = [
            p for p in predictions if p["probability"] >= thresholds[p["tagName"]]
        ]
        stem = os.path.basename(image_path).rsplit(".", 1)[0]
        truth = load_ground_truth(os.path.join(root, "labels", stem + ".txt"))

        matched = set()
        for p in sorted(predictions, key=lambda x: -x["probability"]):
            box = [
                p["boundingBox"]["left"],
                p["boundingBox"]["top"],
                p["boundingBox"]["width"],
                p["boundingBox"]["height"],
            ]
            best_score, best_index = 0.0, None
            for i, (class_id, gt_box) in enumerate(truth):
                if i in matched or labels[class_id] != p["tagName"]:
                    continue
                score = iou(box, gt_box)
                if score > best_score:
                    best_score, best_index = score, i
            if best_score >= 0.5:
                matched.add(best_index)
                tp[p["tagName"]] += 1
            else:
                fp[p["tagName"]] += 1
        for i, (class_id, _) in enumerate(truth):
            if i not in matched:
                fn[labels[class_id]] += 1

    print("\n== %s == (%d images, %.1f ms/frame)" % (name, count, total_ms / max(count, 1)))
    print("%-10s %6s %6s %6s %10s %8s %7s" % ("class", "TP", "FP", "FN", "precision", "recall", "F1"))
    for c in labels + ["ALL"]:
        if c == "ALL":
            t, f, m = sum(tp.values()), sum(fp.values()), sum(fn.values())
        else:
            t, f, m = tp[c], fp[c], fn[c]
        precision = t / float(t + f) if t + f else 0.0
        recall = t / float(t + m) if t + m else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print("%-10s %6d %6d %6d %10.3f %8.3f %7.3f" % (c, t, f, m, precision, recall, f1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset root (contains the split dirs)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--labels", required=True, help="comma separated, in class-id order")
    parser.add_argument("--old", required=True)
    parser.add_argument("--old-backend", default="yolov4")
    parser.add_argument("--old-size", default=DEFAULT_SIZE, help="WxH input geometry")
    parser.add_argument("--old-dataset", help="overrides --dataset for this model")
    parser.add_argument(
        "--old-grey",
        help="single-channel grey specialist; makes the old arm a routed pair",
    )
    parser.add_argument("--new", required=True)
    parser.add_argument("--new-backend", default="ultralytics")
    parser.add_argument("--new-size", default=DEFAULT_SIZE, help="WxH input geometry")
    parser.add_argument("--new-dataset", help="overrides --dataset for this model")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="CLASS=VALUE",
        help="operational threshold per class; defaults to 0.4",
    )
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    thresholds = dict((c, 0.4) for c in labels)
    for item in args.threshold:
        name, _, value = item.partition("=")
        thresholds[name.strip()] = float(value)

    def split_root(dataset):
        return os.path.join(os.path.expanduser(dataset or args.dataset), args.split)

    old_size = parse_size(args.old_size)
    new_size = parse_size(args.new_size)
    print("thresholds: %s" % thresholds)
    for tag, onnx, backend, size, dataset in (
        ("OLD", args.old, args.old_backend, old_size, args.old_dataset),
        ("NEW", args.new, args.new_backend, new_size, args.new_dataset),
    ):
        root = split_root(dataset)
        counts = {"colour": 0, "grey": 0}
        if tag == "OLD" and args.old_grey:
            arm = routed_arm(
                build_model(onnx, backend, labels, size, channels=3),
                build_model(args.old_grey, backend, labels, size, channels=1),
                size,
                counts,
            )
            name = "%s %s + %s routed (%dx%d) %s" % (
                tag, onnx, args.old_grey, size[0], size[1], root)
        else:
            arm = single_arm(build_model(onnx, backend, labels, size), size)
            name = "%s %s (%dx%d) %s" % (tag, onnx, size[0], size[1], root)
        evaluate(arm, name, root, labels, thresholds)
        if counts["colour"] or counts["grey"]:
            print("routing: %d colour, %d grey" % (counts["colour"], counts["grey"]))


if __name__ == "__main__":
    main()
