#!/usr/bin/env python3
"""Replay each exclusion's saved frame through the current model.

An exclusion box is a permanent blind spot: once added, that region stops
reporting that class forever, including when the detection would have been
real. The whole point of saving the frame alongside the geometry is that the
decision can be revisited after a retrain.

For each excludes/*.yaml this loads its paired .jpg, runs the model aicam is
configured to use right now, and asks: does the false positive still happen?

    STILL NEEDED  the model still fires on that spot -- keep the exclusion
    RETIRE        the model no longer fires -- delete the pair, or set
                  disabled: true in the yaml to keep the evidence

It never edits anything; retiring is left to a human because a single frame is
weak evidence and the surrounding scene may have changed.

    python3 recheck_excludes.py --config config.txt
    python3 recheck_excludes.py --config config.txt --only peach_tree
"""

import argparse
import configparser
import glob
import json
import os
import sys

import cv2
import yaml

from camera import set_model_input_sizes
import camera as camera_mod
from detect import bb_intersection_over_union
from main import load_model


def predictions_for(image, color_model, vehicle_model, label, labels_by_model):
    """Run the model that owns `label`, using camera.py's own preprocessing.

    Three channels, unconditionally, because that is what camera.py:resize()
    does since the 2026-09-11 swap to one combined model. There used to be a
    second path here that converted a greyscale frame to a single channel for a
    grey specialist; keeping it after the specialist was gone fed a 1-channel
    image to the 3-channel combined model and raised "axes don't match array"
    on the first IR exclusion. If this ever stops matching resize(), the
    verdict is about a pipeline nobody is running.
    """
    if label in labels_by_model["vehicle"]:
        resized = cv2.cvtColor(
            cv2.resize(image, camera_mod.VEHICLE_INPUT_SIZE), cv2.COLOR_BGR2RGB
        )
        return vehicle_model.predict_image(resized)

    resized = cv2.resize(image, camera_mod.IPCAMS_INPUT_SIZE)
    return color_model.predict_image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.txt")
    ap.add_argument("--dir", default=None, help="exclusions directory")
    ap.add_argument("--only", default=None, help="substring filter on the filename")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="overlap at which a detection counts as the same false positive")
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    det = cfg["detector"]
    exdir = args.dir or det.get("excludes-dir", "excludes")

    with open(det["labelfile-path"]) as f:
        ipcams_labels = [l.strip() for l in f]
    with open(det["vehicle-labelfile-path"]) as f:
        vehicle_labels = [l.strip() for l in f]
    labels_by_model = {"ipcams": set(ipcams_labels), "vehicle": set(vehicle_labels)}

    # One ipcams model since 2026-09-11. [grey-model] is not read at all --
    # camera.py has no greyscale path left to replicate.
    color_model = load_model(cfg["color-model"], ipcams_labels, False)
    vehicle_model = load_model(cfg["vehicle-model"], vehicle_labels, False)
    set_model_input_sizes(color_model, vehicle_model)

    # Thresholds aicam actually acts on -- an exclusion only matters if the
    # detection would have been reported.
    thresholds = {k: float(v) for k, v in cfg["thresholds"].items()} if "thresholds" in cfg else {}
    default_threshold = float(det.get("threshold", 0.7))

    files = sorted(glob.glob(os.path.join(exdir, "*.yaml")))
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    if not files:
        print("no exclusions found in %s" % exdir)
        return 0

    needed = retire = missing = 0
    print("%-34s %-9s %7s  %s" % ("exclusion", "label", "score", "verdict"))
    for fn in files:
        with open(fn) as f:
            doc = yaml.safe_load(f) or {}
        stem = os.path.splitext(os.path.basename(fn))[0]
        img_path = os.path.splitext(fn)[0] + ".jpg"
        if not os.path.exists(img_path):
            print("%-34s %-9s %7s  NO IMAGE (cannot recheck)" % (stem, doc.get("label","?"), "-"))
            missing += 1
            continue
        image = cv2.imread(img_path)
        if image is None:
            print("%-34s %-9s %7s  UNREADABLE IMAGE" % (stem, doc.get("label","?"), "-"))
            missing += 1
            continue

        label = doc["label"]
        box = {k: float(doc["box"][k]) for k in ("left", "top", "width", "height")}
        thr = thresholds.get(label, default_threshold)

        preds = predictions_for(image, color_model, vehicle_model,
                                label, labels_by_model)
        hit = None
        for p in preds:
            if p["tagName"] != label or p["probability"] < thr:
                continue
            if bb_intersection_over_union(box, p["boundingBox"]) > args.iou:
                if hit is None or p["probability"] > hit["probability"]:
                    hit = p

        if hit:
            print("%-34s %-9s %7.2f  STILL NEEDED (threshold %.2f)" % (stem, label, hit["probability"], thr))
            needed += 1
        else:
            best = max((p["probability"] for p in preds if p["tagName"] == label), default=0.0)
            print("%-34s %-9s %7.2f  RETIRE -- no longer detected above %.2f"
                  % (stem, label, best, thr))
            retire += 1

    print("\n%d still needed, %d can be retired, %d unrecheckable" % (needed, retire, missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
