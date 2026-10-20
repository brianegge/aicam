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

By default it only reports. With --apply it acts on provisional exclusions --
the ones a phone tap wrote -- and only those:

    RETIRE        the pair moves to retired-YYYYMMDD/ beside them, which
                  nothing loads, so the evidence is kept
    STILL NEEDED  `models:` is re-stamped with the set running now, which
                  records that it was rechecked against them

This is the step a retrain needs before anything stops being suppressed: a
model change leaves every provisional exclusion in force (excludes.load_dir
logs it as awaiting a recheck) until this has replayed its frame. Hand-written
exclusions in excludes/ are under version control and are only reported.
Unrecheckable ones (no frame) are left in force.

    python3 recheck_excludes.py --config config.txt
    python3 recheck_excludes.py --config config.txt --only peach_tree
    python3 recheck_excludes.py --config config.txt \
        --dir ~/aicam-data/capture/excludes-auto --apply
"""

import argparse
import configparser
import datetime
import glob
import json
import os
import shutil
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


def matches(preds, label, box, iou=0.5):
    """Predictions this exclusion would suppress.

    A "*" exclusion covers every class, because detect.py matches it against
    each prediction without looking at tagName -- these objects do not keep one
    name, and one stake was excluded as a rabbit and alerted as a raccoon from
    the same pixels. Testing `tagName != "*"` matches nothing, which reported
    RETIRE at score 0.00 for every wildcard exclusion however hard the model
    fired on it.
    """
    wildcard = label == "*"
    return [p for p in preds
            if (wildcard or p["tagName"] == label)
            and bb_intersection_over_union(box, p["boundingBox"]) > iou]


def running_models(cfg):
    """Model files aicam is running, as a provisional exclusion names them."""
    return sorted(os.path.basename(cfg[section]["onnx"])
                  for section in ("color-model", "grey-model", "vehicle-model")
                  if cfg.has_option(section, "onnx"))


def retire_pair(yaml_path, today=None):
    """Move an exclusion pair into retired-YYYYMMDD/ beside it; return the dir.

    load_dir does not recurse, so this stops it suppressing anything while the
    frame and the geometry stay where a second look can find them.
    """
    today = today or datetime.date.today()
    dest = os.path.join(os.path.dirname(yaml_path), "retired-%s" % today.strftime("%Y%m%d"))
    os.makedirs(dest, exist_ok=True)
    stem = os.path.splitext(yaml_path)[0]
    for ext in (".yaml", ".jpg"):
        if os.path.exists(stem + ext):
            shutil.move(stem + ext, os.path.join(dest, os.path.basename(stem) + ext))
    return dest


def restamp(yaml_path, models):
    """Rewrite the `models:` list in place, keeping every other line.

    Text, not a yaml round trip: the files open with a comment block saying
    what they are and how to audit them, and yaml.safe_dump would drop it.
    """
    with open(yaml_path) as f:
        lines = f.read().splitlines()
    out, skipping = [], False
    for line in lines:
        if line.startswith("models:"):
            skipping = True
            continue
        if skipping and line.startswith((" ", "-")):
            continue
        skipping = False
        out.append(line)
    out.append("models:")
    out.extend("  - %s" % m for m in models)
    with open(yaml_path, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.txt")
    ap.add_argument("--dir", default=None, help="exclusions directory")
    ap.add_argument("--only", default=None, help="substring filter on the filename")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="overlap at which a detection counts as the same false positive")
    ap.add_argument("--apply", action="store_true",
                    help="retire provisional exclusions the model no longer fires on, "
                         "and re-stamp the rest with the running models")
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
    models = running_models(cfg)

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
        wildcard = label == "*"
        box = {k: float(doc["box"][k]) for k in ("left", "top", "width", "height")}

        preds = predictions_for(image, color_model, vehicle_model,
                                label, labels_by_model)
        here = matches(preds, label, box, args.iou)
        # Each class carries its own threshold, so a wildcard has to be judged
        # per prediction rather than against one number.
        firing = [p for p in here
                  if p["probability"] >= thresholds.get(p["tagName"], default_threshold)]
        hit = max(firing, key=lambda p: p["probability"], default=None)
        best = hit or max(here, key=lambda p: p["probability"], default=None)
        shown = best["tagName"] if (best and wildcard) else label
        thr = thresholds.get(best["tagName"] if best else label, default_threshold)

        act = args.apply and doc.get("provisional")
        if hit:
            print("%-34s %-9s %7.2f  STILL NEEDED (threshold %.2f)%s"
                  % (stem, shown, hit["probability"], thr,
                     " -- re-stamped" if act else ""))
            if act:
                restamp(fn, models)
            needed += 1
        else:
            print("%-34s %-9s %7.2f  RETIRE -- no longer detected above %.2f%s"
                  % (stem, shown, best["probability"] if best else 0.0, thr,
                     " -- moved to %s" % os.path.basename(retire_pair(fn)) if act else ""))
            retire += 1

    print("\n%d still needed, %d can be retired, %d unrecheckable" % (needed, retire, missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
