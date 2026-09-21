#!/usr/bin/env python3
"""Fold Frigate's own false-positive crops into the `none` class.

The ipcams2-derived negatives are scenery and one large boulder on a stone
path. Nothing in them looks like what actually fails: a 56 px pale rock in
green grass on peach_tree. The first rebuild proved the point -- it called
that rock fox at 0.95-1.00, more confidently than the model it replaced.

These crops come straight from Frigate at its own geometry, so they need no
box maths, only the same 40-320 px resample and flip/brightness jitter that
build_grouped.py applies.

HOLDOUT is excluded on purpose. Training on every confirmed false positive
would leave nothing to measure generalisation with, and "the rock is now in
the training set" is not evidence the class works.
"""
import os, sys, glob, math, random
from PIL import Image, ImageEnhance

OUT = os.environ.get("OUT_DIR", "/mnt/storage/ipcams2/crops-grouped-bg2")
REPS = int(os.environ.get("FIELD_REPS", "8"))
SIM_MIN, SIM_MAX = 40, 320
HOLDOUT = set(filter(None, os.environ.get("HOLDOUT", "treeline_obj").split(",")))
random.seed(0)

dest = os.path.join(OUT, "train", "none")
os.makedirs(dest, exist_ok=True)

srcs = []
for d, pat in ((sys.argv[1], "*.jpg"), (sys.argv[2], "*.webp")):
    srcs += sorted(glob.glob(os.path.join(d, pat)))

written, skipped = 0, 0
for p in srcs:
    tag = os.path.basename(p).split("-")[0]
    if tag in HOLDOUT:
        skipped += 1
        continue
    try:
        base = Image.open(p).convert("RGB")
    except Exception:
        continue
    stem = "field_" + os.path.basename(p).rsplit(".", 1)[0][:70]
    for v in range(REPS):
        img = base
        t = int(math.exp(random.uniform(math.log(SIM_MIN), math.log(SIM_MAX))))
        if img.width > t:
            img = img.resize((t, t), Image.LANCZOS)
        img = img.resize((224, 224), Image.LANCZOS)
        if v > 0:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        img.save(os.path.join(dest, "%s_v%d.jpg" % (stem, v)), quality=90)
        written += 1

print("field none crops written: %d (from %d sources, %d held out: %s)"
      % (written, len(srcs) - skipped, skipped, ",".join(sorted(HOLDOUT))))
