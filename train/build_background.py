#!/usr/bin/env python3
"""Emit the source boxes for the classifier's `background` class.

The classifier is a softmax over seven animals, so every crop it is handed
gets one of them. It is only ever handed crops Frigate's COCO detector
flagged as an animal, and that detector fires on rocks, sheds and a wire
reindeer lawn ornament. Those crops have no correct answer, which is how one
pale rock on peach_tree came back as fox, dog and deer on different frames.

Two sources, written as normalised boxes against the native v32 frames so
build_grouped.py can crop them with exactly the geometry it uses for animals:

  hard   COCO-animal detections from yolo11s on ipcams2 frames that match no
         ground-truth box -- what the detector actually false-positives on.
         Hand-verified: mine_fp.py returned 69 crops and 14 were real animals
         the dataset had not labelled, including a deer under a pink IR cast
         that would have taught the model deer-is-background.

  scene  random squares that overlap neither a ground-truth box nor any of
         those detections. Cheap, and covers the scenery the hard set misses
         -- the hard set is dominated by one boulder photographed 20 times.

Rejected hard crops are listed explicitly rather than filtered by a rule:
the giveaway is what the crop shows, and no property of the box predicts it.
"""
import json, os, glob, random, sys

SRC = "/mnt/storage/ipcams2/v32"
FPS = "/mnt/storage/ipcams2/ipcams2_detector_fps.json"
OUT = "/mnt/storage/ipcams2/background_boxes.json"

# Real animals yolo11s found that ipcams2 had not labelled. Index is the
# position in the detector-FP list; the name records what it actually was.
REJECT = {10: "dog in snow", 11: "brown dog", 31: "dog on decking",
          34: "brown dog", 35: "DEER under a pink IR cast", 36: "brown dog",
          40: "dog by the pool", 43: "pale animal on pavement",
          46: "border collie", 48: "brown dog", 57: "white shape, IR, unclear",
          63: "two white shapes, IR, unclear", 65: "bird in flight",
          66: "crow"}

SCENE_PER_IMAGE = float(os.environ.get("SCENE_PER_IMAGE", "0.45"))
MIN_PX, MAX_PX = 48, 300
MIN_SIDE = 24                              # matches build_grouped.py
MAX_OVERLAP = 0.02
random.seed(0)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


fps = json.load(open(FPS))
out, kept_hard = [], 0
unreviewed = 0
for i, f in enumerate(fps):
    if i in REJECT:
        continue
    # crop_fps.py skipped boxes under MIN_SIDE, so those were never eyeballed.
    # Drop them rather than trust them; build_grouped would discard them anyway.
    b = f["nbox"]
    if max(b[2]-b[0], b[3]-b[1]) * 608 < MIN_SIDE:
        unreviewed += 1
        continue
    out.append({"img": f["img"], "nbox": f["nbox"], "src": "hard",
                "note": f"yolo11s {f['coco']} {f['conf']}"})
    kept_hard += 1
print(f"hard:  {kept_hard} kept, {len(REJECT)} rejected as real animals, "
      f"{unreviewed} dropped as too small to have been reviewed")

# Index ground truth and detector boxes per image so scene squares avoid both.
avoid = {}
for f in fps:
    avoid.setdefault(f["img"], []).append(tuple(f["nbox"]))

imgs = []
for s in ("train", "valid", "test"):
    imgs += sorted(glob.glob(os.path.join(SRC, s, "images", "*.jpg")))
print(f"{len(imgs)} source frames")

try:
    from PIL import Image
except ImportError:
    sys.exit("need pillow")

scene = 0
for ip in imgs:
    if random.random() > SCENE_PER_IMAGE:
        continue
    rel = os.path.relpath(ip, SRC)
    lp = ip.replace("/images/", "/labels/").replace(".jpg", ".txt")
    boxes = list(avoid.get(rel, []))
    if os.path.exists(lp):
        for line in open(lp):
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, w, h = (float(v) for v in p[1:])
            boxes.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2))
    try:
        W, H = Image.open(ip).size
    except Exception:
        continue
    for _ in range(12):                      # a few darts, keep the first clean one
        side = random.randint(MIN_PX, min(MAX_PX, max(MIN_PX+1, min(W, H) - 1)))
        x = random.randint(0, W - side)
        y = random.randint(0, H - side)
        nb = (x/W, y/H, (x+side)/W, (y+side)/H)
        if any(iou(nb, b) > MAX_OVERLAP for b in boxes):
            continue
        out.append({"img": rel, "nbox": [round(v, 5) for v in nb], "src": "scene",
                    "note": f"{side}px square"})
        scene += 1
        break

print(f"scene: {scene}")
json.dump(out, open(OUT, "w"))
print(f"{len(out)} background boxes -> {OUT}")
