#!/usr/bin/env python3
"""Build a near-duplicate-aware, event-grouped crop dataset.

Two problems with the previous export:

  1. Capping a class at N was `shuffle(); pool[:N]` -- purely random, so it
     discarded distinct scenes while keeping consecutive frames of the same
     animal. It reduced diversity rather than redundancy.
  2. Near-duplicate frames of one event could land on both sides of the
     train/val split, inflating every accuracy figure. The earlier leakage
     check only caught my own _v0/_v1 variants, not different source frames
     taken seconds apart.

Grouping is done on the pixels, via a 64-bit dHash of each crop, because
ipcams2 filenames use at least four schemes and the UUID uploads carry no
timestamp at all. Crops of the same class within HAMMING bits of each other
are unioned into one event group. Splitting and capping then happen per GROUP,
so near-duplicates cannot straddle the split and a long event cannot dominate
a class.
"""
import os, glob, json, math, random, collections, sys
import numpy as np
from PIL import Image, ImageEnhance

SRC = "/mnt/storage/ipcams2/v32"
OUT = os.environ.get("OUT_DIR", "/mnt/storage/ipcams2/crops-grouped")
NAMES = ["cat", "coyote", "deer", "dog", "fox", "person", "rabbit", "raccoon"]
SKIP = set(filter(None, os.environ.get("SKIP_CLASSES", "person").split(",")))
HAMMING = int(os.environ.get("HAMMING", "6"))     # <= this many bits differ -> same event
MAX_PER_GROUP = int(os.environ.get("MAX_PER_GROUP", "3"))
TARGET = int(os.environ.get("TARGET_PER_CLASS", "1200"))
MAX_OVERSAMPLE = int(os.environ.get("MAX_OVERSAMPLE", "10"))
VAL_FRAC = 0.2
MIN_SIDE = 24
BG_BOXES = os.environ.get("BG_BOXES", "/mnt/storage/ipcams2/background_boxes.json")
BACKGROUND = os.environ.get("BACKGROUND", "1") != "0"
# Frigate reserves this exact name: CustomObjectClassificationProcessor drops
# the consensus label when it equals "none", so the object keeps no sub_label.
# Any other name (background, negative, ...) would be published as a sub_label.
NONE_CLASS = "none"
# The first build made `none` 97% flat scenery -- random grass, pavement,
# stone -- and the model duly learned "none == flat texture". Re-tested on the
# peach_tree rock it went the wrong way: fox at 0.95-1.00, more confident than
# the model it replaced. Frigate's own docs call this out ("otherwise the model
# has no signal for small/ambiguous thing = not one of my known classes").
#
# Three knobs push the hard negatives -- the detector's actual false positives
# -- from ~3% of the class to roughly a third:
#   NONE_MAX_PER_GROUP  one pale boulder was detected 92 times across different
#                       frames and light. That is diversity, not redundancy, so
#                       do not cap it at the animal MAX_PER_GROUP of 3.
#   HARD_REPS           augmented variants per hard box, vs 1 for scene crops.
#   SIM_MIN             the live failure is a 56 px crop upscaled 4x to 224.
#                       A floor of 64 px never showed the model that regime.
NONE_MAX_PER_GROUP = int(os.environ.get("NONE_MAX_PER_GROUP", "12"))
HARD_REPS = int(os.environ.get("HARD_REPS", "6"))
HARD_KEYS = set()
SIM_MIN, SIM_MAX = int(os.environ.get("SIM_MIN", "40")), 320
random.seed(0)


def calculate_region(H, W, xmin, ymin, xmax, ymax, model_size, multiplier=1.0):
    size = int((max(xmax - xmin, ymax - ymin) * multiplier) // 4 * 4)
    if size < model_size:
        size = model_size
    x = int((xmax - xmin) / 2.0 + xmin - size / 2.0)
    x = 0 if x < 0 else min(x, max(0, W - size))
    y = int((ymax - ymin) / 2.0 + ymin - size / 2.0)
    y = 0 if y < 0 else min(y, max(0, H - size))
    return x, y, x + size, y + size


def dhash(img):
    g = img.convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).view(np.uint64)[0]


NAMES = [n for n in NAMES if n not in SKIP]
print("indexing instances...", flush=True)
inst = collections.defaultdict(list)
for split in ("train", "valid", "test"):
    for lp in sorted(glob.glob(os.path.join(SRC, split, "labels", "*.txt"))):
        for i, r in enumerate(l.split() for l in open(lp).read().split("\n") if l.strip()):
            if len(r) != 5:
                continue
            ci = int(r[0])
            if ci < len(NAMES) + len(SKIP) and NAMES.count(
                    ["cat","coyote","deer","dog","fox","person","rabbit","raccoon"][ci]):
                inst[["cat","coyote","deer","dog","fox","person","rabbit","raccoon"][ci]].append(
                    (lp, i, tuple(float(v) for v in r[1:])))

# `none` is not a YOLO class, so it comes from build_background.py as a
# list of normalised boxes rather than from the label files. Entries are given
# a synthetic label path so the rest of this script -- hashing, grouping,
# splitting, the 64-320 px resample, the flip/brightness augmentation -- treats
# them exactly like an animal crop. Without it the classifier is a softmax over
# seven animals and a rock has to be one of them.
if BACKGROUND and os.path.exists(BG_BOXES):
    NAMES = NAMES + [NONE_CLASS]
    for j, e in enumerate(json.load(open(BG_BOXES))):
        x1, y1, x2, y2 = e["nbox"]
        if max(x2 - x1, y2 - y1) <= 0:
            continue
        lp = os.path.join(SRC, e["img"]).replace("/images/", "/labels/").replace(".jpg", ".txt")
        key = (lp, 10000 + j)
        inst[NONE_CLASS].append(
            (lp, 10000 + j, ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)))
        if e.get("src") == "hard":
            HARD_KEYS.add(key)
    print("%s: %d boxes from %s" % (NONE_CLASS, len(inst[NONE_CLASS]), BG_BOXES), flush=True)

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def hamming_matrix(h):
    x = h[:, None] ^ h[None, :]
    b = x.view(np.uint8).reshape(len(h), len(h), 8)
    return POP[b].sum(axis=2)

summary = []
for cls in NAMES:
    items = inst[cls]
    if not items:
        continue
    print("hashing %-8s n=%d" % (cls, len(items)), flush=True)
    hashes, keep = [], []
    for lp, idx, (cx, cy, bw, bh) in items:
        ip = lp.replace("/labels/", "/images/").replace(".txt", ".jpg")
        try:
            im = Image.open(ip)
        except Exception:
            continue
        W, H = im.size
        xmin, xmax = (cx - bw / 2) * W, (cx + bw / 2) * W
        ymin, ymax = (cy - bh / 2) * H, (cy + bh / 2) * H
        if max(xmax - xmin, ymax - ymin) < MIN_SIDE:
            continue
        x, y, x2, y2 = calculate_region(H, W, xmin, ymin, xmax, ymax, int(max(xmax-xmin, ymax-ymin)))
        try:
            c = im.convert("RGB").crop((int(x), int(y), int(x2), int(y2)))
        except Exception:
            continue
        hashes.append(dhash(c)); keep.append((lp, idx, (cx, cy, bw, bh)))
    h = np.array(hashes, dtype=np.uint64)
    n = len(h)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    CH = 1500
    for s in range(0, n, CH):
        e = min(n, s + CH)
        d = hamming_matrix(h[s:e]) if e - s == n else None
        for s2 in range(s, n, CH):
            e2 = min(n, s2 + CH)
            dm = POP[(h[s:e, None] ^ h[None, s2:e2]).view(np.uint8).reshape(e-s, e2-s2, 8)].sum(axis=2)
            ii, jj = np.where(dm <= HAMMING)
            for a, b in zip(ii, jj):
                ga, gb = s + int(a), s2 + int(b)
                if ga != gb: union(ga, gb)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(keep[i])
    summary.append((cls, len(keep), len(groups), groups))

print("\n%-9s %8s %8s %9s %s" % ("class", "boxes", "groups", "dup ratio", "largest group"))
for cls, nb, ng, groups in summary:
    big = max(len(v) for v in groups.values())
    print("%-9s %8d %8d %9.2fx %d" % (cls, nb, ng, nb / max(1, ng), big))

# Split by GROUP, then cap per group, then oversample to balance.
for sub in ("train", "val"):
    for cls, _, _, _ in summary:
        os.makedirs(os.path.join(OUT, sub, cls), exist_ok=True)

written = collections.Counter()
for cls, nb, ng, groups in summary:
    keys = sorted(groups)
    random.Random(hash(cls) & 0xffff).shuffle(keys)
    nval = max(1, int(len(keys) * VAL_FRAC))
    split_of = {k: ("val" if i < nval else "train") for i, k in enumerate(keys)}
    # cap redundancy: at most MAX_PER_GROUP source boxes per event
    cap = NONE_MAX_PER_GROUP if cls == NONE_CLASS else MAX_PER_GROUP
    chosen = []
    for k in keys:
        g = groups[k][:]
        random.shuffle(g)
        for it in g[:cap]:
            chosen.append((split_of[k], it))
    tr = [c for c in chosen if c[0] == "train"]
    reps = min(MAX_OVERSAMPLE, max(1, round(TARGET / max(1, len(tr)))))
    for sub, (lp, idx, (cx, cy, bw, bh)) in chosen:
        ip = lp.replace("/labels/", "/images/").replace(".txt", ".jpg")
        try:
            im = Image.open(ip).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        xmin, xmax = (cx - bw/2)*W, (cx + bw/2)*W
        ymin, ymax = (cy - bh/2)*H, (cy + bh/2)*H
        x, y, x2, y2 = calculate_region(H, W, xmin, ymin, xmax, ymax, int(max(xmax-xmin, ymax-ymin)))
        base = im.crop((int(x), int(y), int(x2), int(y2)))
        if base.width < 8:
            continue
        stem = "%s_%d" % (os.path.basename(lp)[:-4][:70], idx)
        r = reps if sub == "train" else 1     # never inflate the validation set
        if sub == "train" and (lp, idx) in HARD_KEYS:
            r = max(r, HARD_REPS)
        for v in range(r):
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
            img.save(os.path.join(OUT, sub, cls, "%s_v%d.jpg" % (stem, v)), quality=90)
            written[(sub, cls)] += 1

print("\n%-9s %8s %8s" % ("class", "train", "val"))
for cls, _, _, _ in summary:
    print("%-9s %8d %8d" % (cls, written[("train", cls)], written[("val", cls)]))
print("total:", sum(written.values()))
