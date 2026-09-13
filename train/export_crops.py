#!/usr/bin/env python3
"""Export ipcams2 boxes as classifier crops, in Frigate's own geometry.

Frigate's CustomObjectClassificationProcessor does:

    x, y, x2, y2 = calculate_region(frame.shape, *box, max(box_w, box_h), 1.0)
    crop = rgb[y:y2, x:x2]
    resized = cv2.resize(crop, (224, 224))

calculate_region with multiplier=1.0 and model_size=max(w,h) yields a tight
SQUARE centred on the box midpoint, clamped to the frame -- no context padding.
This reproduces that so training crops match inference crops.

It also writes a degraded copy. Frigate crops from the DETECT frame, and over
300 real cat/dog boxes the median crop was 103 px with 86% under 224, i.e. the
model is normally fed upscaled, soft crops. Training on sharp full-resolution
crops would be a train/serve mismatch, so crops-sim/ resamples each crop down
to a size drawn from the measured distribution before the 224 upscale.
"""
import os, sys, glob, math, random
import cv2, numpy as np

SRC = os.path.expanduser('~/train/ipcams2-v29')
OUT_CLEAN = os.path.expanduser('~/train/ipcams2-crops/clean')
OUT_SIM   = os.path.expanduser('~/train/ipcams2-crops/sim')
NAMES = ['cat','coyote','deer','dog','fox','person','rabbit','raccoon']
# Lognormal fitted to the measured crop sizes: median 103 px, p25 78 px.
LOG_MU, LOG_SIGMA, MIN_PX, MAX_PX = math.log(103), 0.40, 48, 400
random.seed(0)

def calculate_region(shape, xmin, ymin, xmax, ymax, model_size, multiplier=1.0):
    size = int((max(xmax-xmin, ymax-ymin) * multiplier) // 4 * 4)
    if size < model_size:
        size = model_size
    x_off = int((xmax-xmin)/2.0 + xmin - size/2.0)
    x_off = 0 if x_off < 0 else min(x_off, max(0, shape[1]-size))
    y_off = int((ymax-ymin)/2.0 + ymin - size/2.0)
    y_off = 0 if y_off < 0 else min(y_off, max(0, shape[0]-size))
    return x_off, y_off, x_off+size, y_off+size

for d in (OUT_CLEAN, OUT_SIM):
    for n in NAMES:
        os.makedirs(os.path.join(d, n), exist_ok=True)

counts = {n: 0 for n in NAMES}
skipped = 0
native = []
for split in ('train','valid','test'):
    labels = sorted(glob.glob(os.path.join(SRC, split, 'labels', '*.txt')))
    for lp in labels:
        rows = [r.split() for r in open(lp).read().strip().splitlines() if r.strip()]
        if not rows:
            continue                      # deliberate null -- no object to crop
        ip = lp.replace('/labels/','/images/').replace('.txt','.jpg')
        img = cv2.imread(ip)
        if img is None:
            skipped += 1; continue
        H, W = img.shape[:2]
        for i, r in enumerate(rows):
            if len(r) != 5:
                skipped += 1; continue    # segment rows, not detection
            ci = int(r[0])
            if ci >= len(NAMES):
                skipped += 1; continue
            cx, cy, bw, bh = (float(v) for v in r[1:])
            xmin, xmax = (cx-bw/2)*W, (cx+bw/2)*W
            ymin, ymax = (cy-bh/2)*H, (cy+bh/2)*H
            side = max(xmax-xmin, ymax-ymin)
            if side < 8:
                skipped += 1; continue
            x, y, x2, y2 = calculate_region(img.shape, xmin, ymin, xmax, ymax, int(side), 1.0)
            crop = img[int(y):int(y2), int(x):int(x2)]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                skipped += 1; continue
            native.append(crop.shape[0])
            stem = '%s_%s_%d' % (split, os.path.basename(lp)[:-4][:60], i)
            cv2.imwrite(os.path.join(OUT_CLEAN, NAMES[ci], stem + '.jpg'),
                        cv2.resize(crop, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 92])
            target = int(min(MAX_PX, max(MIN_PX, random.lognormvariate(LOG_MU, LOG_SIGMA))))
            small = cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)                     if crop.shape[0] > target else crop
            cv2.imwrite(os.path.join(OUT_SIM, NAMES[ci], stem + '.jpg'),
                        cv2.resize(small, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 92])
            counts[NAMES[ci]] += 1

native.sort()
print('crops per class:')
for n in NAMES:
    print('  %-9s %5d' % (n, counts[n]))
print('total %d, skipped %d' % (sum(counts.values()), skipped))
if native:
    print('native crop side: median %d px, p10 %d, p90 %d' % (
        native[len(native)//2], native[len(native)//10], native[9*len(native)//10]))
