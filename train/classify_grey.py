"""Partition the ipcams2 dataset the way camera.py routes frames at runtime.

camera.py:resize() sends a frame to the grey model when its HSV hue sum is
exactly 0. Report both that and mean saturation so we can see whether the
signal is actually bimodal on Roboflow-re-encoded JPEGs -- BI re-encode is
known to add enough chroma noise that a night frame can score a hue sum in
the millions.
"""
import glob, os, sys, json
import cv2, numpy as np

root = sys.argv[1]
out = {}
for split in ("train", "valid", "test"):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, split, "images", "*.jpg"))):
        img = cv2.imread(p)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rows.append((os.path.basename(p), int(np.sum(hsv[:, :, 0])), float(np.mean(hsv[:, :, 1]))))
    out[split] = rows
    hs = np.array([r[1] for r in rows], dtype=float)
    sat = np.array([r[2] for r in rows])
    print("%s: n=%d  hue_sum==0: %d (%.1f%%)" % (split, len(rows), int((hs == 0).sum()), 100.0 * (hs == 0).mean()))
    print("   mean-sat percentiles:", " ".join("p%d=%.1f" % (q, np.percentile(sat, q)) for q in (1, 5, 10, 25, 50, 75, 90, 99)))
    for thr in (1, 2, 3, 5, 8, 12):
        print("   sat<%-2d -> %5d (%.1f%%)" % (thr, int((sat < thr).sum()), 100.0 * (sat < thr).mean()))
json.dump(out, open(os.path.join(root, "colorstats.json"), "w"))
print("wrote colorstats.json")
