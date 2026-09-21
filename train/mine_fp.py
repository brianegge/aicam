"""Find COCO-animal detections on ipcams2 frames that match no ground-truth box.

Frigate's detector is COCO yolo11s and its animal tokens are the only way an
animal reaches the classifier. Anything it fires on that is NOT a labelled
ipcams2 animal is, by construction, a crop the classifier will be handed and
has no correct answer for. That is the background class.

Boxes are emitted in normalised coords. A pure stretch to 608 leaves those
unchanged, so they map straight back onto the native frames.
"""
import json, os, glob, sys
from ultralytics import YOLO

D = os.path.expanduser("~/train/ipcams2-v32-608")
COCO_ANIMAL = {14:"bird",15:"cat",16:"dog",17:"horse",18:"sheep",19:"cow",21:"bear"}
IOU_MAX = 0.10
CONF = 0.35

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

model = YOLO(os.path.expanduser("~/train/yolo11s.pt"))
out = []
imgs = []
for s in ("train","valid","test"):
    imgs += sorted(glob.glob(os.path.join(D,s,"images","*.jpg")))
print(f"{len(imgs)} images", file=sys.stderr, flush=True)

B = 16
for i in range(0, len(imgs), B):
    batch = imgs[i:i+B]
    res = model.predict(batch, imgsz=320, conf=CONF, device="mps", verbose=False)
    for ip, r in zip(batch, res):
        lp = ip.replace("/images/","/labels/").replace(".jpg",".txt")
        gt = []
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) != 5: continue
                cx,cy,w,h = (float(v) for v in p[1:])
                gt.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2))
        H, W = r.orig_shape
        for box in r.boxes:
            ci = int(box.cls)
            if ci not in COCO_ANIMAL: continue
            x1,y1,x2,y2 = (float(v) for v in box.xyxy[0])
            nb = (x1/W, y1/H, x2/W, y2/H)
            if any(iou(nb,g) > IOU_MAX for g in gt): continue
            out.append({"img": os.path.relpath(ip, D), "coco": COCO_ANIMAL[ci],
                        "conf": round(float(box.conf),3), "nbox": [round(v,5) for v in nb]})
    if (i//B) % 40 == 0:
        print(f"  {i+len(batch)}/{len(imgs)} -> {len(out)} fps", file=sys.stderr, flush=True)

json.dump(out, open(os.path.expanduser("~/train/ipcams2_detector_fps.json"),"w"))
import collections
print(f"\n{len(out)} false-positive boxes on {len(set(o['img'] for o in out))} images")
print(collections.Counter(o["coco"] for o in out).most_common())
