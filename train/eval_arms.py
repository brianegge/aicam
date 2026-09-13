"""Score the one-model system against the two-model (routed) system.

The two arrangements are not two models on the same data -- they are two
*systems*:

  one-model : the combined model scores every test image.
  two-model : each test image is routed by the same rule camera.py uses
              (HSV hue sum == 0 -> grey model, else colour model), and the
              routed specialists results are unioned.

Both therefore answer for exactly the same 423 images and the same ground
truth, which is what makes the comparison fair. Scoring is at the thresholds
in aicam config [thresholds], not at trainer defaults.
"""
import argparse, glob, json, os, sys
import cv2, numpy as np, onnxruntime as ort

DEFAULT_THRESHOLD = 0.7
THRESHOLDS = {"dog": 0.95, "person": 0.80, "fox": 0.90, "coyote": 0.85, "deer": 0.90}
CONF_FLOOR = 0.15
NMS_IOU = 0.6


class Model(object):
    """Minimal Ultralytics ONNX runner: output [1, 4+nc, N], cx/cy/w/h pixels."""

    def __init__(self, path, names, size):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.names = names
        self.w, self.h = size

    def predict(self, bgr):
        img = cv2.resize(bgr, (self.w, self.h))
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        out = self.sess.run(None, {self.iname: x})[0]
        p = np.squeeze(out).T  # [N, 4+nc]
        assert p.shape[1] == 4 + len(self.names), (p.shape, len(self.names))
        scores = p[:, 4:]
        best = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= CONF_FLOOR
        p, best, conf = p[keep], best[keep], conf[keep]
        boxes = np.stack([
            (p[:, 0] - p[:, 2] / 2) / self.w, (p[:, 1] - p[:, 3] / 2) / self.h,
            p[:, 2] / self.w, p[:, 3] / self.h], axis=1)
        out_dets = []
        for c in np.unique(best):
            m = best == c
            b, s = boxes[m], conf[m]
            idx = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), CONF_FLOOR, NMS_IOU)
            for i in np.array(idx).reshape(-1):
                out_dets.append((self.names[int(c)], float(s[i]), b[i].tolist()))
        return out_dets


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def truth_for(root, split, stem, names):
    p = os.path.join(root, split, "labels", stem + ".txt")
    out = []
    if os.path.exists(p):
        for line in open(p):
            f = line.split()
            if len(f) >= 5:
                cx, cy, w, h = [float(v) for v in f[1:5]]
                out.append((names[int(f[0])], [cx - w / 2, cy - h / 2, w, h]))
    return out


def score(pairs, names, root, split, tag):
    """pairs: list of (image_path, model). Each image scored by exactly one model."""
    tp = dict((c, 0) for c in names)
    fp = dict((c, 0) for c in names)
    fn = dict((c, 0) for c in names)
    for path, model in pairs:
        img = cv2.imread(path)
        if img is None:
            continue
        dets = [d for d in model.predict(img)
                if d[1] >= THRESHOLDS.get(d[0], DEFAULT_THRESHOLD)]
        stem = os.path.basename(path).rsplit(".", 1)[0]
        gt = truth_for(root, split, stem, names)
        matched = set()
        for cls, conf, box in sorted(dets, key=lambda d: -d[1]):
            best_s, best_i = 0.0, None
            for i, (gcls, gbox) in enumerate(gt):
                if i in matched or gcls != cls:
                    continue
                s = iou(box, gbox)
                if s > best_s:
                    best_s, best_i = s, i
            if best_s >= 0.5:
                matched.add(best_i)
                tp[cls] += 1
            else:
                fp[cls] += 1
        for i, (gcls, _) in enumerate(gt):
            if i not in matched:
                fn[gcls] += 1
    return tag, tp, fp, fn


def report(rows, names):
    hdr = "%-9s" % "class"
    for tag, _, _, _ in rows:
        hdr += "  | %-22s" % tag
    print(hdr)
    print("%-9s" % "" + ("  | %6s %6s %6s %5s" % ("P", "R", "F1", "n")) * len(rows))
    for c in names + ["ALL"]:
        line = "%-9s" % c
        for tag, tp, fp, fn in rows:
            if c == "ALL":
                t, f, m = sum(tp.values()), sum(fp.values()), sum(fn.values())
            else:
                t, f, m = tp[c], fp[c], fn[c]
            p = t / float(t + f) if t + f else 0.0
            r = t / float(t + m) if t + m else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            line += "  | %6.3f %6.3f %6.3f %5d" % (p, r, f1, t + m)
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--combined", required=True)
    ap.add_argument("--color", required=True)
    ap.add_argument("--grey", required=True)
    ap.add_argument("--size", default="1088x608")
    ap.add_argument("--flat", type=float, help="one threshold for every class")
    a = ap.parse_args()
    global THRESHOLDS, DEFAULT_THRESHOLD
    if a.flat is not None:
        THRESHOLDS = {}
        DEFAULT_THRESHOLD = a.flat
    w, _, h = a.size.partition("x")
    size = (int(w), int(h))
    names = [l.strip("- \n") for l in open(os.path.join(a.dataset, "data.yaml")) if l.startswith("- ")]
    stats = json.load(open(os.path.join(a.dataset, "colorstats.json")))[a.split]
    imgdir = os.path.join(a.dataset, a.split, "images")

    combined = Model(a.combined, names, size)
    colorm = Model(a.color, names, size)
    greym = Model(a.grey, names, size)

    one, two = [], []
    ngrey = 0
    for fname, hue, sat in stats:
        p = os.path.join(imgdir, fname)
        one.append((p, combined))
        two.append((p, greym if hue == 0 else colorm))
        ngrey += 1 if hue == 0 else 0
    print("thresholds: default %.2f, overrides %s" % (DEFAULT_THRESHOLD, THRESHOLDS))
    print("%d test images: %d colour -> colour model, %d grey -> grey model\n"
          % (len(one), len(one) - ngrey, ngrey))
    rows = [score(one, names, a.dataset, a.split, "ONE combined"),
            score(two, names, a.dataset, a.split, "TWO routed")]
    report(rows, names)


if __name__ == "__main__":
    main()
