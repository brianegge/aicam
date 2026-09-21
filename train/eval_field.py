#!/usr/bin/env python3
"""Compare the deployed classifier with the `none`-class rebuild on FIELD data.

The validation split is drawn from the same review pipeline as the training
data, so it holds none of the scenery the model actually fails on -- the
README already records a model that scored deer precision 1.000 and then
produced five recurring deer false positives in the field. So this scores
three sets instead:

  fieldset   the 200 crops Frigate saved from real classification attempts,
             named <event>-<ts>-<label>-<score>.webp with the DEPLOYED model's
             own answer. Roughly 174 are the household dog, so this is the
             regression test: a `none` class must not start calling him none.

  fieldneg   19 crops of three objects confirmed by eye to be a pale rock, an
             object on the tree line and a tarp. The deployed model called the
             rock fox, dog and deer on different frames. Every one of these
             should come back `none`.

Scores are read at the configured threshold (0.8), not argmax, because that is
what Frigate acts on.
"""
import os, sys, glob, collections
import numpy as np
import tensorflow as tf
import cv2

THRESH = float(os.environ.get("THRESHOLD", "0.8"))

def load(mdir):
    it = tf.lite.Interpreter(model_path=os.path.join(mdir, "model.tflite"))
    it.allocate_tensors()
    labels = [l.strip() for l in open(os.path.join(mdir, "labelmap.txt")) if l.strip()]
    return it, it.get_input_details()[0], it.get_output_details()[0], labels

def predict(it, inp, out, labels, img):
    # Frigate passes the raw uint8 crop resized to 224; reproduce exactly.
    x = np.expand_dims(cv2.resize(img, (224, 224)), axis=0)
    if inp["dtype"] == np.float32:
        x = x.astype(np.float32) / 255.0
    it.set_tensor(inp["index"], x.astype(inp["dtype"]))
    it.invoke()
    y = it.get_tensor(out["index"])[0].astype(np.float32)
    if out["dtype"] == np.uint8:
        s, z = out["quantization"]
        y = (y - z) * s
    i = int(y.argmax())
    return labels[i], float(y[i])

def read(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

models = {"deployed": load(sys.argv[1]), "rebuilt": load(sys.argv[2])}
for name, m in models.items():
    print(f"{name}: {len(m[3])} classes {m[3]}")

# ---- fieldneg: every one of these should be `none` ----
print(f"\n=== confirmed field false positives (n=19, threshold {THRESH}) ===")
print(f"{'object':14s} {'deployed':>28s} {'rebuilt':>28s}")
per = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for p in sorted(glob.glob(os.path.join(sys.argv[4], "*.jpg"))):
    img = read(p)
    if img is None: continue
    tag = os.path.basename(p).split("-")[0]
    for name, m in models.items():
        lab, sc = predict(*m, img)
        per[tag][name][lab if sc >= THRESH else f"(under {THRESH})"] += 1
for tag in sorted(per):
    a = ", ".join(f"{k}:{v}" for k, v in per[tag]["deployed"].most_common())
    b = ", ".join(f"{k}:{v}" for k, v in per[tag]["rebuilt"].most_common())
    print(f"{tag:14s} {a:>28s} {b:>28s}")

# ---- fieldset: the deployed model's own live answers ----
print(f"\n=== 200 live crops, deployed model's own label vs rebuilt ===")
agree = collections.Counter(); flips = collections.Counter()
for p in sorted(glob.glob(os.path.join(sys.argv[3], "*.webp"))):
    img = read(p)
    if img is None: continue
    stem = os.path.basename(p)[:-5]
    was = stem.rsplit("-", 2)[1]
    lab, sc = predict(*models["rebuilt"], img)
    new = lab if sc >= THRESH else "(under)"
    if new == was: agree[was] += 1
    else: flips[(was, new)] += 1
print("held:")
for k, v in agree.most_common(): print(f"   {k:10s} {v}")
print("changed:")
for (a, b), v in flips.most_common(): print(f"   {a:10s} -> {b:10s} {v}")

# The event from the screenshot: a pale rock on peach_tree that the deployed
# model called fox, dog and deer on different frames of the same object.
ROCK = "1789986117.044585-5j6dd5"
print(f"\n=== the peach_tree rock, event {ROCK} ===")
for p in sorted(glob.glob(os.path.join(sys.argv[3], ROCK + "*.webp"))):
    img = read(p)
    if img is None: continue
    was = os.path.basename(p)[:-5].rsplit("-", 2)
    o_lab, o_sc = predict(*models["deployed"], img)
    n_lab, n_sc = predict(*models["rebuilt"], img)
    print(f"   deployed {o_lab:8s} {o_sc:.2f}   ->   rebuilt {n_lab:8s} {n_sc:.2f}")
