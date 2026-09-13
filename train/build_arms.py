"""Build the three experiment arms from one ipcams2 download.

  combined/  every image           (one model handles both domains)
  color/     hue_sum != 0 only     (specialist)
  grey/      hue_sum == 0 only     (specialist)

Images are symlinked, not copied -- three arms of a 5202-image 1088x608 set
would otherwise be several GB of duplicates. The test split is identical in
all three, so the two-model system can be scored by routing each test image
to its specialist and unioning the result.
"""
import glob, json, os, shutil, sys
import numpy as np

root = sys.argv[1]
base = os.path.dirname(root.rstrip("/"))
stats = json.load(open(os.path.join(root, "colorstats.json")))
names = [l.strip("- \n") for l in open(os.path.join(root, "data.yaml")) if l.startswith("- ")]

def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)

counts = {}
for arm in ("combined", "color", "grey"):
    armdir = os.path.join(base, "ipcams2-arm-" + arm)
    shutil.rmtree(armdir, ignore_errors=True)
    inst = {}
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(armdir, split, sub), exist_ok=True)
        n = 0
        for fname, hue, sat in stats[split]:
            is_grey = hue == 0
            if arm == "color" and is_grey:
                continue
            if arm == "grey" and not is_grey:
                continue
            stem = fname.rsplit(".", 1)[0]
            link(os.path.join(root, split, "images", fname),
                 os.path.join(armdir, split, "images", fname))
            lbl = os.path.join(root, split, "labels", stem + ".txt")
            if os.path.exists(lbl):
                link(lbl, os.path.join(armdir, split, "labels", stem + ".txt"))
                if split == "train":
                    for line in open(lbl):
                        if line.split():
                            c = names[int(line.split()[0])]
                            inst[c] = inst.get(c, 0) + 1
            n += 1
        counts.setdefault(arm, {})[split] = n
    # the test split must stay the full one for combined; specialists are
    # scored only on their own routed subset, which is what the arm holds.
    with open(os.path.join(armdir, "data.yaml"), "w") as f:
        f.write("train: %s/train/images\nval: %s/valid/images\ntest: %s/test/images\nnc: %d\nnames:\n"
                % (armdir, armdir, armdir, len(names)))
        for x in names:
            f.write("- %s\n" % x)
    counts[arm]["instances"] = inst

print("%-10s %7s %7s %7s" % ("arm", "train", "valid", "test"))
for arm in ("combined", "color", "grey"):
    c = counts[arm]
    print("%-10s %7d %7d %7d" % (arm, c["train"], c["valid"], c["test"]))
print()
print("train instances per class")
print("%-9s %8s %8s %8s   %s" % ("class", "combined", "color", "grey", "grey share"))
for cl in names:
    a = counts["combined"]["instances"].get(cl, 0)
    b = counts["color"]["instances"].get(cl, 0)
    g = counts["grey"]["instances"].get(cl, 0)
    print("%-9s %8d %8d %8d   %5.1f%%" % (cl, a, b, g, 100.0 * g / a if a else 0))
