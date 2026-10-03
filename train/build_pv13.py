"""Turn Roboflow packages-vehicles2 v13 into the two-class layout aicam uses.

v13 is the first version with a `person` class (5 boxes, labelled 2026-09-29/30
on review uploads), which pushes `vehicle` from index 1 to 2. aicam's
vehicle-labels.txt is `package,vehicle`, and people are the ipcams model's job,
so person boxes are dropped and vehicle is renumbered. Five boxes could not
train a class anyway.
"""
import glob
import os
import shutil

SRC = "/Users/claw/train/packages-vehicles2-v13"
DST = "/Users/claw/train/pv-v13-2cls"
REMAP = {0: 0, 2: 1}  # package, vehicle; 1 (person) dropped

shutil.rmtree(DST, ignore_errors=True)
shutil.copytree(SRC, DST)
dropped = 0
for f in glob.glob(f"{DST}/*/labels/*.txt"):
    out = []
    for line in open(f).read().splitlines():
        p = line.split()
        if not p:
            continue
        c = int(p[0])
        if c not in REMAP:
            dropped += 1
            continue
        out.append(" ".join([str(REMAP[c])] + p[1:]))
    with open(f, "w") as fh:
        fh.write("\n".join(out) + ("\n" if out else ""))
with open(f"{DST}/data.yaml", "w") as fh:
    fh.write(f"path: {DST}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
             "nc: 2\nnames: [package, vehicle]\n")
print(f"dropped {dropped} person boxes")
