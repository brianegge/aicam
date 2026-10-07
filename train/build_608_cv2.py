"""Stretch a Roboflow YOLO export to 608x608 with cv2.resize, the exact call
camera.py's resize() makes at inference (INTER_LINEAR, no letterbox).
YOLO labels are normalised, so a stretch leaves them correct; copied verbatim.
usage: build_608_cv2.py SRC DST"""
import os, sys, shutil, cv2, yaml
from concurrent.futures import ThreadPoolExecutor
src, dst = sys.argv[1:3]
def one(a):
    s, f = a
    im = cv2.imread(f"{src}/{s}/images/{f}")
    cv2.imwrite(f"{dst}/{s}/images/{f}", cv2.resize(im, (608, 608)), [cv2.IMWRITE_JPEG_QUALITY, 95])
for s in ("train", "valid", "test"):
    os.makedirs(f"{dst}/{s}/images", exist_ok=True)
    shutil.copytree(f"{src}/{s}/labels", f"{dst}/{s}/labels", dirs_exist_ok=True)
    with ThreadPoolExecutor(8) as ex: list(ex.map(one, [(s, f) for f in os.listdir(f"{src}/{s}/images")]))
    print(s, len(os.listdir(f"{dst}/{s}/images")), "images", flush=True)
names = yaml.safe_load(open(f"{src}/data.yaml"))["names"]
open(f"{dst}/data.yaml", "w").write(
    "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
    f"nc: {len(names)}\nnames: {names}\n\n# Built by build_608_cv2.py from {src}\n")
