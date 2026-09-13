#!/bin/zsh
# Does mosaic help or hurt on a fixed-camera setup?
#
# The user's argument: these cameras never move, so an object's location in
# frame is real signal -- deer appear on the ground, not in the sky. Mosaic
# stitches four images into one, which destroys that spatial prior. Against
# that, mosaic is the standard defence against background false positives,
# and its absence (silently forced by rect=True) coincided with the 1088x608
# model firing on dappled driveway shadows.
#
# Single variable: mosaic. mixup/cutmix/copy_paste already default to 0.0, so
# nothing else differs. rect=False in both arms so the flag cannot interfere.
# Everything else matches the deployed pv11 run: 608 square, 100 epochs,
# batch 16, seed 0.
set -e
Y=~/train/.venv/bin/yolo
D=/Users/claw/train/packages-vehicles2-v11/data.yaml

echo "=== mosaic ON  start $(date) ==="
$Y detect train data=$D model=yolo11s.pt epochs=100 imgsz=608 batch=16 \
  device=mps workers=4 project=/Users/claw/train/runs name=mos-on \
  rect=False mosaic=1.0 patience=30 seed=0 plots=True exist_ok=True

echo "=== mosaic OFF start $(date) ==="
$Y detect train data=$D model=yolo11s.pt epochs=100 imgsz=608 batch=16 \
  device=mps workers=4 project=/Users/claw/train/runs name=mos-off \
  rect=False mosaic=0.0 patience=30 seed=0 plots=True exist_ok=True

echo "=== done $(date) ==="
echo ALLDONE
