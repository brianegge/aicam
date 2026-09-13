#!/bin/zsh
# One combined ipcams model to replace the colour/grey pair.
#
# The 2026-09-05 experiment settled the question: routing frames to a dedicated
# grey model buys nothing (F1 0.916 vs 0.916 at a flat 0.50 threshold, and the
# per-class differences all sit in classes with n<12). Two models cost double
# the loading and give each specialist a permanent blind spot -- the colour
# model saw zero rabbits and two coyotes, because those animals are IR-only.
#
# rect=False, which is the whole point: rect=True silently sets mosaic=0, and
# mosaic turned out to matter enormously. On the vehicle model, mosaic-off
# fired on 10/31 empty-driveway shadow frames against 1/31 with it on, while
# simultaneously missing distant street cars (1/31 vs 14/31).
#
# v29 is the 608x608 stretch version, matching camera.py's current geometry so
# the result is a drop-in replacement.
set -e
Y=~/train/.venv/bin/yolo
D=/Users/claw/train/ipcams2-v29/data.yaml

echo "=== start $(date) ==="
$Y detect train data=$D model=yolo11s.pt epochs=80 imgsz=608 batch=16 \
  rect=False device=mps workers=4 project=/Users/claw/train/runs \
  name=ipcams-combined patience=20 seed=0 plots=True exist_ok=True
echo "=== done $(date) ==="
echo ALLDONE
