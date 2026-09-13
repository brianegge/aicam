#!/bin/zsh
# Three arms, identical hyperparameters, sequential -- MPS is one GPU.
#
# yolo11s (not 11n) on purpose: a smaller backbone has less capacity to absorb
# two visual domains at once, which would bias the result toward the
# specialists. Matching production capacity keeps the comparison honest.
#
# batch=8, NOT 16. This mini has 16GB shared with a running aicam. batch=16 at
# 1088x608 wanted 12.4GB + 6.3GB swap, iteration time swung 6-27s/it, and
# aicam was starved into backing off every camera. batch=8 is what the
# packages model trained at for 3h alongside aicam with no impact.
set -e
Y=~/train/.venv/bin/yolo
for arm in combined color grey; do
  echo "=== $arm start $(date) ==="
  $Y detect train \
    data=/Users/claw/train/ipcams2-arm-$arm/data.yaml \
    model=yolo11s.pt epochs=50 imgsz=1088 rect=True batch=8 \
    device=mps workers=4 project=/Users/claw/train/runs name=arm-$arm \
    patience=15 seed=0 plots=True exist_ok=True
  echo "=== $arm done $(date) ==="
done
echo ALLDONE
