#!/bin/bash
# packages-vehicles2 v13, yolo11s at 608 square -- the deployed recipe on more data.
# rect=False keeps mosaic on; batch=8 is the ceiling alongside a running aicam.
nohup /Users/claw/train/.venv/bin/yolo detect train \
  data=/Users/claw/train/pv-v13-2cls/data.yaml \
  model=yolo11s.pt epochs=100 imgsz=608 batch=8 rect=False device=mps workers=4 \
  project=/Users/claw/train/runs name=pv13-yolo11s exist_ok=True \
  > /Users/claw/train/pv13.log 2>&1 &
echo $!
