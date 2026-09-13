#!/bin/zsh
# Combined ipcams animal detector, yolo11m at 608 square.
#
# rect=False is mandatory: ultralytics build_transforms() silently zeroes
# mosaic/mixup/cutmix when rect is True, and args.yaml still records the
# requested value, so a rect run is not comparable to a non-rect one.
# batch=8 is the hard ceiling on this 16 GB mini while aicam is running --
# batch=16 starved aicam into a full detection blackout.
cd ~/train
exec ~/train/.venv/bin/yolo detect train \
  data=$HOME/train/ipcams2-v32-608/data.yaml \
  model=$HOME/train/yolo11m.pt \
  epochs=100 imgsz=608 batch=8 device=mps workers=4 \
  rect=False mosaic=1.0 patience=25 seed=0 \
  project=$HOME/train/runs name=ipcams-v32-yolo11m
