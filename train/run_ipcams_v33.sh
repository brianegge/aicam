#!/bin/zsh
# ipcams v33: the v32 recipe (yolo11m, 608 square) on more data. v33 adds
# bobcat and squirrel, which moves every class id -- deploy needs a matching
# 10-class ipcams-labels.txt alongside the ONNX.
#
# Dataset: ~/train/ipcams2-v33-608, built by build_608_cv2.py from the
# Roboflow v33 export (cv2.resize, the call camera.py makes live; v32 used
# ffmpeg's scaler).
#
# rect=False and batch=8 for the reasons in run_ipcams_y11m.sh.
cd ~/train
exec ~/train/.venv/bin/yolo detect train \
  data=$HOME/train/ipcams2-v33-608/data.yaml \
  model=$HOME/train/yolo11m.pt \
  epochs=100 imgsz=608 batch=8 device=mps workers=4 \
  rect=False mosaic=1.0 patience=25 seed=0 \
  project=$HOME/train/runs name=ipcams-v33-yolo11m
