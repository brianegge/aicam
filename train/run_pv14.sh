#!/bin/bash
# packages-vehicles2 v14, yolo11s at 608 square -- the deployed v13 recipe on
# v14's data (32 new train images; valid and test identical to v13's).
# v14 is already package,vehicle -- the stray person class is gone -- so there
# is no remap step; this only writes an absolute data.yaml.
# rect=False keeps mosaic on; batch=8 is the ceiling alongside a running aicam.
SRC=/Users/claw/train/packages-vehicles2-v14
cat > $SRC/data-abs.yaml <<YAML
path: $SRC
train: train/images
val: valid/images
test: test/images
nc: 2
names: [package, vehicle]
YAML
cd /Users/claw/train
nohup /Users/claw/train/.venv/bin/yolo detect train \
  data=$SRC/data-abs.yaml \
  model=yolo11s.pt epochs=100 imgsz=608 batch=8 rect=False device=mps workers=4 \
  project=/Users/claw/train/runs name=pv14-yolo11s exist_ok=True \
  > /Users/claw/train/pv14.log 2>&1 &
echo $!
