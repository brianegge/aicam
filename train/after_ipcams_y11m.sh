#!/bin/zsh
# Wait for the ipcams yolo11m run, export it, and score it against the
# deployed routed colour/grey yolov4 pair on the same held-out split.
#
# Waits on the *process*, not on best.pt: ultralytics rewrites best.pt after
# every improving epoch, so the file appears within minutes and means nothing.
set -u
RUN=$HOME/train/runs/ipcams-v32-yolo11m
LOG=$HOME/train/ipcams_y11m_compare.log
DATA=$HOME/train/ipcams2-v32-608
LABELS=cat,coyote,deer,dog,fox,person,rabbit,raccoon

echo "=== waiting for training (started $(date)) ===" > $LOG
while pgrep -f "name=ipcams-v32-yolo11m" > /dev/null; do sleep 120; done
echo "=== training done $(date) ===" >> $LOG
tail -3 $RUN/results.csv >> $LOG 2>&1

if [[ ! -f $RUN/weights/best.pt ]]; then
  echo "=== NO best.pt -- training failed, nothing to compare ===" >> $LOG
  exit 1
fi

echo "=== exporting 608 square ===" >> $LOG
$HOME/train/.venv/bin/yolo export model=$RUN/weights/best.pt format=onnx imgsz=608 opset=12 >> $LOG 2>&1
cp $RUN/weights/best.onnx $HOME/train/ipcams_v32_yolo11m_608.onnx
echo "  wrote ipcams_v32_yolo11m_608.onnx" >> $LOG

# Model files moved to ~/aicam-models on 2026-09-13; they used to sit in
# the aicam checkout, where a git stash -u during a deploy destroyed them.
cd $HOME/aicam || exit 1   # for compare_models.py itself
run_at() {
  echo "" >> $LOG
  echo "=== $1 ===" >> $LOG
  shift
  ./.venv/bin/python compare_models.py \
    --dataset $DATA --split test --labels $LABELS \
    --old $HOME/aicam-models/ipcams_color_yolov4.onnx \
    --old-grey $HOME/aicam-models/ipcams_grey_yolov4.onnx \
    --old-backend yolov4 \
    --new $HOME/train/ipcams_v32_yolo11m_608.onnx --new-backend ultralytics \
    "$@" >> $LOG 2>&1
}

# What aicam acts on today. These were tuned for the yolov4 confidence
# distribution and are known to be wrong for yolo11 (dog=0.95 gave 1.9%
# recall on a yolo11 model with mAP50 0.90), so this run measures a naive
# drop-in swap, not the ceiling.
run_at "aicam live thresholds" \
  --threshold dog=0.95 --threshold person=0.80 --threshold fox=0.90 \
  --threshold coyote=0.85 --threshold deer=0.90

# Architecture-fair: same bar for both models, no inherited calibration.
for t in 0.50 0.25; do
  run_at "flat $t" $(for c in ${(s:,:)LABELS}; do echo --threshold $c=$t; done)
done

echo "" >> $LOG
echo "=== done $(date) ===" >> $LOG
