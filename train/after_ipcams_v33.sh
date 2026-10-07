#!/bin/zsh
# Wait for the ipcams v33 run, export it, and score it against the deployed
# v32 model on v33's test split (v32's 424 test images plus 59 new).
#
# v32 has 8 classes and v33 10, in a different id order, so each model gets
# its own label list. v32 cannot detect bobcat or squirrel at all; those rows
# measure what the new classes add, not a like-for-like comparison.
set -u
RUN=$HOME/train/runs/ipcams-v33-yolo11m
LOG=$HOME/train/ipcams_v33_compare.log
DATA=$HOME/train/ipcams2-v33-608
LABELS=bobcat,cat,coyote,deer,dog,fox,person,rabbit,raccoon,squirrel
OLD_LABELS=cat,coyote,deer,dog,fox,person,rabbit,raccoon

echo "=== waiting for training (started $(date)) ===" > $LOG
while pgrep -f "name=ipcams-v33-yolo11m" > /dev/null; do sleep 120; done
echo "=== training done $(date) ===" >> $LOG
tail -3 $RUN/results.csv >> $LOG 2>&1

if [[ ! -f $RUN/weights/best.pt ]]; then
  echo "=== NO best.pt -- training failed, nothing to compare ===" >> $LOG
  exit 1
fi

echo "=== exporting 608 square ===" >> $LOG
$HOME/train/.venv/bin/yolo export model=$RUN/weights/best.pt format=onnx imgsz=608 opset=12 >> $LOG 2>&1
cp $RUN/weights/best.onnx $HOME/train/ipcams_v33_yolo11m_608.onnx
echo "  wrote ipcams_v33_yolo11m_608.onnx" >> $LOG

cd $HOME/aicam || exit 1
run_at() {
  echo "" >> $LOG
  echo "=== $1 ===" >> $LOG
  shift
  ./.venv/bin/python compare_models.py \
    --dataset $DATA --split test --labels $LABELS \
    --old $HOME/aicam-models/ipcams_v32_yolo11m_608.onnx --old-backend ultralytics \
    --old-labels $OLD_LABELS \
    --new $HOME/train/ipcams_v33_yolo11m_608.onnx --new-backend ultralytics \
    "$@" >> $LOG 2>&1
}

# config.txt [thresholds] as of 2026-10-06; bobcat/squirrel have none yet.
run_at "aicam live thresholds" \
  --threshold cat=0.60 --threshold coyote=0.50 --threshold deer=0.60 \
  --threshold dog=0.50 --threshold fox=0.70 --threshold person=0.55 \
  --threshold rabbit=0.85 --threshold raccoon=0.50 \
  --threshold bobcat=0.50 --threshold squirrel=0.50

for t in 0.50 0.25; do
  run_at "flat $t" $(for c in ${(s:,:)LABELS}; do echo --threshold $c=$t; done)
done

echo "" >> $LOG
echo "=== done $(date) ===" >> $LOG
