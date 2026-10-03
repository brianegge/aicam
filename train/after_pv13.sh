#!/bin/bash
# Wait for the pv13 run, export 608 ONNX, score against the deployed model on
# the v13 test split (a superset of v11's 61) at aicam's real thresholds.
set -u
RUN=/Users/claw/train/runs/pv13-yolo11s
LOG=/Users/claw/train/pv13_compare.log
V=/Users/claw/train/.venv/bin/python
OUT=/Users/claw/train/packages_vehicles_v13_yolo11s.onnx
exec >> $LOG 2>&1
echo "=== waiting for training (started $(date)) ==="
while pgrep -f 'yolo detect train' >/dev/null; do sleep 60; done
if [ ! -f $RUN/weights/best.pt ]; then echo 'training did not produce best.pt'; exit 1; fi
echo "=== training done $(date) ==="
tail -3 $RUN/results.csv | cut -c1-120
$V -c "
from ultralytics import YOLO
YOLO('$RUN/weights/best.pt').export(format='onnx', imgsz=608, opset=12)
import shutil; shutil.move('$RUN/weights/best.onnx', '$OUT')
print('  wrote $OUT')"
cd /Users/claw/aicam
for T in "--threshold package=0.80 --threshold vehicle=0.70" "--threshold package=0.80 --threshold vehicle=0.90"; do
  echo "=== deployed v11 yolo11s  vs  v13 yolo11s  [$T] ==="
  /Users/claw/aicam/.venv/bin/python compare_models.py --dataset /Users/claw/train/pv-v13-2cls --split test \
    --labels package,vehicle \
    --old /Users/claw/aicam-models/packages_vehicles_yolo11s.onnx --old-backend ultralytics \
    --new $OUT --new-backend ultralytics $T 2>&1 | grep -vE 'onnxruntime|CoreML'
done
echo "=== done $(date) ==="
