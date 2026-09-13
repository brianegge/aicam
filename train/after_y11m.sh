#!/bin/bash
# Wait for the yolo11m run to finish, then export both geometries and score
# all three against the deployed model at aicam's real thresholds.
#
# Two exports from ONE square-trained model:
#   608x608    isolates 'bigger model'
#   1088x608   isolates 'wider input'
# YOLO is fully convolutional, so a rectangular ONNX can be exported from
# square training. That matters because the Sep 4 wide model was trained with
# rect=True, which silently zeroes mosaic/mixup/cutmix -- so its dappled-light
# false positives measured missing augmentation, not geometry. This separates
# the two for the first time.
set -u
RUN=/Users/claw/train/runs/pv11-yolo11m
LOG=/Users/claw/train/y11m_compare.log
V=/Users/claw/train/.venv/bin/python
exec >> $LOG 2>&1
echo "=== waiting for training (started $(date)) ==="
for i in $(seq 1 720); do
  [ -f $RUN/weights/best.pt ] && ! pgrep -f 'yolo detect train' >/dev/null && break
  sleep 60
done
if [ ! -f $RUN/weights/best.pt ]; then echo 'training did not produce best.pt'; exit 1; fi
echo "=== training done $(date) ==="
tail -3 $RUN/results.csv | cut -c1-120

echo '=== exporting 608 square ==='
$V -c "
from ultralytics import YOLO
YOLO('$RUN/weights/best.pt').export(format='onnx', imgsz=608, opset=12)
import shutil; shutil.move('$RUN/weights/best.onnx', '/Users/claw/train/pv11_yolo11m_608.onnx')
print('  wrote pv11_yolo11m_608.onnx')"

echo '=== exporting 1088x608 ==='
$V -c "
from ultralytics import YOLO
YOLO('$RUN/weights/best.pt').export(format='onnx', imgsz=[608,1088], opset=12)
import shutil; shutil.move('$RUN/weights/best.onnx', '/Users/claw/train/pv11_yolo11m_1088x608.onnx')
print('  wrote pv11_yolo11m_1088x608.onnx')"

cd /Users/claw/aicam
echo '=== A: deployed yolo11s 608  vs  new yolo11m 608 (isolates model size) ==='
/Users/claw/aicam/.venv/bin/python compare_models.py   --dataset /Users/claw/train/packages-vehicles2-v11 --split test --labels package,vehicle   --old /Users/claw/aicam-models/packages_vehicles_yolo11s.onnx --old-backend ultralytics   --new /Users/claw/train/pv11_yolo11m_608.onnx --new-backend ultralytics   --threshold package=0.80 --threshold vehicle=0.70 2>&1 | grep -vE 'onnxruntime|CoreML'

echo '=== B: new yolo11m 608  vs  new yolo11m 1088x608 (isolates geometry) ==='
/Users/claw/aicam/.venv/bin/python compare_models.py   --dataset /Users/claw/train/packages-vehicles2-v11 --split test --labels package,vehicle   --old /Users/claw/train/pv11_yolo11m_608.onnx --old-backend ultralytics --old-size 608x608   --old-dataset /Users/claw/train/packages-vehicles2-v11   --new /Users/claw/train/pv11_yolo11m_1088x608.onnx --new-backend ultralytics --new-size 1088x608   --new-dataset /Users/claw/train/pv-v12-rect   --threshold package=0.80 --threshold vehicle=0.70 2>&1 | grep -vE 'onnxruntime|CoreML'
echo "=== done $(date) ==="
