# aicam

## Deployment

Runs on **claw-mini** (`openclaw.home`, Apple M4, macOS). Moved off egge-nano
2026-09-02: the Jetson spent ~80% of each frame decoding 4K JPEGs on its CPU
(471 ms/frame end to end), against 24 ms on the M4.

- SSH: `ssh openclaw.home` (user `claw`)
- Code deployed at: `/Users/claw/aicam/` (clone of this repo)
- Python venv: `/Users/claw/aicam/.venv` (3.12, built with `uv`)
- Capture/log data: `/Users/claw/aicam-data/`
- Config file: `/Users/claw/aicam/config.txt` (mode 0600 — holds tokens)
- Service: launchd agent `com.brianegge.aicam`, installed from
  `com.brianegge.aicam.plist` in this repo to `~/Library/LaunchAgents/`

No `--trt` on this host: it runs the ONNX Runtime backend with the CoreML
execution provider. CoreML computes in fp16, so confidences differ from
TensorRT in the third decimal — watch class thresholds that sit on a boundary
(`dog=0.95`).

## Models

Each model section picks its decoder with `backend=`, so they do not all have
to be the same architecture:

| model | file | backend |
|-------|------|---------|
| color | `ipcams_color_yolov4.onnx` | `yolov4` (default) |
| grey | `ipcams_grey_yolov4.onnx` | `yolov4` (default) |
| vehicle | `packages_vehicles_yolo11s.onnx` | `ultralytics` |

- `yolov4` — the darknet-lineage models, output `boxes[1,N,1,4]` +
  `confs[1,N,nc]`, normalised corners.
- `ultralytics` — YOLO11/v8, one output `[1, 4+nc, N]`, boxes as `cx,cy,w,h` in
  input pixels, no objectness.

`vehicles_yolov4.onnx` is still on the host; rolling back is `onnx=` plus
removing `backend=` in `[vehicle-model]`, then a restart.

## Retraining a model on claw-mini

Darknet has **no GPU path on Apple silicon**, so the original YOLOv4 pipeline
cannot be reproduced here — CPU training of the old cfg's 6000 batches would
take days. Retrain with Ultralytics instead and set `backend=ultralytics`.

```bash
ssh openclaw.home
# one-time
/opt/homebrew/bin/uv venv -p 3.12 ~/train/.venv
/opt/homebrew/bin/uv pip install -p ~/train/.venv/bin/python ultralytics roboflow onnx onnxslim

# fetch the dataset (YOLOv8 format; YOLO11 is the same layout)
export RFKEY=$(python3 -c "import configparser;c=configparser.ConfigParser();c.read('/Users/claw/aicam/config.txt');print(c['roboflow']['api-key'])")
~/train/.venv/bin/python -c "
from roboflow import Roboflow
import os
Roboflow(api_key=os.environ['RFKEY']).workspace('egge-public').project('packages-vehicles2') \
    .version(11).download('yolov8', location='/Users/claw/train/packages-vehicles2-v11')"

# train (~70 s/epoch for yolo11s, so ~2 h for 100 epochs)
nohup ~/train/.venv/bin/yolo detect train \
  data=/Users/claw/train/packages-vehicles2-v11/data.yaml \
  model=yolo11s.pt epochs=100 imgsz=608 batch=16 device=mps workers=4 \
  project=/Users/claw/train/runs name=pv11-yolo11s > ~/train/train.log 2>&1 &

# export, then install
~/train/.venv/bin/python -c "
from ultralytics import YOLO
YOLO('/Users/claw/train/runs/pv11-yolo11s/weights/best.pt').export(format='onnx', imgsz=608, opset=12)"
cp /Users/claw/train/runs/pv11-yolo11s/weights/best.onnx ~/aicam/<name>.onnx
```

**Train at `imgsz=608`, not the 640 default.** `camera.py`'s `resize()` hands
every model a 608x608 array, so matching it avoids touching the capture path.

`/Users/claw/train/compare.py` scores an old and a new model against the same
held-out split, using the same preprocessing aicam uses, and reports per-class
precision/recall/F1 at the thresholds actually configured in `[thresholds]`.
Run it before swapping a model in — the validation mAP the trainer prints is
not measured at your operational thresholds.

### Network constraints

claw-mini sits on the isolated OpenClaw subnet (192.168.251.0/24, a dedicated
EdgeRouter eth1 port). It **cannot reach the IPCAMS VLAN** (192.168.253.0/24) —
`OPENCLAW_IN` rule 30 drops it. Reachable hosts are the `OPENCLAW_ALLOW` group
only: Home Assistant (.254.30), Blue Iris (.254.31), LibreNMS (.254.35) and
MQTT (.254.16).

Consequences baked into the config:
- Blue Iris is addressed by its **LAN** IP `192.168.254.31`, not
  `blueiris-3.home` (.253.7, unreachable).
- `blueiris-only=true` in `[detector]`: the direct `snapshot.cgi` fallback has
  no route and would only burn its 20 s connect timeout, stalling the sweep.
- No `ftp-path`: cameras cannot push into this subnet. The interval sweep is
  the only trigger, so cameras run `interval=3` instead of the old 30 s. A full
  15-camera sweep costs ~1.3 s (~0.11 s of that inference), which beats the
  latency the FTP push used to give.

## Services
- `com.brianegge.aicam` (launchd) — main camera detection loop
- `com.brianegge.aicam-review` (launchd) — Roboflow review upload server
  (`roboflow_upload.py --port 5050`)
- `com.brianegge.aicam-cleanup` (launchd) — hourly capture trim

The review server **must run on the same host as aicam**: the "Flag for Review"
link in a Pushover alert resolves to a file under `save-path/review/`, which
the server reads off local disk. Home Assistant's
`rest_command.aicam_roboflow_upload` has to point at this host — it still
pointed at `egge-nano.home:5050` after the move, so every flag returned
`404 file not found`.

```bash
curl -s http://openclaw.home:5050/health
ssh openclaw.home "tail -n 50 ~/aicam-data/aicam-review.log"
```

## Logs
```bash
# View aicam logs
ssh openclaw.home "tail -n 100 ~/aicam-data/aicam.log"

# Follow
ssh openclaw.home "tail -f ~/aicam-data/aicam.log"

# Search for errors
ssh openclaw.home "grep -i error ~/aicam-data/aicam.log"

# Service state / restart
ssh openclaw.home "launchctl print gui/\$(id -u)/com.brianegge.aicam | head -20"
ssh openclaw.home "launchctl kickstart -k gui/\$(id -u)/com.brianegge.aicam"
```

Diagnostic lines are also shipped to the LibreNMS syslog sink (see `[syslog]`
in config and `logsetup.py`).

## Deploy Changes
```bash
ssh openclaw.home "cd ~/aicam && git pull && launchctl kickstart -k gui/\$(id -u)/com.brianegge.aicam"
```

## egge-nano (decommissioned for aicam)

The nano still exists and `aicam.service` is left installed but stopped, so the
old path is one `systemctl start aicam` away if the Mac has to be rolled back.
Do not run both at once — they publish to the same MQTT topics and would send
duplicate Pushover notifications.

- SSH: `ssh egge@egge-nano.home`, code at `/home/egge/aicam/`
- Interpreter: `/home/egge/detector/bin/python3` (Python 3.6.9), from the
  separate `brianegge/yolov3` fork at `/home/egge/detector/`
- ExecStart was `/home/egge/detector/bin/python3 -u /home/egge/aicam/main.py --trt`
- `journalctl -u aicam -n 100 --no-pager` for its logs
