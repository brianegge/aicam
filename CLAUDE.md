# aicam

## Deployment
- Runs on **egge-nano** (Jetson Nano, Python 3.6.9)
- SSH: `ssh egge@egge-nano.home`
- Code deployed at: `/home/egge/aicam/` (clone of this repo)
- Python venv: `/home/egge/detector/bin/python3` — lives in a separate fork (`brianegge/yolov3` at `/home/egge/detector/`) that only provides the interpreter + site-packages; the aicam source tree is not inside it.
- aicam.service ExecStart: `/home/egge/detector/bin/python3 -u /home/egge/aicam/main.py --trt`
- Config file: `/home/egge/aicam/config.txt`

## Services (systemd system units)
- `aicam.service` - Main camera detection loop (`main.py --trt`)
- `aicam-review.service` - Roboflow review upload server (`roboflow_upload.py --port 5050`)

## Logs
```bash
# View aicam logs
ssh egge@egge-nano.home "journalctl -u aicam -n 100 --no-pager"

# View review upload server logs
ssh egge@egge-nano.home "journalctl -u aicam-review -n 100 --no-pager"

# Search for errors
ssh egge@egge-nano.home "journalctl -u aicam --no-pager" | grep -i error

# Restart service
ssh egge@egge-nano.home "sudo systemctl restart aicam"
```

## Deploy Changes
```bash
ssh egge@egge-nano.home "cd /home/egge/aicam && git pull && sudo systemctl restart aicam"
```
