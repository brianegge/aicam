# aicam

## Deployment
- Runs on **egge-nano** (Jetson Nano)
- SSH: `ssh egge@egge-nano.home`
- Code deployed at: `/home/egge/aicam/`
- Python venv: `/home/egge/detector/bin/python3`
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
