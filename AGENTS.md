# Agent Instructions

## Project Overview

This is an AI camera detection system that monitors IP cameras for object detection (people, vehicles, animals, packages) using ONNX/TensorRT models on an NVIDIA Jetson Nano. It integrates with Home Assistant, MQTT, and Pushover for notifications.

## Logs

All logs are on the remote Jetson Nano at `egge@egge-nano.home`. Always use SSH to access them — there are no local log files.

```bash
# Check service status (uptime, PID, recent output)
ssh egge@egge-nano.home "systemctl status aicam"

# View today's logs
ssh egge@egge-nano.home "journalctl -u aicam --since today --no-pager"

# Search for errors, crashes, restarts
ssh egge@egge-nano.home "journalctl -u aicam --since today --no-pager | grep -iE 'error|exception|traceback|watchdog|started|stopped|failed'"

# Follow logs in real-time
ssh egge@egge-nano.home "journalctl -u aicam -f"

# Check all system errors today (not just aicam)
ssh egge@egge-nano.home "journalctl --since today | grep -iE 'crash|fatal|traceback|exception|error' | tail -80"
```

### Investigating restarts and reboots

The Jetson Nano is on a HA-controlled smart switch. Home Assistant will power-cycle the Nano if aicam becomes unresponsive (sensor goes unavailable for 10 minutes). The HA automation is called "Jetson Nano down" and increments `counter.jetson_nano_crashes` on each power cycle.

The availability mechanism: aicam sets an MQTT Last Will and Testament on `aicam/status` (payload "offline"). All HA sensors use this as their `availability_topic`. When the MQTT broker detects the client is gone, it publishes the LWT, and sensors go unavailable. After 10 minutes unavailable, HA power-cycles the smart switch.

```bash
# Check if the whole system rebooted (lists boot sessions)
ssh egge@egge-nano.home "journalctl --list-boots"

# Check reset source from Tegra PMC (power-on vs watchdog vs software)
ssh egge@egge-nano.home "dmesg | grep 'PMC reset source'"
# TEGRA_POWER_ON_RESET = power loss or HA smart-switch power cycle
# TEGRA_WATCHDOG_RESET = watchdog triggered
# TEGRA_SOFTWARE_RESET = software reboot (e.g. sudo reboot)

# Check previous boot's aicam logs (if journal retained them)
# NOTE: hard power cuts lose the previous boot's journal entirely
ssh egge@egge-nano.home "journalctl -b -1 -u aicam --no-pager | tail -30"

# Check for OOM kills, thermal shutdowns
ssh egge@egge-nano.home "dmesg | grep -iE 'oom|killed|thermal|shutdown|voltage'"
```

## Deployment

### Deploy to egge-nano

```bash
# 1. Commit and push changes to GitHub
git push

# 2. Pull and restart on egge-nano (pulls from root of repo, not simplescan/)
ssh egge@egge-nano.home "cd /home/egge/detector && git pull && sudo cp simplescan/aicam.service /etc/systemd/system/aicam.service && sudo systemctl daemon-reload && sudo systemctl restart aicam"

# 3. Verify service is running
ssh egge@egge-nano.home "systemctl status aicam | head -15"
```

### Troubleshooting: GitHub SSH on egge-nano

If `git pull` fails with "Host key verification failed", GitHub's IP keys have rotated:

```bash
# Remove stale IP key and re-scan
ssh egge@egge-nano.home "ssh-keygen -R \$(ssh -T git@github.com 2>&1 | grep -oP '[\d.]+') 2>/dev/null; ssh-keyscan -t ecdsa github.com >> ~/.ssh/known_hosts 2>&1"
```

### Configuration

Config file location: `egge@egge-nano:~/detector/simplescan/config.txt`

Key sections:
- `[mqtt]` - MQTT broker credentials
- `[homeassistant]` - Home Assistant API URL and token
- `[pushover]` - Pushover notification credentials
- `[codeproject]` - CodeProject AI ALPR URL
- `[cam0]` through `[camN]` - Camera configurations

## Testing

### Run tests locally with Python 3.6.9 container

```bash
# Build test container
podman build -t simplescan-test -f Containerfile .

# Run tests
podman run --rm -v "$(pwd)":/app -w /app simplescan-test pytest -v

# Compile check
podman run --rm -v "$(pwd)":/app -w /app simplescan-test python -m compileall -q .
```

### Start podman machine (if needed)

```bash
podman machine start
```

## Key Files

- `main.py` - Entry point, camera polling loop
- `detect.py` - Object detection logic
- `notify.py` - Notification handling (Pushover, Home Assistant)
- `camera.py` - Camera capture and MQTT publishing
- `homeassistant.py` - Home Assistant API integration
- `codeproject.py` - CodeProject AI ALPR integration
- `Containerfile` - Python 3.6.9 test container
