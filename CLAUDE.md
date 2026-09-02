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
- `com.brianegge.aicam` (launchd, claw-mini) — main camera detection loop
- `aicam-review.service` (systemd, egge-nano) — Roboflow review upload server
  (`roboflow_upload.py --port 5050`), still on the nano

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
