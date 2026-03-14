# SimpleScan - AI Camera Detection System

AI-powered camera monitoring system that detects people, vehicles, animals, and packages using ONNX/TensorRT models on an NVIDIA Jetson Nano. Integrates with Home Assistant, MQTT, and Pushover for real-time notifications.

## Architecture

```
IP Cameras (FTP/HTTP) -> Jetson Nano (YOLOv4 inference) -> MQTT / Pushover / Home Assistant
```

- **15 cameras** polled via FTP motion events or HTTP snapshots
- **3 ONNX models**: color, grayscale, and vehicle/package detection
- **MQTT** publishes per-camera object counts for Home Assistant sensors
- **Pushover** sends priority-based mobile notifications
- **Home Assistant** provides mode (home/away/night) and controls (e.g. power cycling the Jetson)

## Road Classification

Cameras can define a `road_line` in `config.txt` to classify detections near public roads differently. Objects above the road line are reclassified as `_road` variants (e.g. `person_road`, `vehicle_road`, `dog_road`), which have separate notification priorities.

### Configuration

Per-camera `road_line` key in `config.txt`:

| Format | Meaning | Example |
|--------|---------|---------|
| `x1:y1, x2:y2, ...` | Piecewise linear road boundary | `0:0.5, 1.0:0.2` |
| `all` | Entire frame is road | `all` |
| *(omitted)* | No road classification | |

Current cameras with road lines:

| Camera | Config | Effect |
|--------|--------|--------|
| **driveway** | `road_line = 0:0.5, 1.0:0.2` | Diagonal line, objects above are `_road` |
| **peach tree** | `road_line = 0:0.31, 0.651:0.348, 1.0:0.479` | 3-point piecewise line |
| **mailbox** | `road_line = all` | All detections are `_road` variants |

The `Camera.road_y_at(x)` method interpolates the road line at any x position.

## Notification Modes

The system operates in one of three modes, determined by Home Assistant state:

1. **night** -- `input_boolean.night_mode` is on
2. **home** -- anyone in `group.egge` has presence state `"home"`
3. **away** -- otherwise

Mode selects which priority config section is used (`[priority-night]`, `[priority-home]`, or `[priority-away]`).

## Dog Notification Triggers

A notification is sent (priority = 1) when any of the following conditions are met:

- **Dog on the road without a person** -- `dog_road` detected with no `person_road` nearby. If a person is on the road, priority drops to -3 (suppressed as "person walking dog").
- **Dog in the garage** -- `dog` on the garage camera with confidence > 90% and no person detected.
- **Dog near a package** -- both `dog` and `package` detected in the same frame.

## MQTT Reconnection

On unexpected MQTT disconnect (e.g. Home Assistant restart), the client automatically retries with exponential backoff (1-30s) for up to 5 minutes. If reconnection fails after 5 minutes, the process shuts down and systemd restarts it.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, camera polling loop, MQTT setup |
| `detect.py` | Object detection, road classification, exclusion zones |
| `notify.py` | Notification logic (Pushover, Home Assistant) |
| `camera.py` | Camera capture, MQTT publishing, road line parsing |
| `homeassistant.py` | Home Assistant API integration |
| `codeproject.py` | CodeProject AI ALPR integration |
| `config.txt` | Per-deployment configuration (not in repo) |
| `config-test.txt` | Test configuration with mock values |
| `excludes.json` | Static bounding box exclusion zones |
