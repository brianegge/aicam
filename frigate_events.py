"""Tell Frigate about the animals it cannot see, so it keeps the clip.

Frigate runs COCO, which has no coyote, fox, raccoon, rabbit or deer, and at
its detect resolution the animals that are in COCO are a few dozen pixels
across. On 2026-09-30 a coyote crossed the garden and on 2026-10-01 a
raccoon spent a minute on the tree-line lawn; aicam alerted on both and
Frigate recorded no event for either -- its own model scored the raccoon
"person" 0.37 at best and the coyote "dog" 0.50 on one frame in twelve. The
coyote's video had to be pulled off the NVR.

So when aicam alerts on an animal, it creates a Frigate event on the same
camera (POST /api/events/<camera>/<label>/create). Frigate starts it the
camera's pre-capture before the call, keeps the recording with the event,
and lists it in Review under aicam's label. A manual event is not published
on the /events MQTT topic, so lpr-enrich never sees these.

Configured under [frigate]:

    url = http://192.168.254.31:5000
    event-labels = cat,coyote,deer,dog,fox,rabbit,raccoon   (default)
    event-duration = 60                                      (seconds)

Never in the way of the alert: the call runs on its own thread with a short
timeout, and a failure is a log line.
"""

import logging
import threading

import requests

from frigate_lpr import frigate_camera_name

logger = logging.getLogger(__name__)

DEFAULT_LABELS = "cat,coyote,deer,dog,fox,rabbit,raccoon"
# notify() marks predictions it will not alert on with this priority.
IGNORED = -4


def _labels(config):
    raw = config["frigate"].get("event-labels", DEFAULT_LABELS)
    return set(l.strip() for l in raw.split(",") if l.strip())


def events_for(cam, predictions, config):
    """The (camera, label, body) events worth creating for one alert."""
    if "frigate" not in config or not config["frigate"].get("url"):
        return []
    camera = frigate_camera_name(cam)
    if not camera:
        return []
    wanted = _labels(config)
    duration = config["frigate"].getint("event-duration", 60)
    by_label = {}
    for p in predictions:
        if "ignore" in p or p.get("departed"):
            continue
        if p.get("priority", IGNORED) <= IGNORED:
            continue
        label = p.get("tagName", "")
        if label.endswith("_road"):
            label = label[:-5]
        if label not in wanted:
            continue
        by_label.setdefault(label, []).append(p)
    out = []
    for label, preds in sorted(by_label.items()):
        best = max(p.get("probability", 0) for p in preds)
        boxes = [{"box": [p["boundingBox"]["left"], p["boundingBox"]["top"],
                          p["boundingBox"]["width"], p["boundingBox"]["height"]],
                  "score": int(round(p.get("probability", 0) * 100))}
                 for p in preds if "boundingBox" in p]
        out.append((camera, label, {
            "sub_label": "aicam",
            "score": round(best, 3),
            "duration": duration,
            "include_recording": True,
            "draw": {"boxes": boxes},
        }))
    return out


def _post(base_url, camera, label, body):
    try:
        r = requests.post("%s/api/events/%s/%s/create" % (base_url.rstrip("/"), camera, label),
                          json=body, timeout=10)
        r.raise_for_status()
        logger.info("frigate: %s on %s recorded as event %s", label, camera,
                    r.json().get("event_id"))
    except Exception as e:
        logger.warning("frigate: could not create a %s event on %s: %s", label, camera, e)


def mark(cam, predictions, config):
    """Create the events for this alert in the background. Returns how many."""
    events = events_for(cam, predictions, config)
    base_url = config["frigate"]["url"] if events else None
    for camera, label, body in events:
        threading.Thread(target=_post, args=(base_url, camera, label, body),
                         daemon=True).start()
    return len(events)
