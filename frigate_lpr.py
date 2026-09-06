"""Read licence plates from Frigate's native LPR.

Replaces the CodeProject AI round trip. aicam used to crop each vehicle and
POST it to a service on the Blue Iris host; that host was rebuilt as the
Frigate box on 2026-09-05 and the service went with it. Frigate 0.17 does plate
recognition itself, on the same GPU it already uses for detection, and crops
the plate off its own car detections -- so there is nothing to upload.

Frigate also owns the plate-to-person mapping now, via `lpr.known_plates`
generated from license-plates.json. A recognised plate that matches one sets
the event's `sub_label`, which is carried through here as `plate_owner`.

Matching is by camera and time, not by bounding box. aicam and Frigate detect
independently, so their boxes will not correspond; but a plate recognised on
the same camera within LOOKBACK_SECONDS is the same vehicle in every practical
case, and the alternative -- geometric matching across two detectors on two
different frames -- would be far more fragile than the thing it replaced.
"""

import logging
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

# How recently Frigate must have recognised a plate for it to be attributed to
# a vehicle aicam is asking about. Generous, because a car sits in the driveway
# far longer than this and a plate is only readable for part of its approach.
LOOKBACK_SECONDS = 180

# Frigate's own confidence in the OCR. Below this the read is not worth
# announcing -- aicam's is_plausible_plate() shape check is no longer applied,
# since Frigate has already filtered to plate-shaped regions.
MIN_PLATE_SCORE = 0.8

REQUEST_TIMEOUT = 10


def frigate_camera_name(cam):
    """The Frigate stream name for this camera, or None.

    Derived from the configured snapshot URL rather than a second config key:
    the URL already names the stream (`?src=peach_tree`) and a separate setting
    would be one more thing to keep in sync.
    """
    uri = getattr(cam, "blueiris_uri", None)
    if not uri:
        return None
    src = parse_qs(urlparse(uri).query).get("src")
    return src[0] if src else None


def fetch_recent_plates(base_url, camera, now_ts):
    """Plates Frigate recognised on `camera` within the lookback window.

    Returns a list of (plate, owner, score), most recent first.
    """
    try:
        resp = requests.get(
            base_url.rstrip("/") + "/api/events",
            params={
                "cameras": camera,
                "after": now_ts - LOOKBACK_SECONDS,
                "limit": 20,
                "include_thumbnails": 0,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception:
        logger.exception("Could not read plates from Frigate for %s", camera)
        return []

    out = []
    for event in events:
        data = event.get("data") or {}
        plate = data.get("recognized_license_plate")
        if not plate:
            continue
        score = data.get("recognized_license_plate_score") or 0.0
        if score < MIN_PLATE_SCORE:
            logger.debug("Ignoring %s on %s: score %.2f", plate, camera, score)
            continue
        out.append((plate, event.get("sub_label"), score))
    return out


def read_plates(cam, vehicles, config):
    """Attach any plate Frigate has recognised to the vehicles awaiting one.

    Mirrors alpr.read_plates(): records `plate`, `plate_owner`, `plate_read`
    and `alpr_count` on the prediction, and returns plates seen for the first
    time so the caller can decide whether to notify.
    """
    if not vehicles:
        return []
    base_url = config["frigate"]["url"] if "frigate" in config else None
    if not base_url:
        return []
    camera = frigate_camera_name(cam)
    if not camera:
        logger.debug("No Frigate stream name for %s; skipping LPR", cam.name)
        return []

    import time

    found = fetch_recent_plates(base_url, camera, time.time())
    if not found:
        # Record the attempt so the retry cadence in wants_alpr() advances;
        # without this a vehicle would be re-queried on every single frame.
        for vehicle in vehicles:
            vehicle["alpr_count"] = vehicle.get("alpr_count", 0) + 1
        return []

    plate, owner, score = found[0]
    new_plates = []
    for vehicle in vehicles:
        vehicle["alpr_count"] = vehicle.get("alpr_count", 0) + 1
        if vehicle.get("plate") == plate:
            continue
        vehicle["plate"] = plate
        if owner:
            vehicle["plate_owner"] = owner
        # Read once; stop retrying this vehicle.
        vehicle["plate_read"] = True
        new_plates.append(plate)
        logger.info(
            "Frigate read plate %s on %s (%.2f)%s",
            plate,
            cam.name,
            score,
            " -> %s" % owner if owner else "",
        )
    return new_plates
