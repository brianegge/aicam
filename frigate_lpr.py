"""Read licence plates from Frigate's native LPR.

Replaces the CodeProject AI round trip. aicam used to crop each vehicle and
POST it to a service on the Blue Iris host; that host was rebuilt as the
Frigate box on 2026-09-05 and the service went with it. Frigate 0.17 does plate
recognition itself, on the same GPU it already uses for detection, and crops
the plate off its own car detections -- so there is nothing to upload.

Frigate also owns the plate-to-person mapping now, via `lpr.known_plates`
generated from license-plates.json. A recognised plate that matches one sets
the event's `sub_label`, which is carried through here as `plate_owner`.

Matching is by camera, time AND position. Time alone was enough while cars
arrived one at a time, but on 2026-09-08 two vehicles were parked in the
driveway and the most recent plate -- the Subaru's BT70150 -- was attached to
the Jeep, which is CT 419875. The same lookup drives
script.house_cleaners_arrive, so a mis-attributed plate can open the garage for
the wrong car; that is the reason this is not merely a cosmetic fix.

aicam and Frigate detect independently on different frames, so their boxes will
never agree exactly -- but both are normalised to the same frame, so overlap is
still meaningful. Where it is ambiguous the plate is dropped rather than
guessed: an unnamed "Vehicle" is a much cheaper failure than the wrong name.
"""

import logging
from urllib.parse import parse_qs, urlparse

import requests

from utils import bb_intersection_over_union

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

# Minimum overlap between aicam's vehicle box and Frigate's, below which the
# plate is not considered to belong to that vehicle. Deliberately loose: the two
# detectors box the same car differently and on frames up to a second apart.
MIN_PLATE_IOU = 0.25

# How much better the best match must be than the runner-up before it is
# trusted. Two cars side by side on the same camera produce similar overlaps,
# and that is precisely the case where guessing is wrong.
IOU_MARGIN = 0.10


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
        box = data.get("box") or []
        # Frigate gives [x, y, w, h] normalised; aicam uses the same dict shape
        # as its own predictions so bb_intersection_over_union can compare them.
        region = None
        if len(box) == 4:
            region = {"left": box[0], "top": box[1],
                      "width": box[2], "height": box[3]}
        out.append((plate, event.get("sub_label"), score, region))
    return out


def match_plate(vehicle, candidates):
    """Which recognised plate belongs to this vehicle, if any.

    Returns (plate, owner, score) or None. With a single candidate and a single
    vehicle the old time-only behaviour is kept -- there is nothing to confuse
    it with. Otherwise the best spatial overlap wins, and only if it is clearly
    better than the runner-up.
    """
    usable = [c for c in candidates if c[3]]
    if not usable:
        # No boxes to compare (older Frigate, or a plate with no object box).
        # Only safe when there is exactly one candidate.
        return candidates[0][:3] if len(candidates) == 1 else None

    scored = sorted(
        ((bb_intersection_over_union(vehicle["boundingBox"], c[3]), c) for c in usable),
        key=lambda x: -x[0])
    best_iou, best = scored[0]
    if best_iou < MIN_PLATE_IOU:
        return None
    if len(scored) > 1 and best_iou - scored[1][0] < IOU_MARGIN:
        return None
    return best[:3]


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

    new_plates = []
    for vehicle in vehicles:
        vehicle["alpr_count"] = vehicle.get("alpr_count", 0) + 1
        matched = match_plate(vehicle, found)
        if not matched:
            logger.debug(
                "No plate confidently matches a vehicle on %s (%d candidate(s)); "
                "leaving it unnamed", cam.name, len(found))
            continue
        plate, owner, score = matched
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
