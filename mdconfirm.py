"""Let MegaDetector vouch for a one-frame wildlife sighting.

confirm.py holds an animal back until it is seen on fresh ground a second time
on the same camera. That is what stopped the rocks and stumps -- and it also
drops every animal that crosses a camera inside one sweep. Over 2026-11-15 to
22 it held 320 outdoor wildlife visits that never alerted. MegaDetector
(MDv1000-redwood, run by md_server.py) found an animal on the held box in 39 of
them; by eye 29 of those were real -- foxes on the shed, play and peach tree
cameras, a buck on garage-l, deer on the tree line, the coyote on peach tree
at 01:49 on 11-15 that it scored 0.93 -- 4 were Rufus in daylight, 1 the cat,
and 3-5 showed nothing.

So it may confirm, and it may never veto. Its blind spots are real ones: it
scored the cat seen from above in the garage 0.00-0.07 on every frame, and
Rufus in daylight 0.02 to 0.82. Used to silence, those would cost real alerts;
used only to let a first sighting through, they cost nothing -- the movement
gate still runs for everything MegaDetector does not vouch for.

Night only by default (dark-only): all four Rufus cases were daylight, and the
29 real ones were all infra-red. Outdoor only: the garage is where it is blind.

Every failure -- service down, timeout, bad reply -- answers "no", which leaves
the sighting to the movement gate exactly as before this existed.
"""
import io
import json
import logging
import time

import requests
from PIL import Image

from utils import bb_intersection_over_union

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:5060/detect"
DEFAULT_CLASSES = "deer,fox,coyote,raccoon,rabbit,bobcat"
# A failure is logged once per this many seconds, not once per frame.
QUIET_SECONDS = 600
_last_failure = [0.0]


def _box(b):
    """aicam boundingBox dict -> [x, y, w, h]."""
    return [b["left"], b["top"], b["width"], b["height"]]


def overlaps(aicam_box, md_box):
    """The same animal, give or take how two detectors draw a box.

    IoU alone is too strict between models -- the detector judged a 608x608
    stretch and MegaDetector the frame at its own size -- so a box whose centre
    lies inside the other also counts.
    """
    a = _box(aicam_box) if isinstance(aicam_box, dict) else list(aicam_box)
    m = list(md_box)
    as_dict = lambda b: {"left": b[0], "top": b[1], "width": b[2], "height": b[3]}
    if bb_intersection_over_union(as_dict(a), as_dict(m)) > 0.1:
        return True
    def inside(b, c):
        cx, cy = c[0] + c[2] / 2.0, c[1] + c[3] / 2.0
        return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]
    return inside(a, m) or inside(m, a)


def _jpeg(image):
    if isinstance(image, Image.Image):
        pil = image
    else:
        # OpenCV frames are BGR
        pil = Image.fromarray(image[:, :, ::-1])
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class Voucher(object):
    """One frame's worth of MegaDetector, asked at most once per frame."""

    def __init__(self, cfg, cam_name, image, dark, session=None):
        self.url = cfg.get("url", DEFAULT_URL)
        self.timeout = cfg.getfloat("timeout", 5.0)
        self.min_score = cfg.getfloat("min-score", 0.6)
        self.classes = set(c.strip() for c in cfg.get("classes", DEFAULT_CLASSES).split(",")
                           if c.strip())
        excluded = set(c.strip() for c in cfg.get("exclude-cameras", "garage").split(",")
                       if c.strip())
        self.eligible_camera = cam_name not in excluded
        self.eligible_light = dark or not cfg.getboolean("dark-only", True)
        self.cam_name = cam_name
        self.image = image
        self.http = session or requests
        self._detections = None
        self.last = None

    def _ask(self):
        if self._detections is None:
            self._detections = []
            try:
                start = time.time()
                r = self.http.post(self.url, data=_jpeg(self.image), timeout=self.timeout,
                                   headers={"Content-Type": "image/jpeg"})
                r.raise_for_status()
                self._detections = [d for d in r.json().get("detections") or []
                                    if d.get("category") == "animal"]
                logger.debug("%s: MegaDetector answered in %.2fs", self.cam_name,
                             time.time() - start)
            except Exception as e:
                if time.time() - _last_failure[0] > QUIET_SECONDS:
                    _last_failure[0] = time.time()
                    logger.warning("MegaDetector unavailable (%s); one-frame sightings "
                                   "wait for the movement gate", e)
        return self._detections

    def __call__(self, p):
        """True if MegaDetector sees an animal on this detection's box."""
        self.last = None
        if p["tagName"] not in self.classes or not self.eligible_camera or not self.eligible_light:
            return False
        best = max((d for d in self._ask() if overlaps(p["boundingBox"], d["bbox"])),
                   key=lambda d: d["conf"], default=None)
        if best is None:
            return False
        self.last = best["conf"]
        if best["conf"] < self.min_score:
            logger.info("%s: MegaDetector sees %s only at %.2f (< %.2f); waiting for "
                        "movement", self.cam_name, p["tagName"], best["conf"], self.min_score)
            return False
        return True


def voucher_for(config, cam_name, image, dark):
    """A Voucher for this frame, or None when [megadetector] is off."""
    if not config.has_section("megadetector"):
        return None
    cfg = config["megadetector"]
    if not cfg.getboolean("enabled", False):
        return None
    return Voucher(cfg, cam_name, image, dark)
