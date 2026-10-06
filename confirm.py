"""Hold animal alerts until something has been seen to move.

Animals move; rocks, stumps and IR-lit ground texture do not. The detector
cannot tell them apart on a single frame -- it called bare ground at the peach
tree a rabbit hundreds of times, and the tree line a 72% deer -- and while
verify.py was paid for, a vision model threw those out. Without it they all
alerted.

So a class in [confirm] classes does not alert on its first sighting. It
becomes *active* when a second sighting lands on fresh ground within the
window: on a different camera, or on the same camera in a box that overlaps
no earlier sighting there. A false positive sits in one place and keeps
overlapping itself, so it never gets there. Once active, a class stays
active until a whole window passes with no sighting of it at all, and every
track of that class held back meanwhile is announced.

Sightings in the same frame never confirm each other: two stones in one
picture are not movement. Sightings only from the hysteresis hold do not
count either -- they exist only by matching an established track.

Deliberately no motion requirement upstream: aicam polls frames whether or not
anything moved, which is what lets it see small things motion detection
misses. This gate is the movement check, applied after the fact, at the scale
of the whole box.
"""

import logging
from datetime import timedelta

from utils import bb_intersection_over_union

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60


class MovementGate:
    def __init__(self, classes, window_seconds=WINDOW_SECONDS):
        self.classes = set(classes)
        self.window = timedelta(seconds=window_seconds)
        # class -> [(time, camera, box)], oldest first, pruned to the window
        self.sightings = {}
        # class -> time of the sighting that activated it
        self.activated = {}

    def gated(self, tag_name):
        return tag_name in self.classes

    def _prune(self, tag_name, now):
        kept = [s for s in self.sightings.get(tag_name, [])
                if now - s[0] < self.window]
        self.sightings[tag_name] = kept
        if not kept:
            self.activated.pop(tag_name, None)
        return kept

    def observe(self, tag_name, cam_name, box, now):
        """Record a sighting; True if this one activated the class."""
        if not self.gated(tag_name):
            return False
        earlier = [s for s in self._prune(tag_name, now) if s[0] < now]
        self.sightings[tag_name].append((now, cam_name, dict(box)))
        if tag_name in self.activated or not earlier:
            return False
        if any(c == cam_name and bb_intersection_over_union(b, box) > 0
               for _, c, b in earlier):
            return False
        self.activated[tag_name] = now
        logger.info("%s confirmed by movement on %s (%d earlier sighting(s))",
                    tag_name, cam_name, len(earlier))
        return True

    def is_active(self, tag_name, now):
        if not self.gated(tag_name):
            return True
        self._prune(tag_name, now)
        return tag_name in self.activated


_gate = None
_gate_config = None


def gate_for(config):
    """The process-wide gate, or None when [confirm] is not configured.

    One gate for every camera: a deer seen on the tree line and then the
    deck has moved, whichever camera saw it first.
    """
    global _gate, _gate_config
    if "confirm" not in config:
        return None
    if _gate is None or _gate_config is not config:
        cfg = config["confirm"]
        classes = {c.strip() for c in cfg.get("classes", "").split(",")
                   if c.strip()}
        _gate = MovementGate(classes, cfg.getint("window", WINDOW_SECONDS))
        _gate_config = config
    return _gate


def apply(gate, valid_predictions, new_predictions, tracked_pairs, now):
    """Hold back unconfirmed animal arrivals; release them once confirmed.

    Returns the predictions that count as arrivals this frame: new tracks of
    ungated or active classes, plus held tracks whose class is now active.
    Held tracks carry "unconfirmed" until then, and the current frame's copy
    of each is marked too, so the caller can keep it out of the alert.
    """
    for p in valid_predictions:
        if not p.get("hold_only"):
            gate.observe(p["tagName"], p["camName"], p["boundingBox"], now)
    track_of = {id(p): prev for p, prev in tracked_pairs}
    new_ids = set(id(p) for p in new_predictions)
    arrivals = []
    for p in valid_predictions:
        track = track_of.get(id(p))
        if id(p) in new_ids:
            if gate.is_active(p["tagName"], now):
                arrivals.append(p)
            else:
                p["unconfirmed"] = True  # p is the new track itself
        elif track is not None and track.get("unconfirmed"):
            if gate.is_active(p["tagName"], now):
                del track["unconfirmed"]
                # notify() treats a matched track as already announced.
                p.pop("iou", None)
                arrivals.append(p)
            else:
                p["unconfirmed"] = True
    return arrivals
