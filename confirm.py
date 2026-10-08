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

People are different: they stand still, so movement proves nothing, and a
person is never what a rock looks like. What does mistake itself for a person
is something else passing through -- the dog walking away from the deck camera
at night scored person 0.66 on one frame and dog 0.72 to 0.91 on the frames
after. So [confirm] second_frame = person:0.75 holds a new person track that
scores under 0.75 until that same track is seen again in a later frame, as a
real detection rather than a hysteresis hold. A confident person alerts at
once, as before; a borderline one waits at most one poll; one that was the dog
never alerts. This is per track, not per class: a long-standing low-scoring
"person" somewhere else must not vouch for this one.

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
    def __init__(self, classes, window_seconds=WINDOW_SECONDS, second_frame=None):
        self.classes = set(classes)
        # class -> score below which a new track waits for a second frame
        self.second_frame = dict(second_frame or {})
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

    def needs_second_frame(self, p):
        limit = self.second_frame.get(p["tagName"])
        return limit is not None and (p.get("probability") or 0) < limit

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
        _gate = MovementGate(classes, cfg.getint("window", WINDOW_SECONDS),
                             parse_second_frame(cfg.get("second_frame", "")))
        _gate_config = config
    return _gate


def parse_second_frame(text):
    """"person:0.75, cat:0.7" -> {"person": 0.75, "cat": 0.7}."""
    limits = {}
    for item in text.split(","):
        if item.strip():
            tag, _, limit = item.partition(":")
            limits[tag.strip()] = float(limit)
    return limits


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
            if not gate.is_active(p["tagName"], now):
                p["unconfirmed"] = True  # p is the new track itself
            elif gate.needs_second_frame(p):
                p["unconfirmed"] = True
                p["second_frame"] = True
            else:
                arrivals.append(p)
        elif track is not None and track.get("second_frame"):
            if p.get("hold_only"):
                p["unconfirmed"] = True
            else:
                del track["unconfirmed"]
                del track["second_frame"]
                p.pop("iou", None)
                arrivals.append(p)
        elif track is not None and track.get("unconfirmed"):
            if gate.is_active(p["tagName"], now):
                del track["unconfirmed"]
                # notify() treats a matched track as already announced.
                p.pop("iou", None)
                arrivals.append(p)
            else:
                p["unconfirmed"] = True
    return arrivals
