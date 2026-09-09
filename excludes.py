"""Exclusion geometries stored one per file, beside the frame that caused them.

excludes.json records a box and nothing else, so once a model is retrained
there is no way to tell whether an exclusion is still earning its place. Every
entry is permanent by default, and a stale one silently suppresses real
detections.

Here each exclusion is a pair sharing a stem:

    peach_tree-deer-23-90.yaml   the geometry, and why it exists
    peach_tree-deer-23-90.jpg    the frame the false positive came from

The stem is camera, label, and the box centre as whole percent of the frame,
so files sort together, a duplicate is obvious, and the position is readable
without opening anything. recheck_excludes.py replays each jpg through the
current model and reports which exclusions have become unnecessary.

excludes.json is still honoured; entries from both are merged.
"""

import glob
import json
import logging
import os
import re

logger = logging.getLogger("aicam")


def slug(text):
    """Filename-safe form of a camera or label name ('peach tree' -> peach_tree)."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def exclusion_stem(camera, label, box):
    """camera-label-centreX-centreY, each centre as whole percent."""
    # Half-up, not round(): Python rounds halves to even, so a centre at
    # exactly 22.5% would become 22 while 23.5% becomes 24. For a name a human
    # has to match against a file listing, predictable beats statistically
    # unbiased.
    cx = int((box["left"] + box["width"] / 2.0) * 100 + 0.5)
    cy = int((box["top"] + box["height"] / 2.0) * 100 + 0.5)
    return "%s-%s-%02d-%02d" % (slug(camera), slug(label), cx, cy)


def _as_box(d):
    return {k: float(d[k]) for k in ("left", "top", "width", "height")}


def load_dir(path):
    """Read every *.yaml in path into {camera: {label: [box, ...]}}."""
    found = {}
    if not path or not os.path.isdir(path):
        return found
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; ignoring exclusion directory %s", path)
        return found

    for fn in sorted(glob.glob(os.path.join(path, "*.yaml"))):
        try:
            with open(fn) as f:
                doc = yaml.safe_load(f) or {}
            if doc.get("disabled"):
                logger.info("exclusion %s is disabled, skipping", os.path.basename(fn))
                continue
            camera = doc["camera"]
            label = doc["label"]
            box = _as_box(doc["box"])
        except Exception as e:
            # One malformed file must not cost us every other exclusion.
            logger.warning("skipping exclusion %s: %s", os.path.basename(fn), e)
            continue
        box["comment"] = doc.get("comment") or os.path.basename(fn)
        found.setdefault(camera, {}).setdefault(label, []).append(box)
    return found


def merge(base, extra):
    """Merge extra into a copy of base, without mutating either."""
    out = {c: {l: list(v) for l, v in labels.items()} for c, labels in base.items()}
    for camera, labels in extra.items():
        for label, boxes in labels.items():
            out.setdefault(camera, {}).setdefault(label, []).extend(boxes)
    return out


def load(json_path=None, dir_path=None):
    """Merged exclusions from the legacy json file and the per-file directory."""
    base = {}
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            base = json.load(f)
    from_dir = load_dir(dir_path)
    total = sum(len(b) for l in from_dir.values() for b in l.values())
    if total:
        logger.info("loaded %d exclusion(s) from %s", total, dir_path)
    return merge(base, from_dir)
