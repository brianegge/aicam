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


def load_dir(path, models=None):
    """Read every *.yaml in path into {camera: {label: [box, ...]}}.

    `models` is the set of model files aicam is running right now. A
    provisional exclusion -- one written from a phone tap rather than from an
    archive audit -- names the models it was created against and stops applying
    the moment that set changes. That is what "silence it until the model
    learns" means in practice: the tap also uploads the frame to Roboflow as a
    background example, so the next retrain is the thing that should have fixed
    it, and the retrain is what puts the blind spot back on trial. If the rock
    still reads as a rabbit, the next alert says so and costs one more tap.

    Passing models=None applies them regardless, which is what a tool that only
    wants to read the geometry (recheck_excludes.py) should get.
    """
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
            if doc.get("provisional") and models is not None:
                was = set(doc.get("models") or ())
                if was != set(models):
                    logger.info(
                        "provisional exclusion %s was written against %s and the "
                        "models are now %s; letting it fire again",
                        os.path.basename(fn), ", ".join(sorted(was)) or "nothing",
                        ", ".join(sorted(models)))
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


def load(json_path=None, dir_path=None, auto_dir=None, models=None):
    """Merged exclusions from the json file and the hand-written and auto dirs.

    Two directories, because the machine-written ones must not land in the
    repo: a `git stash -u` during a deploy on 2026-09-13 swept every untracked
    file in the checkout. `excludes/` is audited geometry under version
    control; the auto dir lives beside the captures and holds what a phone tap
    wrote.
    """
    base = {}
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            base = json.load(f)
    merged = merge(base, {})
    for path in (dir_path, auto_dir):
        if not path:
            continue
        from_dir = load_dir(path, models)
        total = sum(len(b) for l in from_dir.values() for b in l.values())
        if total:
            logger.info("loaded %d exclusion(s) from %s", total, path)
        merged = merge(merged, from_dir)
    return merged


def signature(path, auto_dir=None):
    """A value that changes when any exclusion file does.

    main.py reloads on it, so a tap on a phone silences a false positive within
    one sweep instead of at the next restart. Names and mtimes rather than
    contents: this runs every few seconds against a directory of a dozen small
    files, and an edit that changes neither is not a thing that happens.
    """
    stamps = []
    for d in (path, auto_dir):
        if not d or not os.path.isdir(d):
            continue
        for fn in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            try:
                stamps.append((fn, os.stat(fn).st_mtime))
            except OSError:
                pass
    return tuple(stamps)
