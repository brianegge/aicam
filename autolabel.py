"""Turn suppressed detections into training negatives, without a human step.

The exclusions in excludes/ are hand-written geometry that says "nothing real
is ever here". Each one is a permanent patch over a model that has not learned
the fact, and the only way to retire them is to teach the model instead --
which needs the frames as background examples in ipcams2.

Every part of that already exists except the join. verify.py asks a vision
model about fresh wildlife detections and gets back "nothing", with a
confidence, about thirty times a night. excludes/ suppresses the recurring
scenery for free. roboflow_upload.py can upload an image and annotate it with
zero boxes. Nothing was connecting them: frames reached Roboflow only when
someone tapped "Flag for Review" on a phone, arrived unannotated, and waited
for hand-labelling. 1767 of them were sitting unflagged on claw-mini.

When a frame is a negative and when it is not
---------------------------------------------
Only when *every* detection in it was suppressed. A frame where the mulch-bed
rock was excluded but a real raccoon is also present is not a background
example, and uploading it as one would teach the detector that raccoons are
scenery -- the exact opposite of what this is for. That check is the whole
safety of the thing, so it is done on the full prediction list rather than on
whatever survived filtering.

The full frame is uploaded, never the crop verify.py sent to the model. The
detector sees whole frames stretched to 608x608, so a crop is a different
distribution and training on it would teach the wrong scale.

Why it is rate limited
----------------------
The peach tree rock alone produced 96 detections across three days. Uploading
each one would bury ipcams2 in near-duplicates of a single object and skew the
class balance of a dataset that only holds 5251 images. A handful of views of
a static object is all the model needs; a hundred is harmful.
"""
import datetime
import hashlib
import io
import json
import logging
import os

from PIL import Image

import roboflow_upload
from utils import bb_intersection_over_union

logger = logging.getLogger(__name__)

# Suppressions that are about position or configuration rather than about what
# the pixels hold. A vehicle ignored for being over the road is still a
# vehicle, and filing that frame as a negative would teach otherwise.
#
# A denylist, not an allowlist. The alternative -- listing the reasons worth
# teaching -- meant matching free text that whoever wrote the exclusion chose:
# "pale stone at the lawn edge", "small stake or marker in the lawn", "bare
# shrub branch tip". No keyword set anticipates those, and a miss fails
# silently by never uploading. These three are set literally by detect.py and
# are the complete set of position-based suppressions, so they are matched
# exactly; an exclusion comment mentioning a road in passing still counts.
NOT_ABOUT_THE_PIXELS = ("road", "neighbor", "neighbour", "in grass")


def _state_path(config):
    return os.path.join(config["detector"]["save-path"], "autolabel-state.json")


def _load(config):
    try:
        with open(_state_path(config)) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(config, state):
    try:
        tmp = _state_path(config) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, _state_path(config))
    except Exception:
        logger.exception("could not write autolabel state")


def _trainable(reason):
    if not reason:
        return False
    return reason.strip().lower() not in NOT_ABOUT_THE_PIXELS


def is_negative(predictions):
    """True if every detection was suppressed and at least one is trainable.

    An empty frame is not a negative example worth uploading -- the dataset has
    plenty of those and they cost a request each. What is wanted is a frame the
    detector got wrong.
    """
    if not predictions:
        return False
    if any("ignore" not in p for p in predictions):
        return False
    return any(_trainable(p.get("ignore")) for p in predictions)


def _seen_before(state, cam, predictions, iou_floor):
    """Has this same object already been uploaded today?"""
    for prior in state.get("boxes", {}).get(cam, []):
        for p in predictions:
            if bb_intersection_over_union(prior, p["boundingBox"]) >= iou_floor:
                return True
    return False


def maybe_upload_negative(cam, image_bytes, predictions, config):
    """Upload the frame to Roboflow as a background example, if it qualifies."""
    section = config["roboflow"] if config.has_section("roboflow") else None
    if section is None or not section.getboolean("auto-null-uploads", False):
        return None
    if not is_negative(predictions):
        return None

    per_day = section.getint("auto-null-per-camera-daily", 5)
    iou_floor = section.getfloat("auto-null-dedupe-iou", 0.6)
    today = datetime.date.today().isoformat()

    state = _load(config)
    if state.get("date") != today:
        state = {"date": today, "counts": {}, "boxes": {}}
    if state["counts"].get(cam, 0) >= per_day:
        return None
    if _seen_before(state, cam, predictions, iou_floor):
        return None

    reasons = sorted(set(p.get("ignore", "") for p in predictions))
    labels = sorted(set(p.get("tagName", "") for p in predictions))
    stem = hashlib.sha1(image_bytes).hexdigest()[:8]
    name = roboflow_upload.upload_name(cam, set(labels), stem)
    projects = [pid for pid, classes in _projects(section).items()
                if set(labels) & classes]
    if not projects:
        return None

    # The VOC annotation carries the frame's real dimensions; Roboflow rejects
    # the annotation outright if it cannot be parsed.
    try:
        width, height = Image.open(io.BytesIO(image_bytes)).size
    except Exception:
        logger.exception("%s: could not read frame dimensions", cam)
        return None

    uploaded = []
    for project_id in projects:
        try:
            image_id = roboflow_upload.upload_image(
                section["api-key"], project_id, name, image_bytes,
                tags=[cam.replace(" ", "_"), "auto-null"])
            if not image_id:
                logger.warning("%s: upload to %s returned no id", cam, project_id)
                continue
            roboflow_upload.annotate_null(
                section["api-key"], project_id, image_id, name, width, height)
            uploaded.append(project_id)
        except Exception:
            logger.exception("%s: negative upload to %s failed", cam, project_id)

    if not uploaded:
        return None

    state["counts"][cam] = state["counts"].get(cam, 0) + 1
    state.setdefault("boxes", {}).setdefault(cam, []).extend(
        dict(p["boundingBox"]) for p in predictions)
    _save(config, state)
    logger.info("%s: uploaded a negative to %s as %s (%s) [%d/%d today]",
                cam, ",".join(uploaded), name, "; ".join(r for r in reasons if r),
                state["counts"][cam], per_day)
    return name


def _projects(section):
    out = {}
    for key, value in section.items():
        if key.startswith("project."):
            out[key[len("project."):]] = set(c.strip() for c in value.split(","))
    return out
