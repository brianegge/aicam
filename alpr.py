"""Reading license plates off tracked vehicles.

This runs on its own cadence rather than as part of building a notification.
A vehicle that arrives and parks holds a stable track and stops producing new
objects, so anything gated on a notification stops looking at it — which is
exactly backwards, because a parked vehicle is when the plate is most readable.
"""

import logging
import os
from datetime import date, datetime
from io import BytesIO

import codeproject

logger = logging.getLogger(__name__)

# A vehicle is rarely readable on the frames where it is first seen: still
# moving, angled, or half out of frame. Keep looking on a decaying cadence
# until the plate is read, or it has sat long enough that it never will be.
# At a 3s interval this is ~15 attempts spread over the first six minutes.
ALPR_INITIAL_FRAMES = 3
ALPR_RETRY_EVERY = 10
ALPR_MAX_AGE = 120

# State that has to survive onto the tracked prediction; detect.py copies these
# back after each frame, since only age/ignore/priority did before.
ALPR_STATE_KEYS = ("plate_read", "plate", "alpr_count")

# Below this the plate cannot be more than a few pixels across, so the request
# is guaranteed waste. A shed crop of 206x109 was being sent.
MIN_CROP_PX = 300

# Padding around the vehicle box, as a fraction of the frame.
CROP_PAD = 0.05


def wants_alpr(p):
    """Should ALPR run for this prediction on this frame?"""
    if p.get("tagName") != "vehicle":
        return False
    if "ignore" in p or "departed" in p:
        return False
    if p.get("plate_read"):
        return False
    age = p.get("age", 0)
    if age < ALPR_INITIAL_FRAMES:
        return True
    if age > ALPR_MAX_AGE:
        return False
    return age % ALPR_RETRY_EVERY == 0


def vehicle_crop_box(vehicle, width, height):
    """Pixel crop box for one vehicle.

    Deliberately per-vehicle. Taking min/max across every vehicle in frame
    returns their union, which on the shed camera meant one 2301x504 crop
    spanning a mower and a truck with empty pavement in between — the plate
    ends up a tiny fraction of the image and ALPR finds nothing.
    """
    box = vehicle["boundingBox"]
    left = max(0, (box["left"] - CROP_PAD) * width)
    right = min(width, (box["left"] + box["width"] + CROP_PAD) * width)
    top = max(0, (box["top"] - CROP_PAD) * height)
    bottom = min(height, (box["top"] + box["height"] + CROP_PAD) * height)
    return (left, top, right, bottom)


def read_plates(cam, image, vehicles, config):
    """Run ALPR on each vehicle due for it.

    Records the result on the prediction and returns the plates read for the
    first time on this frame, so the caller can decide to notify about them.
    """
    width, height = image.size
    save_dir = os.path.join(
        config["detector"]["save-path"], date.today().strftime("%Y%m%d")
    )
    # detect.py normally makes this first, but don't depend on call order —
    # a missing directory otherwise surfaces as a failed ALPR request.
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError:
        logger.warning("Could not create %s for ALPR crops", save_dir)
    codeproject_url = config["codeproject"]["url"] if "codeproject" in config else None
    new_plates = []

    for vehicle in vehicles:
        crop_box = vehicle_crop_box(vehicle, width, height)
        crop_w = crop_box[2] - crop_box[0]
        crop_h = crop_box[3] - crop_box[1]
        if crop_w < MIN_CROP_PX or crop_h < MIN_CROP_PX / 2:
            logger.debug(
                "Skipping ALPR for %s: crop %dx%d too small to hold a plate",
                cam.name,
                crop_w,
                crop_h,
            )
            continue

        vehicle_image = image.crop(crop_box)
        stamp = datetime.now().strftime("%H%M%S")
        basename = os.path.join(
            save_dir, "{}-{}-codeproject".format(stamp, cam.name.replace(" ", "_"))
        )
        try:
            vehicle_image.save(basename + ".jpg")
        except OSError:
            logger.warning("Could not save ALPR crop for %s", cam.name)

        vehicle_bytes = BytesIO()
        vehicle_image.save(vehicle_bytes, "jpeg")
        vehicle_bytes.seek(0)
        try:
            enrichments = codeproject.enrich(
                vehicle_bytes.read(), basename + ".txt", url=codeproject_url
            )
        except Exception:
            logger.exception("Failed to enrich via codeproject")
            continue

        vehicle["alpr_count"] = enrichments["count"]
        if enrichments["plates"]:
            plate = enrichments["plates"][0]
            vehicle["plate"] = plate
            # Read it once; stop retrying this vehicle.
            vehicle["plate_read"] = True
            new_plates.append(plate)
            logger.info("Read plate %s on %s", plate, cam.name)

    return new_plates
