import io
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pprint import pformat
from timeit import default_timer as timer

import cv2
import humanize
from PIL import Image

import alpr
import autolabel
import frigate_lpr
import verify
from alpr import ALPR_STATE_KEYS, wants_alpr
from notify import notify
from utils import bb_intersection_over_union, draw_bbox, draw_road

logger = logging.getLogger(__name__)


def add_centers(predictions):
    for p in predictions:
        bbox = p["boundingBox"]
        p["center"] = {}
        center = p["center"]
        center["x"] = bbox["left"] + bbox["width"] / 2.0
        center["y"] = bbox["top"] + bbox["height"] / 2.0


# Once a class has been accepted on a camera, keep accepting it at a much lower
# confidence for a while. A stationary object's score wobbles: the car parked at
# peach tree ranged 0.58-0.93 against a 0.70 threshold and dropped in and out of
# the published count 156 times in one run. cam.objects already did this, but it
# is rebuilt every frame, so it only ever bridged a single missed frame -- not
# two consecutive dips, and not a capture error.
OBJECT_HOLD_SECONDS = 60

# Two boxes this close are the same object, in the same place, that has not
# moved. Compared against the box the track *started* on, not the previous
# frame, so a slow drift cannot creep past it one frame at a time.
STATIC_IOU = 0.6

# How much longer a track that has never moved survives without being
# detected. A lululemon package sat on the driveway edge on 2026-09-19 and
# was announced "departed" while it was still plainly there: it cleared the
# 0.80 package threshold exactly once in nine sightings (0.82, with the rest
# 0.18-0.56 carried by the hold below), and when the light changed and the
# detector stopped emitting any box at all, the track expired 2.3 minutes
# later.
#
# The asymmetry is the argument. A package that has not shifted a pixel in
# five minutes is furniture; when it stops being detected, a detection
# failure is far likelier than someone having removed it. Something that was
# *moving* and then vanishes really has usually left. So only stationary
# tracks get the benefit of the doubt, and the ones most likely to be
# genuinely departing are unaffected.
STATIC_EXPIRY_MULTIPLIER = 4
# Matched to the detector's conf floor, so a weak frame on an object we are
# already holding actually reaches this check instead of being discarded twice.
HOLD_PROBABILITY = 0.15


def recently_seen(recent_objects, tag_name, now, hold_seconds=OBJECT_HOLD_SECONDS):
    """Was this class accepted on this camera recently enough to hold it?"""
    seen_at = recent_objects.get(tag_name)
    if seen_at is None:
        return False
    return (now - seen_at).total_seconds() < hold_seconds


def label_with_confidence(tag_names, predictions):
    """"dog 79% (vs deer 93%),person 87% (confirmed 95%, amazon)" for the alert.

    Carries both opinions, the detector's and verify.py's, because neither is
    reliably right when they disagree. On 2026-09-14 the detector called the
    family dog at 79% on the play camera and the model said "deer 93%, young
    deer in yard"; an hour earlier the detector called the same dog a deer at
    77% on garage-l and the model correctly said dog. Printing one and
    discarding the other loses the only signal that tells those two cases
    apart -- that the two disagreed at all.

    A bare score means the class was never verified: not in the configured
    classes, or the gate could not be reached, in which case unverified_note
    says why.

    Highest score per class, because that is the one that cleared the bar.
    """
    best = {}
    for p in predictions:
        tag = p.get("tagName")
        if tag in tag_names:
            score = p.get("probability") or 0
            if tag not in best or score > (best[tag].get("probability") or 0):
                best[tag] = p

    out = []
    for tag in sorted(tag_names):
        p = best.get(tag)
        if p is None:
            out.append(tag)
            continue
        part = "%s %.0f%%" % (tag, (p.get("probability") or 0) * 100)
        verdict = p.get("verified") or {}
        label, confidence = verdict.get("label"), verdict.get("confidence")
        # Only ever set when the model both agreed the box holds a person and
        # saw a service on them, so it needs no guard of its own here; see
        # COURIERS in verify.py. "amazon" is worth more than "person" on a
        # phone at 7am, which is the whole point of asking.
        courier = verdict.get("courier")
        if label and confidence is not None:
            if label == tag:
                part += " (confirmed %.0f%%%s)" % (
                    confidence * 100, ", " + courier if courier else "")
            else:
                part += " (vs %s %.0f%%)" % (label, confidence * 100)
        out.append(part)
    return ",".join(out)


def _frame_jpeg(image, quality=90):
    """The full frame as JPEG bytes, from whichever form it arrived in.

    The whole frame, not a crop: the detector judges whole frames stretched to
    608x608, so a crop trains the wrong scale.
    """
    if isinstance(image, Image.Image):
        pil = image
    else:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def threshold_for(tag_name, thresholds, dark_thresholds, default,
                  cam_thresholds=None):
    """New-object threshold for a class, preferring the narrower setting.

    IR floodlight glare reads as a vehicle to the detector: overnight on
    2026-09-03 the driveway and peach tree produced 147 vehicle detections
    topping out at 0.88, none of them real. Raising the bar only while it is
    actually dark keeps daytime recall, where the same class legitimately
    scores 0.85-0.95.

    A camera's own setting is narrower still and wins outright, dark or not.
    One bar per class has to serve fifteen cameras, and what it costs is not
    the same at each: package=0.80 is right where the class has nothing to
    find and every detection is scenery, and on the front entry it is what
    lost the UPS parcel of 2026-09-23 -- boxed correctly at 0.723, left on the
    porch, never announced. A camera that says what it is for is the one place
    that difference can be written down.
    """
    if cam_thresholds:
        override = cam_thresholds.get(tag_name)
        if override is not None:
            return float(override)
    if dark_thresholds is not None:
        override = dark_thresholds.get(tag_name)
        if override is not None:
            return float(override)
    configured = thresholds.get(tag_name)
    return float(configured) if configured is not None else default


def apply_thresholds(
    predictions, thresholds, dark_thresholds, default, recent_objects, now,
    cam_thresholds=None
):
    """Drop predictions under their class threshold, except held-over ones.

    A prediction that only survives on the hysteresis hold is marked
    `hold_only`. Such a detection may sustain an already-tracked object but
    must never create a new one -- see track_predictions(). The hold is keyed
    by class, so two cars parked on the peach tree camera kept `vehicle`
    permanently fresh; without the mark, every scrap of detector noise above
    HOLD_PROBABILITY became an unmatched new object at age=0 and notified.
    """
    kept = []
    for p in predictions:
        if p["probability"] > threshold_for(
            p["tagName"], thresholds, dark_thresholds, default, cam_thresholds
        ):
            kept.append(p)
        elif (
            recently_seen(recent_objects, p["tagName"], now)
            and p["probability"] > HOLD_PROBABILITY
        ):
            p["hold_only"] = True
            kept.append(p)
    return kept


def expiry_minutes(tracked, interval):
    """How long a track survives with no detection at all, in minutes.

    Grows with age so that something long-established is not dropped over a
    brief gap, and is capped so nothing is held for ever. A track that has
    never moved gets STATIC_EXPIRY_MULTIPLIER times as long -- see the
    constant for why the asymmetry is deliberate.
    """
    minutes = 1 + tracked["age"] * interval / 60
    # age >= 2 so the bonus needs the object to have actually been seen in
    # the same place more than once; a single sighting has not yet shown it
    # is stationary, it has only failed to show otherwise.
    if tracked.get("static") and tracked["age"] >= 2:
        minutes *= STATIC_EXPIRY_MULTIPLIER
    return min(minutes, 60)


def track_predictions(valid_predictions, prev_predictions, new_predictions):
    """Match this frame's predictions to the tracks carried from earlier ones.

    Returns (tracked_pairs, unmatched_holds). A matched prediction inherits the
    track's age and state; an unmatched one starts a new track and is appended
    to `new_predictions` -- unless it is `hold_only`, in which case it is
    under-threshold noise that matched nothing and is returned for the caller
    to discard.
    """
    tracked_pairs = []
    unmatched_holds = []
    for p in valid_predictions:
        this_box = p["boundingBox"]
        this_name = p["tagName"]
        prev_class = prev_predictions.setdefault(this_name, [])
        for prev in prev_class:
            iou = bb_intersection_over_union(prev["boundingBox"], this_box)
            logger.debug(f"iou {this_name} = prev_box & this_box = {iou}")
            if iou > 0.5:
                p["iou"] = iou
                # Against where the track *began*: comparing with the previous
                # frame would let an object drift across the whole scene while
                # every single step looked stationary.
                if bb_intersection_over_union(
                    prev.get("anchor_box", this_box), this_box
                ) <= STATIC_IOU:
                    prev["static"] = False
                prev["boundingBox"] = this_box  # move the box to current
                prev["last_time"] = datetime.now()
                prev["age"] = prev["age"] + 1
                for t in ["age", "ignore", "priority", "priority_type", "static"] + list(
                    ALPR_STATE_KEYS
                ):
                    if t in prev:
                        p[t] = prev[t]
                tracked_pairs.append((p, prev))
        if "iou" not in p:
            if p.get("hold_only"):
                unmatched_holds.append(p)
                continue
            p["start_time"] = datetime.now()
            p["last_time"] = datetime.now()
            p["age"] = 0
            # Copied: prev["boundingBox"] is reassigned every match, and the
            # anchor has to keep describing where the track began.
            p["anchor_box"] = dict(this_box)
            p["static"] = True
            prev_class.append(p)
            new_predictions.append(p)
    return tracked_pairs, unmatched_holds


def detect(cam, color_model, grey_model, vehicle_model, config, ha):
    threshold = config["detector"].getfloat("threshold")
    # Only ask Home Assistant if a dark override is actually configured.
    dark_thresholds = None
    if "thresholds-dark" in config:
        try:
            if ha.is_dark():
                dark_thresholds = config["thresholds-dark"]
        except Exception:
            logger.warning("Could not read is_dark; using daytime thresholds")
    image = cam.image
    if image is None:
        return 0, 0, "{}=[err={}]".format(cam.name, cam.error)
    if cam.resized is None:
        return 0, 0, "{}=[err=resized is None]".format(cam.name)
    prediction_start = timer()
    try:
        if len(cam.resized.shape) == 3:
            predictions = color_model.predict_image(cam.resized)
            model_name = "color"
        elif len(cam.resized.shape) == 2:
            predictions = grey_model.predict_image(cam.resized)
            model_name = "grey"
        else:
            return 0, 0, "Unknown image shape {}".format(cam.resized.shape)
    except OSError:
        return 0, 0, "{}=error:{}".format(cam.name, sys.exc_info()[0])
    cam.age = cam.age + 1
    vehicle_predictions = []
    if cam.vehicle_check and vehicle_model is not None:
        vehicle_predictions = vehicle_model.predict_image(cam.resized2)
        logger.debug(f"{cam.name} vehicles={vehicle_predictions}")
        # include all vehicle predictions for now
        predictions += vehicle_predictions
        model_name += "+vehicle"
    prediction_time = timer() - prediction_start
    notify_time = 0.0
    hold_now = datetime.now()
    predictions = apply_thresholds(
        predictions,
        config["thresholds"],
        dark_thresholds,
        threshold,
        cam.recent_objects,
        hold_now,
        getattr(cam, "thresholds", None),
    )
    for p in predictions:
        p["camName"] = cam.name
    add_centers(predictions)
    # remove road
    if cam.road_line == "all":
        for p in predictions:
            if p["tagName"] == "person":
                p["tagName"] = "person_road"
            elif p["tagName"] == "vehicle":
                p["tagName"] = "vehicle_road"
            elif p["tagName"] == "dog":
                p["tagName"] = "dog_road"
    elif cam.road_line:
        for p in predictions:
            x = p["center"]["x"]
            road_y = cam.road_y_at(x)
            p["road_y"] = road_y
            if p["center"]["y"] < road_y and p["tagName"] in ["vehicle", "person", "package", "dog"]:
                if p["tagName"] == "person":
                    p["tagName"] = "person_road"
                if p["tagName"] == "dog":
                    p["tagName"] = "dog_road"
                if p["tagName"] == "vehicle":
                    p["tagName"] = "vehicle_road"
                    p["ignore"] = "road"
    if cam.name == "garage-l":
        for p in predictions:
            if (
                p["boundingBox"]["top"] + p["boundingBox"]["height"] < 0.24
                and (p["tagName"] in ["vehicle", "person"])
                and p["boundingBox"]["left"] > 0.8
            ):
                p["ignore"] = "neighbor"
            if p["boundingBox"]["top"] + p["boundingBox"]["height"] < 0.24 and (
                p["tagName"] in ["vehicle", "person"]
            ):
                p["ignore"] = "road"
                if p["tagName"] == "person":
                    p["tagName"] = "person_road"
                if p["tagName"] == "vehicle":
                    p["tagName"] = "vehicle_road"
    if cam.name in ["front entry"]:
        for p in filter(lambda p: p["tagName"] == "package", predictions):
            if p["center"]["x"] < 0.178125:
                p["ignore"] = "in grass"
    for p in filter(lambda p: "ignore" not in p, predictions):
        if "*" in cam.excludes:
            for e in cam.excludes["*"]:
                iou = bb_intersection_over_union(e, p["boundingBox"])
                if iou > 0.5:
                    p["ignore"] = e.get("comment", "static")
                    break
        if p["tagName"] in cam.excludes:
            for i, e in enumerate(cam.excludes[p["tagName"]]):
                iou = bb_intersection_over_union(e, p["boundingBox"])
                if iou > 0.5:
                    p["ignore"] = e.get("comment", "static iou {}".format(iou))
                    break

    # A second opinion on fresh wildlife detections, from a vision model that
    # gets the original pixels rather than the 608x608 stretch the detector
    # judged. Placed with the exclusions, and setting the same "ignore" key, so
    # a suppressed detection is invisible to everything downstream instead of
    # being special-cased at the notify call.
    verify.verify_predictions(cam, image, predictions, config)

    valid_predictions = list(filter(lambda p: not ("ignore" in p), predictions))
    valid_objects = set(p["tagName"] for p in valid_predictions)
    departed_objects = cam.objects - valid_objects

    # A frame where everything was suppressed is a background example, and the
    # only way to retire the hand-written entries in excludes/ is to put those
    # frames in the dataset. Deliberately after valid_predictions, so the
    # "nothing survived" test is on the same list the rest of this function
    # trusts. Rate limited and deduplicated inside; see autolabel.py.
    if autolabel.is_negative(predictions):
        try:
            autolabel.maybe_upload_negative(
                cam.name, _frame_jpeg(image), predictions, config)
        except Exception:
            logger.exception("negative upload failed for %s", cam.name)

    # cam.image is already the full-resolution frame the model ran on, so the
    # ALPR crop and the boxes drawn below describe the same instant.
    vehicles_due = [p for p in valid_predictions if wants_alpr(p)]

    yyyymmdd = date.today().strftime("%Y%m%d")
    save_dir = os.path.join(config["detector"]["save-path"], yyyymmdd)
    os.makedirs(save_dir, exist_ok=True)
    today_dir = os.path.join(config["detector"]["save-path"], "today")
    try:
        os.symlink(yyyymmdd, today_dir)
    except FileExistsError:
        try:
            if os.readlink(today_dir) != yyyymmdd:
                os.unlink(today_dir)
                os.symlink(yyyymmdd, today_dir)
        except OSError as e:
            logger.warning(f"Failed to update today symlink: {e}")

    if len(departed_objects) > 0 and cam.prior_priority > -3:
        logger.info(
            "{} current={}, prior={}, departed={}".format(
                cam.name,
                ",".join(valid_objects),
                ",".join(cam.objects),
                ",".join(departed_objects),
            )
        )
        basename = os.path.join(
            save_dir,
            datetime.now().strftime("%H%M%S")
            + "-"
            + cam.name.replace(" ", "_")
            + "-"
            + "_".join(departed_objects)
            + "-departed",
        )
        if isinstance(image, Image.Image):
            image.save(basename + ".jpg")
        else:
            cv2.imwrite(basename + ".jpg", image)
        cam.objects = valid_objects

    colors = config["colors"]
    new_predictions = []
    # (current frame prediction, tracked prediction) for everything matched to
    # an existing track, so state notify() sets can be written back afterwards.
    tracked_pairs, unmatched_holds = track_predictions(
        valid_predictions, cam.prev_predictions, new_predictions
    )
    if unmatched_holds:
        dropped = set(id(p) for p in unmatched_holds)
        predictions = [p for p in predictions if id(p) not in dropped]
        valid_predictions = [p for p in valid_predictions if id(p) not in dropped]
    # Refresh the hold, now that we know which detections matched a track.
    # Anything that survived here either cleared its threshold on its own or is
    # a held detection still tracking a real object, and both should keep the
    # low bar armed -- a car parked in poor light scored 0.17-0.53 for nine
    # straight sweeps, and refreshing only from above-threshold detections
    # capped the hold at OBJECT_HOLD_SECONDS, dropped the car, expired its
    # track and made the next good frame a fresh arrival. Unmatched holds are
    # already gone by this point, so noise cannot arm its own bar.
    for tag_name in set(p["tagName"] for p in valid_predictions):
        cam.recent_objects[tag_name] = hold_now
    expired = []
    for prev_tag, prev_class in cam.prev_predictions.items():
        for x in prev_class:
            if x["last_time"] < datetime.now() - timedelta(
                minutes=expiry_minutes(x, cam.interval)
            ):
                expired.append(x)
        prev_class[:] = [x for x in prev_class if x not in expired]

    if len(valid_predictions) >= 0:
        if isinstance(image, Image.Image):
            im_pil = image.copy()  # for drawing on
        else:
            im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for p in predictions:
            if "ignore" in p:
                width = 2
            else:
                width = 4
            color = colors.get(p["tagName"], fallback="red")
            draw_bbox(im_pil, p, color, width=width)
        if cam.road_line and cam.road_line != "all":
            draw_road(im_pil, cam.road_line)
        if cam.name in ["garage-l"]:
            draw_road(im_pil, [(0, 0.24), (1.0, 0.24)])
    notify_expired = []
    for e in expired:
        draw_bbox(im_pil, e, "grey", width=4)
        t = (
            humanize.naturaltime(datetime.now() - e["start_time"])
            .replace(" ago", "")
            .replace("a minute", "minute")
        )
        e[
            "msg"
        ] = f"{e['tagName']} departed from {cam.name} after being seen {e['age']} times over the past {t}"
        e["departed"] = True
        logger.info(e["msg"])
        if datetime.now() - e["start_time"] > timedelta(minutes=2) and e["age"] > 4:
            notify_expired.append(e)
    if len(notify_expired):
        logger.debug(pformat(notify_expired))
        msg = ", ".join([x["msg"] for x in notify_expired])
        notify_start = timer()
        notify(
            cam,
            msg,
            im_pil,
            notify_expired,
            config,
            ha,
            model_name=model_name,
            original_image=image,
        )
        notify_time += timer() - notify_start

    new_objects = set(p["tagName"] for p in new_predictions)
    min_age = 1000000
    for p in valid_predictions:
        if "age" in p:
            min_age = min(p["age"], min_age)

    # Only notify deer if not seen
    if "deer" in new_objects:
        ha.deer_alert(cam.name)
    # mosquitto_pub -h mqtt.home -t "homeassistant/sensor/deck-dog/config" -r -m '{"name": "deck dog count", "state_topic": "deck/dog/count", "state_class": "measurement", "uniq_id": "deck-dog", "availability_topic": "aicam/status"}'
    show_count = 0
    for o in cam.mqtt:
        count = len(
            list(
                filter(
                    lambda p: p["tagName"] == o and (o != "package" or p["age"] > 0),
                    valid_predictions,
                )
            )
        )
        show_count += count
        if cam.counts.get(o, -1) != count:
            logger.info(f"Publishing count {cam.ha_name}/{o}/count={count}")
            cam.publish(f"{cam.ha_name}/{o}/count", count, retain=False)
            cam.counts[o] = count
    if show_count != cam.last_show_count:
        cam.publish(f"{cam.ha_name}/show", show_count > 0, retain=False)
        cam.last_show_count = show_count

    # Read plates on their own cadence, before deciding whether to notify. A
    # vehicle that parks holds a stable track and produces no new objects, so
    # gating this on movement meant a parked vehicle -- the case where the
    # plate is most readable -- was never looked at again.
    new_plates = []
    if vehicles_due:
        try:
            if "frigate" in config:
                # Frigate does the recognition itself, on its own crops, so
                # there is no image to send -- see frigate_lpr for why matching
                # is by camera and time rather than by bounding box.
                new_plates = frigate_lpr.read_plates(cam, vehicles_due, config)
            else:
                # Clean pixels, not im_pil: no reason to hand the reader an
                # image with bounding boxes drawn over it.
                if isinstance(image, Image.Image):
                    alpr_image = image
                else:
                    alpr_image = Image.fromarray(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    )
                new_plates = alpr.read_plates(cam, alpr_image, vehicles_due, config)
        except Exception:
            logger.exception("ALPR pass failed for %s", cam.name)

    # Notify on movement, and also when a plate is read for the first time.
    if len(new_objects) or new_plates:
        seen = label_with_confidence(valid_objects, valid_predictions)
        # Say when the second opinion could not be had. A gate that fails open
        # silently reads exactly like a gate that passed the detection.
        unverified = verify.unverified_note(valid_predictions)
        if cam.name in ["driveway", "garage"]:
            message = "%s in %s" % (seen, cam.name) + unverified
        elif cam.name == "shed":
            message = "%s in front of garage" % seen + unverified
        elif cam.name == "garage-r":
            message = "%s in front of left garage" % seen + unverified
        elif cam.name == "garage-l":
            message = "%s in front of right garage" % seen + unverified
        else:
            message = "%s near %s" % (seen, cam.name) + unverified
        if cam.age > 2 or "once" in config["detector"]:
            notify_start = timer()
            priority = notify(cam, message, im_pil, valid_predictions, config, ha, model_name=model_name, original_image=image)
            notify_time += timer() - notify_start
        else:
            logger.info("Skipping notifications until after warm up")
            priority = -4
    elif len(valid_predictions) > 0:
        priority = cam.prior_priority
    else:
        priority = -4

    # Notify records a successful plate read on the current frame's prediction;
    # persist it onto the track so the retries stop.
    for current, tracked in tracked_pairs:
        for key in ALPR_STATE_KEYS:
            if key in current:
                tracked[key] = current[key]

    # Notify may also mark objects as ignore
    valid_predictions = list(filter(lambda p: not ("ignore" in p), predictions))
    cam.objects = set(p["tagName"] for p in valid_predictions)

    if priority > -3 and not cam.is_file and min_age < 2:
        # don't save file if we're reading from a file
        if cam.prior_image is not None:
            priorname = (
                os.path.join(
                    save_dir,
                    cam.prior_time.strftime("%H%M%S")
                    + "-"
                    + cam.name.replace(" ", "_")
                    + "-"
                    + "_".join(valid_objects)
                    + "-prior",
                )
                + ".jpg"
            )
            if isinstance(cam.prior_image, Image.Image):
                cam.prior_image.save(priorname)
            else:
                cv2.imwrite(priorname, cam.prior_image)
            cam.prior_image = None
            utime = time.mktime(cam.prior_time.timetuple())
            os.utime(priorname, (utime, utime))
        basename = os.path.join(
            save_dir,
            datetime.now().strftime("%H%M%S")
            + "-"
            + cam.name.replace(" ", "_")
            + "-"
            + "_".join(valid_objects),
        )
        if isinstance(image, Image.Image):
            image.save(basename + ".jpg")
        else:
            cv2.imwrite(basename + ".jpg", image)
        with open(basename + ".txt", "w") as file:
            j = {
                "source": str(cam.source),
                "time": str(datetime.now()),
                "predictions": predictions,
            }
            file.write(json.dumps(j, indent=4, default=str))
        im_pil.save(basename + "-annotated.jpg")
    else:
        cam.prior_image = image
    cam.prior_time = datetime.now()
    cam.prior_priority = priority

    def format_prediction(p):
        o = "{}:{:.2f}".format(p["tagName"], p["probability"])
        if "iou" in p:
            o += ":iou={:.2f}".format(p["iou"])
        if "ignore" in p:
            o += ":ignore={}".format(p["ignore"])
        if "priority" in p:
            o += ":p={}".format(p["priority"])
        if "priority_type" in p:
            o += ":pt={}".format(p["priority_type"])
        if "age" in p:
            o += ":age={}".format(p["age"])
        return o

    return (
        prediction_time,
        notify_time,
        "{}=[".format(cam.name)
        + ",".join(format_prediction(p) for p in predictions)
        + "]",
    )
