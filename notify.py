import json
import logging
import os
import uuid
from datetime import date, datetime
from io import BytesIO
from pprint import pformat

import cv2
import requests
from PIL import Image

import codeproject

logger = logging.getLogger(__name__)

license_plates = {}
# Maps variant strings (exact + edits1) of known plates to their original plate key.
# At query time, checking edits1(query) against this dict covers edit distance <= 2.
_plate_variants = {}


def _edits1(word):
    "All edits that are one edit away from `word`."
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + replaces + inserts)


def _load_license_plates():
    global license_plates, _plate_variants
    if not license_plates:
        try:
            with open("license-plates.json") as f:
                license_plates = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load license-plates.json: {e}")
            license_plates = {}
        # Pre-expand known plates so detection-time lookup is O(edits1) not O(edits2)
        for plate_key in license_plates:
            _plate_variants[plate_key] = plate_key
            for variant in _edits1(plate_key):
                _plate_variants.setdefault(variant, plate_key)
    return license_plates


def _match_plate(plate):
    """Find the best matching known plate within edit distance 2.

    Checks: exact match, then edits1(query) against pre-computed edits1(known).
    edits1(query) ∩ edits1(known) covers all pairs within edit distance 2.
    """
    # Distance 0 or 1: query itself may be in the pre-expanded variants
    if plate in _plate_variants:
        return _plate_variants[plate]
    # Distance 1-2: check edits1(query) against pre-expanded variants
    for variant in _edits1(plate):
        if variant in _plate_variants:
            return _plate_variants[variant]
    return None


def notify(cam, message, image, predictions, config, ha, model_name="color", original_image=None):
    mode = ha.mode()
    mode_key = "priority-%s" % mode
    if mode_key in config:
        mode_priorities = config[mode_key]
    else:
        mode_priorities = {}
    priorities = config["priority"]
    priority = None
    has_dog = False
    vehicles = list(
        filter(lambda p: p["tagName"] == "vehicle" and "ignore" not in p, predictions)
    )
    has_vehicles = len(vehicles) > 0
    has_visible_vehicles = len(
        list(
            filter(
                lambda p: p["tagName"] == "vehicle"
                and "ignore" not in p
                and "departed" not in p
                and p["age"] < 3,  # Run ALPR on first 3 frames to catch plates that become visible
                predictions,
            )
        )
    )
    people = list(
        filter(
            lambda p: p["tagName"] == "person" and "ignore" not in p,
            predictions,
        )
    )
    has_person = len(people) > 0
    has_dog = len(list(filter(lambda p: p["tagName"] == "dog", predictions))) > 0
    has_person_road = (
        len(list(filter(lambda p: p["tagName"] == "person_road", predictions))) > 0
    )
    has_dog_road = (
        len(list(filter(lambda p: p["tagName"] == "dog_road", predictions))) > 0
    )
    dog_inside = ha.is_dog_inside() if (has_dog or has_dog_road) else False
    packages = list(
        filter(lambda p: p["tagName"] == "package" and "departed" not in p, predictions)
    )
    has_package = len(packages) > 0
    if has_vehicles and cam.name != "mailbox":
        notify_vehicle = ha.should_notify_vehicle()
        logging.info(f"ha.should_notify_vehicle={notify_vehicle}")
    else:
        notify_vehicle = False
    if has_person:
        notify_person = ha.should_notify_person()
        logging.info(f"see {len(people)} people, notify_person={notify_person}")
    else:
        notify_person = False
    # if notify_person:
    #    door_left = ha.get_door_left()
    #    door_right = ha.get_door_right()
    #    do_ignore = (
    #        cam.name in ["deck", "play"] and (door_left or door_left) and mode == "home"
    #    )
    #    if do_ignore:
    #        logging.info(
    #            f"Ignoring person ignore={do_ignore} because door left={door_left}, door_right={door_right}, cam={cam.name} and mode={mode}"
    #        )
    #        notify_person = False
    #        ha.suppress_notify_person()
    #    else:
    #        logging.info(
    #            f"Notifying person because door left={door_left}, door_right={door_right}, cam={cam.name} and mode={mode}"
    #        )
    # else:
    #    # If person detection is off, override night or away mode
    #    mode = "home"
    sound = "pushover"
    for p in list(filter(lambda p: "ignore" in p or "iou" in p, predictions)):
        p["priority"] = -4
    for p in list(filter(lambda p: "priority" not in p, predictions)):
        tagName = p["tagName"]
        probability = p["probability"]
        i_type = None
        if tagName == "person_road":
            if mode == "night" and ha.is_time_after_midnight_and_before_six():
                notify_person = True
                i = 0
            else:
                notify_person = False
                i = -4
        elif tagName == "dog_road":
            if has_person_road:
                i = -3
                i_type = "person walking dog"
            else:
                i = 1
                i_type = "dog without person"
        elif tagName == "vehicle" and cam.name == "front entry":
            i = -3
            i_type = "vehicle rule"
            # cars should not be possible here, unless in road
        elif tagName == "cat" and cam.name == "garage":
            i = -2
            i_type = "cat in garage rule"
        elif tagName == "person" and cam.name == "garage":
            i = -3
            i_type = "person in garage rule"
        elif tagName == "person" and has_person and not notify_person:
            i = -4
            # we are still outside, keep detection off
            # ha.suppress_notify_person()
            i_type = "person detection off"
        elif tagName == "deer" and has_person:
            i = -1
            # this should never occur
            i_type = "deer and person not possible"
        elif tagName == "vehicle" and not notify_vehicle:
            i = -4
            i_type = "vehicle detection off"
        elif tagName in mode_priorities:
            i = mode_priorities.getint(p["tagName"])
            i_type = mode_key
        elif tagName in priorities:
            i = priorities.getint(p["tagName"])
            i_type = "class {}".format(tagName)
        else:
            i_type = "default"
            i = 0
        if cam.name == "peach tree" and mode == "night" and i < 1:
            i = 1
            i_type = "fruit robber"
        if "departed" in p:
            sound = config["sounds"]["departed"]
        elif tagName in config["sounds"]:
            sound = config["sounds"][tagName]
        # if tagName == "dog" and (p["camName"] == "deck") and probability < 0.9:
        #    i = 0
        #    i_type = "maybe dog rule"
        if (
            tagName == "dog"
            and (p["camName"] == "garage")
            and probability > 0.9
            and not has_person
            and dog_inside
        ):
            i = 1
            i_type = "dog in garage rule"
        # elif tagName == "dog" and p["camName"] == "deck" and i > -3:
        #    i = -3
        #    i_type = f"{tagName} on {p['camName']}"
        if tagName in ["fox", "coyote"] and p["camName"] == "deck" and i < 1:
            i = 1
            i_type = f"{tagName} on {p['camName']}"
        if i is not None:
            p["priority"] = i
            p["priority_type"] = i_type
            if priority is None:
                priority = i
            else:
                priority = max(i, priority)
    # raise priority if dog is near package
    if has_package and has_dog and dog_inside:
        priority = 1
    if priority is None:
        for p in predictions:
            if "priority" in p:
                priority = p["priority"]
                logging.info(f"Using prior priority={priority}")
    if priority is None:
        priority = 0
        logging.info("Using default priority")

    # Return early if no predictions to crop
    if len(predictions) == 0:
        return priority

    # crop to area of interest
    width, height = image.size
    left = min(p["boundingBox"]["left"] - 0.05 for p in predictions) * width
    right = (
        max(
            p["boundingBox"]["left"] + p["boundingBox"]["width"] + 0.05
            for p in predictions
        )
        * width
    )
    top = min(p["boundingBox"]["top"] - 0.05 for p in predictions) * height
    bottom = (
        max(
            p["boundingBox"]["top"] + p["boundingBox"]["height"] + 0.05
            for p in predictions
        )
        * height
    )
    center_x = left + (right - left) / 2
    center_y = top + (bottom - top) / 2
    show_width = max(width / 4, right - left)
    show_height = max(height / 4, bottom - top)
    left = min(left, center_x - show_width / 2)
    left = max(0, left)
    top = max(0, top)
    if left + show_width > width:
        show_width = width - left
    if top + show_height > height:
        show_height = height - top
    top = min(top, center_y - show_height / 2)
    top = max(0, top)
    crop_rectangle = (left, top, left + show_width, top + show_height)
    # logging.info("Cropping to %d,%d,%d,%d" % crop_rectangle)
    cropped_image = image.crop(crop_rectangle)

    static_dir = os.path.join(config["detector"]["save-path"], "static")
    for p in predictions:
        cropped_image.save(os.path.join(static_dir, f"{p['tagName']}.jpg"))

    # Run ALPR for vehicles regardless of notification priority
    if has_visible_vehicles and len(vehicles) > 0:
        logging.info(pformat(vehicles))
        left = max(0, min(p["boundingBox"]["left"] - 0.05 for p in vehicles) * width)
        right = min(
            width,
            max(
                p["boundingBox"]["left"] + p["boundingBox"]["width"] + 0.05
                for p in vehicles
            )
            * width,
        )
        top = max(0, min(p["boundingBox"]["top"] - 0.05 for p in vehicles) * height)
        bottom = min(
            height,
            max(
                p["boundingBox"]["top"] + p["boundingBox"]["height"] + 0.05
                for p in vehicles
            )
            * height,
        )
        crop_rectangle = (left, top, right, bottom)
        vehicle_image = image.crop(crop_rectangle)
        save_dir = os.path.join(
            config["detector"]["save-path"], date.today().strftime("%Y%m%d")
        )
        save_vehicle = os.path.join(
            save_dir,
            datetime.now().strftime("%H%M%S")
            + "-"
            + vehicles[0]["camName"].replace(" ", "_")
            + "-"
            + "codeproject.jpg",
        )
        vehicle_image.save(save_vehicle)
        vehicle_bytes = BytesIO()
        vehicle_image.save(vehicle_bytes, "jpeg")
        vehicle_bytes.seek(0)
        try:
            save_json = os.path.join(
                save_dir,
                datetime.now().strftime("%H%M%S")
                + "-"
                + vehicles[0]["camName"].replace(" ", "_")
                + "-"
                + "codeproject.txt",
            )
            codeproject_url = config["codeproject"]["url"] if "codeproject" in config else None
            enrichments = codeproject.enrich(vehicle_bytes.read(), save_json, url=codeproject_url)
            vehicle_message = ""
            if enrichments["count"] == 0:
                # Don't announce if ALPR can't find a vehicle
                notify_vehicle = False
            plates_db = _load_license_plates()
            house_cleaner_found = False
            for plate in enrichments["plates"]:
                matched_key = _match_plate(plate) or _match_plate(plate.replace(" ", ""))
                if matched_key:
                    r = plates_db[matched_key]
                    if len(vehicle_message) > 0:
                        vehicle_message += " and "
                    if "owner" in r:
                        vehicle_message += r["owner"] + "'s "
                        if r["owner"].lower() == "house cleaner":
                            house_cleaner_found = True
                    if "color" in r:
                        vehicle_message += r["color"] + " "
                    if "make" in r:
                        vehicle_message += r["make"] + " "
                        if "model" in r:
                            vehicle_message += r["model"]
                    else:
                        vehicle_message += "vehicle"
                    if r.get("announce", True) is False:
                        logging.info(
                            "Ignoring {}'s vehicle with plate {}".format(
                                r["owner"], plate
                            )
                        )
                        vehicle_message = None
                if vehicle_message is not None:
                    if vehicle_message == "":
                        vehicle_message = "Vehicle"
                    if notify_vehicle:
                        if cam.name == "shed":
                            ha.echo_speaks(f"{vehicle_message} in front of garage")
                        else:
                            ha.echo_speaks(f"{vehicle_message} in driveway")
                    # don't announce plate
                    message += "\n" + vehicle_message + " " + plate
            if house_cleaner_found:
                ha.house_cleaners_arrived()

        except Exception:
            logging.exception("Failed to enrich via codeproject")

    #    if has_package and (priority >= 0 or has_dog):
    #        prob = max(map(lambda x: x["probability"], packages))
    #        logging.info(
    #            f"has_package={has_package}, has_dog={has_dog}, prob={prob}, mode={mode}"
    #        )
    #        # if has_package and has_dog:
    #        #    ha.echo_speaks(
    #        #        "Rufus is opening package near {}".format(packages[0]["camName"])
    #        #    )
    #        if prob > 0.9:
    #            if len(packages) == 1:
    #                ha.echo_speaks(
    #                    "Package delivered near {}".format(packages[0]["camName"])
    #                )
    #            else:
    #                ha.echo_speaks(
    #                    "{} packages delivered near {}".format(
    #                        len(packages), packages[0]["camName"]
    #                    )
    #                )
    #        else:
    #            logging.info(
    #                "Not speaking package delivery because probability {} < 0.9".format(
    #                    prob
    #                )
    #            )
    #            if not has_dog:
    #                priority = -1

    if priority >= -3 and ha.vacation_mode() is False:
        # prepare post
        output_bytes = BytesIO()
        cropped_image.save(output_bytes, "jpeg")
        output_bytes.seek(0)
        # send as -2 to generate no notification/alert, -1 to always send as a quiet notification, 1 to display as high-priority and bypass the user's quiet hours, or 2 to also require confirmation from the user
        pushover_data = {
            "token": config["pushover"]["token"],
            "user": config["pushover"]["user"],
            "message": message,
            "priority": priority,
            "sound": sound,
        }
        # Save review image and add Flag for Review link if roboflow is configured
        if "roboflow" in config:
            try:
                review_dir = os.path.join(config["detector"]["save-path"], "review")
                os.makedirs(review_dir, exist_ok=True)
                review_id = uuid.uuid4().hex[:8]
                review_file = "%s.jpg" % review_id
                review_image = original_image if original_image is not None else image
                if not isinstance(review_image, Image.Image):
                    review_image = Image.fromarray(cv2.cvtColor(review_image, cv2.COLOR_BGR2RGB))
                review_image.save(os.path.join(review_dir, review_file))
                webhook_url = config["roboflow"]["webhook-url"]
                detection_tags = set(p["tagName"] for p in predictions if "ignore" not in p)
                pushover_data["url"] = "%s?file=%s&model=%s&cam=%s&tags=%s" % (
                    webhook_url, review_file, model_name,
                    cam.name.replace(" ", "_"),
                    ",".join(sorted(detection_tags)),
                )
                pushover_data["url_title"] = "Flag for Review"
            except Exception:
                logger.exception("Failed to save review image")
        try:
            r = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=pushover_data,
                files={"attachment": ("image.jpg", output_bytes, "image/jpeg")},
            )
            if r.status_code != 200:
                logging.warning(pformat(r))
                logging.warning(pformat(r.headers))
        except Exception:
            logger.exception("Failed to call Pushover")

    return priority
