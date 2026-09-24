"""Ask a frontier vision model whether a fresh wildlife detection is real.

Why this exists
---------------
Swapping the yolov4 pair for ipcams_v32_yolo11m took wildlife alerts from
about 20 a day to 123. Most of the new ones are scenery -- a twig tip on the
treeline shrub read as a rabbit eight times in one evening, a pale stone at the
garage lawn edge seventeen times, a paint patch on the shed wall as a fox.

Two cheaper fixes were measured first and neither reaches these:

* Thresholds. The treeline deer peaks at 0.96. Any bar that stops it stops
  real deer too, and the 424-image test split reports deer precision 1.000
  because it holds none of this scenery -- which is exactly why the sweep that
  chose the current thresholds could not see the problem.
* Intersecting with the old model. Of the nine exclusions on disk, five fire
  under *both* models, so intersection removes four and keeps the five deer
  ones that prompted the complaint. It also caps recall at the old model's
  0.752 against the new model's 0.871, and the old colour model trained on
  zero rabbits, so it cannot confirm a daytime rabbit at all.

A vision model can, and it gets a materially better look than the detector
did: the detector judged a 608x608 stretch of the whole frame, where this
sends the original pixels around the box. The stone is 46 pixels wide in the
4K frame and about 7 in the tensor the detector saw.

It also answers a question no threshold can -- "that is a squirrel" -- which
is the actual content of most of these alerts.

What it may and may not override
--------------------------------
Measured against eight deer detections from the capture archive, scoring 0.51
to 0.97. The model returned "nothing" for seven, and every note matched what
the crop actually showed: "paved area edge only", "just ornamental grass",
"garden stake and mulch", "base of sapling tree". Six of those seven scored
between 0.92 and 0.97.

That kills the obvious safety rail. A ceiling that declines to second-guess a
confident detector would have blocked six of the seven real catches and left
the feature doing nothing for the class that needed it most -- the detector's
confidence carries almost no information about whether a deer is there.
`max-score` still exists for anyone who wants that trade, but it defaults off.

The protection instead comes from the evidence, not from deference: "that is a
squirrel" is a positive identification, while "nothing" is the answer the model
also gives when it simply could not read a small, dark or blurry crop. So
"nothing" is held to a higher bar than a named animal.

The eighth detection is a caution in the other direction. On a 0.51 detection
at the swing that this repo had already written off as "dark foliage and
shadow", the model said "deer, partially hidden in brush" at 0.85 -- and
looking again, it may well be right. Being wrong in that direction is what the
excludes/ files risk and this gate does not: the gate re-decides every arrival,
where an exclusion is permanent until someone deletes it.

Failing open, and saying so
---------------------------
Every failure path keeps the alert. A missed coyote costs more than a
duplicate notification, so a timeout, a bad key, an unparseable reply or a
model that is merely unsure all fall through to notifying.

The alert then says why, because a silent fallback is indistinguishable from
a working gate. On 2026-09-13 the account ran out of credit at about 01:00 and
the only symptom was wildlife alerts resuming -- twenty-two failed calls, each
one a 3am notification that looked exactly like a verified one. Now it reads
"raccoon 54% near peach tree (unverified: no OpenRouter credit)".

A key that is out of credit or rejected stays that way, so the first such
answer opens a breaker and later detections skip the call for
retry-after-minutes rather than paying a round trip per detection to learn
the same thing.
"""
import base64
import io
import json
import logging
import re
import time

import cv2
import requests
from PIL import Image, ImageDraw

from utils import bb_intersection_over_union

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Constrained choice rather than free text: matching "grey squirrel" or
# "a small rodent" back onto a class list is a second guessing problem, and
# the model is markedly steadier when the answers are enumerated.
# "bear" is listed for the same reason "coyote" is: not because it is common,
# but because "other" is suppressed by default, so anything missing from this
# list is silently dropped when the model is confident about it. Black bears
# are resident in Connecticut and would otherwise land in "other".
# "package" is here so the class can be verified rather than merely
# thresholded. A parcel on the porch is exactly the case the detector is worst
# at ranking: the UPS delivery of 2026-09-23 scored 0.723 and a false one on
# the same camera scored 0.804, so no bar separates them -- but a model shown
# the crop can say which is a parcel and which is a doormat.
LABELS = ["deer", "fox", "coyote", "bear", "dog", "cat", "rabbit", "raccoon",
          "squirrel", "bird", "person", "vehicle", "package", "other",
          "nothing"]

PROMPT = (
    "A motion detector on a home security camera claims the red box contains "
    "a %s. Judge only what is inside the red box.%s\n"
    'Reply with JSON and nothing else: {"label": one of [%s], '
    '"confidence": 0.0-1.0, "note": "at most five words"}\n'
    'Use "nothing" when the box holds only vegetation, shadow, bare ground, '
    'snow, rain, or part of a building or vehicle. Use "other" for an animal '
    "that is none of the listed ones."
)

# Couriers, enumerated for the same reason the animal labels are: "the Amazon
# lady", "a delivery guy" and "UPS" all have to be matched back onto one thing,
# and the model is steadier when the answers are listed.
#
# This costs no extra request. Every person detection already sends the crop
# and pays for the image; the reply was being collapsed to "person 0.99" and
# the rest discarded. The added field is about twenty completion tokens.
#
# Asked about the person class only, and read only when the model agrees the
# box holds a person -- a courier verdict on a crop the model called "nothing"
# is an answer about an object that is not there.
COURIERS = ["amazon", "ups", "fedex", "usps", "other", "none"]

# The uniform is the evidence, not the errand. A resident carrying shopping in
# from the car is not a courier and neither is a neighbour, so "none" has to be
# an easy answer -- otherwise every person holding anything becomes a delivery.
# Same failure the classifier's "none" class exists to prevent, and the reason
# a visible service marking is named as the thing to look for rather than
# "does this look like a delivery".
#
# The last sentence is load-bearing and was added after measuring. Without it
# the extra question drags the label's own confidence down -- the same crop of
# a real Amazon driver scored person 0.99 twice without it and person 0.80
# twice with it. That number is not cosmetic: "nothing" on a person is held to
# NOTHING_FLOOR_STRICT, and 31 of the 178 person->nothing verdicts in the log
# clear 0.95 while the median is 0.75, so a 0.19 shift would put most of the
# suppressed lawn chairs and patio boxes back into the alert stream.
COURIER_PROMPT = (
    ' Also include "courier": one of [%s] -- which delivery service this '
    "person works for, judged from a uniform, vest, logo or handheld scanner. "
    'Use "none" unless a service is actually visible on them; carrying a '
    'parcel is not on its own enough. Use "other" for a service not listed. '
    '"confidence" still refers to "label" alone -- it is not lowered by any '
    "uncertainty about the courier."
)

# Told to the model when other things are detected in the same frame. Two
# separate problems, one fix.
#
# It keeps answering about the wrong subject. The crop carries three times the
# box for context, so when the family is outside with the dog a child is often
# inside it, and the model describes the child: "dog 83% -> person 100% (young
# child crouching)" eleven times on 2026-09-14, every one of them a correctly
# detected dog standing near a correctly detected person.
#
# And it cannot apply the obvious prior without knowing the rest of the frame.
# Deer do not graze beside people. Across the whole capture archive dog+person
# is the commonest pairing at 51 frames, while deer+person happens twice --
# and both of those, checked by eye, are the dog. A model told a person is
# present has what it needs to prefer dog over deer; blind to it, it answered
# "deer 93%, young deer in yard" about the family dog walking with a child.
FRAME_CONTEXT = (
    " The detector also reports elsewhere in this frame, outside the red box: "
    "%s. Use that as context for what is plausible here -- this is a "
    "residential garden -- but describe only the red box."
)

# Set while the key is known bad, so we stop paying a round trip per detection
# to be told the same thing. Cleared by time alone: topping the account up is
# invisible from here.
_breaker = {"until": 0.0, "reason": ""}

# {camera: [(tag, box, label, expires_at, hits)]}. Only suppressions are
# cached: a static false positive re-acquires every few minutes once its track
# expires, and re-billing seventeen times a night for the same stone is the
# whole cost of the feature. Confirmations are not cached, because a real
# animal moves and the next frame is a genuinely different question.
_suppressed = {}


def _failure_reason(exc):
    """Short cause, phrased for the notification rather than the log."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 402:
        return "no OpenRouter credit"
    if status in (401, 403):
        return "OpenRouter rejected the key"
    if status == 429:
        return "rate limited"
    if status:
        return "OpenRouter HTTP %d" % status
    if isinstance(exc, requests.Timeout):
        return "model timed out"
    if isinstance(exc, requests.RequestException):
        return "model unreachable"
    return "verification failed"


# Classes whose alert may only be silenced by an affirmative identification of
# something else, never by absence of evidence.
#
# A person alert is not symmetric with a deer alert. Dropping a deer that was
# really there costs nothing; dropping a person does. So "nothing" -- which is
# also what the model answers when it simply could not read the crop -- is not
# allowed to silence a person, and neither is "other". Only a concrete
# alternative will: the model naming a dog, a cat or a vehicle.
#
# This exists because the dog on the deck reads as a person. On 2026-09-14 it
# produced person 0.64 and 0.83 while, on the same camera and in the same
# minute, the gate was confirming dog 51% -> dog 99%, dog 88% -> dog 99% and
# correctly calling a toddler's balance bike a vehicle rather than a dog.
# Override per class with suppress-<class> in config.
POSITIVE_ONLY = {"person": ("dog", "cat", "vehicle")}

# "nothing" may silence a POSITIVE_ONLY class, but only from much higher up.
#
# Refusing it outright was too strict. Over 2026-09-14 the model returned
# "nothing" for ten person detections and was right every time -- a trash can,
# an empty plastic swing, a can of insect repellent, a shadow, a push handle,
# a toy balance bike -- while confirming all 89 real people as person. It has
# not once answered "nothing" about a person who was there.
#
# But it is noticeably less certain about an absent person than an absent
# animal: those ten ran 0.70 to 0.99, where wildlife "nothing" clusters at
# 0.94-0.99. A person-shaped object is genuinely more ambiguous. So the bar is
# 0.95 rather than the 0.90 used elsewhere, which on that day's evidence
# catches the unmistakable cases (a chair handle at 0.99, a balance bike at
# 0.98) and leaves the merely-probable ones to alert.
NOTHING_FLOOR_STRICT = 0.95

# Verdicts aicam can report itself, so a wrong detection could be renamed
# rather than silenced. "squirrel" and "bird" are not here: aicam has no such
# class, so they can only suppress.
#
# Off by default, because overriding the detector turned out not to be
# supported by the evidence. Within an hour of enabling it the detector
# correctly called the family dog at 79% on the play camera, the model
# answered "deer 93%, young deer in yard", and the alert went out as deer --
# worse than the wrong-but-harmless "deer 77%" on garage-l that motivated it.
# The model is demonstrably good at confirming a detection and at rejecting
# scenery; that it is *more accurate than the detector when they disagree* is
# a different claim, and one sample went each way. Both verdicts now appear in
# the alert instead, which needs no such claim to be true.
#
# On 2026-09-14 the dog walked past garage-l and was detected only as
# deer 0.77. The model answered "dog 0.98, dog walking on pavement" and the
# alert went out as "deer 77%" anyway, because dog is not in deer's suppress
# set -- correctly, since suppressing it would have lost the alert entirely.
# There was no separate dog detection on that frame to fall back on. What was
# wanted was the same alert with the right word in it.
RELABEL_TO = ("deer", "fox", "coyote", "bear", "dog", "cat",
              "rabbit", "raccoon", "person", "vehicle")

# Wild animals that do not stay beside people, and the domestic ones that do.
#
# When the detector and the model disagree across these two sets and a person
# is in the same frame, the domestic reading wins. This is a prior about the
# garden, not about either model, and it is the only thing that separates two
# cases neither model can:
#
#   play 15:19      detector dog  0.79, model deer 0.93  -> dog   (correct)
#   west_lawn 16:29 detector deer 0.75, model dog  0.98  -> dog   (correct)
#
# Measured over the whole capture archive: dog+person is the commonest pairing
# at 51 frames; deer+person occurs twice, and both of those are the dog,
# confirmed by eye at full resolution. So P(real deer | person in frame) is
# indistinguishable from zero here.
#
# Prompting does not achieve this. Told plainly that a person was in the frame
# and that a large animal beside one is more likely a pet, the model still
# answered "deer 0.88, deer browsing near child" about the collie.
# One object cannot be two of these at once, unlike person+package where a
# delivery driver genuinely holds a parcel and both boxes are correct.
MUTUALLY_EXCLUSIVE = ("deer", "fox", "coyote", "bear", "dog", "cat",
                      "rabbit", "raccoon")

WARY_OF_PEOPLE = ("deer", "fox", "coyote")
DOMESTIC_NEAR_PEOPLE = ("dog", "cat")


# Causes a retry cannot fix, so they arm the breaker.
FATAL = ("no OpenRouter credit", "OpenRouter rejected the key")


def unverified_note(predictions):
    """" (unverified: ...)" for the alert text, or "" when all was well."""
    reasons = []
    for p in predictions:
        r = p.get("unverified")
        if r and r not in reasons:
            reasons.append(r)
    return " (unverified: %s)" % ", ".join(reasons) if reasons else ""


def _cfg(config):
    return config["verify"] if config.has_section("verify") else None


def _crop(image, box, context, max_edge, min_window=400):
    """The box plus context, marked, as JPEG bytes.

    The mark matters as much as the pixels: at three times the box there is
    usually more than one thing in the crop, and "is this a fox" is not
    answerable without saying which thing.

    min_window is the part that makes small boxes answerable. Three times a
    46-pixel box is 138 pixels, which upscales to an unreadable smear -- the
    first run of this against the garage stone produced a blocky grey blob the
    model called a cottontail rabbit at 0.86. Widening to a fixed window of
    original pixels puts the driveway edge and the grass in frame, so there is
    something to judge the scale against.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    w, h = image.size
    x0, y0 = box["left"] * w, box["top"] * h
    x1, y1 = x0 + box["width"] * w, y0 + box["height"] * h
    pad_x = max((x1 - x0) * (context - 1) / 2, (min_window - (x1 - x0)) / 2)
    pad_y = max((y1 - y0) * (context - 1) / 2, (min_window - (y1 - y0)) / 2)
    cx0, cy0 = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
    cx1, cy1 = min(w, int(x1 + pad_x)), min(h, int(y1 + pad_y))
    crop = image.crop((cx0, cy0, cx1, cy1)).convert("RGB")

    draw = ImageDraw.Draw(crop)
    draw.rectangle([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0],
                   outline=(255, 0, 0), width=max(2, crop.size[0] // 200))

    # Scale to a fixed long edge in both directions. Shrinking keeps the
    # request small; enlarging a 46-pixel crop does not add information but
    # does stop the model reading it as noise.
    long_edge = max(crop.size)
    if long_edge != max_edge and long_edge > 0:
        scale = max_edge / long_edge
        crop = crop.resize((max(1, int(crop.size[0] * scale)),
                            max(1, int(crop.size[1] * scale))))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def ask(image_bytes, tag, cfg, others=(), courier=False):
    """The model's verdict, or None if it could not be obtained.

    others names the classes detected elsewhere in the same frame.
    courier adds the delivery-service question; see COURIER_PROMPT.
    """
    context = FRAME_CONTEXT % ", ".join(sorted(others)) if others else ""
    prompt = PROMPT % (tag, context, ", ".join(LABELS))
    if courier:
        prompt += COURIER_PROMPT % ", ".join(COURIERS)
    body = {
        "model": cfg.get("model", "google/gemini-3.8-flash"),
        # Generous, because the reasoning tokens are charged against this
        # budget before any JSON appears. At 200 the model spent the lot
        # thinking and returned '{"label": "nothing' with finish_reason
        # "length", which is a parse failure and therefore a wasted call.
        "max_tokens": cfg.getint("max-tokens", 1000),
        # Removes the ```json fences rather than stripping them afterwards.
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," +
                       base64.b64encode(image_bytes).decode()}},
        ]}],
    }
    r = requests.post(ENDPOINT,
                      headers={"Authorization": "Bearer " + cfg["api-key"]},
                      json=body, timeout=cfg.getfloat("timeout", 20))
    r.raise_for_status()
    data = r.json()
    text = (data["choices"][0]["message"].get("content") or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    out = json.loads(text)
    out["cost"] = (data.get("usage") or {}).get("cost")
    return out


def _is_fresh(cam, p):
    """True if nothing already being tracked covers this box.

    A held object is asked about once, when it arrives, not on every sweep.
    """
    for prev in cam.prev_predictions.get(p["tagName"], []):
        if bb_intersection_over_union(prev["boundingBox"], p["boundingBox"]) > 0.5:
            return False
    return True


def _cached(cam, p, iou_floor, now):
    entries = _suppressed.get(cam.name, [])
    entries[:] = [e for e in entries if e["expires"] > now]
    for e in entries:
        if e["tag"] == p["tagName"] and bb_intersection_over_union(
                e["box"], p["boundingBox"]) >= iou_floor:
            e["hits"] += 1
            return e
    return None


def verify_predictions(cam, image, predictions, config):
    """Mark fresh detections the model says are not what they claim.

    Sets p["ignore"], so suppression flows through valid_predictions exactly
    as a static exclusion does -- the message, the counts and the notify
    decision all see the same thing.
    """
    cfg = _cfg(config)
    if cfg is None or not cfg.getboolean("enabled", False) or not cfg.get("api-key"):
        return
    classes = {c.strip() for c in cfg.get("classes", "").split(",") if c.strip()}
    suppress = {c.strip() for c in
                cfg.get("suppress", "squirrel,bird,nothing,other").split(",")
                if c.strip()}
    floor = cfg.getfloat("min-confidence", 0.6)

    def suppress_for(tag):
        """What may silence this class, which is not the same for every class."""
        override = cfg.get("suppress-%s" % tag)
        if override is not None:
            return {c.strip() for c in override.split(",") if c.strip()}
        if tag in POSITIVE_ONLY:
            # "nothing" is allowed in, but held to NOTHING_FLOOR_STRICT below.
            return set(POSITIVE_ONLY[tag]) | {"nothing"}
        return suppress

    # 0.70. Raised to 0.90 at first, on the reasoning that "nothing" is also
    # what the model answers when it cannot read the crop, so a hedged
    # "nothing" should not be trusted. Measuring it says otherwise.
    #
    # Across 2026-09-14, 31 wildlife detections were let through by the 0.90
    # bar. Every one of their notes names a specific object rather than
    # hedging about whether anything is there: "rock and shadow", "only a
    # leaf", "back of a chair", "folds in a bag", "marks on pavement edge",
    # "tree stump and wood". Two of the highest-risk were checked at full
    # resolution -- cat 0.80 over folds in a white bag, cat 0.61 over a pale
    # rock in the peach tree bed -- and both are scenery. The ten person
    # cases, at 0.70 to 0.99, were correct as well.
    #
    # The confidence on a "nothing" verdict tracks how well the model can
    # *name* what is there, not how sure it is the box is empty. When it is
    # genuinely torn it returns an animal label, not "nothing". Filtering on
    # that number was filtering on the wrong axis.
    #
    # 0.70 rather than 0 because a handful do hedge about presence itself --
    # "unclear dark shape or noise" at 0.43, "tree base and shadows only" at
    # 0.68 -- and those should still alert. person keeps its own bar at 0.95;
    # see NOTHING_FLOOR_STRICT.
    floor_nothing = cfg.getfloat("min-confidence-nothing", 0.70)
    person_floor = cfg.getfloat("person-present-confidence", 0.6)
    dup_iou = cfg.getfloat("duplicate-box-iou", 0.8)
    # Off by default; see the note above on detector confidence.
    ceiling = cfg.getfloat("max-score", 1.01)
    iou_floor = cfg.getfloat("cache-iou", 0.8)
    ttl = cfg.getfloat("cache-minutes", 60) * 60
    now = time.time()

    for p in predictions:
        if "ignore" in p or p.get("hold_only"):
            continue
        if p["tagName"] not in classes or not _is_fresh(cam, p):
            continue
        if (p.get("probability") or 0) >= ceiling:
            logger.debug("%s: %s at %.2f is above the %.2f ceiling, not asking",
                         cam.name, p["tagName"], p["probability"], ceiling)
            continue

        hit = _cached(cam, p, iou_floor, now)
        if hit:
            p["ignore"] = "verified: %s" % hit["label"]
            if hit["hits"] in (5, 10, 20):
                logger.warning(
                    "%s: %s at %.2f,%.2f has been called %s %d times -- worth "
                    "an entry in excludes/ so it stops costing a model call",
                    cam.name, p["tagName"], p["center"]["x"], p["center"]["y"],
                    hit["label"], hit["hits"])
            continue

        if now < _breaker["until"]:
            p["unverified"] = _breaker["reason"]
            continue

        try:
            jpeg = _crop(image, p["boundingBox"],
                         cfg.getfloat("context", 3.0), cfg.getint("max-edge", 768),
                         cfg.getint("min-window", 400))
            # Everything else the detector sees in this frame, so the model
            # can judge what is plausible and knows the other subjects are
            # not what it was asked about.
            others = {q.get("tagName") for q in predictions
                      if q is not p and q.get("tagName")
                      and not q.get("hold_only")} - {p["tagName"]}
            got = ask(jpeg, p["tagName"], cfg, others,
                      courier=(p["tagName"] == "person"
                               and cfg.getboolean("courier", True)))
        except Exception as e:
            reason = _failure_reason(e)
            p["unverified"] = reason
            if reason in FATAL:
                _breaker.update(until=now + cfg.getfloat("retry-after-minutes", 15) * 60,
                                reason=reason)
                logger.error("%s: %s -- not asking again for %s minutes; "
                             "alerts will say so", cam.name, reason,
                             cfg.get("retry-after-minutes", "15"))
            elif isinstance(e, requests.RequestException):
                # Expected and self-describing; a stack trace per detection
                # buried the 402s last night.
                logger.warning("%s: verification of %s failed (%s), alerting anyway",
                               cam.name, p["tagName"], reason)
            else:
                logger.exception("%s: verification of %s failed, alerting anyway",
                                 cam.name, p["tagName"])
            continue
        if not got or got.get("label") not in LABELS:
            p["unverified"] = "model gave no usable answer"
            logger.warning("%s: unusable verdict for %s (%s), alerting anyway",
                           cam.name, p["tagName"], got)
            continue

        label, conf = got["label"], float(got.get("confidence") or 0)

        # Informational only. It never suppresses, never relabels and is not
        # consulted by anything below -- every other branch here answers "is
        # this real", and letting a hallucinated vest into that decision would
        # mean a prowler could be reasoned away. Dropped unless the model also
        # agreed the box holds a person, and "none" is dropped rather than
        # printed so the alert only ever gains a word by saying something.
        courier = str(got.pop("courier", "") or "").strip().lower()
        if label == "person" and courier in COURIERS and courier != "none":
            got["courier"] = courier

        logger.info("%s: %s %.0f%% -> %s %.0f%%%s (%s) $%s", cam.name,
                    p["tagName"], (p.get("probability") or 0) * 100,
                    label, conf * 100,
                    " " + got["courier"] if got.get("courier") else "",
                    got.get("note", ""), got.get("cost"))
        p["verified"] = got

        # The detector sometimes puts two labels on one animal -- "coyote 0.86"
        # and "dog 0.54" on boxes differing in the third decimal, tree line
        # 2026-09-14. When the model names one of the two, it is choosing
        # between the detector's own proposals rather than overriding it, so
        # the loser is a duplicate and not a second animal. Nothing is lost:
        # the object still alerts under the other box.
        #
        # Restricted to mutually exclusive classes. person+package overlaps
        # twelve times in the archive and is usually a real person holding a
        # real parcel.
        if label != p["tagName"] and label in MUTUALLY_EXCLUSIVE \
                and p["tagName"] in MUTUALLY_EXCLUSIVE:
            twin = next((q for q in predictions
                         if q is not p and q.get("tagName") == label
                         and "ignore" not in q
                         and bb_intersection_over_union(
                             q["boundingBox"], p["boundingBox"]) > dup_iou), None)
            if twin is not None:
                logger.info("  %s is the same box as the %s detection; "
                            "keeping the %s", p["tagName"], label, label)
                p["ignore"] = "duplicate of the %s box" % label
                continue

        # Common sense before either model's opinion: a wild animal standing
        # beside a person in a garden is a pet.
        person_present = any(
            q.get("tagName") == "person" and (q.get("probability") or 0) >= person_floor
            and "ignore" not in q
            for q in predictions if q is not p)
        pair = {p["tagName"], label}
        if (person_present and cfg.getboolean("people-imply-pets", True)
                and pair & set(WARY_OF_PEOPLE) and pair & set(DOMESTIC_NEAR_PEOPLE)):
            domestic = (label if label in DOMESTIC_NEAR_PEOPLE else p["tagName"])
            if p["tagName"] != domestic:
                logger.info("  %s -> %s: a person is in frame, so the pet reading wins",
                            p["tagName"], domestic)
                p["relabelled_from"] = p["tagName"]
                p["tagName"] = domestic
            p["prior"] = "person in frame"
            continue

        if label not in suppress_for(p["tagName"]):
            # Not something to stay quiet about -- but if the model named a
            # class aicam has, and it is not the one the detector chose, the
            # alert should carry the model's word rather than the detector's.
            if (cfg.getboolean("relabel", False) and label != p["tagName"]
                    and label in RELABEL_TO
                    and conf >= cfg.getfloat("min-confidence-relabel", 0.85)):
                logger.info("  relabelling %s -> %s on the model's %.0f%%",
                            p["tagName"], label, conf * 100)
                p["relabelled_from"] = p["tagName"]
                p["tagName"] = label
            continue
        if label == "nothing":
            needed = (cfg.getfloat("min-confidence-nothing-strict",
                                   NOTHING_FLOOR_STRICT)
                      if p["tagName"] in POSITIVE_ONLY else floor_nothing)
        else:
            needed = floor
        if conf < needed:
            # Unsure is not a reason to stay quiet.
            logger.info("  %s below the %.2f floor, alerting anyway", label, needed)
            continue
        p["ignore"] = "verified: %s" % label
        _suppressed.setdefault(cam.name, []).append({
            "tag": p["tagName"], "box": dict(p["boundingBox"]), "label": label,
            "expires": now + ttl, "hits": 1})
