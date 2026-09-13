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
LABELS = ["deer", "fox", "coyote", "dog", "cat", "rabbit", "raccoon",
          "squirrel", "bird", "person", "vehicle", "other", "nothing"]

PROMPT = (
    "A motion detector on a home security camera claims the red box contains "
    "a %s. Judge only what is inside the red box.\n"
    'Reply with JSON and nothing else: {"label": one of [%s], '
    '"confidence": 0.0-1.0, "note": "at most five words"}\n'
    'Use "nothing" when the box holds only vegetation, shadow, bare ground, '
    'snow, rain, or part of a building or vehicle. Use "other" for an animal '
    "that is none of the listed ones."
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


def ask(image_bytes, tag, cfg):
    """The model's verdict, or None if it could not be obtained."""
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
            {"type": "text", "text": PROMPT % (tag, ", ".join(LABELS))},
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
    # 0.90, not 0.95: the model's "nothing" verdicts clustered at 0.94-0.98,
    # so a 0.95 bar drops a third of the real catches for no gain.
    floor_nothing = cfg.getfloat("min-confidence-nothing", 0.90)
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
            got = ask(jpeg, p["tagName"], cfg)
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
        logger.info("%s: %s %.0f%% -> %s %.0f%% (%s) $%s", cam.name,
                    p["tagName"], (p.get("probability") or 0) * 100,
                    label, conf * 100, got.get("note", ""), got.get("cost"))
        p["verified"] = got

        if label not in suppress:
            continue
        needed = floor_nothing if label == "nothing" else floor
        if conf < needed:
            # Unsure is not a reason to stay quiet.
            logger.info("  %s below the %.2f floor, alerting anyway", label, needed)
            continue
        p["ignore"] = "verified: %s" % label
        _suppressed.setdefault(cam.name, []).append({
            "tag": p["tagName"], "box": dict(p["boundingBox"]), "label": label,
            "expires": now + ttl, "hits": 1})
