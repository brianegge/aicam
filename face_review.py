"""Frigate face review over Pushover, the way detections are reviewed.

Frigate files every face it cannot name into its Train tab
(`clips/faces/train/`), and naming one is how its library grows. The Train tab
is unusable on the phone -- the page goes white after a second -- so this does
the same job through the channel aicam alerts already use:

  poll    `/api/faces` on the Frigate host for Train images it has not seen
  send    one Pushover per sighting -- a Frigate person event, split further
          if its attempts confidently name two different people -- with a
          mosaic of every attempt, the one that a tap will train marked 1
          (see best_attempt), linking to face-review.html on Home Assistant
  tap     the page calls the existing aicam_roboflow_review webhook with
          model=face, file=<train file>, cam=<name>; Home Assistant forwards
          that to the review server here, which calls classify() below

`cam` carries the person's name because the webhook forwards exactly four
fields (file, model, cam, tags) and changing that means editing YAML on the
Home Assistant box -- see CLAUDE.md, "A webhook automation cannot answer the
phone". Reserved names: SKIP leaves the face in the Train tab; DELETE,
CANT_TELL and VISITOR remove it without training (see DISCARD).

Configured by an optional `[face-review]` section; without one nothing runs.
"""
import json
import logging
import os
import threading
import time
from urllib.parse import quote, urlencode

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

SKIP = "__skip__"
DELETE = "__delete__"
CANT_TELL = "__cant_tell__"
VISITOR = "__visitor__"
# Answers that take a face out of the Train tab without teaching Frigate
# anything. A visitor (the UPS driver) is a real face, but naming one would
# put a stranger in the library the cameras compare everyone against; one
# blurred past recognition is no use either. Both just go.
DISCARD = {
    DELETE: "Not a face, removed",
    CANT_TELL: "Can't tell, removed without training",
    VISITOR: "Visitor, removed without training",
}

# Pushover's own caps (see notify.py): url 512 chars, attachment 2.5 MB.
URL_LIMIT = 512


class FaceReviewConfig(object):
    def __init__(self, config):
        section = config["face-review"]
        self.frigate_url = section.get("frigate-url", "http://192.168.254.31:5000").rstrip("/")
        self.page_url = section["page-url"]
        self.poll_seconds = section.getint("poll-seconds", 60)
        # Frigate's own floor (face_recognition.min_area). This was a 70 px
        # minimum *side*, which threw away a sharp 64x86 crop Frigate had
        # already scored Kyle 0.85 and sent a tilted, soft 83x92 one of the
        # same visit instead (2026-10-10 01:41).
        self.min_area = section.getint("min-area", 4800)
        # More would make a lock-screen picture of postage stamps.
        self.max_tiles = section.getint("max-tiles", 6)
        # A face nobody could name is not worth asking about, and naming one
        # anyway poisons the library: Frigate averages each person's images,
        # so washed-out and side-on crops make that person's average a
        # generic face others match (2026-10-03: nine such images in one
        # person's folder drew 14 of 17 wrong names). Set on the 62 events in the
        # Train tab that day, every rejected crop checked by eye.
        self.max_blown = section.getfloat("max-blown", 0.25)
        self.min_sharpness = section.getfloat("min-sharpness", 80.0)
        self.max_yaw = section.getfloat("max-yaw", 0.5)
        # YuNet, the same file Frigate uses (model_cache/facedet/facedet.onnx).
        # Without it the side-on check is skipped and the other two still run.
        self.face_detector = section.get("face-detector", "")
        # A burst (a party, a delivery crew) should not become forty
        # notifications at once; the rest wait for the next poll.
        self.max_per_poll = section.getint("max-per-poll", 5)
        self.priority = section.getint("priority", 0)
        self.sound = section.get("sound", "none")
        save_path = config["detector"]["save-path"]
        self.state_path = section.get("state-file",
                                      os.path.join(save_path, "face-review-seen.json"))
        self.pushover = (config["pushover"]["token"], config["pushover"]["user"]) \
            if config.has_section("pushover") else None

    @staticmethod
    def enabled(config):
        return config.has_section("face-review") and \
            config["face-review"].getboolean("enabled", True)


def parse_train_name(filename):
    """Frigate's Train-tab name: <event_id>-<timestamp>-<guess>-<score>.webp.

    Event ids are themselves "<ts>-<random>", hence parts[0]-parts[1]; this is
    the same split web/src/pages/FaceLibrary.tsx does.
    """
    stem = filename[:-5] if filename.endswith(".webp") else filename
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    try:
        return {"file": filename, "event": "%s-%s" % (parts[0], parts[1]),
                "ts": float(parts[2]), "guess": parts[3], "score": float(parts[4])}
    except ValueError:
        return None


def group_by_event(train_files):
    events = {}
    for f in train_files:
        info = parse_train_name(f)
        if info:
            events.setdefault(info["event"], []).append(info)
    return events


def face_quality(img, detector_path=""):
    """Measure what makes a Train crop unusable. Returns (blown, sharpness, yaw).

    blown      share of pixels at 245 or above in the middle half of the
               crop, where the eyes, nose and mouth are. The whole crop
               counted a bright wall, white hair and glare on glasses the
               same as a whited-out face, and dropped two clear faces (17%
               and 25%). The middle half reads 11% and 16% for those, and 76%
               for the garage-lit face of 21:40 on 2026-10-03.
    sharpness  variance of the Laplacian, at the crop's own size.
    yaw        how far the nose sits from midway between the eyes, in eye
               widths: 0 is facing the camera. None when no detector is
               configured or it finds no face, which is 18 of 177 crops --
               an unknown is not held against the face.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    blown = float((grey[h // 4:h - h // 4, w // 4:w - w // 4] >= 245).mean())
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    yaw = None
    if detector_path and os.path.exists(detector_path):
        h, w = img.shape[:2]
        found = cv2.FaceDetectorYN.create(detector_path, "", (w, h), 0.3).detect(img)[1]
        if found is not None and len(found):
            right_eye, left_eye, nose = found[0][4:10].reshape(3, 2)
            eye_width = abs(left_eye[0] - right_eye[0])
            if eye_width > 1:
                yaw = float(abs(nose[0] - (right_eye[0] + left_eye[0]) / 2) / eye_width)
    return blown, sharpness, yaw


# Sharpness compared at one size: the Laplacian variance of a raw crop grows
# as the crop shrinks, so ranking raw crops by it prefers the smallest.
RANK_SIDE = 112
# A named guess at least this good is Frigate saying the face is readable.
NAMED_SCORE = 0.8


def rank_sharpness(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = min(grey.shape[:2]) > RANK_SIDE
    grey = cv2.resize(grey, (RANK_SIDE, RANK_SIDE),
                      interpolation=cv2.INTER_AREA if small else cv2.INTER_CUBIC)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def best_attempt(fit):
    """The attempt a tap will train, from [(attempt, img)] that are fit.

    Frigate naming a face confidently is the strongest sign it is readable,
    so that comes first; then sharpness at a common size. Largest-first chose
    the worst of Kyle's four attempts on 2026-10-10.
    """
    def key(pair):
        a, img = pair
        named = a["score"] if a["guess"] != "unknown" and a["score"] >= NAMED_SCORE else 0.0
        return (named, rank_sharpness(img))
    return max(fit, key=key)


def split_people(attempts):
    """One group per person. A Frigate event is one tracked person, and two
    people arriving together are two events (Josiana and her helper,
    2026-10-09 07:57). Should the tracker swap people mid-event, attempts that
    confidently name two different people are split by name, and the
    unnamed ones -- which could be either -- are left out."""
    named = {}
    for a in attempts:
        if a["guess"] != "unknown" and a["score"] >= NAMED_SCORE:
            named.setdefault(a["guess"], []).append(a)
    if len(named) < 2:
        return [attempts]
    return list(named.values())


TILE = 240


def mosaic(tiles):
    """[(img, label, is_best, dim)] -> one image, best first, numbered."""
    cells = []
    for img, label, is_best, dim in tiles:
        h, w = img.shape[:2]
        scale = float(TILE) / max(h, w)
        cell = np.full((TILE, TILE, 3), 24, np.uint8)
        im = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_CUBIC)
        y, x = (TILE - im.shape[0]) // 2, (TILE - im.shape[1]) // 2
        if dim:
            im = (im * 0.4).astype(np.uint8)
        cell[y:y + im.shape[0], x:x + im.shape[1]] = im
        colour = (60, 200, 60) if is_best else (200, 200, 200)
        if is_best:
            cv2.rectangle(cell, (1, 1), (TILE - 2, TILE - 2), colour, 4)
        cv2.rectangle(cell, (0, TILE - 26), (TILE, TILE), (0, 0, 0), -1)
        cv2.putText(cell, label, (6, TILE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, 1, cv2.LINE_AA)
        cells.append(cell)
    cols = min(3, len(cells))
    rows = (len(cells) + cols - 1) // cols
    out = np.full((rows * TILE, cols * TILE, 3), 24, np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        out[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = cell
    return out


def page_link(page_url, filename, names):
    """The review page link, carrying the names to offer as buttons.

    Names are dropped from the end if the link would pass Pushover's 512; the
    page always offers a free-text box as well, so nobody becomes unnameable.
    """
    names = sorted(names, key=str.lower)
    while True:
        query = urlencode({"file": filename, "names": ",".join(names)}, quote_via=quote)
        url = page_url + ("&" if "?" in page_url else "?") + query
        if len(url) <= URL_LIMIT or not names:
            return url
        names = names[:-1]


class FaceReview(object):
    def __init__(self, cfg, session=None):
        self.cfg = cfg
        self.http = session or requests.Session()
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ state
    def _load_seen(self):
        try:
            with open(self.cfg.state_path) as f:
                got = json.load(f)
            return got if isinstance(got, dict) else None
        except (OSError, ValueError):
            return None

    def _save_seen(self, seen):
        # Only events still in the Train tab need remembering; Frigate prunes
        # it to save_attempts, so this stays as small as the tab.
        tmp = self.cfg.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(seen, f, indent=1, sort_keys=True)
        os.replace(tmp, self.cfg.state_path)

    # ------------------------------------------------------------------ frigate
    def faces(self):
        r = self.http.get(self.cfg.frigate_url + "/api/faces", timeout=15)
        r.raise_for_status()
        return r.json()

    def train_image(self, filename):
        r = self.http.get("%s/clips/faces/train/%s" % (self.cfg.frigate_url, quote(filename)),
                          timeout=15)
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        return img

    def camera_of(self, event_id):
        try:
            r = self.http.get("%s/api/events/%s" % (self.cfg.frigate_url, event_id), timeout=10)
            if r.ok:
                return r.json().get("camera")
        except (requests.RequestException, ValueError):
            pass
        return None

    # ------------------------------------------------------------------ poll
    def poll_once(self):
        """Send a notification for each new sighting. Returns how many were sent."""
        with self.lock:
            faces = self.faces()
            train = faces.get("train") or []
            names = [n for n in faces if n != "train"]
            events = group_by_event(train)
            seen = self._load_seen()
            if seen is None:
                # First run: everything already in the tab is backlog, not
                # news. Ninety notifications at once would bury the point.
                self._save_seen({e: "backlog" for e in events})
                logger.info("face review: first run, %d existing sightings marked as backlog",
                            len(events))
                return 0

            sent = 0
            for event_id in sorted(set(events) - set(seen), key=lambda e: events[e][0]["ts"]):
                if sent >= self.cfg.max_per_poll:
                    break  # the rest go out next poll
                status = self._review_event(event_id, events[event_id], names)
                seen[event_id] = status
                sent += status == "sent"
            # Forget events Frigate has already pruned from the tab.
            self._save_seen({e: s for e, s in seen.items() if e in events})
            return sent

    def _unfit(self, img):
        """Why this crop is not worth asking about, or None if it is."""
        if img.shape[0] * img.shape[1] < self.cfg.min_area:
            return "too small"
        blown, sharpness, yaw = face_quality(img, self.cfg.face_detector)
        if blown > self.cfg.max_blown:
            return "washed out (%.0f%%)" % (blown * 100)
        if sharpness < self.cfg.min_sharpness:
            return "blurred (%.0f)" % sharpness
        if yaw is not None and yaw > self.cfg.max_yaw:
            return "turned away (%.2f)" % yaw
        return None

    def _review_event(self, event_id, attempts, names):
        loaded = []
        for a in sorted(attempts, key=lambda a: a["ts"]):
            try:
                img = self.train_image(a["file"])
            except requests.RequestException:
                continue
            if img is not None:
                loaded.append((a, img))
        if not loaded:
            return "unreadable"
        camera = None
        statuses = []
        for group in split_people([a for a, _ in loaded]):
            files = set(a["file"] for a in group)
            pairs = [(a, img) for a, img in loaded if a["file"] in files]
            judged = [(a, img, self._unfit(img)) for a, img in pairs]
            fit = [(a, img) for a, img, why in judged if not why]
            if not fit:
                reasons = [why for _, _, why in judged]
                if all(r == "too small" for r in reasons):
                    statuses.append("too small")
                else:
                    logger.info("face review: not asking about %s, no attempt fit to "
                                "name: %s", event_id, ", ".join(reasons))
                    statuses.append("unfit")
                continue
            best, best_img = best_attempt(fit)
            if not self.cfg.pushover:
                logger.info("face review: no Pushover configured, not sending %s",
                            best["file"])
                statuses.append("no pushover")
                continue
            if camera is None:
                camera = self.camera_of(event_id) or ""
            self.send(best, self._picture(best, judged), camera or None, names,
                      shown=min(len(judged), self.cfg.max_tiles))
            statuses.append("sent")
        for status in ("sent", "unfit", "no pushover", "too small"):
            if status in statuses:
                return status
        return "unfit"

    def _picture(self, best, judged):
        """The notification picture: one face, or every attempt with the one
        a tap trains marked 1 and the unusable ones dimmed with the reason."""
        if len(judged) == 1:
            return judged[0][1]
        ordered = [j for j in judged if j[0] is best]
        ordered += [j for j in judged if j[0] is not best][:self.cfg.max_tiles - 1]
        tiles = []
        for n, (a, img, why) in enumerate(ordered, 1):
            label = "%d  %s" % (n, why or (
                "%s %.0f%%" % (a["guess"], a["score"] * 100)
                if a["guess"] != "unknown" else "fit"))
            tiles.append((img, label + ("  (trains)" if a is best else ""),
                          a is best, bool(why)))
        return mosaic(tiles)

    def send(self, attempt, img, camera, names, shown=1):
        # Train crops are small; scale up so the lock screen shows a face, not
        # a thumbnail of one.
        side = max(img.shape[:2])
        if side < 400:
            img = cv2.resize(img, None, fx=400.0 / side, fy=400.0 / side,
                             interpolation=cv2.INTER_CUBIC)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        guess = attempt["guess"]
        if guess == "unknown":
            message = "Unknown face"
        else:
            message = "Looks like %s (%.0f%%)" % (guess, attempt["score"] * 100)
        if camera:
            message += " at %s" % camera.replace("_", " ")
        if shown > 1:
            message += " -- best of %d, marked 1" % shown
        token, user = self.cfg.pushover
        data = {
            "token": token, "user": user,
            "title": "Who is this?",
            "message": message,
            "timestamp": int(attempt["ts"]),
            "priority": self.cfg.priority,
            "sound": self.cfg.sound,
            "url": page_link(self.cfg.page_url, attempt["file"], names),
            "url_title": "Name this face",
        }
        r = self.http.post("https://api.pushover.net/1/messages.json", data=data,
                           files={"attachment": ("face.jpg", jpg.tobytes(), "image/jpeg")},
                           timeout=30)
        # Pushover can say no with a 200 and status 0 -- see roboflow_upload.announce.
        try:
            answer = r.json()
        except ValueError:
            answer = {}
        if r.status_code != 200 or answer.get("status") != 1:
            logger.warning("face review: Pushover refused %s: %s %s", attempt["file"],
                           r.status_code, r.text.strip()[:300])
        else:
            logger.info("face review: sent %s (%s)", attempt["file"], message)

    # ------------------------------------------------------------------ tap
    def classify(self, filename, name):
        """Act on a tap. Returns (http_code, result) like roboflow_upload._do_upload."""
        filename = os.path.basename(filename or "")
        name = (name or "").strip()
        if name.startswith("__") and name not in DISCARD and name != SKIP:
            return 400, {"error": "unknown answer %s" % name}
        if not filename or not name:
            return 400, {"error": "a face tap needs a file and a name"}
        if name == SKIP:
            return 200, {"message": "Left in the Train tab"}
        try:
            train = self.faces().get("train") or []
        except requests.RequestException as e:
            return 502, {"error": "Frigate unreachable: %s" % e}
        if filename not in train:
            # Classified or deleted already -- from another tap, or in the GUI.
            return 200, {"message": "Already handled"}
        try:
            if name in DISCARD:
                r = self.http.post(self.cfg.frigate_url + "/api/faces/train/delete",
                                   json={"ids": [filename]}, timeout=15)
                done = DISCARD[name]
            else:
                r = self.http.post("%s/api/faces/train/%s/classify" % (self.cfg.frigate_url, quote(name)),
                                   json={"training_file": filename}, timeout=15)
                done = "Added to %s" % name
        except requests.RequestException as e:
            return 502, {"error": "Frigate unreachable: %s" % e}
        if not r.ok:
            return r.status_code, {"error": "Frigate said %s: %s" % (r.status_code, r.text[:200])}
        logger.info("face review: %s -> %s", filename, name)
        return 200, {"message": done}

    # ------------------------------------------------------------------ loop
    def run_forever(self):
        while True:
            try:
                self.poll_once()
            except Exception:
                logger.exception("face review poll failed")
            time.sleep(self.cfg.poll_seconds)


def start(config):
    """Start the poller thread if [face-review] is configured. Returns the FaceReview or None."""
    if not FaceReviewConfig.enabled(config):
        return None
    fr = FaceReview(FaceReviewConfig(config))
    threading.Thread(target=fr.run_forever, name="face-review", daemon=True).start()
    logger.info("face review: polling %s every %ds", fr.cfg.frigate_url, fr.cfg.poll_seconds)
    return fr
