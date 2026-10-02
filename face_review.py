"""Frigate face review over Pushover, the way detections are reviewed.

Frigate files every face it cannot name into its Train tab
(`clips/faces/train/`), and naming one is how its library grows. The Train tab
is unusable on the phone -- the page goes white after a second -- so this does
the same job through the channel aicam alerts already use:

  poll    `/api/faces` on the Frigate host for Train images it has not seen
  send    one Pushover per sighting (the largest attempt of a Frigate event),
          picture attached, linking to face-review.html on Home Assistant
  tap     the page calls the existing aicam_roboflow_review webhook with
          model=face, file=<train file>, cam=<name>; Home Assistant forwards
          that to the review server here, which calls classify() below

`cam` carries the person's name because the webhook forwards exactly four
fields (file, model, cam, tags) and changing that means editing YAML on the
Home Assistant box -- see CLAUDE.md, "A webhook automation cannot answer the
phone". Two reserved names: SKIP leaves the face in the Train tab, DELETE
removes it (not a face, or nobody worth naming).

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

# Pushover's own caps (see notify.py): url 512 chars, attachment 2.5 MB.
URL_LIMIT = 512


class FaceReviewConfig(object):
    def __init__(self, config):
        section = config["face-review"]
        self.frigate_url = section.get("frigate-url", "http://192.168.254.31:5000").rstrip("/")
        self.page_url = section["page-url"]
        self.poll_seconds = section.getint("poll-seconds", 60)
        # Smaller than this is not worth a person's time: Frigate's own
        # min_area is 4800 (~70x70), and nothing below it can be recognised.
        self.min_side = section.getint("min-side", 70)
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

    def _review_event(self, event_id, attempts, names):
        best, best_img = None, None
        for a in attempts:
            try:
                img = self.train_image(a["file"])
            except requests.RequestException:
                continue
            if img is None:
                continue
            if best_img is None or img.shape[0] * img.shape[1] > best_img.shape[0] * best_img.shape[1]:
                best, best_img = a, img
        if best is None:
            return "unreadable"
        if min(best_img.shape[:2]) < self.cfg.min_side:
            return "too small"
        if not self.cfg.pushover:
            logger.info("face review: no Pushover configured, not sending %s", best["file"])
            return "no pushover"
        self.send(best, best_img, self.camera_of(event_id), names)
        return "sent"

    def send(self, attempt, img, camera, names):
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
            if name == DELETE:
                r = self.http.post(self.cfg.frigate_url + "/api/faces/train/delete",
                                   json={"ids": [filename]}, timeout=15)
                done = "Deleted"
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
