"""Confirm Remini preschool photos of one child, then share the day's batch.

The preschool posts ~100 photos a day to Remini. remini-chloe (on ubuntu24,
which holds the Remini login) downloads them, finds candidate faces with
Frigate's face classifier and POSTs each candidate here. Frigate alone is not
trusted to decide: on the 2026-09 test weeks its large model still called a
classmate "Chloe" at 0.90-0.94 about twice a day. So:

  candidate   POST /remini/candidate  (photo + the matched face box)
              -> one Pushover "Is this Chloe?", the face boxed on the photo,
                 linking to remini-review.html on Home Assistant
  tap         webhook model=remini, file=<id>, cam=yes|no -> confirm()
              yes: the face crop is also registered to Frigate's library,
                   so every confirmation makes the next match better
  digest      once nothing is pending and the last answer is
              `digest-delay-minutes` old, the confirmed photos go out over
              iMessage (BlueBubbles, on this same Mac) as one batch

BlueBubbles' password is read from its own config.db at send time rather than
copied into config.txt: the server runs on this host and owns that file.

Configured by an optional `[remini-review]` section; without it nothing runs.
"""
import datetime
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from urllib.parse import quote, urlencode

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

YES, NO = "yes", "no"
BLUEBUBBLES_DB = os.path.expanduser(
    "~/Library/Application Support/bluebubbles-server/config.db")


# AP-style months, as people write them: "Sept 15", not "Sep 15".
_MONTHS = ("Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct",
           "Nov", "Dec")


def pretty_date(date):
    try:
        d = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return date or "today"
    return "%s %d" % (_MONTHS[d.month - 1], d.day)


class ReminiReviewConfig(object):
    def __init__(self, config):
        s = config["remini-review"]
        self.page_url = s["page-url"]
        self.name = s.get("name", "Chloe")
        self.chats = [c.strip() for c in s.get("imessage-chats", "").split(",") if c.strip()]
        self.digest_delay = s.getint("digest-delay-minutes", 15) * 60
        # One message per Remini post, ahead of its photos. {caption} is the
        # teacher's own text from that post; without one the header is just
        # "{name} {date}".
        self.digest_text = s.get("digest-text", "{name} {date}: {caption}")
        self.train_on_confirm = s.getboolean("train-on-confirm", True)
        # "No" files the face here, so Frigate learns what not-her looks like. With
        # only adults beside her in the library, every unknown preschooler's
        # nearest class was Chloe (or Kyle); a class of other children gives them
        # somewhere else to land. Empty to just drop a No.
        self.negative_name = s.get("negative-name", "Classmates").strip()
        self.frigate_url = s.get("frigate-url", "http://192.168.254.31:5000").rstrip("/")
        self.bluebubbles_url = s.get("bluebubbles-url", "http://localhost:1234").rstrip("/")
        self.bluebubbles_db = s.get("bluebubbles-db", BLUEBUBBLES_DB)
        # Rejected candidates (a classmate) are only worth keeping long enough
        # to answer "Already answered" to a repeat tap. Confirmed, trained and
        # sent ones stay: they are the photos of her.
        self.rejected_days = s.getint("keep-rejected-days", 30)
        self.dir = s.get("dir", os.path.join(config["detector"]["save-path"], "remini"))
        self.pushover = (config["pushover"]["token"], config["pushover"]["user"]) \
            if config.has_section("pushover") else None

    @staticmethod
    def enabled(config):
        return config.has_section("remini-review") and \
            config["remini-review"].getboolean("enabled", True)


class ReminiReview(object):
    def __init__(self, cfg, session=None, now=time.time):
        self.cfg = cfg
        self.http = session or requests.Session()
        self.now = now
        self.lock = threading.Lock()
        os.makedirs(cfg.dir, exist_ok=True)

    # ------------------------------------------------------------------ store
    def _path(self, cid, ext):
        return os.path.join(self.cfg.dir, "%s.%s" % (cid, ext))

    def _load(self, cid):
        try:
            with open(self._path(cid, "json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _save(self, cid, meta):
        tmp = self._path(cid, "json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        os.replace(tmp, self._path(cid, "json"))

    def _all(self):
        out = []
        for fn in sorted(os.listdir(self.cfg.dir)):
            if fn.endswith(".json"):
                meta = self._load(fn[:-5])
                if meta:
                    out.append(meta)
        return out

    # ------------------------------------------------------------------ candidate
    def add_candidate(self, image_bytes, box, score=None, source=None, date=None, share=True,
                      caption=None):
        """Store a candidate and ask about it. Returns (code, result)."""
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return 400, {"error": "not an image"}
        # The same photo twice (a re-run, a re-post) is one question, not two.
        for meta in self._all():
            if source and meta.get("source") == source and meta.get("box") == box:
                return 200, {"id": meta["id"], "status": "duplicate"}
        cid = uuid.uuid4().hex[:12]
        with open(self._path(cid, "jpg"), "wb") as f:
            f.write(image_bytes)
        # share=False: a training question about an old photo. A Yes teaches
        # Frigate but does not send weeks-old pictures out as "today".
        meta = {"id": cid, "state": "pending", "box": box, "score": score,
                "source": source, "date": date, "added": self.now(), "share": bool(share),
                "caption": caption or ""}
        self._save(cid, meta)
        self._ask(cid, img, meta)
        return 200, {"id": cid, "status": "pending"}

    def _ask(self, cid, img, meta):
        if not self.cfg.pushover:
            logger.info("remini review: no Pushover configured, %s waits unasked", cid)
            return
        shown = img.copy()
        if meta.get("box"):
            x, y, w, h = meta["box"]
            pad = max(4, int(0.15 * max(w, h)))
            t = max(3, img.shape[1] // 300)
            cv2.rectangle(shown, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 220, 255), t)
        side = max(shown.shape[:2])
        if side > 1600:  # Pushover's attachment cap is 2.5 MB
            shown = cv2.resize(shown, None, fx=1600.0 / side, fy=1600.0 / side,
                               interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 85])
        message = "Remini %s" % (meta.get("date") or "photo")
        if not meta.get("share", True):
            message += " (training only, not shared)"
        if meta.get("score") is not None:
            message += ", Frigate %.0f%%" % (100 * float(meta["score"]))
        token, user = self.cfg.pushover
        page = self.cfg.page_url + ("&" if "?" in self.cfg.page_url else "?") + \
            urlencode({"file": cid, "name": self.cfg.name}, quote_via=quote)
        r = self.http.post("https://api.pushover.net/1/messages.json", timeout=30, data={
            "token": token, "user": user, "title": "Is this %s?" % self.cfg.name,
            "message": message, "priority": 0, "url": page,
            "url_title": "Yes / No"},
            files={"attachment": ("remini.jpg", jpg.tobytes(), "image/jpeg")})
        try:
            ok = r.status_code == 200 and r.json().get("status") == 1
        except ValueError:
            ok = False
        if ok:
            logger.info("remini review: asked about %s (%s)", cid, message)
        else:
            logger.warning("remini review: Pushover refused %s: %s %s", cid, r.status_code,
                           r.text.strip()[:300])

    # ------------------------------------------------------------------ tap
    def confirm(self, cid, answer):
        """Act on a tap. Returns (code, result) like roboflow_upload._do_upload."""
        cid = os.path.basename(cid or "")
        answer = (answer or "").strip().lower()
        if answer not in (YES, NO):
            return 400, {"error": "answer must be yes or no"}
        with self.lock:
            meta = self._load(cid)
            if meta is None:
                return 404, {"error": "no such photo %s" % cid}
            if meta["state"] != "pending":
                return 200, {"message": "Already answered: %s" % meta["state"]}
            if answer == NO:
                meta["state"] = "rejected"
            else:
                meta["state"] = "confirmed" if meta.get("share", True) else "trained"
            meta["answered"] = self.now()
            self._save(cid, meta)
        if answer == NO:
            if self.cfg.negative_name and self.cfg.train_on_confirm and \
                    self._train(cid, meta, self.cfg.negative_name):
                return 200, {"message": "Not %s, filed under %s" % (self.cfg.name, self.cfg.negative_name)}
            return 200, {"message": "Not %s, dropped" % self.cfg.name}
        trained = self._train(cid, meta) if self.cfg.train_on_confirm else False
        taught = ", and taught to Frigate" if trained else ""
        if not meta.get("share", True):
            return 200, {"message": "%s confirmed%s (not shared)" % (self.cfg.name, taught)}
        return 200, {"message": "%s confirmed, in the next batch%s" % (self.cfg.name, taught)}

    def _train(self, cid, meta, name=None):
        """Register the face with Frigate under `name` (default the child). A padded crop, so Frigate's own
        detector finds the face again -- a tight one comes back with no face."""
        img = cv2.imread(self._path(cid, "jpg"))
        if img is None or not meta.get("box"):
            return False
        x, y, w, h = meta["box"]
        m = int(0.4 * max(w, h))
        crop = img[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
        ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        try:
            name = name or self.cfg.name
            r = self.http.post("%s/api/faces/%s/register" % (self.cfg.frigate_url, quote(name)),
                               files={"file": ("face.jpg", jpg.tobytes(), "image/jpeg")}, timeout=30)
            if r.ok:
                logger.info("remini review: taught Frigate %s from %s", name, cid)
                return True
            logger.warning("remini review: Frigate would not register %s: %s", cid, r.text[:200])
        except requests.RequestException as e:
            logger.warning("remini review: Frigate unreachable registering %s: %s", cid, e)
        return False

    # ------------------------------------------------------------------ digest
    def digest_due(self):
        items = self._all()
        if any(m["state"] == "pending" for m in items):
            return []
        ready = [m for m in items if m["state"] == "confirmed"]
        if not ready:
            return []
        last = max(m.get("answered", 0) for m in ready)
        return ready if self.now() - last >= self.cfg.digest_delay else []

    def send_digest(self):
        with self.lock:
            ready = self.digest_due()
            if not ready or not self.cfg.chats:
                return 0
            password = self._bluebubbles_password()
            # Grouped by post, in date order: each group is the teacher's
            # message for that day, then the photos of Chloe from it.
            groups = {}
            for m in sorted(ready, key=lambda m: (m.get("date") or "", m.get("added", 0))):
                groups.setdefault((m.get("date") or "", m.get("caption") or ""), []).append(m)
            for chat in self.cfg.chats:
                for (date, caption), items in groups.items():
                    self._bb_text(chat, password, self.header(date, caption))
                    for m in items:
                        self._bb_photo(chat, password, self._path(m["id"], "jpg"))
            for m in ready:
                m["state"] = "sent"
                m["sent"] = self.now()
                self._save(m["id"], m)
            logger.info("remini review: sent %d photo(s) to %s", len(ready), ", ".join(self.cfg.chats))
            return len(ready)

    def header(self, date, caption):
        text = self.cfg.digest_text.format(name=self.cfg.name, date=pretty_date(date),
                                           caption=caption or "").strip()
        return text.rstrip(":").strip() if not caption else text

    def _bluebubbles_password(self):
        db = sqlite3.connect("file:%s?mode=ro" % self.cfg.bluebubbles_db, uri=True)
        try:
            return db.execute("select value from config where name='password'").fetchone()[0]
        finally:
            db.close()

    def _bb_text(self, chat, password, text):
        self.http.post("%s/api/v1/message/text" % self.cfg.bluebubbles_url, params={"password": password},
                       json={"chatGuid": chat, "tempGuid": str(uuid.uuid4()), "message": text,
                             "method": "apple-script"}, timeout=60).raise_for_status()

    def _bb_photo(self, chat, password, path):
        with open(path, "rb") as fh:
            self.http.post("%s/api/v1/message/attachment" % self.cfg.bluebubbles_url,
                           params={"password": password}, timeout=120,
                           data={"chatGuid": chat, "tempGuid": str(uuid.uuid4()),
                                 "name": os.path.basename(path), "method": "apple-script"},
                           files={"attachment": (os.path.basename(path), fh, "image/jpeg")}
                           ).raise_for_status()

    def trim(self):
        """Delete rejected candidates older than keep-rejected-days. Returns how many."""
        cutoff = self.now() - self.cfg.rejected_days * 86400
        gone = 0
        with self.lock:
            for m in self._all():
                if m["state"] == "rejected" and m.get("answered", m.get("added", 0)) < cutoff:
                    for ext in ("jpg", "json"):
                        try:
                            os.remove(self._path(m["id"], ext))
                        except OSError:
                            pass
                    gone += 1
        if gone:
            logger.info("remini review: removed %d rejected candidate(s) older than %d days",
                        gone, self.cfg.rejected_days)
        return gone

    # ------------------------------------------------------------------ loop
    def run_forever(self):
        last_trim = 0
        while True:
            try:
                self.send_digest()
                if self.now() - last_trim > 3600:
                    self.trim()
                    last_trim = self.now()
            except Exception:
                logger.exception("remini review: digest failed")
            time.sleep(60)


def start(config):
    if not ReminiReviewConfig.enabled(config):
        return None
    rr = ReminiReview(ReminiReviewConfig(config))
    threading.Thread(target=rr.run_forever, name="remini-review", daemon=True).start()
    logger.info("remini review: confirming %s, digest to %s", rr.cfg.name,
                ", ".join(rr.cfg.chats) or "nobody (imessage-chats unset)")
    return rr
