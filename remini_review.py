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
import base64
import hashlib
import datetime
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from urllib.parse import quote, urlencode

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

YES, NO, NOT_A_FACE, ACCEPT = "yes", "no", "notface", "accept"
# The file name an "Accept the rest" tap carries: every pending candidate with
# a prediction takes it.
REST = "__rest__"
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


# Box colours (BGR) for a candidate's `faces`: every face in the photo, labelled
# by the model, so a person sees at a glance who it thinks is who.
FACE_COLOURS = {"chloe": (60, 200, 60), "classmate": (230, 130, 30), "unknown": (235, 235, 235)}
LEGEND = "\U0001F7E9 Chloe  \U0001F7E6 Classmate  \u2B1C Unknown"


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
        # A school day with no photo of her (off sick, or none found) still gets
        # the teacher's note: it carries the next day's plans (Brian, 2026-10-19).
        self.note_text = s.get("note-text", "{name}'s class {date} (no photos of {name}): {caption}")
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
        # Face matches skip the question and go straight to the group (Brian,
        # 2026-10-16): 29 of 29 face guesses had been right. Outfit matches are
        # still asked. auto-share-min keeps a floor on the face score.
        self.auto_share = s.getboolean("auto-share-faces", False)
        self.auto_share_min = s.getfloat("auto-share-min", 0.0)
        self.sent_page_url = s.get("sent-page-url", self.page_url.replace("remini-review.html", "remini-sent.html"))
        # Photos with no detectable face go to a vision model (aicam's
        # [verify] OpenRouter account) with a contact sheet of the day's
        # confirmed photos of her -- children wear one outfit all day. On the
        # 2026-09-17 test it was 8/8 on answered photos at ~$0.003 each.
        # Its own `api-key` here, so its spend shows separately on OpenRouter;
        # [verify]'s key is the fallback.
        v = config["verify"] if config.has_section("verify") else None
        self.outfit_model = s.get("outfit-model", v.get("model", "google/gemini-3.8-flash") if v else "")
        key = s.get("api-key") or (v.get("api-key") if v is not None else None)
        self.outfit_key = key if s.getboolean("outfit-check", True) else None
        self.held_days = s.getint("keep-held-days", 3)
        self.dir = s.get("dir", os.path.join(config["detector"]["save-path"], "remini"))
        self.pushover = (config["pushover"]["token"], config["pushover"]["user"]) \
            if config.has_section("pushover") else None
        # Its own Pushover application (2026-10-13), so Remini questions arrive
        # under their own name and quota rather than as aicam alerts.
        if self.pushover and s.get("pushover-token"):
            self.pushover = (s["pushover-token"], self.pushover[1])

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
                      caption=None, predicted=None, label=None, train=True, faces=None):
        """Store a candidate and ask about it. Returns (code, result)."""
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return 400, {"error": "not an image"}
        # The same photo twice is one question, not two: a re-run (same source),
        # or Remini posting identical bytes under a second URL (2026-10-15: two
        # photos each went to the group twice). Content, not name, decides.
        sha = hashlib.sha1(image_bytes).hexdigest()
        for meta in self._all():
            if (source and meta.get("source") == source and meta["state"] == "pending") or \
                    (meta.get("sha") == sha and meta.get("date") == date and meta["state"] != "held"):
                return 200, {"id": meta["id"], "status": "duplicate"}
        cid = uuid.uuid4().hex[:12]
        with open(self._path(cid, "jpg"), "wb") as f:
            f.write(image_bytes)
        # share=False: a training question about an old photo. A Yes teaches
        # Frigate but does not send weeks-old pictures out as "today".
        # predicted/label: the model's own answer, shown so a person only has to
        # correct the wrong ones; "Accept the rest" applies it to the untouched.
        # train=False: the box is only a pointer for the eye ("is she anywhere in
        # this photo?") -- it may well be a classmate, so it must never register.
        meta = {"id": cid, "state": "pending", "score": score, "sha": sha,
                "source": source, "date": date, "added": self.now(), "share": bool(share),
                "caption": caption or "", "predicted": predicted, "label": label,
                "faces": sorted(faces or [], key=lambda f: f["box"][0])}
        meta["box" if train else "box_shown"] = box
        if self._auto(meta, score):
            meta.update(state="confirmed", accepted=True, auto=True, answered=self.now(),
                        summarized=self.now())
            self._save(cid, meta)
            logger.info("remini review: %s shared without asking (%s)", cid, meta.get("label"))
            return 200, {"id": cid, "status": "auto"}
        self._save(cid, meta)
        self._ask(cid, img, meta)
        return 200, {"id": cid, "status": "pending"}

    def _auto(self, meta, score):
        """A face match ("Chloe 0.xx") meant for sharing, when auto-share is on."""
        face_match = meta.get("predicted") == "yes" and \
            (meta.get("label") or "").startswith("%s 0." % self.cfg.name)
        if not (self.cfg.auto_share and face_match and meta.get("share", True)):
            return False
        try:
            s = float(score) if score is not None else float(meta["label"].split()[1])
        except (ValueError, IndexError):
            return False
        return s >= self.cfg.auto_share_min

    def _ask(self, cid, img, meta):
        if not self.cfg.pushover:
            logger.info("remini review: no Pushover configured, %s waits unasked", cid)
            return
        shown = img.copy()
        t = max(3, img.shape[1] // 300)
        for n, f in enumerate(meta.get("faces") or [], 1):
            # Every face, coloured by the model's call; Chloe drawn last-but-one
            # and thicker so she stands out in a crowd.
            x, y, w, h = f["box"]
            pad = max(3, int(0.12 * max(w, h)))
            colour = FACE_COLOURS.get(f.get("cls"), FACE_COLOURS["unknown"])
            cv2.rectangle(shown, (x - pad, y - pad), (x + w + pad, y + h + pad), colour,
                          t * 2 if f.get("cls") == "chloe" else t)
            # The face's number, for "Face 2 is Chloe" on the page, and the
            # classmate's name when remini-chloe's model knows it.
            label = "%d %s" % (n, f["name"]) if f.get("name") else str(n)
            scale = max(0.8, img.shape[1] / 1100.0)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 * t // 3 + 1)
            lx, ly = max(0, x - pad), max(th + 8, y - pad)
            cv2.rectangle(shown, (lx, ly - th - 8), (lx + tw + 8, ly), colour, -1)
            cv2.putText(shown, label, (lx + 4, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                        2 * t // 3 + 1, cv2.LINE_AA)
        if (meta.get("box") or meta.get("box_shown")) and not meta.get("faces"):
            x, y, w, h = meta.get("box") or meta["box_shown"]
            pad = max(4, int(0.15 * max(w, h)))
            t = max(3, img.shape[1] // 300)
            cv2.rectangle(shown, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 220, 255), t)
        side = max(shown.shape[:2])
        if side > 1600:  # Pushover's attachment cap is 2.5 MB
            shown = cv2.resize(shown, None, fx=1600.0 / side, fy=1600.0 / side,
                               interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 85])
        message = "Remini %s" % (meta.get("date") or "photo")
        if meta.get("label"):
            message = "Looks like: %s\n%s" % (meta["label"], message)
        if meta.get("faces"):
            message += "\n" + LEGEND
        if not meta.get("share", True):
            message += " (training only, not shared)"
        # The score was Frigate's once; since the preschool model it is already
        # in the label ("Chloe 0.72"), so it is shown only when there is no label.
        if meta.get("score") is not None and not meta.get("label"):
            message += ", score %.2f" % float(meta["score"])
        token, user = self.cfg.pushover
        page = self.cfg.page_url + ("&" if "?" in self.cfg.page_url else "?") + \
            urlencode({k: v for k, v in (("file", cid), ("name", self.cfg.name),
                                         ("guess", meta.get("label")),
                                         ("faces", len(meta.get("faces") or []) or None),
                                         ("fn", "|".join(f.get("name") or "" for f in meta.get("faces") or [])
                                          if any(f.get("name") for f in meta.get("faces") or []) else None)) if v},
                      quote_via=quote)
        r = self.http.post("https://api.pushover.net/1/messages.json", timeout=30, data={
            "token": token, "user": user, "title": "Is this %s?" % self.cfg.name,
            "message": message, "priority": 0, "url": page,
            "url_title": "Correct it" if meta.get("label") else "Yes / No"},
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
        if cid == REST and answer == ACCEPT:
            return self.accept_rest()
        face_no = None
        if answer.startswith("face:"):
            try:
                face_no = int(answer[5:])
            except ValueError:
                return 400, {"error": "bad face number %s" % answer}
            answer = YES
        if answer not in (YES, NO, NOT_A_FACE):
            return 400, {"error": "answer must be yes, no, notface or face:N"}
        with self.lock:
            meta = self._load(cid)
            if meta is None:
                return 404, {"error": "no such photo %s" % cid}
            if meta["state"] != "pending":
                # A guess applied by "Accept the rest" is the model's answer, not
                # the person's: a later tap corrects it (2026-10-02, two
                # corrections were refused this way, silently). A person's own
                # answer, or a photo already shared, stays as it is -- and says
                # so with a 409, which announce() does not keep quiet.
                if meta["state"] == "sent" and meta.get("auto") and answer in (NO, NOT_A_FACE):
                    # Shared without asking, and wrong: it stays in the chat, but
                    # the model must learn it was not her.
                    meta.update(state="rejected", corrected=True, answered=self.now(),
                                why="auto-shared, corrected")
                    self._save(cid, meta)
                    return 200, {"message": "Marked not %s (already in the chat; delete it there)" % self.cfg.name}
                if not meta.get("accepted") or meta["state"] == "sent":
                    return 409, {"error": "Already answered: %s" % meta["state"]}
                meta["accepted"] = False
                meta["corrected"] = True
                meta["summarized"] = None
            faces = meta.get("faces") or []
            if face_no is not None and not 1 <= face_no <= len(faces):
                return 400, {"error": "no face %d in this photo" % face_no}
            if answer in (NO, NOT_A_FACE):
                meta["state"] = "rejected"
            else:
                meta["state"] = "confirmed" if meta.get("share", True) else "trained"
            meta["answered"] = self.now()
            self._save(cid, meta)
        if answer == NOT_A_FACE:
            # A box on a hat or a shoe: nothing to learn, and filing it under
            # Classmates would teach Frigate a non-face as a person.
            taught, message = None, "Not a face, dropped"
        elif answer == NO:
            taught = self.cfg.negative_name if self.cfg.negative_name and self.cfg.train_on_confirm \
                and self._train(cid, meta, self.cfg.negative_name) else None
            message = "Not %s, %s" % (self.cfg.name, "filed under %s" % taught if taught else "dropped")
        elif face_no is not None:
            # A person pointed at her face: train on exactly that one, even on a
            # photo-only (train=False) question -- the box is no longer a guess.
            box = faces[face_no - 1]["box"]
            taught = self.cfg.name if self.cfg.train_on_confirm and \
                self._train(cid, dict(meta, box=box)) else None
            with self.lock:
                m2 = self._load(cid) or meta
                m2["box"], m2["chloe_face"] = box, face_no
                self._save(cid, m2)
            message = "%s confirmed (face %d)%s" % (self.cfg.name, face_no,
                                                    ", and taught to Frigate" if taught else "")
        else:
            taught = self.cfg.name if self.cfg.train_on_confirm and self._train(cid, meta) else None
            message = "%s confirmed%s%s" % (
                self.cfg.name, ", and taught to Frigate" if taught else "",
                " (not shared)" if not meta.get("share", True) else ", in the next batch")
        with self.lock:
            meta = self._load(cid) or meta
            meta["taught"] = taught
            self._save(cid, meta)
        self.summarize_if_done()
        return 200, {"message": message}

    # ------------------------------------------------------------------ no-face photos
    def add_noface(self, image_bytes, source=None, date=None, caption=None, share=True, faces=None):
        """A photo with no detectable face: held, not asked about, until the day
        has a confirmed photo of her to compare outfits against."""
        sha = hashlib.sha1(image_bytes).hexdigest()
        if any((source and m.get("source") == source) or (m.get("sha") == sha and m.get("date") == date)
               for m in self._all()):
            return 200, {"status": "duplicate"}
        if cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) is None:
            return 400, {"error": "not an image"}
        cid = uuid.uuid4().hex[:12]
        with open(self._path(cid, "jpg"), "wb") as f:
            f.write(image_bytes)
        # faces: the face model's boxes, kept so an outfit match is asked about
        # with numbered faces -- tapping her number then teaches the face model a
        # face it missed (looking down, at an angle), not just "she is in it".
        self._save(cid, {"id": cid, "state": "held", "source": source, "date": date, "sha": sha,
                         "caption": caption or "", "share": bool(share), "added": self.now(),
                         "faces": sorted(faces or [], key=lambda f: f["box"][0]),
                         "summarized": self.now()})
        return 200, {"id": cid, "status": "held"}

    def reference_sheet(self, date, limit=8, height=360):
        """Her, cropped from the day's confirmed photos: a generous region below
        and around each confirmed face, so the clothes are in it."""
        tiles = []
        for m in self._all():
            # Her face: one picked by number, or the face a guess was about once
            # the guess was confirmed or accepted (box_shown) -- not a pointer box
            # on a photo-only question, which may sit on a classmate.
            face_guess = (m.get("label") or "").startswith("%s 0." % self.cfg.name)
            box = m.get("box") or (m.get("box_shown") if face_guess else None)
            if m.get("date") != date or m["state"] not in ("confirmed", "trained", "sent") or not box:
                continue
            img = cv2.imread(self._path(m["id"], "jpg"))
            if img is None:
                continue
            x, y, w, h = box
            crop = img[max(0, y - h // 2):min(img.shape[0], y + 5 * h), max(0, x - w):min(img.shape[1], x + 2 * w)]
            if crop.size == 0:
                continue
            s = float(height) / crop.shape[0]
            tiles.append(cv2.copyMakeBorder(cv2.resize(crop, (max(1, int(crop.shape[1] * s)), height)),
                                            4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
            if len(tiles) >= limit:
                break
        if not tiles:
            return None
        sheet = np.hstack(tiles)
        if sheet.shape[1] > 2000:
            sheet = cv2.resize(sheet, (2000, int(sheet.shape[0] * 2000.0 / sheet.shape[1])))
        return cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    OUTFIT_PROMPT = (
        "Image 1 is a contact sheet of one little girl, photographed several times at preschool today; "
        "she wears the same clothes all day (she may take a layer off). Image 2 is another photo from the same "
        "classroom on the same day. Is the girl from image 1 in image 2? Her face may be hidden, turned away, "
        "blurred or too small, so compare clothing, accessories, hair and build. Other children wear similar "
        "colours, so only say yes when the details match. Also say how much of her the photo shows: "
        '"subject" if she is a main subject, "clear" if she is fully visible but not the focus, '
        '"partial" if only part of her is in frame (an arm, a sleeve, cut off by the edge), '
        '"tiny" if she is far away or very small. Reply only with JSON '
        '{"present": true|false, "visibility": "subject"|"clear"|"partial"|"tiny", '
        '"confidence": "high"|"medium"|"low", "where": "<short, or empty>", "why": "<one sentence>"}')

    # A sleeve at the edge of the frame or a speck in the background is her, but
    # not a photo of her (Brian, 2026-10-13). Only these are asked about.
    WORTH_ASKING = ("subject", "clear")

    @classmethod
    def worth_asking(cls, verdict):
        """Present, and enough of her to be a photo of her. A verdict without a
        visibility (an older prompt) counts, rather than silently dropping her."""
        return bool(verdict.get("present")) and verdict.get("visibility", "clear") in cls.WORTH_ASKING

    def ask_outfit(self, sheet, photo):
        """The vision model's verdict on one photo, or None if it could not be had."""
        b64 = lambda b: "data:image/jpeg;base64," + base64.b64encode(b).decode()
        try:
            r = self.http.post("https://openrouter.ai/api/v1/chat/completions", timeout=60,
                               headers={"Authorization": "Bearer " + self.cfg.outfit_key}, json={
                "model": self.cfg.outfit_model, "max_tokens": 3000,  # reasoning is charged first; 1000 truncated the JSON
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": self.OUTFIT_PROMPT},
                    {"type": "image_url", "image_url": {"url": b64(sheet)}},
                    {"type": "image_url", "image_url": {"url": b64(photo)}}]}]})
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            return json.loads(re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip())
        except Exception:
            logger.warning("remini review: outfit check failed", exc_info=True)
            return None

    def process_held(self):
        """Ask about held no-face photos for every day that now has references.
        A yes becomes an ordinary candidate guessed "Chloe (outfit)"; a no is
        dropped quietly. Days without references wait, up to keep-held-days."""
        held = [m for m in self._all() if m["state"] == "held"]
        if not held:
            return 0
        asked = 0
        for date in sorted({m.get("date") for m in held}):
            sheet = self.reference_sheet(date) if self.cfg.outfit_key else None
            for m in [m for m in held if m.get("date") == date]:
                if sheet is None:
                    if self.now() - m.get("added", 0) > self.cfg.held_days * 86400:
                        m.update(state="rejected", answered=self.now(), why="no references that day")
                        self._save(m["id"], m)
                    continue
                with open(self._path(m["id"], "jpg"), "rb") as f:
                    verdict = self.ask_outfit(sheet, f.read())
                if verdict is None:
                    continue  # try again next pass
                m["outfit"] = verdict
                if self.worth_asking(verdict):
                    m.update(state="pending", predicted="yes", summarized=None,
                             label="%s (outfit%s)" % (self.cfg.name,
                                                      ": " + verdict["where"] if verdict.get("where") else ""))
                    self._save(m["id"], m)
                    img = cv2.imread(self._path(m["id"], "jpg"))
                    self._ask(m["id"], img, m)
                    asked += 1
                else:
                    m.update(state="rejected", answered=self.now())
                    self._save(m["id"], m)
                logger.info("remini review: outfit check %s %s: %s", m["id"],
                            "match" if self.worth_asking(verdict) else
                            "present but %s, not asked" % verdict.get("visibility") if verdict.get("present") else
                            "no match", verdict.get("why", "")[:120])
        return asked

    def accept_rest(self):
        """Apply the model's prediction to every pending candidate that has one.

        Accepted predictions decide which photos are hers; they never train
        Frigate -- only a person's tap does that."""
        n = {YES: 0, NO: 0}
        with self.lock:
            for m in self._all():
                if m["state"] != "pending" or m.get("predicted") not in (YES, NO):
                    continue
                if m["predicted"] == YES:
                    m["state"] = "confirmed" if m.get("share", True) else "trained"
                else:
                    m["state"] = "rejected"
                m["answered"] = self.now()
                m["accepted"] = True
                m["taught"] = None
                self._save(m["id"], m)
                n[m["predicted"]] += 1
        self.summarize_if_done()
        return 200, {"message": "Accepted %d as %s and %d as not" % (n[YES], self.cfg.name, n[NO])}

    def labels(self, days=30):
        """Every answered candidate of the last `days`, without images: what the
        person said, and which face was hers. remini-chloe pulls this to train
        its own face model (preschool.py) -- training no longer happens here."""
        cutoff = self.now() - days * 86400
        keep = ("id", "source", "date", "state", "box", "box_shown", "chloe_face", "predicted", "label",
                "accepted", "answered")
        out = []
        for m in self._all():
            if m.get("answered") and m["answered"] >= cutoff and m["state"] not in ("pending", "held"):
                row = {k: m.get(k) for k in keep}
                # She is in the photo even though it was closed as rejected: the
                # outfit check saw her but only partly (a sleeve, far away), or the
                # question was withdrawn for that reason. Its faces must not be
                # learned as not-her.
                row["outfit_present"] = bool((m.get("outfit") or {}).get("present")) or \
                    str(m.get("why", "")).startswith("withdrawn")
                row["faces"] = [f["box"] for f in m.get("faces") or []]
                out.append(row)
        return out

    # ------------------------------------------------------------------ summary
    def summarize_if_done(self):
        """One Pushover for a whole round of answers, once the last pending one is
        answered -- by then every face is already in Frigate's library. Per-tap
        confirmations were a wall of near-identical lines (2026-10-02): the
        summary says what changed, with the faces themselves as the picture.
        Returns how many answers it covered."""
        with self.lock:
            items = self._all()
            if any(m["state"] == "pending" for m in items):
                return 0
            fresh = [m for m in items if m.get("answered") and not m.get("summarized")]
            if not fresh:
                return 0
            for m in fresh:
                m["summarized"] = self.now()
                self._save(m["id"], m)
        accepted = sum(1 for m in fresh if m.get("accepted"))
        yes = [m for m in fresh if m["state"] != "rejected"]
        no = [m for m in fresh if m["state"] == "rejected"]
        lines = ["\u2713 %d confirmed as %s" % (len(yes), self.cfg.name)]
        shared = sum(1 for m in yes if m["state"] in ("confirmed", "sent"))
        if shared:
            lines[0] += ", %d queued for iMessage" % shared
        if no:
            lines.append("\u2717 %d %s" % (len(no), "filed under %s" % self.cfg.negative_name
                                            if any(m.get("taught") for m in no) else "dropped"))
        # Only faces that were meant to train: a photo-only question (no box --
        # "is she anywhere in this photo?") never registers anything.
        untaught = [m for m in fresh if m.get("box") and not m.get("taught") and self.cfg.train_on_confirm]
        if untaught:
            lines.append("%d could not be added to Frigate (no face found)" % len(untaught))
        if accepted:
            lines.append("(%d of those were the predictions, accepted untouched)" % accepted)
        library = self._library_sizes()
        if library:
            lines.append("Frigate library: " + ", ".join("%s %d" % kv for kv in library))
        self._push("Remini review done", "\n".join(lines), self._mosaic(yes, no))
        logger.info("remini review: summary sent for %d answer(s)", len(fresh))
        return len(fresh)

    def _library_sizes(self):
        try:
            faces = self.http.get(self.cfg.frigate_url + "/api/faces", timeout=15).json()
        except Exception:
            return []
        names = [self.cfg.name] + ([self.cfg.negative_name] if self.cfg.negative_name else [])
        return [(n, len(faces.get(n) or [])) for n in names]

    def _mosaic(self, yes, no, cell=160, cols=6, limit=36):
        """The answered faces, cropped, Yes framed green and No framed grey."""
        tiles = []
        for m, colour in [(m, (60, 200, 60)) for m in yes] + [(m, (140, 140, 140)) for m in no]:
            img = cv2.imread(self._path(m["id"], "jpg"))
            if img is None:
                continue
            if m.get("box"):
                x, y, w, h = m["box"]
                pad = int(0.25 * max(w, h))
                face = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
            else:
                # A photo-only answer: the box was a pointer that may sit on a
                # classmate, so show the photo (centre square), not that face.
                hh, ww = img.shape[:2]
                side = min(hh, ww)
                face = img[(hh - side) // 2:(hh + side) // 2, (ww - side) // 2:(ww + side) // 2]
            if face.size == 0:
                continue
            t = cv2.resize(face, (cell, cell), interpolation=cv2.INTER_AREA)
            cv2.rectangle(t, (0, 0), (cell - 1, cell - 1), colour, 6)
            tiles.append(t)
            if len(tiles) >= limit:
                break
        if not tiles:
            return None
        tiles += [np.zeros((cell, cell, 3), np.uint8)] * (-len(tiles) % cols)
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        ok, jpg = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes() if ok else None

    def _push(self, title, message, image=None):
        if not self.cfg.pushover:
            return
        token, user = self.cfg.pushover
        files = {"attachment": ("faces.jpg", image, "image/jpeg")} if image else None
        try:
            r = self.http.post("https://api.pushover.net/1/messages.json", timeout=30, files=files,
                               data={"token": token, "user": user, "title": title,
                                     "message": message, "priority": 0})
            if r.status_code != 200:
                logger.warning("remini review: Pushover refused the summary: %s %s",
                               r.status_code, r.text.strip()[:300])
        except Exception:
            logger.warning("remini review: could not send the summary", exc_info=True)

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
        # Open questions hold the batch so a day goes out as one message -- unless
        # face matches share themselves: then waiting on an outfit question that
        # may sit unanswered overnight would hold back the photos already known.
        if not self.cfg.auto_share and any(m["state"] == "pending" for m in items):
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
            # A day already in the chat gets "More of Chloe Oct 15", not the
            # teacher's note again.
            shared_days = {m.get("date") for m in self._all() if m["state"] == "sent"} | self._noted()
            for chat in self.cfg.chats:
                for (date, caption), items in groups.items():
                    header = "More of %s %s" % (self.cfg.name, pretty_date(date)) if date in shared_days \
                        else self.header(date, caption)
                    self._bb_text(chat, password, header)
                    for m in items:
                        self._bb_photo(chat, password, self._path(m["id"], "jpg"))
            for m in ready:
                m["state"] = "sent"
                m["sent"] = self.now()
                self._save(m["id"], m)
            logger.info("remini review: sent %d photo(s) to %s", len(ready), ", ".join(self.cfg.chats))
        self._announce_sent(ready)
        return len(ready)

    def _announce_sent(self, sent):
        """After a batch goes out: one Pushover with what was sent, numbered, and a
        page to say "not Chloe" for any of them -- it cannot unsend, but it
        teaches the model. Only when something was shared without being asked."""
        if not any(m.get("auto") for m in sent):
            return
        tiles = []
        for n, m in enumerate(sent, 1):
            img = cv2.imread(self._path(m["id"], "jpg"))
            if img is None:
                continue
            h, w = img.shape[:2]
            side = min(h, w)
            t = cv2.resize(img[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2], (220, 220))
            cv2.rectangle(t, (0, 0), (46, 38), (0, 0, 0), -1)
            cv2.putText(t, str(n), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(t)
        image = None
        if tiles:
            tiles += [np.zeros((220, 220, 3), np.uint8)] * (-len(tiles) % 4)
            grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
            image = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        url = self.cfg.sent_page_url + ("&" if "?" in self.cfg.sent_page_url else "?") + \
            urlencode({"ids": ",".join(m["id"] for m in sent), "name": self.cfg.name}, quote_via=quote)
        days = sorted({pretty_date(m.get("date")) for m in sent})
        if not self.cfg.pushover:
            return
        token, user = self.cfg.pushover
        try:
            self.http.post("https://api.pushover.net/1/messages.json", timeout=30,
                           files={"attachment": ("sent.jpg", image, "image/jpeg")} if image else None,
                           data={"token": token, "user": user, "priority": 0,
                                 "title": "Sent %d photo%s of %s" % (len(sent), "" if len(sent) == 1 else "s", self.cfg.name),
                                 "message": "%s, to the family chat. Not her in one? Tap to mark it." % ", ".join(days),
                                 "url": url[:512], "url_title": "Mark one as not %s" % self.cfg.name})
        except Exception:
            logger.warning("remini review: could not announce the batch", exc_info=True)

    def _notes_path(self):
        return os.path.join(self.cfg.dir, "notes-sent.txt")   # not .json: _all() reads those

    def _noted(self):
        try:
            with open(self._notes_path()) as f:
                return {l.strip() for l in f if l.strip()}
        except OSError:
            return set()

    def send_note(self, date, caption):
        """The teacher's note alone, for a day with no photo of her to share.
        Once per day; skipped when the day already went out with photos. A photo
        of her found later still goes out, headed "More of Chloe <day>"."""
        if not date or not (caption or "").strip():
            return 400, {"error": "date and caption are needed"}
        with self.lock:
            if date in self._noted():
                return 200, {"sent": False, "why": "note already sent"}
            if any(m.get("date") == date and m["state"] in ("sent", "confirmed", "pending")
                   for m in self._all()):
                return 200, {"sent": False, "why": "the day has photos of %s" % self.cfg.name}
            if not self.cfg.chats:
                return 200, {"sent": False, "why": "no chats configured"}
            password = self._bluebubbles_password()
            text = self.cfg.note_text.format(name=self.cfg.name, date=pretty_date(date), caption=caption.strip())
            for chat in self.cfg.chats:
                self._bb_text(chat, password, text)
            with open(self._notes_path(), "a") as f:
                f.write(date + "\n")
        logger.info("remini review: sent the teacher's note for %s (no photos of %s)", date, self.cfg.name)
        return 200, {"sent": True}

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
                self.process_held()
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
