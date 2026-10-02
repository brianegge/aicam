import json
from configparser import ConfigParser

import cv2
import numpy as np

import remini_review as rr


def jpeg(w=400, h=300):
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 100, np.uint8))
    return buf.tobytes()


class Resp(object):
    def __init__(self, code=200, body=None):
        self.status_code, self._body, self.ok = code, body or {}, code < 400
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise rr.requests.HTTPError(str(self.status_code))


class FakeHTTP(object):
    def __init__(self):
        self.pushes, self.registers, self.texts, self.photos = [], [], [], []

    def post(self, url, data=None, files=None, json=None, params=None, timeout=None):
        if "pushover" in url:
            self.pushes.append(data)
            return Resp(body={"status": 1})
        if "/register" in url:
            self.registers.append(url)
            return Resp(body={"success": True})
        if url.endswith("/message/text"):
            self.texts.append((json["chatGuid"], json["message"]))
            return Resp()
        if url.endswith("/message/attachment"):
            self.photos.append(data["chatGuid"])
            return Resp()
        return Resp(404)


class Clock(object):
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(tmp_path, **opts):
    c = ConfigParser()
    c["detector"] = {"save-path": str(tmp_path)}
    c["pushover"] = {"token": "t", "user": "u"}
    c["remini-review"] = {"page-url": "https://ha.example/local/remini-review.html",
                          "imessage-chats": "any;-;me@example.com", **opts}
    db = tmp_path / "bb.db"
    con = __import__("sqlite3").connect(str(db))
    con.execute("create table config(name text, value text)")
    con.execute("insert into config values('password','pw')")
    con.commit(); con.close()
    c["remini-review"]["bluebubbles-db"] = str(db)
    http, clock = FakeHTTP(), Clock()
    return rr.ReminiReview(rr.ReminiReviewConfig(c), session=http, now=clock), http, clock


def test_candidate_asks_once_and_duplicates_are_ignored(tmp_path):
    r, http, _ = make(tmp_path)
    code, res = r.add_candidate(jpeg(), [50, 60, 80, 90], 0.93, "https://cdn/x.jpg", "2026-10-01")
    assert code == 200 and res["status"] == "pending"
    assert http.pushes[0]["title"] == "Is this Chloe?"
    assert "file=%s" % res["id"] in http.pushes[0]["url"]
    code, again = r.add_candidate(jpeg(), [50, 60, 80, 90], 0.93, "https://cdn/x.jpg", "2026-10-01")
    assert again["status"] == "duplicate" and len(http.pushes) == 1


def test_not_an_image(tmp_path):
    r, _, _ = make(tmp_path)
    assert r.add_candidate(b"nope", None)[0] == 400


def test_yes_trains_frigate_and_no_does_not(tmp_path):
    r, http, _ = make(tmp_path)
    a = r.add_candidate(jpeg(), [50, 60, 80, 90], source="a")[1]["id"]
    b = r.add_candidate(jpeg(), [50, 60, 80, 90], source="b")[1]["id"]
    code, res = r.confirm(a, "yes")
    assert code == 200 and "taught to Frigate" in res["message"]
    assert http.registers == ["http://192.168.254.31:5000/api/faces/Chloe/register"]
    code, res = r.confirm(b, "no")
    assert code == 200 and len(http.registers) == 1
    assert r.confirm(a, "yes")[1]["message"].startswith("Already answered")
    assert r.confirm(a, "maybe")[0] == 400
    assert r.confirm("missing", "yes")[0] == 404


def test_digest_waits_for_pending_and_the_quiet_period(tmp_path):
    r, http, clock = make(tmp_path, **{"digest-delay-minutes": "15"})
    a = r.add_candidate(jpeg(), [1, 1, 80, 80], source="a")[1]["id"]
    b = r.add_candidate(jpeg(), [1, 1, 80, 80], source="b")[1]["id"]
    r.confirm(a, "yes")
    clock.t += 3600
    assert r.send_digest() == 0                 # b still unanswered
    r.confirm(b, "no")                          # a "no" does not restart the quiet period
    assert r.send_digest() == 1
    assert http.texts == [("any;-;me@example.com", "Here are some pics of Chloe from school today")]
    assert http.photos == ["any;-;me@example.com"]
    assert r.send_digest() == 0                 # sent is sent


def test_digest_quiet_period_after_last_yes(tmp_path):
    r, http, clock = make(tmp_path, **{"digest-delay-minutes": "15"})
    a = r.add_candidate(jpeg(), [1, 1, 80, 80], source="a")[1]["id"]
    r.confirm(a, "yes")
    clock.t += 14 * 60
    assert r.send_digest() == 0
    clock.t += 60
    assert r.send_digest() == 1
