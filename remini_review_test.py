import json
import os
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
        self.outfit_calls, self.outfit_answer = 0, {"present": True, "where": "left", "why": "same pink pants"}

    def get(self, url, timeout=None):
        return Resp(body={"Chloe": ["a"] * 21, "Classmates": ["b"] * 13})

    def post(self, url, data=None, files=None, json=None, params=None, timeout=None, headers=None):
        if "pushover" in url:
            self.pushes.append(data)
            return Resp(body={"status": 1})
        if "/register" in url:
            self.registers.append(url)
            return Resp(body={"success": True})
        if url.endswith("/message/text"):
            self.texts.append((json["chatGuid"], json["message"]))
            return Resp()
        if "openrouter" in url:
            self.outfit_calls += 1
            return Resp(body={"choices": [{"message": {"content": "```json\n" + __import__("json").dumps(self.outfit_answer) + "\n```"}}]})
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
    c["verify"] = {"api-key": "k", "model": "m"}
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
    assert code == 200 and res["message"] == "Not Chloe, filed under Classmates"
    assert http.registers[1].endswith("/api/faces/Classmates/register")
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
    assert http.texts == [("any;-;me@example.com", "Chloe today")]
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


def test_no_without_a_negative_class_just_drops(tmp_path):
    r, http, _ = make(tmp_path, **{"negative-name": ""})
    a = r.add_candidate(jpeg(), [1, 1, 80, 80], source="a")[1]["id"]
    assert r.confirm(a, "no")[1]["message"] == "Not Chloe, dropped"
    assert http.registers == []


def test_training_only_candidate_teaches_but_is_never_shared(tmp_path):
    r, http, clock = make(tmp_path)
    a = r.add_candidate(jpeg(), [1, 1, 80, 80], source="old", date="2026-09-15", share=False)[1]["id"]
    assert "training only" in http.pushes[0]["message"]
    code, res = r.confirm(a, "yes")
    assert res["message"] == "Chloe confirmed, and taught to Frigate (not shared)"
    assert len(http.registers) == 1
    clock.t += 3600
    assert r.send_digest() == 0 and http.photos == []


def test_digest_heads_each_post_with_the_teachers_message(tmp_path):
    r, http, clock = make(tmp_path)
    for i, (date, cap) in enumerate([("2026-09-15", "Hi Families!\nApples today."),
                                     ("2026-09-15", "Hi Families!\nApples today."),
                                     ("2026-09-16", "")]):
        cid = r.add_candidate(jpeg(), [1, 1, 80, 80], source=str(i), date=date, caption=cap)[1]["id"]
        r.confirm(cid, "yes")
    clock.t += 3600
    assert r.send_digest() == 3
    assert [t for _, t in http.texts] == ["Chloe Sept 15: Hi Families!\nApples today.", "Chloe Sept 16"]
    assert len(http.photos) == 3


def test_pretty_date():
    assert rr.pretty_date("2026-09-15") == "Sept 15"
    assert rr.pretty_date("2026-03-02") == "March 2"
    assert rr.pretty_date(None) == "today"


def test_trim_removes_only_old_rejections(tmp_path):
    r, http, clock = make(tmp_path)
    no = r.add_candidate(jpeg(), [1, 1, 80, 80], source="no")[1]["id"]
    yes = r.add_candidate(jpeg(), [1, 1, 80, 80], source="yes", share=False)[1]["id"]
    r.confirm(no, "no"); r.confirm(yes, "yes")
    clock.t += 29 * 86400
    assert r.trim() == 0
    clock.t += 2 * 86400
    assert r.trim() == 1
    left = sorted(os.listdir(r.cfg.dir))
    assert left == sorted([yes + ".jpg", yes + ".json"])


def test_one_summary_with_faces_once_the_round_is_answered(tmp_path):
    r, http, _ = make(tmp_path)
    ids = [r.add_candidate(jpeg(), [100, 100, 80, 80], source=str(i), share=False)[1]["id"]
           for i in range(3)]
    asked = len(http.pushes)
    r.confirm(ids[0], "yes")
    r.confirm(ids[1], "no")
    assert len(http.pushes) == asked              # one still pending: no summary yet
    r.confirm(ids[2], "yes")
    summary = http.pushes[-1]
    assert len(http.pushes) == asked + 1
    assert summary["title"] == "Remini review done"
    assert "2 confirmed as Chloe" in summary["message"]
    assert "1 filed under Classmates" in summary["message"]
    assert "Frigate library: Chloe 21, Classmates 13" in summary["message"]
    assert r.summarize_if_done() == 0             # nothing new: no second summary


def test_predictions_shown_corrected_and_the_rest_accepted(tmp_path):
    r, http, _ = make(tmp_path)
    a = r.add_candidate(jpeg(), [1, 1, 80, 80], source="a", share=False, predicted="yes",
                        label="Chloe 0.67", train=False)[1]["id"]
    b = r.add_candidate(jpeg(), [1, 1, 80, 80], source="b", share=False, predicted="no",
                        label="Classmates 0.44", train=False)[1]["id"]
    c = r.add_candidate(jpeg(), [1, 1, 80, 80], source="c", share=False, predicted="no",
                        label="Classmates 0.30", train=False)[1]["id"]
    assert http.pushes[0]["message"].startswith("Looks like: Chloe 0.67")
    assert "guess=Chloe%200.67" in http.pushes[0]["url"]
    r.confirm(c, "yes")                               # a correction
    code, res = r.accept_rest()
    assert res["message"] == "Accepted 1 as Chloe and 1 as not"
    states = {m["id"]: m["state"] for m in r._all()}
    assert states == {a: "trained", b: "rejected", c: "trained"}
    assert http.registers == []                       # train=False: no box ever registers
    assert "1 of those were the predictions" not in http.pushes[-1]["message"]
    assert "2 of those were the predictions" in http.pushes[-1]["message"]


def test_accept_rest_via_confirm_and_not_a_face(tmp_path):
    r, http, _ = make(tmp_path)
    a = r.add_candidate(jpeg(), [1, 1, 80, 80], source="a", predicted="no", label="x")[1]["id"]
    b = r.add_candidate(jpeg(), [1, 1, 80, 80], source="b")[1]["id"]
    assert r.confirm(b, "notface")[1]["message"] == "Not a face, dropped"
    assert http.registers == []                       # not filed under Classmates
    assert r.confirm("__rest__", "accept")[0] == 200
    assert {m["id"]: m["state"] for m in r._all()}[a] == "rejected"


def test_every_face_is_boxed_in_its_class_colour(tmp_path, monkeypatch):
    drawn = []
    real = rr.cv2.rectangle
    monkeypatch.setattr(rr.cv2, "rectangle", lambda img, a, b, colour, t: drawn.append(colour) or real(img, a, b, colour, t))
    r, http, _ = make(tmp_path)
    faces = [{"box": [10, 10, 40, 40], "cls": "chloe"}, {"box": [100, 10, 40, 40], "cls": "classmate"},
             {"box": [200, 10, 40, 40], "cls": "unknown"}]
    r.add_candidate(jpeg(), [10, 10, 40, 40], source="a", predicted="yes", label="Chloe 0.6",
                    train=False, faces=faces)
    assert drawn[::2] == [rr.FACE_COLOURS["chloe"], rr.FACE_COLOURS["classmate"], rr.FACE_COLOURS["unknown"]]  # box, then its number tag
    assert rr.LEGEND in http.pushes[0]["message"]


def test_numbered_face_answer_trains_exactly_that_face(tmp_path):
    r, http, _ = make(tmp_path)
    faces = [{"box": [200, 10, 40, 40], "cls": "unknown"}, {"box": [10, 10, 40, 40], "cls": "classmate"}]
    cid = r.add_candidate(jpeg(), [200, 10, 40, 40], source="a", predicted="no", label="x",
                          train=False, faces=faces)[1]["id"]
    assert "faces=2" in http.pushes[0]["url"]
    assert r.confirm(cid, "face:3")[0] == 400                      # only two faces
    code, res = r.confirm(cid, "face:2")                           # sorted left-to-right: x=200 is face 2
    assert code == 200 and res["message"] == "Chloe confirmed (face 2), and taught to Frigate"
    assert http.registers == ["http://192.168.254.31:5000/api/faces/Chloe/register"]
    m = r._load(cid)
    assert m["state"] == "confirmed" and m["box"] == [200, 10, 40, 40] and m["chloe_face"] == 2


def test_noface_photo_waits_for_a_reference_then_asks_on_a_match(tmp_path):
    r, http, clock = make(tmp_path)
    held = r.add_noface(jpeg(), source="back", date="2026-09-17", caption="note")[1]["id"]
    assert r.process_held() == 0 and http.outfit_calls == 0      # no reference yet: nothing to compare
    ref = r.add_candidate(jpeg(), [100, 50, 40, 40], source="face", date="2026-09-17")[1]["id"]
    r.confirm(ref, "yes")
    pushes = len(http.pushes)
    assert r.process_held() == 1 and http.outfit_calls == 1
    m = r._load(held)
    assert m["state"] == "pending" and m["predicted"] == "yes" and m["label"] == "Chloe (outfit: left)"
    assert len(http.pushes) == pushes + 1 and http.pushes[-1]["message"].startswith("Looks like: Chloe (outfit")
    assert r.add_noface(jpeg(), source="back", date="2026-09-17")[1]["status"] == "duplicate"


def test_noface_no_match_is_dropped_quietly(tmp_path):
    r, http, _ = make(tmp_path)
    http.outfit_answer = {"present": False, "why": "only a boy"}
    held = r.add_noface(jpeg(), source="x", date="2026-09-17")[1]["id"]
    ref = r.add_candidate(jpeg(), [100, 50, 40, 40], source="face", date="2026-09-17")[1]["id"]
    r.confirm(ref, "yes"); pushes = len(http.pushes)
    r.process_held()
    assert r._load(held)["state"] == "rejected" and len(http.pushes) == pushes


def test_held_photo_expires_without_references(tmp_path):
    r, http, clock = make(tmp_path)
    held = r.add_noface(jpeg(), source="x", date="2026-09-20")[1]["id"]
    clock.t += 4 * 86400
    r.process_held()
    assert r._load(held)["state"] == "rejected" and http.outfit_calls == 0


def test_labels_lists_answers_with_the_face_picked(tmp_path):
    r, http, _ = make(tmp_path)
    faces = [{"box": [10, 10, 40, 40], "cls": "unknown"}, {"box": [200, 10, 40, 40], "cls": "chloe"}]
    a = r.add_candidate(jpeg(), [200, 10, 40, 40], source="a", date="2026-09-17", train=False, faces=faces)[1]["id"]
    b = r.add_candidate(jpeg(), [1, 1, 80, 80], source="b", date="2026-09-17")[1]["id"]
    r.add_candidate(jpeg(), [1, 1, 80, 80], source="c", date="2026-09-17")      # still pending: not listed
    r.confirm(a, "face:2"); r.confirm(b, "no")
    got = {l["source"]: l for l in r.labels()}
    assert set(got) == {"a", "b"}
    assert got["a"]["chloe_face"] == 2 and got["a"]["box"] == [200, 10, 40, 40]
    assert got["a"]["faces"] == [[10, 10, 40, 40], [200, 10, 40, 40]]
    assert got["b"]["state"] == "rejected"


def test_accepted_face_guess_is_a_reference_for_the_outfit_check(tmp_path):
    r, http, _ = make(tmp_path)
    held = r.add_noface(jpeg(), source="back", date="2026-09-24")[1]["id"]
    r.add_candidate(jpeg(), [100, 50, 40, 40], source="face", date="2026-09-24", predicted="yes",
                    label="Chloe 0.62", train=False)
    r.accept_rest()
    assert r.reference_sheet("2026-09-24") is not None
    assert r.process_held() == 1 and r._load(held)["state"] == "pending"
