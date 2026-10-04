import json
from configparser import ConfigParser
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pytest

import face_review as fr


def webp(w, h, level=128, blur=0):
    """A textured crop: sharp and not blown out unless asked to be."""
    rng = np.random.default_rng(w * 1000 + h)
    img = np.clip(rng.normal(level, 40, (h, w, 3)), 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 100])
    return buf.tobytes()


class Resp(object):
    def __init__(self, code=200, body=None, content=b""):
        self.status_code, self._body, self.content = code, body, content
        self.ok = code < 400
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise fr.requests.HTTPError(str(self.status_code))


class FakeHTTP(object):
    """Frigate + Pushover, in memory."""

    def __init__(self, train, images, library=("Chloe", "Lee")):
        self.train, self.images, self.library = list(train), images, library
        self.pushes, self.posts = [], []

    def get(self, url, timeout=None):
        if url.endswith("/api/faces"):
            out = {n: [] for n in self.library}
            out["train"] = list(self.train)
            return Resp(body=out)
        if "/clips/faces/train/" in url:
            name = url.rsplit("/", 1)[1]
            return Resp(content=self.images[name]) if name in self.images else Resp(404)
        if "/api/events/" in url:
            return Resp(body={"camera": "front_entry"})
        return Resp(404)

    def post(self, url, data=None, files=None, json=None, timeout=None):
        if "pushover" in url:
            self.pushes.append(data)
            return Resp(body={"status": 1})
        self.posts.append((url, json))
        if url.endswith("/api/faces/train/delete") or "/classify" in url:
            self.train.remove(json.get("training_file") or json["ids"][0])
            return Resp(body={"success": True})
        return Resp(404)


def make(tmp_path, http, **opts):
    c = ConfigParser()
    c["detector"] = {"save-path": str(tmp_path)}
    c["pushover"] = {"token": "t", "user": "u"}
    c["face-review"] = {"page-url": "https://ha.example/local/face-review.html", **opts}
    return fr.FaceReview(fr.FaceReviewConfig(c), session=http)


A1 = "1790000000.1-abc123-1790000001.5-unknown-0.12.webp"
A2 = "1790000000.1-abc123-1790000002.5-Chloe-0.83.webp"   # same event, bigger
B1 = "1790000100.2-def456-1790000101.0-unknown-0.30.webp"
TINY = "1790000200.3-ghi789-1790000201.0-unknown-0.10.webp"


def test_parse_train_name():
    got = fr.parse_train_name(A2)
    assert got["event"] == "1790000000.1-abc123"
    assert got["guess"] == "Chloe" and got["score"] == 0.83 and got["ts"] == 1790000002.5
    assert fr.parse_train_name("garbage.webp") is None


def test_first_run_is_backlog_not_a_flood(tmp_path):
    http = FakeHTTP([A1, A2, B1], {A1: webp(80, 80), A2: webp(120, 120), B1: webp(90, 90)})
    rev = make(tmp_path, http)
    assert rev.poll_once() == 0
    assert http.pushes == []
    # nothing new since: still nothing
    assert rev.poll_once() == 0


def test_new_sighting_sends_largest_attempt_once(tmp_path):
    http = FakeHTTP([], {A1: webp(80, 80), A2: webp(120, 120), B1: webp(90, 90)})
    rev = make(tmp_path, http)
    rev.poll_once()                       # seed: empty tab
    http.train += [A1, A2]
    assert rev.poll_once() == 1
    push = http.pushes[0]
    q = parse_qs(urlparse(push["url"]).query)
    assert q["file"] == [A2]              # the 120px attempt, not the 80px one
    assert q["names"] == ["Chloe,Lee"]
    assert "Chloe" in push["message"] and "front entry" in push["message"]
    assert rev.poll_once() == 0           # same event again: silent


def test_too_small_is_not_sent(tmp_path):
    http = FakeHTTP([], {TINY: webp(40, 40)})
    rev = make(tmp_path, http)
    rev.poll_once()
    http.train.append(TINY)
    assert rev.poll_once() == 0 and http.pushes == []


def test_burst_is_capped_per_poll(tmp_path):
    files = ["17900%05d.1-e%04d-17900%05d.5-unknown-0.10.webp" % (i, i, i) for i in range(7)]
    http = FakeHTTP([], {f: webp(100, 100) for f in files})
    rev = make(tmp_path, http, **{"max-per-poll": "3"})
    rev.poll_once()
    http.train += files
    assert rev.poll_once() == 3
    assert rev.poll_once() == 3
    assert rev.poll_once() == 1


def test_classify_moves_the_file_and_repeat_is_already_handled(tmp_path):
    http = FakeHTTP([A2], {})
    rev = make(tmp_path, http)
    code, res = rev.classify(A2, "Chloe")
    assert code == 200 and res["message"] == "Added to Chloe"
    assert http.posts[0][0].endswith("/api/faces/train/Chloe/classify")
    code, res = rev.classify(A2, "Chloe")
    assert code == 200 and res["message"] == "Already handled"


def test_classify_new_person_with_space(tmp_path):
    http = FakeHTTP([B1], {})
    rev = make(tmp_path, http)
    code, _ = rev.classify(B1, "Nhyle Lapitan")
    assert code == 200
    assert http.posts[0][0].endswith("/api/faces/train/Nhyle%20Lapitan/classify")


def test_skip_and_delete(tmp_path):
    http = FakeHTTP([A2], {})
    rev = make(tmp_path, http)
    assert rev.classify(A2, fr.SKIP) == (200, {"message": "Left in the Train tab"})
    assert http.posts == [] and A2 in http.train
    code, res = rev.classify(A2, fr.DELETE)
    assert code == 200 and res["message"] == "Not a face, removed" and A2 not in http.train


@pytest.mark.parametrize("answer,said", [(fr.CANT_TELL, "Can't tell"), (fr.VISITOR, "Visitor")])
def test_cant_tell_and_visitor_remove_without_training(tmp_path, answer, said):
    http = FakeHTTP([A2], {})
    rev = make(tmp_path, http)
    code, res = rev.classify(A2, answer)
    assert code == 200 and res["message"].startswith(said)
    assert http.posts[0][0].endswith("/api/faces/train/delete")   # never /classify
    assert A2 not in http.train


def test_unknown_reserved_answer_is_refused(tmp_path):
    http = FakeHTTP([A2], {})
    assert make(tmp_path, http).classify(A2, "__oops__")[0] == 400
    assert http.posts == []


def test_page_link_drops_names_to_fit_pushover():
    names = ["Person%02d With A Long Name" % i for i in range(40)]
    url = fr.page_link("https://ha.example/local/face-review.html", A2, names)
    assert len(url) <= fr.URL_LIMIT
    assert parse_qs(urlparse(url).query)["file"] == [A2]


@pytest.mark.parametrize("name", ["", None])
def test_classify_needs_a_name(tmp_path, name):
    rev = make(tmp_path, FakeHTTP([A2], {}))
    assert rev.classify(A2, name)[0] == 400


# --- only faces fit to name are sent -----------------------------------------

def test_a_washed_out_face_is_not_sent(tmp_path):
    """21:40 on 2026-10-03: the garage light whited out every attempt."""
    http = FakeHTTP([], {B1: webp(140, 160, level=250)})
    rev = make(tmp_path, http)
    rev.poll_once()
    http.train.append(B1)
    assert rev.poll_once() == 0 and http.pushes == []
    assert json.load(open(rev.cfg.state_path))[B1.split("-1790000101")[0]] == "unfit"


def test_a_blurred_face_is_not_sent(tmp_path):
    http = FakeHTTP([], {B1: webp(140, 160, blur=4)})
    rev = make(tmp_path, http)
    rev.poll_once()
    http.train.append(B1)
    assert rev.poll_once() == 0 and http.pushes == []


def test_the_largest_fit_attempt_wins_over_a_larger_unfit_one(tmp_path):
    """Largest alone picked the most blown-out of the three."""
    http = FakeHTTP([], {A1: webp(90, 90), A2: webp(150, 150, level=250)})
    rev = make(tmp_path, http)
    rev.poll_once()
    http.train += [A1, A2]
    assert rev.poll_once() == 1
    assert parse_qs(urlparse(http.pushes[0]["url"]).query)["file"] == [A1]


def test_quality_thresholds_are_configurable(tmp_path):
    http = FakeHTTP([], {B1: webp(140, 160, level=250)})
    rev = make(tmp_path, http, **{"max-blown": "0.9"})
    rev.poll_once()
    http.train.append(B1)
    assert rev.poll_once() == 1


def test_face_quality_measures_the_crop():
    flat = np.full((100, 100, 3), 128, np.uint8)
    white = np.full((100, 100, 3), 255, np.uint8)
    assert fr.face_quality(white)[0] == 1.0
    assert fr.face_quality(flat)[1] == 0.0
    assert fr.face_quality(flat)[2] is None        # no detector configured
