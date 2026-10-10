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


# --- mosaic, ranking, size, people (2026-10-10) ---------------------------------

K0 = "1791610887.5-qmhvye-1791610891.5-Kyle-0.85.webp"      # 64x86, sharp
K1 = "1791610887.5-qmhvye-1791610892.4-unknown-0.36.webp"   # 83x92, soft
K2 = "1791610887.5-qmhvye-1791610893.1-unknown-0.00.webp"   # 89x101, blurred


def _attachment(push_files):
    return cv2.imdecode(np.frombuffer(push_files, np.uint8), cv2.IMREAD_COLOR)


class FilesHTTP(FakeHTTP):
    """FakeHTTP that also keeps the attachments."""
    def __init__(self, *a, **k):
        super(FilesHTTP, self).__init__(*a, **k)
        self.attachments = []

    def post(self, url, data=None, files=None, json=None, timeout=None):
        if "pushover" in url and files:
            self.attachments.append(files["attachment"][1])
        return super(FilesHTTP, self).post(url, data=data, files=files, json=json,
                                           timeout=timeout)


def _visit(tmp_path, images, **opts):
    http = FilesHTTP([], images)
    rev = make(tmp_path, http, **opts)
    rev.poll_once()
    http.train += list(images)
    return rev, http


def test_the_sharp_named_face_trains_not_the_bigger_soft_one(tmp_path):
    """Kyle at 01:41 on 2026-10-10: the notification showed the tilted, soft
    83x92 crop and left out the sharp 64x86 one Frigate had at Kyle 0.85."""
    rev, http = _visit(tmp_path, {K0: webp(64, 86), K1: webp(83, 92, blur=1.2),
                                  K2: webp(89, 101, blur=3)})
    assert rev.poll_once() == 1
    q = parse_qs(urlparse(http.pushes[0]["url"]).query)
    assert q["file"] == [K0]
    assert "best of 3" in http.pushes[0]["message"]


def test_a_64x86_face_is_big_enough(tmp_path):
    """Frigate's floor is an area (4800); a 70 px minimum side refused it."""
    rev, http = _visit(tmp_path, {K0: webp(64, 86)})
    assert rev.poll_once() == 1


def test_several_attempts_arrive_as_one_mosaic(tmp_path):
    rev, http = _visit(tmp_path, {K0: webp(64, 86), K1: webp(83, 92, blur=1.2),
                                  K2: webp(89, 101, blur=3)})
    rev.poll_once()
    pic = _attachment(http.attachments[0])
    assert pic.shape[:2] == (fr.TILE, 3 * fr.TILE)


def test_one_attempt_is_sent_as_the_face_itself(tmp_path):
    rev, http = _visit(tmp_path, {K0: webp(64, 86)})
    rev.poll_once()
    pic = _attachment(http.attachments[0])
    assert pic.shape[0] != fr.TILE        # the upscaled face, not a tile
    assert "best of" not in http.pushes[0]["message"]


def test_the_mosaic_is_capped(tmp_path):
    files = {"1790000000.1-many01-17900000%02d.5-unknown-0.10.webp" % i: webp(100 + i, 100)
             for i in range(9)}
    rev, http = _visit(tmp_path, files, **{"max-tiles": "4"})
    rev.poll_once()
    pic = _attachment(http.attachments[0])
    assert pic.shape[:2] == (2 * fr.TILE, 3 * fr.TILE)   # 4 tiles, 3 a row
    assert "best of 4" in http.pushes[0]["message"]


def test_an_event_naming_two_people_is_split():
    """Josiana's helper walks in with her. Frigate tracks them as two events;
    should the tracker swap them inside one, each still gets their own ask."""
    atts = [fr.parse_train_name(f) for f in (
        "1790000000.1-two001-1790000001.5-Josiana-0.92.webp",
        "1790000000.1-two001-1790000002.5-Lee-0.88.webp",
        "1790000000.1-two001-1790000003.5-unknown-0.20.webp")]
    groups = fr.split_people(atts)
    assert sorted(g[0]["guess"] for g in groups) == ["Josiana", "Lee"]
    assert all(len(g) == 1 for g in groups)


def test_one_named_person_and_unknowns_stay_together():
    atts = [fr.parse_train_name(f) for f in (K0, K1, K2)]
    assert fr.split_people(atts) == [atts]


def test_two_people_in_one_event_get_two_notifications(tmp_path):
    a = "1790000000.1-two002-1790000001.5-Chloe-0.92.webp"
    b = "1790000000.1-two002-1790000002.5-Lee-0.90.webp"
    rev, http = _visit(tmp_path, {a: webp(100, 100), b: webp(100, 100)})
    rev.poll_once()
    sent = sorted(parse_qs(urlparse(p["url"]).query)["file"][0] for p in http.pushes)
    assert sent == [a, b]
