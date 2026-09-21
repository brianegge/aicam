"""Tests for the notification decisions in notify.py.

First tests for this module. They cover the plate-driven side effects, which
are the ones that reach outside aicam: opening the garage for the house
cleaner, and pausing person detection for a crew. Both act on a matched plate
record, so a wrong match has consequences beyond a mis-worded alert.
"""

import json
import logging
from unittest import mock

import pytest

import notify


@pytest.fixture(autouse=True)
def plates_db():
    """A licence-plate database, without touching the real file."""
    db = {
        "BV41507": {"owner": "Jorge's Lawn Care", "color": "white",
                    "make": "Mitsubishi", "model": "Fuso FE HD",
                    "suppress_person": True},
        "AUDUSD": {"owner": "Brian", "color": "gray", "make": "Ford",
                   "model": "Mustang Mach-E"},
        "2AVJU3": {"owner": "House cleaner"},
    }
    with mock.patch.object(notify, "license_plates", dict(db)), \
         mock.patch.object(notify, "_plate_variants",
                           {p: p for p in db}):
        yield db


def _ha():
    """A Home Assistant that says yes to everything and records the calls.

    vacation_mode True short-circuits the Pushover post, which needs a real
    image; the plate side effects run before it.
    """
    ha = mock.Mock()
    ha.mode.return_value = "home"
    ha.vacation_mode.return_value = True
    ha.should_notify_person.return_value = True
    ha.should_notify_vehicle.return_value = True
    ha.is_dog_inside.return_value = False
    return ha


def _vehicle(plate):
    return {"tagName": "vehicle", "probability": 0.95, "plate": plate,
            "camName": "garage_left",
            "boundingBox": {"left": 0.3, "top": 0.3, "width": 0.3, "height": 0.3},
            "center": {"x": 0.45, "y": 0.45}}


def _config(tmp_path):
    """The sections notify() reads. It crops and saves a static image before it
    ever looks at a plate, so save-path has to be real."""
    import configparser
    (tmp_path / "static").mkdir(exist_ok=True)
    c = configparser.ConfigParser()
    c["priority"] = {"vehicle": "0", "person": "1", "dog": "0"}
    c["sounds"] = {"departed": "none", "vehicle": "pushover", "person": "pushover"}
    c["detector"] = {"save-path": str(tmp_path)}
    return c


def _notify(ha, predictions, tmp_path):
    from PIL import Image
    cam = mock.Mock()
    cam.name = "garage-l"
    cam.road_line = None
    return notify.notify(cam, "", Image.new("RGB", (1920, 1080), (40, 40, 40)),
                         predictions, _config(tmp_path), ha)


def test_a_flagged_vehicle_pauses_person_detection(caplog, tmp_path):
    """Jorge's crew works the property for an hour and trips person detection
    continuously; the truck being here is what explains the people."""
    ha = _ha()
    with caplog.at_level(logging.INFO):
        _notify(ha, [_vehicle("BV41507")], tmp_path)
    assert ha.suppress_notify_person.called
    assert "Pausing person detector" in caplog.text


def test_an_ordinary_vehicle_does_not(plates_db, tmp_path):
    """Suppression has to be opt-in: it silences people who are not the crew."""
    ha = _ha()
    _notify(ha, [_vehicle("AUDUSD")], tmp_path)
    assert not ha.suppress_notify_person.called


def test_an_unknown_plate_does_not(tmp_path):
    ha = _ha()
    _notify(ha, [_vehicle("ZZ99999")], tmp_path)
    assert not ha.suppress_notify_person.called


def test_no_vehicle_at_all_does_not(tmp_path):
    ha = _ha()
    _notify(ha, [{"tagName": "person", "probability": 0.9, "camName": "garage_left",
                  "boundingBox": {"left": 0.1, "top": 0.1, "width": 0.1, "height": 0.2},
                  "center": {"x": 0.15, "y": 0.2}}], tmp_path)
    assert not ha.suppress_notify_person.called


def test_the_house_cleaner_rule_still_fires(plates_db, tmp_path):
    """The flag was added beside this one; it must not have displaced it."""
    ha = _ha()
    _notify(ha, [_vehicle("2AVJU3")], tmp_path)
    assert ha.house_cleaners_arrived.called
    assert not ha.suppress_notify_person.called


class TestReviewLinks:
    """The three verdicts a Pushover alert offers, and what they carry.

    The tap does not reach aicam. It goes to a Home Assistant webhook, which
    hands four fixed query fields to a rest_command whose payload lives in YAML
    on that box, so anything new has to travel inside a field that already
    exists. These tests pin the two halves of that: what goes in the link, and
    what goes on disk because it could not.
    """

    WEBHOOK = "https://example.ui.nabu.casa/api/webhook/aicam_roboflow_review"

    def test_the_verdict_travels_on_the_file_name(self):
        url = notify.verdict_url(self.WEBHOOK, "abc123.jpg", "correct")
        assert url == self.WEBHOOK + "?file=abc123.jpg%7Ccorrect"

    def test_a_flag_link_is_the_bare_name_the_old_path_expects(self):
        assert notify.verdict_url(self.WEBHOOK, "abc123.jpg", "flag").endswith("file=abc123.jpg")

    def test_a_verdict_link_carries_no_ampersand(self):
        """It is sent inside HTML; "&" vs "&amp;" in an href is not worth risking."""
        assert "&" not in notify.verdict_url(self.WEBHOOK, "abc123.jpg", "false")

    def test_both_verdicts_are_offered(self):
        markup = notify.verdict_html(self.WEBHOOK, "abc123.jpg")
        assert markup.count("<a href=") == 2
        assert "%7Ccorrect" in markup and "%7Cfalse" in markup

    def test_the_sidecar_records_the_boxes_the_alert_was_about(self, tmp_path):
        predictions = [
            {"tagName": "deer", "probability": 0.72,
             "boundingBox": {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}},
            {"tagName": "rabbit", "probability": 0.6, "ignore": "pale stone",
             "boundingBox": {"left": 0.5, "top": 0.5, "width": 0.1, "height": 0.1}},
        ]
        path = notify.write_sidecar(str(tmp_path), "abc123.jpg", "peach tree", "ipcams",
                                    {"deer"}, predictions, (2688, 1520))
        doc = json.load(open(path))
        assert path.endswith("abc123.json")
        assert doc["cam"] == "peach tree" and doc["tags"] == ["deer"]
        assert (doc["width"], doc["height"]) == (2688, 1520)
        # The suppressed one is left out on purpose: an excluded box is scenery,
        # and annotating it would teach the model to keep seeing it.
        assert [b["label"] for b in doc["boxes"]] == ["deer"]
        assert doc["boxes"][0]["left"] == 0.1


def _review_post(predictions, message, tmp_path):
    """Run notify() far enough to see the Pushover payload it would send."""
    from PIL import Image
    ha = _ha()
    ha.vacation_mode.return_value = False
    c = _config(tmp_path)
    c["pushover"] = {"token": "t", "user": "u"}
    c["roboflow"] = {"webhook-url": "https://example.ui.nabu.casa/api/webhook/aicam"}
    cam = mock.Mock()
    cam.name = "garage"
    cam.road_line = None
    with mock.patch.object(notify.requests, "post") as post:
        notify.notify(cam, message, Image.new("RGB", (1920, 1080), (40, 40, 40)),
                      predictions, c, ha)
    return post.call_args[1]["data"] if post.called else None


def _cat(**extra):
    p = {"tagName": "cat", "probability": 0.79, "camName": "garage",
         "boundingBox": {"left": 0.60, "top": 0.87, "width": 0.15, "height": 0.10},
         "center": {"x": 0.67, "y": 0.92}}
    p.update(extra)
    return p


class TestNothingToJudge:
    """A departure alert has no object in it, so it gets no verdicts.

    The frame attached to "cat departed from garage" is the one where the cat
    has gone; the box is where it last was. Offering "all correct" there means
    offering to annotate an empty garage floor as a cat.
    """

    def test_a_departure_offers_review_but_not_a_verdict(self, tmp_path):
        data = _review_post([_cat(departed=True)],
                            "cat departed from garage after being seen 25 times",
                            tmp_path)
        assert "url" in data and data["url_title"] == "Flag for Review"
        assert "html" not in data
        assert "<a href=" not in data["message"]

    def test_a_live_detection_still_offers_both(self, tmp_path):
        data = _review_post([_cat()], "cat in garage", tmp_path)
        assert data["html"] == 1
        assert data["message"].count("<a href=") == 2

    def test_a_departed_box_never_reaches_the_sidecar(self):
        assert notify.sidecar_boxes([_cat(departed=True)]) == []
        assert notify.sidecar_boxes([_cat(ignore="mulch bed rock")]) == []
        assert [b["label"] for b in notify.sidecar_boxes([_cat()])] == ["cat"]

    def test_a_wholly_suppressed_frame_never_gets_as_far_as_an_alert(self, tmp_path):
        """An excluded prediction is priority -4, under the bar to send at all.

        So the "no judgeable box" branch is reached by departures in practice.
        autolabel files these frames as negatives without anyone tapping.
        """
        assert _review_post([_cat(ignore="mulch bed rock")], "cat in garage",
                            tmp_path) is None


class TestReviewPage:
    """One link to a page, instead of two links in the message.

    Three vehicle alerts arrived within a second of each other on 2026-09-21
    and each spent two lines of its lock-screen preview on "All correct / All
    false" -- markup where the picture should be. With review-page-url set the
    verdicts move to a page and the notification carries one supplementary
    link, which Pushover shows as a row rather than body text.
    """

    PAGE = "https://example.ui.nabu.casa/local/review.html"

    def _with_page(self, tmp_path, predictions=None):
        from PIL import Image
        ha = _ha()
        ha.vacation_mode.return_value = False
        c = _config(tmp_path)
        c["pushover"] = {"token": "t", "user": "u"}
        c["roboflow"] = {"webhook-url": "https://example.ui.nabu.casa/api/webhook/aicam",
                         "review-page-url": self.PAGE}
        cam = mock.Mock()
        cam.name = "garage right"
        cam.road_line = None
        with mock.patch.object(notify.requests, "post") as post:
            notify.notify(cam, "vehicle 97% in front of right garage",
                          Image.new("RGB", (1920, 1080), (40, 40, 40)),
                          predictions or [_cat()], c, ha)
        return post.call_args[1]["data"]

    def test_the_message_carries_no_markup(self, tmp_path):
        data = self._with_page(tmp_path)
        assert "<a href=" not in data["message"]
        assert "html" not in data

    def test_the_one_link_goes_to_the_page(self, tmp_path):
        data = self._with_page(tmp_path)
        assert data["url"].startswith(self.PAGE + "?")
        assert "file=" in data["url"] and "cam=garage_right" in data["url"]
        assert data["url_title"] == "Review Detection"

    def test_the_page_link_fits_pushovers_limit(self, tmp_path):
        assert len(self._with_page(tmp_path)["url"]) <= 512
