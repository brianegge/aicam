"""Tests for the notification decisions in notify.py.

First tests for this module. They cover the plate-driven side effects, which
are the ones that reach outside aicam: opening the garage for the house
cleaner, and pausing person detection for a crew. Both act on a matched plate
record, so a wrong match has consequences beyond a mis-worded alert.
"""

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
