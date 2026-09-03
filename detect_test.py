"""Tests for the new-object threshold selection, including the dark override."""

import configparser

import pytest

pytest.importorskip("cv2")

from detect import threshold_for

DEFAULT = 0.7


def _section(pairs):
    config = configparser.ConfigParser()
    config["s"] = pairs
    return config["s"]


def test_falls_back_to_the_detector_default():
    assert threshold_for("vehicle", _section({}), None, DEFAULT) == DEFAULT


def test_uses_the_configured_class_threshold():
    assert threshold_for("dog", _section({"dog": "0.95"}), None, DEFAULT) == 0.95


def test_dark_override_wins():
    """IR glare read as a vehicle up to 0.88 overnight; daytime stays at 0.70."""
    day = _section({"person": "0.80"})
    dark = _section({"vehicle": "0.90"})
    assert threshold_for("vehicle", day, dark, DEFAULT) == 0.90


def test_dark_override_only_applies_to_listed_classes():
    day = _section({"dog": "0.95"})
    dark = _section({"vehicle": "0.90"})
    # dog is not overridden at night, so it keeps its daytime threshold
    assert threshold_for("dog", day, dark, DEFAULT) == 0.95
    # and an unlisted class still falls through to the default
    assert threshold_for("cat", day, dark, DEFAULT) == DEFAULT


def test_no_dark_section_means_daytime_behaviour():
    day = _section({"vehicle": "0.70"})
    assert threshold_for("vehicle", day, None, DEFAULT) == 0.70


def test_dark_override_can_lower_as_well_as_raise():
    day = _section({"person": "0.80"})
    dark = _section({"person": "0.60"})
    assert threshold_for("person", day, dark, DEFAULT) == 0.60


def test_observed_overnight_false_positives_are_all_suppressed():
    """Every overnight vehicle detection on 2026-09-03 was below 0.90."""
    observed_max = 0.88
    dark = _section({"vehicle": "0.90"})
    assert observed_max < threshold_for("vehicle", _section({}), dark, DEFAULT)


# --- durable low-confidence hold -------------------------------------------

from datetime import datetime, timedelta  # noqa: E402

from detect import OBJECT_HOLD_SECONDS, recently_seen  # noqa: E402

NOW = datetime(2026, 9, 3, 13, 0, 0)


def test_unseen_class_is_not_held():
    assert not recently_seen({}, "vehicle", NOW)


def test_just_seen_class_is_held():
    assert recently_seen({"vehicle": NOW}, "vehicle", NOW)


def test_hold_survives_a_run_of_missed_frames():
    """The old cam.objects hold only bridged one frame; a stationary object
    dipping for several consecutive frames broke it."""
    seen = {"vehicle": NOW - timedelta(seconds=30)}
    assert recently_seen(seen, "vehicle", NOW)


def test_hold_expires():
    seen = {"vehicle": NOW - timedelta(seconds=OBJECT_HOLD_SECONDS + 1)}
    assert not recently_seen(seen, "vehicle", NOW)


def test_hold_is_per_class():
    seen = {"vehicle": NOW}
    assert not recently_seen(seen, "person", NOW)


def test_hold_window_is_configurable():
    seen = {"vehicle": NOW - timedelta(seconds=10)}
    assert not recently_seen(seen, "vehicle", NOW, hold_seconds=5)
