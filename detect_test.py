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


from detect import HOLD_PROBABILITY, apply_thresholds, track_predictions  # noqa: E402


def _pred(tag, prob, box=(0.1, 0.1, 0.2, 0.2)):
    return {
        "tagName": tag,
        "probability": prob,
        "boundingBox": {
            "left": box[0],
            "top": box[1],
            "width": box[2],
            "height": box[3],
        },
    }


def _thresholds():
    return _section({"vehicle": "0.70"})


def test_over_threshold_is_kept_and_not_marked_held():
    now = datetime.now()
    kept = apply_thresholds(
        [_pred("vehicle", 0.80)], _thresholds(), None, 0.7, {"vehicle": now}, now
    )
    assert len(kept) == 1
    assert "hold_only" not in kept[0]


def test_under_threshold_without_a_hold_is_dropped():
    now = datetime.now()
    kept = apply_thresholds([_pred("vehicle", 0.40)], _thresholds(), None, 0.7, {}, now)
    assert kept == []


def test_under_threshold_with_a_hold_is_kept_but_marked():
    now = datetime.now()
    kept = apply_thresholds(
        [_pred("vehicle", 0.40)], _thresholds(), None, 0.7, {"vehicle": now}, now
    )
    assert len(kept) == 1
    assert kept[0]["hold_only"] is True


def test_below_the_hold_floor_is_dropped_even_with_a_hold():
    now = datetime.now()
    kept = apply_thresholds(
        [_pred("vehicle", HOLD_PROBABILITY - 0.01)],
        _thresholds(),
        None,
        0.7,
        {"vehicle": now},
        now,
    )
    assert kept == []


def _tracked(box=(0.1, 0.1, 0.2, 0.2), age=5):
    p = _pred("vehicle", 0.9, box)
    p["age"] = age
    p["start_time"] = datetime.now()
    p["last_time"] = datetime.now()
    return p


def test_held_detection_sustains_a_matching_track():
    """A parked car dipping under threshold must keep its age, not re-arrive."""
    prev = {"vehicle": [_tracked(age=5)]}
    held = _pred("vehicle", 0.40)
    held["hold_only"] = True
    new_predictions = []
    tracked, unmatched = track_predictions([held], prev, new_predictions)
    assert len(tracked) == 1
    assert unmatched == []
    assert new_predictions == []  # not an arrival
    assert held["age"] == 6


def test_held_detection_matching_nothing_is_discarded():
    """The peach tree bug: two parked cars kept `vehicle` held forever, so
    every stray low-confidence box became a new object and notified."""
    prev = {"vehicle": [_tracked(box=(0.1, 0.1, 0.2, 0.2))]}
    noise = _pred("vehicle", 0.19, box=(0.8, 0.8, 0.05, 0.05))  # nowhere near
    noise["hold_only"] = True
    new_predictions = []
    tracked, unmatched = track_predictions([noise], prev, new_predictions)
    assert tracked == []
    assert unmatched == [noise]
    assert new_predictions == []  # no notification
    assert len(prev["vehicle"]) == 1  # and no new track was started


def test_over_threshold_detection_matching_nothing_is_a_real_arrival():
    """The drop must not suppress genuine arrivals."""
    prev = {"vehicle": [_tracked(box=(0.1, 0.1, 0.2, 0.2))]}
    arrival = _pred("vehicle", 0.85, box=(0.8, 0.8, 0.1, 0.1))
    new_predictions = []
    tracked, unmatched = track_predictions([arrival], prev, new_predictions)
    assert unmatched == []
    assert new_predictions == [arrival]
    assert arrival["age"] == 0
    assert len(prev["vehicle"]) == 2
