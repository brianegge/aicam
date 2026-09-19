"""Tests for the new-object threshold selection, including the dark override,
and for how long a track survives once the detector stops seeing it."""

import configparser
from datetime import datetime

import pytest

pytest.importorskip("cv2")

import detect
from detect import label_with_confidence, threshold_for

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


# --- confidence in the alert text ---------------------------------------

def _p(tag, prob):
    return {"tagName": tag, "probability": prob}


def test_the_alert_carries_the_score_that_triggered_it():
    """The swing false positive acquired at 0.510 against a 0.45 threshold."""
    assert label_with_confidence(
        {"deer"}, [_p("deer", 0.510)]) == "deer 51%"


def test_the_highest_score_per_class_is_the_one_reported():
    """Several boxes of one class; the alert fired on the best of them."""
    out = label_with_confidence(
        {"deer"}, [_p("deer", 0.31), _p("deer", 0.76), _p("deer", 0.52)])
    assert out == "deer 76%"


def test_several_classes_are_listed_in_a_stable_order():
    out = label_with_confidence(
        {"deer", "dog"}, [_p("dog", 0.93), _p("deer", 0.51)])
    assert out == "deer 51%,dog 93%"


def test_predictions_for_other_classes_are_ignored():
    out = label_with_confidence(
        {"deer"}, [_p("deer", 0.51), _p("vehicle", 0.99)])
    assert out == "deer 51%"


def test_a_class_with_no_matching_prediction_still_appears():
    """Never drop an object from the message just because the score is missing."""
    assert label_with_confidence({"deer"}, []) == "deer"
    assert label_with_confidence(
        {"deer"}, [{"tagName": "deer"}]) == "deer 0%"


# --- both opinions in the alert ------------------------------------------

def _pv(tag, prob, label, confidence):
    """A prediction the gate has looked at."""
    return {"tagName": tag, "probability": prob,
            "verified": {"label": label, "confidence": confidence}}


def test_a_disagreement_shows_both_verdicts():
    """play, 2026-09-14: detector dog 0.79, model "deer 0.93, young deer".

    The detector was right. An hour earlier on garage-l it was the other way
    round. Printing one and dropping the other loses what distinguishes them.
    """
    assert label_with_confidence(
        {"dog"}, [_pv("dog", 0.79, "deer", 0.93)]) == "dog 79% (vs deer 93%)"


def test_agreement_is_shown_as_confirmation():
    assert label_with_confidence(
        {"dog"}, [_pv("dog", 0.92, "dog", 0.99)]) == "dog 92% (confirmed 99%)"


def test_an_unverified_class_shows_the_score_alone():
    """Not in the configured classes, or the gate could not be reached."""
    assert label_with_confidence({"deer"}, [_p("deer", 0.51)]) == "deer 51%"


def test_both_are_carried_for_each_class_independently():
    out = label_with_confidence(
        {"dog", "person"},
        [_pv("dog", 0.79, "deer", 0.93), _pv("person", 0.87, "person", 0.95)])
    assert out == "dog 79% (vs deer 93%),person 87% (confirmed 95%)"


def test_a_verdict_without_a_confidence_is_not_printed():
    """Never render "(vs deer None%)"."""
    p = {"tagName": "dog", "probability": 0.79, "verified": {"label": "deer"}}
    assert label_with_confidence({"dog"}, [p]) == "dog 79%"


# --- how long a track survives with no detection -------------------------

def _tracked(age, static=True, last_time=None):
    return {"age": age, "static": static,
            "last_time": last_time or datetime.now()}


def test_expiry_grows_with_age():
    """Something long established is not dropped over a brief gap."""
    assert detect.expiry_minutes(_tracked(0, static=False), 10) == pytest.approx(1.0)
    assert detect.expiry_minutes(_tracked(12, static=False), 10) == pytest.approx(3.0)


def test_expiry_is_capped():
    assert detect.expiry_minutes(_tracked(100000, static=False), 10) == 60


def test_a_track_that_never_moved_survives_longer():
    """The 2026-09-19 lululemon package: age 8, interval 10s, announced
    departed after 2.3 minutes while still plainly sitting there."""
    moving = detect.expiry_minutes(_tracked(8, static=False), 10)
    parked = detect.expiry_minutes(_tracked(8, static=True), 10)
    assert moving == pytest.approx(2.333, abs=0.01)
    assert parked == pytest.approx(9.333, abs=0.01)


def test_a_single_sighting_earns_no_bonus():
    """One sighting has not shown the object is stationary, only that it has
    not yet shown otherwise."""
    assert detect.expiry_minutes(_tracked(1, static=True), 10) == \
        detect.expiry_minutes(_tracked(1, static=False), 10)


def test_the_static_bonus_is_still_capped():
    assert detect.expiry_minutes(_tracked(100000, static=True), 10) == 60


def test_a_track_with_no_static_key_behaves_as_moving():
    """Tracks carried across a restart or built by older code."""
    assert detect.expiry_minutes({"age": 8, "last_time": datetime.now()}, 10) \
        == pytest.approx(2.333, abs=0.01)


# --- deciding that a track has moved -------------------------------------

def _box(left, top, w=0.1, h=0.1):
    return {"left": left, "top": top, "width": w, "height": h}


def test_a_new_track_starts_static_with_an_anchor():
    prev = {}
    new = []
    p = {"tagName": "package", "boundingBox": _box(0.5, 0.5), "probability": 0.9}
    detect.track_predictions([p], prev, new)
    assert p["static"] is True
    assert p["anchor_box"] == _box(0.5, 0.5)


def test_a_stationary_object_stays_static():
    prev = {}
    detect.track_predictions(
        [{"tagName": "package", "boundingBox": _box(0.5, 0.5), "probability": 0.9}],
        prev, [])
    p2 = {"tagName": "package", "boundingBox": _box(0.502, 0.501), "probability": 0.3}
    detect.track_predictions([p2], prev, [])
    assert prev["package"][0]["static"] is True


def test_an_object_that_moves_stops_being_static():
    prev = {}
    detect.track_predictions(
        [{"tagName": "person", "boundingBox": _box(0.5, 0.5, 0.2, 0.2),
          "probability": 0.9}], prev, [])
    moved = {"tagName": "person", "boundingBox": _box(0.60, 0.60, 0.2, 0.2),
             "probability": 0.9}
    detect.track_predictions([moved], prev, [])
    assert prev["person"][0]["static"] is False


def test_a_slow_drift_cannot_creep_past_the_check():
    """Compared against where the track began, not the previous frame --
    otherwise an object crosses the scene while every step looks stationary."""
    prev = {}
    detect.track_predictions(
        [{"tagName": "person", "boundingBox": _box(0.30, 0.5, 0.2, 0.2),
          "probability": 0.9}], prev, [])
    for left in (0.33, 0.36, 0.39, 0.42, 0.45):
        detect.track_predictions(
            [{"tagName": "person", "boundingBox": _box(left, 0.5, 0.2, 0.2),
              "probability": 0.9}], prev, [])
    assert prev["person"][0]["static"] is False


def test_once_moved_it_does_not_become_static_again():
    """A car that parks has still arrived; it must not earn the furniture
    bonus by sitting still afterwards."""
    prev = {}
    detect.track_predictions(
        [{"tagName": "vehicle", "boundingBox": _box(0.10, 0.5, 0.2, 0.2),
          "probability": 0.9}], prev, [])
    detect.track_predictions(
        [{"tagName": "vehicle", "boundingBox": _box(0.40, 0.5, 0.2, 0.2),
          "probability": 0.9}], prev, [])
    for _ in range(5):
        detect.track_predictions(
            [{"tagName": "vehicle", "boundingBox": _box(0.40, 0.5, 0.2, 0.2),
              "probability": 0.9}], prev, [])
    assert prev["vehicle"][0]["static"] is False
