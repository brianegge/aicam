import configparser
from datetime import datetime, timedelta

import pytest

pytest.importorskip("cv2")

import confirm  # noqa: E402
from confirm import MovementGate  # noqa: E402
from detect import track_predictions  # noqa: E402

T0 = datetime(2026, 10, 6, 8, 34, 0)


def _box(left, top, w=0.1, h=0.1):
    return {"left": left, "top": top, "width": w, "height": h}


def _at(seconds):
    return T0 + timedelta(seconds=seconds)


def test_a_first_sighting_does_not_activate():
    g = MovementGate({"deer"})
    assert not g.observe("deer", "tree line", _box(0.4, 0.4), _at(0))
    assert not g.is_active("deer", _at(0))


def test_a_stationary_object_never_activates():
    g = MovementGate({"deer"})
    for s in range(0, 600, 10):
        g.observe("deer", "tree line", _box(0.4 + (s % 20) / 1000, 0.4), _at(s))
    assert not g.is_active("deer", _at(600))


def test_moving_to_fresh_ground_activates():
    g = MovementGate({"deer"})
    g.observe("deer", "tree line", _box(0.1, 0.4), _at(0))
    g.observe("deer", "tree line", _box(0.15, 0.4), _at(10))  # overlaps
    assert not g.is_active("deer", _at(10))
    assert g.observe("deer", "tree line", _box(0.3, 0.4), _at(20))
    assert g.is_active("deer", _at(20))


def test_fresh_ground_must_clear_every_earlier_box_not_just_the_first():
    g = MovementGate({"deer"})
    g.observe("deer", "tree line", _box(0.1, 0.4), _at(0))
    g.observe("deer", "tree line", _box(0.3, 0.4), _at(0))  # same frame
    assert not g.is_active("deer", _at(0))
    # Both stones again in the next frame: each overlaps its own earlier box.
    g.observe("deer", "tree line", _box(0.1, 0.4), _at(10))
    g.observe("deer", "tree line", _box(0.3, 0.4), _at(10))
    assert not g.is_active("deer", _at(10))


def test_another_camera_activates():
    g = MovementGate({"deer"})
    g.observe("deer", "tree line", _box(0.4, 0.4), _at(0))
    assert g.observe("deer", "deck", _box(0.4, 0.4), _at(5))


def test_a_sighting_outside_the_window_does_not_count():
    g = MovementGate({"deer"}, window_seconds=60)
    g.observe("deer", "tree line", _box(0.1, 0.4), _at(0))
    assert not g.observe("deer", "deck", _box(0.4, 0.4), _at(61))


def test_stays_active_while_sightings_continue_and_lapses_after_a_quiet_window():
    g = MovementGate({"deer"}, window_seconds=60)
    g.observe("deer", "tree line", _box(0.1, 0.4), _at(0))
    g.observe("deer", "deck", _box(0.4, 0.4), _at(5))
    for s in range(50, 300, 50):
        g.observe("deer", "deck", _box(0.4, 0.4), _at(s))
        assert g.is_active("deer", _at(s))
    assert g.is_active("deer", _at(300 - 50 + 59))
    assert not g.is_active("deer", _at(300 - 50 + 60))


def test_classes_are_independent_and_ungated_ones_always_pass():
    g = MovementGate({"deer", "rabbit"})
    g.observe("deer", "tree line", _box(0.1, 0.4), _at(0))
    assert not g.observe("rabbit", "deck", _box(0.4, 0.4), _at(5))
    assert g.is_active("person", _at(5))


def _pred(tag, box, cam="tree line", prob=0.8):
    return {"tagName": tag, "probability": prob, "boundingBox": box,
            "camName": cam}


def _frame(gate, prev, preds, now):
    new = []
    pairs, _ = track_predictions(preds, prev, new)
    return confirm.apply(gate, preds, new, pairs, now)


def test_a_held_track_is_announced_once_its_class_activates():
    gate = MovementGate({"deer"})
    prev = {}
    p0 = _pred("deer", _box(0.10, 0.4))
    assert _frame(gate, prev, [p0], _at(0)) == []
    assert p0.get("unconfirmed")
    # Same deer, slightly moved: matches its track, still unconfirmed.
    p1 = _pred("deer", _box(0.11, 0.4))
    assert _frame(gate, prev, [p1], _at(10)) == []
    assert p1.get("unconfirmed")
    # A second deer on fresh ground confirms the class; both are announced.
    p2a = _pred("deer", _box(0.12, 0.4))
    p2b = _pred("deer", _box(0.6, 0.4))
    arrivals = _frame(gate, prev, [p2a, p2b], _at(20))
    assert arrivals == [p2a, p2b]
    assert "iou" not in p2a  # notify() would otherwise silence it
    assert not any(t.get("unconfirmed") for t in prev["deer"])
    # From then on the track is ordinary: no second announcement.
    p3 = _pred("deer", _box(0.12, 0.4))
    assert _frame(gate, prev, [p3], _at(30)) == []
    assert "iou" in p3


def test_a_track_held_on_one_camera_is_released_by_another():
    gate = MovementGate({"deer"})
    prev_a, prev_b = {}, {}
    a0 = _pred("deer", _box(0.4, 0.4), cam="tree line")
    _frame(gate, prev_a, [a0], _at(0))
    b0 = _pred("deer", _box(0.4, 0.4), cam="deck")
    assert _frame(gate, prev_b, [b0], _at(3)) == [b0]
    a1 = _pred("deer", _box(0.4, 0.4), cam="tree line")
    assert _frame(gate, prev_a, [a1], _at(10)) == [a1]


def test_ungated_classes_arrive_as_before():
    gate = MovementGate({"deer"})
    prev = {}
    person = _pred("person", _box(0.4, 0.4))
    assert _frame(gate, prev, [person], _at(0)) == [person]


def test_hold_only_sightings_neither_activate_nor_are_recorded():
    gate = MovementGate({"deer"})
    prev = {}
    _frame(gate, prev, [_pred("deer", _box(0.1, 0.4))], _at(0))
    held = _pred("deer", _box(0.6, 0.4), prob=0.2)
    held["hold_only"] = True
    confirm.apply(gate, [held], [], [], _at(10))
    assert not gate.is_active("deer", _at(10))
    assert len(gate.sightings["deer"]) == 1


def test_gate_is_off_without_a_confirm_section():
    config = configparser.ConfigParser()
    assert confirm.gate_for(config) is None


def test_gate_reads_classes_and_window():
    config = configparser.ConfigParser()
    config["confirm"] = {"classes": "deer, rabbit", "window": "30"}
    gate = confirm.gate_for(config)
    assert gate.classes == {"deer", "rabbit"}
    assert gate.window == timedelta(seconds=30)
    assert confirm.gate_for(config) is gate
