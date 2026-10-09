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


# --- second_frame: borderline people ------------------------------------------

def _people_gate():
    return MovementGate({"dog"}, second_frame={"person": 0.75})


def test_a_confident_person_alerts_at_once():
    gate, prev = _people_gate(), {}
    p = _pred("person", _box(0.4, 0.4), cam="deck", prob=0.91)
    assert _frame(gate, prev, [p], _at(0)) == [p]


def test_the_dog_called_a_person_for_one_frame_never_alerts():
    """The deck at night: person 0.66 on one frame, then dog on every frame
    after. Nothing of class person ever matches the held track again."""
    gate, prev = _people_gate(), {}
    p0 = _pred("person", _box(0.45, 0.45, 0.1, 0.2), cam="deck", prob=0.66)
    assert _frame(gate, prev, [p0], _at(0)) == []
    assert p0.get("unconfirmed")
    for s, prob in ((3, 0.72), (6, 0.75), (9, 0.91)):
        dog = _pred("dog", _box(0.45, 0.47, 0.1, 0.18), cam="deck", prob=prob)
        assert all(a["tagName"] != "person" for a in _frame(gate, prev, [dog], _at(s)))
    assert prev["person"][0].get("unconfirmed")


def test_a_borderline_person_seen_again_alerts_on_the_second_frame():
    gate, prev = _people_gate(), {}
    p0 = _pred("person", _box(0.4, 0.4, 0.1, 0.3), cam="deck", prob=0.62)
    assert _frame(gate, prev, [p0], _at(0)) == []
    p1 = _pred("person", _box(0.41, 0.4, 0.1, 0.3), cam="deck", prob=0.58)
    assert _frame(gate, prev, [p1], _at(3)) == [p1]
    assert "iou" not in p1  # notify() would otherwise treat it as old news
    # Announced once; the track is ordinary from here.
    p2 = _pred("person", _box(0.41, 0.4, 0.1, 0.3), cam="deck", prob=0.70)
    assert _frame(gate, prev, [p2], _at(6)) == []


def test_a_hysteresis_hold_does_not_count_as_the_second_frame():
    gate, prev = _people_gate(), {}
    p0 = _pred("person", _box(0.4, 0.4, 0.1, 0.3), cam="deck", prob=0.60)
    _frame(gate, prev, [p0], _at(0))
    held = _pred("person", _box(0.4, 0.4, 0.1, 0.3), cam="deck", prob=0.30)
    held["hold_only"] = True
    assert _frame(gate, prev, [held], _at(3)) == []
    assert held.get("unconfirmed")
    p2 = _pred("person", _box(0.4, 0.4, 0.1, 0.3), cam="deck", prob=0.60)
    assert _frame(gate, prev, [p2], _at(6)) == [p2]


def test_another_cameras_long_standing_person_does_not_vouch_for_this_one():
    """Per track, not per class: front entry has carried a low-scoring
    "person" for hundreds of frames, and it must not release the deck's."""
    gate = _people_gate()
    front, deck = {}, {}
    for s in range(0, 30, 3):
        _frame(gate, front, [_pred("person", _box(0.2, 0.2), cam="front entry",
                                   prob=0.6)], _at(s))
    p0 = _pred("person", _box(0.45, 0.45), cam="deck", prob=0.66)
    assert _frame(gate, deck, [p0], _at(30)) == []


def test_classes_without_a_limit_are_untouched():
    gate, prev = _people_gate(), {}
    car = _pred("vehicle", _box(0.4, 0.4), prob=0.56)
    assert _frame(gate, prev, [car], _at(0)) == [car]


def test_gate_reads_second_frame():
    config = configparser.ConfigParser()
    config["confirm"] = {"classes": "dog", "second_frame": "person:0.75, cat: 0.7"}
    confirm._gate = None
    gate = confirm.gate_for(config)
    assert gate.second_frame == {"person": 0.75, "cat": 0.7}


# --- release_absent: the held sighting that moved on ----------------------------

def _present(prev, preds):
    """ids of tracks this frame touched, as detect.py computes them."""
    new = []
    pairs, _ = track_predictions(preds, prev, new)
    return pairs, new


def test_the_raccoon_held_at_087_is_released_when_it_confirms_elsewhere():
    """2026-10-09 at the peach tree: 0.87 held, an empty frame, then 0.77 on
    fresh ground. The 0.77 alerted; the 0.87 must now be released too."""
    gate, prev = MovementGate({"raccoon"}), {}
    first = _pred("raccoon", _box(0.30, 0.45), cam="peach tree", prob=0.87)
    _frame(gate, prev, [first], _at(0))
    first["held_at"] = _at(0)
    _frame(gate, prev, [], _at(5))
    second = _pred("raccoon", _box(0.55, 0.50), cam="peach tree", prob=0.77)
    pairs, new = _present(prev, [second])
    assert confirm.apply(gate, [second], new, pairs, _at(11)) == [second]
    present = set(id(t) for _, t in pairs) | set(id(p) for p in new)
    released = confirm.release_absent(gate, prev, present, _at(11))
    assert released == [first]
    assert "unconfirmed" not in first
    # Once only.
    assert confirm.release_absent(gate, prev, present, _at(12)) == []


def test_nothing_is_released_while_the_class_is_unconfirmed():
    gate, prev = MovementGate({"raccoon"}), {}
    first = _pred("raccoon", _box(0.30, 0.45), cam="peach tree")
    _frame(gate, prev, [first], _at(0))
    first["held_at"] = _at(0)
    assert confirm.release_absent(gate, prev, set(), _at(5)) == []
    assert first.get("unconfirmed")


def test_a_hold_older_than_the_window_is_not_news():
    gate, prev = MovementGate({"raccoon"}, window_seconds=60), {}
    old = _pred("raccoon", _box(0.1, 0.1), cam="peach tree")
    _frame(gate, prev, [old], _at(0))
    old["held_at"] = _at(0)
    _frame(gate, prev, [_pred("raccoon", _box(0.5, 0.5), cam="peach tree")], _at(50))
    _frame(gate, prev, [_pred("raccoon", _box(0.8, 0.8), cam="peach tree")], _at(90))
    assert old not in confirm.release_absent(gate, prev, set(), _at(90))


def test_a_borderline_person_is_never_released_by_another_sighting():
    gate = MovementGate({"dog"}, second_frame={"person": 0.75})
    prev = {}
    p0 = _pred("person", _box(0.45, 0.45), cam="deck", prob=0.66)
    _frame(gate, prev, [p0], _at(0))
    p0["held_at"] = _at(0)
    assert confirm.release_absent(gate, prev, set(), _at(3)) == []
    assert p0.get("unconfirmed")


def test_an_old_hold_lets_go_of_its_frame():
    gate, prev = MovementGate({"deer"}, window_seconds=60), {}
    rock = _pred("deer", _box(0.4, 0.4), cam="tree line")
    _frame(gate, prev, [rock], _at(0))
    rock.update(held_at=_at(0), held_image="big", held_original="bigger")
    confirm.release_absent(gate, prev, set(), _at(30))
    assert rock.get("held_image") == "big"
    confirm.release_absent(gate, prev, set(), _at(61))
    assert "held_image" not in rock and "held_original" not in rock
    assert rock.get("unconfirmed")
