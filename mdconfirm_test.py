import configparser
from datetime import datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("cv2")

import confirm  # noqa: E402
import mdconfirm  # noqa: E402
from confirm import MovementGate  # noqa: E402
from detect import track_predictions  # noqa: E402

T0 = datetime(2026, 11, 15, 1, 49, 0)
FRAME = np.zeros((90, 160, 3), dtype=np.uint8)


def _box(left, top, w=0.1, h=0.1):
    return {"left": left, "top": top, "width": w, "height": h}


def _pred(tag, box, cam="peach tree", prob=0.8):
    return {"tagName": tag, "probability": prob, "boundingBox": box, "camName": cam}


class _Resp(object):
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakeMD(object):
    """md_server.py, in memory."""

    def __init__(self, detections=(), fail=False):
        self.detections = list(detections)
        self.fail = fail
        self.calls = 0

    def post(self, url, data=None, timeout=None, headers=None):
        self.calls += 1
        if self.fail:
            raise mdconfirm.requests.ConnectionError("connection refused")
        return _Resp({"detections": self.detections})


def _cfg(**opts):
    c = configparser.ConfigParser()
    c["megadetector"] = dict({"enabled": "true"}, **opts)
    return c


def _voucher(md, cam="peach tree", dark=True, **opts):
    return mdconfirm.Voucher(_cfg(**opts)["megadetector"], cam, FRAME, dark, session=md)


def _frame(gate, prev, preds, vouch, at=0):
    new = []
    pairs, _ = track_predictions(preds, prev, new)
    return confirm.apply(gate, preds, new, pairs, T0 + timedelta(seconds=at), vouch=vouch)


def _animal(conf, bbox):
    return {"category": "animal", "conf": conf, "bbox": bbox}


def test_a_one_frame_coyote_vouched_for_alerts_at_once():
    """01:49 on 2026-11-15: a coyote crossed peach tree in one frame, scored
    0.93 by MegaDetector, and never alerted -- there was no second frame."""
    gate = MovementGate({"coyote"})
    md = FakeMD([_animal(0.93, [0.41, 0.42, 0.09, 0.08])])
    p = _pred("coyote", _box(0.4, 0.4))
    assert _frame(gate, {}, [p], _voucher(md)) == [p]
    assert not p.get("unconfirmed")
    assert gate.is_active("coyote", T0, "peach tree")
    assert not gate.is_active("coyote", T0, "tree line")


def test_a_weak_megadetector_answer_leaves_it_to_the_movement_gate():
    gate = MovementGate({"deer"})
    p = _pred("deer", _box(0.4, 0.4))
    assert _frame(gate, {}, [p], _voucher(FakeMD([_animal(0.45, [0.4, 0.4, 0.1, 0.1])]))) == []
    assert p.get("unconfirmed")


def test_an_animal_somewhere_else_in_the_frame_does_not_vouch_for_this_box():
    gate = MovementGate({"deer"})
    p = _pred("deer", _box(0.1, 0.1))
    assert _frame(gate, {}, [p], _voucher(FakeMD([_animal(0.95, [0.7, 0.7, 0.1, 0.1])]))) == []
    assert p.get("unconfirmed")


def test_daylight_is_not_asked_by_default():
    """All four Rufus-as-deer cases MegaDetector would have let through were
    in daylight; the 29 real animals were all at night."""
    gate = MovementGate({"deer"})
    md = FakeMD([_animal(0.9, [0.4, 0.4, 0.1, 0.1])])
    p = _pred("deer", _box(0.4, 0.4))
    assert _frame(gate, {}, [p], _voucher(md, dark=False)) == []
    assert md.calls == 0
    assert _frame(MovementGate({"deer"}), {}, [_pred("deer", _box(0.4, 0.4))],
                  _voucher(md, dark=False, **{"dark-only": "false"})) != []


def test_the_garage_is_never_asked():
    """MegaDetector scored the cat seen from above 0.00-0.07 every time."""
    gate = MovementGate({"raccoon"})
    md = FakeMD([_animal(0.9, [0.4, 0.4, 0.1, 0.1])])
    p = _pred("raccoon", _box(0.4, 0.4), cam="garage")
    assert _frame(gate, {}, [p], _voucher(md, cam="garage")) == []
    assert md.calls == 0


def test_pets_are_not_asked():
    gate = MovementGate({"dog", "cat"})
    md = FakeMD([_animal(0.9, [0.4, 0.4, 0.1, 0.1])])
    _frame(gate, {}, [_pred("dog", _box(0.4, 0.4)), _pred("cat", _box(0.6, 0.6))], _voucher(md))
    assert md.calls == 0


def test_a_dead_service_changes_nothing():
    gate = MovementGate({"fox"})
    md = FakeMD(fail=True)
    p = _pred("fox", _box(0.4, 0.4))
    assert _frame(gate, {}, [p], _voucher(md)) == []
    assert p.get("unconfirmed")
    assert md.calls == 1


def test_one_request_per_frame():
    gate = MovementGate({"fox", "deer"})
    md = FakeMD([_animal(0.9, [0.1, 0.1, 0.1, 0.1]), _animal(0.9, [0.6, 0.6, 0.1, 0.1])])
    a, b = _pred("fox", _box(0.1, 0.1)), _pred("deer", _box(0.6, 0.6))
    assert _frame(gate, {}, [a, b], _voucher(md)) == [a, b]
    assert md.calls == 1


def test_no_section_means_no_voucher():
    assert mdconfirm.voucher_for(configparser.ConfigParser(), "peach tree", FRAME, True) is None
    off = _cfg(enabled="false")
    assert mdconfirm.voucher_for(off, "peach tree", FRAME, True) is None


def test_a_box_drawn_differently_by_the_two_detectors_still_matches():
    # Centre of MegaDetector's tighter box inside aicam's looser one.
    assert mdconfirm.overlaps(_box(0.30, 0.30, 0.20, 0.20), [0.36, 0.36, 0.05, 0.05])
    assert not mdconfirm.overlaps(_box(0.30, 0.30, 0.10, 0.10), [0.60, 0.60, 0.05, 0.05])
