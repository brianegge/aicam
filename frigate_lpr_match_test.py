"""Spatial matching of Frigate plates to aicam vehicles.

Regression cover for 2026-09-08: two cars parked in the driveway, and the most
recent plate on that camera (the Subaru's BT70150) was attached to the Jeep.
The same lookup opens the garage for the house cleaner, so a wrong name is not
a cosmetic failure.
"""

import pytest

pytest.importorskip("cv2")

from frigate_lpr import IOU_MARGIN, MIN_PLATE_IOU, match_plate


def veh(left, top, w=0.2, h=0.2):
    return {"boundingBox": {"left": left, "top": top, "width": w, "height": h}}


def cand(plate, left=None, top=None, w=0.2, h=0.2, owner=None, score=0.95):
    region = None if left is None else {"left": left, "top": top, "width": w, "height": h}
    return (plate, owner, score, region)


def test_single_candidate_with_no_box_is_still_used():
    """Older events carry no box; one candidate is unambiguous anyway."""
    assert match_plate(veh(0.1, 0.1), [cand("BT70150")])[0] == "BT70150"


def test_two_candidates_without_boxes_are_refused():
    got = match_plate(veh(0.1, 0.1), [cand("BT70150"), cand("CT419875")])
    assert got is None


def test_overlapping_plate_wins():
    jeep = veh(0.10, 0.10)
    got = match_plate(jeep, [cand("BT70150", 0.70, 0.70), cand("CT419875", 0.11, 0.11)])
    assert got[0] == "CT419875"


def test_the_september_8_mix_up():
    """The Subaru's plate must not land on the Jeep."""
    jeep = veh(0.05, 0.10, 0.30, 0.35)
    subaru_plate = cand("BT70150", 0.60, 0.55, 0.30, 0.35)
    assert match_plate(jeep, [subaru_plate]) is None


def test_ambiguous_overlap_is_refused():
    """Two cars side by side produce similar overlap -- decline, do not guess."""
    v = veh(0.40, 0.40, 0.20, 0.20)
    a = cand("AAA1111", 0.38, 0.38, 0.20, 0.20)
    b = cand("BBB2222", 0.42, 0.42, 0.20, 0.20)
    assert match_plate(v, [a, b]) is None


def test_no_overlap_at_all_is_refused():
    assert match_plate(veh(0.0, 0.0, 0.1, 0.1),
                       [cand("BT70150", 0.8, 0.8, 0.1, 0.1)]) is None


def test_owner_is_carried_through():
    got = match_plate(veh(0.1, 0.1), [cand("HC00001", 0.1, 0.1, owner="House cleaner")])
    assert got[1] == "House cleaner"


def test_thresholds_are_sane():
    assert 0 < MIN_PLATE_IOU < 1 and 0 < IOU_MARGIN < 1
