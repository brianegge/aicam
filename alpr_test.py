"""Tests for when ALPR is attempted on a tracked vehicle, and on what crop."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PIL")

from PIL import Image

from alpr import (
    ALPR_MAX_AGE,
    ALPR_RETRY_EVERY,
    MIN_CROP_PX,
    vehicle_crop_box,
    wants_alpr,
)


def _vehicle(age=0, box=None, **extra):
    p = {"tagName": "vehicle", "age": age}
    if box:
        p["boundingBox"] = box
    p.update(extra)
    return p


def _box(left, top, width, height):
    return {"left": left, "top": top, "width": width, "height": height}


def test_runs_on_the_first_frames():
    for age in (0, 1, 2):
        assert wants_alpr(_vehicle(age))


def test_keeps_retrying_while_the_vehicle_sits_there():
    """The truck that prompted this parked readable at age 99-105."""
    assert wants_alpr(_vehicle(ALPR_RETRY_EVERY))
    assert wants_alpr(_vehicle(ALPR_RETRY_EVERY * 10))
    assert not wants_alpr(_vehicle(ALPR_RETRY_EVERY + 1))


def test_gives_up_eventually():
    assert not wants_alpr(_vehicle(ALPR_MAX_AGE + ALPR_RETRY_EVERY))


def test_bounded_number_of_attempts():
    attempts = sum(1 for age in range(ALPR_MAX_AGE * 2) if wants_alpr(_vehicle(age)))
    assert attempts <= 16, attempts


def test_stops_once_the_plate_is_read():
    assert not wants_alpr(_vehicle(ALPR_RETRY_EVERY, plate_read=True))
    assert not wants_alpr(_vehicle(0, plate_read=True))


def test_ignores_non_vehicles_and_departed():
    p = _vehicle(0)
    p["tagName"] = "person"
    assert not wants_alpr(p)
    assert not wants_alpr(_vehicle(0, ignore=True))
    assert not wants_alpr(_vehicle(0, departed=True))


def test_crop_is_per_vehicle_not_the_union():
    """Two vehicles in frame must not be cropped as one box spanning both.

    On the shed camera a mower at the left and a truck at the right produced a
    single 2301x504 crop, mostly empty pavement, and ALPR found nothing.
    """
    mower = _vehicle(box=_box(0.05, 0.10, 0.10, 0.20))
    truck = _vehicle(box=_box(0.60, 0.10, 0.25, 0.30))
    width, height = 3840, 2160

    mower_box = vehicle_crop_box(mower, width, height)
    truck_box = vehicle_crop_box(truck, width, height)

    union_width = max(mower_box[2], truck_box[2]) - min(mower_box[0], truck_box[0])
    assert truck_box[2] - truck_box[0] < union_width / 2
    # the truck crop must not reach back to the mower
    assert truck_box[0] > mower_box[2]


def test_crop_is_clamped_to_the_frame():
    edge = _vehicle(box=_box(0.0, 0.0, 0.2, 0.2))
    left, top, right, bottom = vehicle_crop_box(edge, 1000, 1000)
    assert left == 0 and top == 0
    assert right <= 1000 and bottom <= 1000


def test_tiny_crops_are_not_sent(monkeypatch, tmp_path):
    """A 206x109 crop cannot hold a plate; it is not worth the request."""
    import alpr

    called = []
    monkeypatch.setattr(
        alpr.codeproject,
        "enrich",
        lambda *a, **k: called.append(a) or {"plates": [], "count": 0},
    )

    tiny = _vehicle(box=_box(0.5, 0.5, 0.01, 0.01))
    left, _, right, _ = vehicle_crop_box(tiny, 640, 480)
    assert (right - left) < MIN_CROP_PX  # precondition for this test

    image = Image.new("RGB", (640, 480))
    cam = SimpleNamespace(name="shed")
    config = {"detector": {"save-path": str(tmp_path)}}

    assert alpr.read_plates(cam, image, [tiny], config) == []
    assert called == []
