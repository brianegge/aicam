"""Tests for when ALPR is attempted on a tracked vehicle."""

import pytest

pytest.importorskip("cv2")

from notify import ALPR_MAX_AGE, ALPR_RETRY_EVERY, wants_alpr


def _vehicle(age, **extra):
    p = {"tagName": "vehicle", "age": age}
    p.update(extra)
    return p


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
