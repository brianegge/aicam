"""Tests for reading plates out of Frigate's native LPR."""

from unittest import mock

import frigate_lpr


class _Cam:
    def __init__(self, uri, name="peach tree"):
        self.blueiris_uri = uri
        self.name = name


def _cfg(url="http://frigate.home:5000"):
    return {"frigate": {"url": url}}


def test_camera_name_comes_from_the_snapshot_url():
    """One source of truth: the URL already names the stream."""
    cam = _Cam("http://frigate.home:1984/api/frame.jpeg?src=peach_tree&w=1088")
    assert frigate_lpr.frigate_camera_name(cam) == "peach_tree"


def test_camera_name_absent_when_not_a_frigate_source():
    assert frigate_lpr.frigate_camera_name(_Cam(None)) is None
    assert frigate_lpr.frigate_camera_name(_Cam("http://bi.home:81/image/deck")) is None


def _event(plate, score, sub_label=None):
    return {
        "sub_label": sub_label,
        "data": {
            "recognized_license_plate": plate,
            "recognized_license_plate_score": score,
        },
    }


def test_plate_and_owner_attach_to_the_vehicle():
    cam = _Cam("http://frigate.home:1984/api/frame.jpeg?src=driveway")
    vehicles = [{}]
    with mock.patch.object(
        frigate_lpr, "fetch_recent_plates", return_value=[("AT34047", "Brian", 0.95)]
    ):
        new = frigate_lpr.read_plates(cam, vehicles, _cfg())
    assert new == ["AT34047"]
    assert vehicles[0]["plate"] == "AT34047"
    assert vehicles[0]["plate_owner"] == "Brian"
    assert vehicles[0]["plate_read"] is True


def test_same_plate_is_not_reported_twice():
    """notify() fires on a first read; re-reporting would re-notify."""
    cam = _Cam("http://frigate.home:1984/api/frame.jpeg?src=driveway")
    vehicles = [{"plate": "AT34047"}]
    with mock.patch.object(
        frigate_lpr, "fetch_recent_plates", return_value=[("AT34047", "Brian", 0.95)]
    ):
        assert frigate_lpr.read_plates(cam, vehicles, _cfg()) == []


def test_attempt_counted_when_nothing_recognised():
    """alpr_count drives the retry cadence; without it a vehicle with no plate
    would be re-queried on every frame forever."""
    cam = _Cam("http://frigate.home:1984/api/frame.jpeg?src=driveway")
    vehicles = [{}]
    with mock.patch.object(frigate_lpr, "fetch_recent_plates", return_value=[]):
        assert frigate_lpr.read_plates(cam, vehicles, _cfg()) == []
    assert vehicles[0]["alpr_count"] == 1


def test_no_frigate_section_is_a_no_op():
    cam = _Cam("http://frigate.home:1984/api/frame.jpeg?src=driveway")
    assert frigate_lpr.read_plates(cam, [{}], {}) == []


def test_low_confidence_reads_are_discarded():
    session = mock.Mock()
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = [
        _event("BADREAD", 0.4),
        _event("BW41507", 0.97, "Brian"),
    ]
    with mock.patch.object(frigate_lpr.requests, "get", return_value=resp):
        out = frigate_lpr.fetch_recent_plates("http://frigate.home:5000", "driveway", 0)
    assert out == [("BW41507", "Brian", 0.97)]


def test_fetch_failure_is_not_fatal():
    """A Frigate outage must not take detection down with it."""
    with mock.patch.object(
        frigate_lpr.requests, "get", side_effect=OSError("connection refused")
    ):
        assert frigate_lpr.fetch_recent_plates("http://frigate.home:5000", "x", 0) == []
