"""Tests for Camera capture paths (Blue Iris primary, direct fallback)."""

import configparser
from unittest import mock

import pytest

# The documented local test flow runs pytest inside the python:3.6.9
# container, which deliberately excludes opencv; skip rather than break
# collection for the whole suite there.
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import camera as camera_mod
from camera import Camera


@pytest.fixture(autouse=True)
def _reset_bi_breaker():
    camera_mod._bi_fail_count = 0
    camera_mod._bi_skip_until = 0.0
    yield
    camera_mod._bi_fail_count = 0
    camera_mod._bi_skip_until = 0.0


def _jpeg_bytes(seed=0):
    img = np.full((48, 64, 3), seed, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_camera(
    blueiris_url="http://blueiris-3.home:81", blueiris_only=False, **extra
):
    config = configparser.ConfigParser()
    config["cam0"] = {
        "uri": "http://test-cam.home/cgi-bin/snapshot.cgi",
        "name": "test",
        "user": "admin",
        "password": "x",
    }
    for k, v in extra.items():
        config["cam0"][k] = v
    return Camera(
        config["cam0"],
        {},
        mqtt_client=None,
        blueiris_url=blueiris_url,
        blueiris_only=blueiris_only,
    )


def _response(content, status=200):
    resp = mock.Mock()
    resp.content = content
    resp.status_code = status
    if status >= 400:
        import requests

        resp.raise_for_status.side_effect = requests.HTTPError(str(status))
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_default_blueiris_name_strips_cam_suffix():
    cam = _make_camera()
    assert cam.blueiris_uri == "http://blueiris-3.home:81/image/test?q=100&s=100"


def test_blueiris_name_override():
    cam = _make_camera(**{"blueiris-name": "front_entry"})
    assert "image/front_entry" in cam.blueiris_uri


def test_no_blueiris_url_disables_blueiris():
    cam = _make_camera(blueiris_url=None)
    assert cam.blueiris_uri is None


def test_capture_prefers_blueiris():
    cam = _make_camera()
    with mock.patch.object(camera_mod, "_bi_session") as session:
        session.get.return_value = _response(_jpeg_bytes())
        direct = mock.patch.object(
            cam, "_get_session", side_effect=AssertionError("direct fetch used")
        )
        with direct:
            result = cam.capture()
    assert result is cam
    assert cam.image is not None
    assert cam.error is None
    assert cam.source == cam.blueiris_uri
    assert cam.fails == 0


def test_capture_falls_back_to_direct_when_blueiris_fails():
    cam = _make_camera()
    direct_resp = mock.Mock()
    direct_resp.__enter__ = mock.Mock(return_value=direct_resp)
    direct_resp.__exit__ = mock.Mock(return_value=False)
    direct_resp.raise_for_status.return_value = None
    direct_resp.raw.read.return_value = _jpeg_bytes(seed=7)
    direct_session = mock.Mock()
    direct_session.get.return_value = direct_resp
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.side_effect = OSError("connection refused")
        with mock.patch.object(cam, "_get_session", return_value=direct_session):
            cam.capture()
    assert cam.image is not None
    assert cam.error is None
    assert cam.source == cam.config["uri"]


def test_blueiris_only_skips_direct_fallback():
    """With no route to the camera, the direct request can only time out."""
    cam = _make_camera(blueiris_only=True)
    direct_session = mock.Mock()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.side_effect = OSError("connection refused")
        with mock.patch.object(cam, "_get_session", return_value=direct_session):
            cam.capture()
    direct_session.get.assert_not_called()
    assert cam.image is None
    assert cam.error == "no-bi"
    # Still backs off, so a camera BI cannot serve is not retried every cycle.
    assert cam.skip == 1
    assert cam.fails == 1


def test_blueiris_only_does_not_back_off_on_repeated_frames():
    """BI limit-decodes idle cameras, so repeats are normal, not a fault.

    Backing off here would sample a quiet camera less and less often.
    """
    cam = _make_camera(blueiris_only=True)
    content = _jpeg_bytes()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(content)
        for _ in range(camera_mod.BI_FROZEN_THRESHOLD + 3):
            cam.capture()
    assert cam.error == "dup"
    assert cam.skip == 0
    assert cam.fails == 0


def test_blueiris_only_still_uses_a_good_blueiris_frame():
    cam = _make_camera(blueiris_only=True)
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(_jpeg_bytes(seed=3))
        cam.capture()
    assert cam.image is not None
    assert cam.error is None


def test_capture_blueiris_frozen_stream_falls_back_after_threshold():
    """Identical frames tolerate a static scene, then signal frozen."""
    cam = _make_camera()
    content = _jpeg_bytes()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(content)
        assert cam._capture_blueiris() is True  # fresh
        # Below the threshold a duplicate is served, not treated as frozen.
        for _ in range(camera_mod.BI_FROZEN_THRESHOLD - 1):
            assert cam._capture_blueiris() is True
        assert cam._capture_blueiris() is False  # frozen -> fall back
        assert cam._capture_blueiris() is False  # stays frozen


def test_capture_blueiris_dup_check_survives_direct_fallback():
    """The freshness hash must not be de-armed by the direct path writing
    self.image_hash — the frozen frame stays frozen across a fallback."""
    cam = _make_camera()
    content = _jpeg_bytes()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(content)
        for _ in range(camera_mod.BI_FROZEN_THRESHOLD):
            cam._capture_blueiris()
        # Simulate the direct fallback overwriting the shared hash fields.
        cam.image_hash = "direct-hash"
        assert cam._capture_blueiris() is False  # still frozen


def test_capture_blueiris_recovers_when_frame_changes():
    cam = _make_camera()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(_jpeg_bytes())
        for _ in range(camera_mod.BI_FROZEN_THRESHOLD + 1):
            cam._capture_blueiris()
        bi.get.return_value = _response(_jpeg_bytes(seed=9))
        assert cam._capture_blueiris() is True
        assert cam.bi_dups == 0


def test_capture_blueiris_circuit_breaker_opens_after_failures():
    cam = _make_camera()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.side_effect = OSError("down")
        for _ in range(camera_mod.BI_BREAKER_FAILURES):
            assert cam._capture_blueiris() is False
        assert camera_mod._bi_skip_until > 0
        bi.get.reset_mock()
        assert cam._capture_blueiris() is False  # breaker open
        bi.get.assert_not_called()


def test_capture_blueiris_processing_failure_leaves_no_partial_state():
    """A resize() exception must not leave self.image set, or the direct
    path's `if self.image is None` failure counting is disabled."""
    cam = _make_camera()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(_jpeg_bytes())
        with mock.patch.object(cam, "resize", side_effect=cv2.error("bad frame")):
            assert cam._capture_blueiris() is False
    assert cam.image is None
    assert cam.resized is None


def test_capture_blueiris_undecodable_signals_fallback():
    cam = _make_camera()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(b"<html>error</html>")
        assert cam._capture_blueiris() is False
    assert cam.image is None


def test_capture_blueiris_empty_signals_fallback():
    cam = _make_camera()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(b"")
        assert cam._capture_blueiris() is False


def test_capture_backoff_is_capped():
    """A long-dead camera must not earn unbounded skip blackouts."""
    cam = _make_camera(blueiris_url=None)
    cam.fails = 50
    direct_session = mock.Mock()
    direct_session.get.side_effect = OSError("still down")
    with mock.patch.object(cam, "_get_session", return_value=direct_session):
        with mock.patch.object(cam, "reboot"):
            cam.capture()
    assert cam.skip == 64
    assert cam.fails == 51
