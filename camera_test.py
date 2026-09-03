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
    assert cam.blueiris_uri.startswith("http://blueiris-3.home:81/image/test?")
    assert cam.full_uri.startswith("http://blueiris-3.home:81/image/test?")


def test_blueiris_name_override():
    cam = _make_camera(**{"blueiris-name": "front_entry"})
    assert "image/front_entry" in cam.blueiris_uri


def test_no_blueiris_url_disables_blueiris():
    cam = _make_camera(blueiris_url=None)
    assert cam.blueiris_uri is None


def test_snapshot_url_overrides_blueiris():
    """Frigate's go2rtc holds the main stream open, so its frames do not flip
    between sharp and upscaled substream the way Blue Iris does."""
    url = "http://ubuntu24.home:1984/api/frame.jpeg?src=garage_right"
    cam = _make_camera(**{"snapshot-url": url})
    assert cam.blueiris_uri == url


def test_snapshot_url_works_without_a_blueiris_url():
    url = "http://ubuntu24.home:1984/api/frame.jpeg?src=shed"
    cam = _make_camera(blueiris_url=None, **{"snapshot-url": url})
    assert cam.blueiris_uri == url


def test_snapshot_url_wins_over_blueiris_name():
    """An explicit source must not be silently replaced by name derivation."""
    url = "http://ubuntu24.home:1984/api/frame.jpeg?src=shed"
    cam = _make_camera(**{"snapshot-url": url, "blueiris-name": "front_entry"})
    assert cam.blueiris_uri == url


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
    # A single failure costs nothing but the next retry; sustained failure
    # still backs off (see BACKOFF_AFTER_FAILURES).
    assert cam.skip == 0
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


def test_blueiris_only_repeats_do_not_warn(caplog):
    """Idle cameras must not spam the syslog sink at WARNING."""
    cam = _make_camera(blueiris_only=True)
    with caplog.at_level("WARNING", logger="camera"):
        with mock.patch.object(camera_mod, "_bi_session") as bi:
            bi.get.return_value = _response(_jpeg_bytes())
            for _ in range(camera_mod.BI_FROZEN_THRESHOLD + 5):
                cam.capture()
    assert caplog.records == []


def test_blueiris_only_warns_once_on_a_genuinely_stuck_stream(caplog):
    cam = _make_camera(blueiris_only=True)
    with caplog.at_level("WARNING", logger="camera"):
        with mock.patch.object(camera_mod, "_bi_session") as bi:
            bi.get.return_value = _response(_jpeg_bytes())
            for _ in range(camera_mod.BI_STUCK_THRESHOLD + 5):
                cam.capture()
    assert len(caplog.records) == 1
    assert "stuck stream" in caplog.records[0].message


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


# --- backoff grace ----------------------------------------------------------


def test_transient_failures_do_not_cost_cycles():
    """Blue Iris drops ~1 request in 10. Backing off from the first failure
    turned that into 65% of camera samples being skipped."""
    from camera import backoff_cycles

    assert backoff_cycles(1) == 0
    assert backoff_cycles(2) == 0
    assert backoff_cycles(3) == 0


def test_sustained_failure_still_backs_off():
    from camera import backoff_cycles

    assert backoff_cycles(4) == 1
    assert backoff_cycles(5) == 2
    assert backoff_cycles(6) == 4


def test_backoff_is_still_capped():
    """A camera down for hours previously earned ~20-minute blackouts."""
    from camera import backoff_cycles

    assert backoff_cycles(100) == 64
    assert max(backoff_cycles(n) for n in range(1, 200)) == 64


def test_backoff_is_monotonic():
    from camera import backoff_cycles

    seq = [backoff_cycles(n) for n in range(1, 30)]
    assert seq == sorted(seq)


# --- detection frame vs full-resolution frame -------------------------------


def test_detect_and_full_uris_differ():
    """Routine polling must not pull a 4MB frame; the model only needs 608x608."""
    cam = _make_camera()
    assert "w=1088" in cam.blueiris_uri
    assert "s=100" in cam.full_uri
    assert cam.blueiris_uri != cam.full_uri


def test_detect_uri_is_at_least_the_model_input():
    cam = _make_camera()
    width = int(cam.blueiris_uri.split("w=")[1].split("&")[0])
    assert width >= 608


def test_full_image_is_fetched_lazily_and_cached():
    cam = _make_camera()
    cam.capture.__self__  # sanity: bound method exists
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(_jpeg_bytes(seed=5))
        first = cam.full_image()
        second = cam.full_image()
    assert first is not None
    assert second is first
    assert bi.get.call_count == 1  # cached, not refetched


def test_full_image_falls_back_to_the_detection_frame():
    """A failed full fetch must not lose the event."""
    cam = _make_camera()
    cam.image = "detection-frame"
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.side_effect = OSError("boom")
        assert cam.full_image() == "detection-frame"


def test_snapshot_url_camera_does_not_double_fetch():
    """With one explicit source there is no cheaper detection variant."""
    url = "http://ubuntu24.home:1984/api/frame.jpeg?src=shed"
    cam = _make_camera(**{"snapshot-url": url})
    cam.image = "detection-frame"
    assert cam.full_uri == url
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        assert cam.full_image() == "detection-frame"
        bi.get.assert_not_called()
