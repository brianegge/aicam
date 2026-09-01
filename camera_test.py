"""Tests for Camera capture paths (Blue Iris primary, direct fallback)."""
import configparser
from unittest import mock

import cv2
import numpy as np

import camera as camera_mod
from camera import Camera


def _jpeg_bytes(seed=0):
    img = np.full((48, 64, 3), seed, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_camera(blueiris_url="http://blueiris-3.home:81", **extra):
    config = configparser.ConfigParser()
    config["cam0"] = {
        "uri": "http://test-cam.home/cgi-bin/snapshot.cgi",
        "name": "test",
        "user": "admin",
        "password": "x",
    }
    for k, v in extra.items():
        config["cam0"][k] = v
    return Camera(config["cam0"], {}, mqtt_client=None, blueiris_url=blueiris_url)


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
    assert cam.blueiris_uri == "http://blueiris-3.home:81/image/test?q=85&w=1920&h=1080"


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


def test_capture_blueiris_duplicate_frame_signals_fallback():
    """An identical frame from BI means the stream is frozen; fall back."""
    cam = _make_camera()
    content = _jpeg_bytes()
    with mock.patch.object(camera_mod, "_bi_session") as bi:
        bi.get.return_value = _response(content)
        assert cam._capture_blueiris() is True  # first frame is fresh
        assert cam._capture_blueiris() is False  # identical frame -> fallback


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
