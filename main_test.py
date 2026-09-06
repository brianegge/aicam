"""Tests for the Frigate motion trigger in main.py."""

from unittest import mock

import pytest

pytest.importorskip("cv2")

import main


class _Cam:
    def __init__(self, name="peach tree"):
        self.name = name
        self.motion_pending = False


class _Client:
    def __init__(self, cams):
        self._aicam_frigate_cams = cams
        self.subscribed = []

    def subscribe(self, topic):
        self.subscribed.append(topic)

    def publish(self, *a, **kw):
        pass


def test_motion_on_queues_the_camera():
    cam = _Cam()
    main._handle_frigate_motion(_Client({"peach_tree": cam}), "peach_tree", b"ON")
    assert cam.motion_pending is True


def test_motion_off_does_not_queue():
    cam = _Cam()
    main._handle_frigate_motion(_Client({"peach_tree": cam}), "peach_tree", b"OFF")
    assert cam.motion_pending is False


def test_unknown_camera_is_ignored():
    cam = _Cam()
    client = _Client({"peach_tree": cam})
    main._handle_frigate_motion(client, "not_a_camera", b"ON")
    assert cam.motion_pending is False


def test_payload_is_tolerant_of_whitespace_and_case():
    cam = _Cam()
    main._handle_frigate_motion(_Client({"peach_tree": cam}), "peach_tree", b" on\n")
    assert cam.motion_pending is True


def test_undecodable_payload_does_not_raise():
    cam = _Cam()
    main._handle_frigate_motion(_Client({"peach_tree": cam}), "peach_tree", b"\xff\xfe")
    assert cam.motion_pending is False


def test_on_message_routes_motion_without_touching_flag_path():
    """A motion message must not fall through into the flag handler.

    The flag path expects aicam/{cam}/flag_*/set and would otherwise try to
    read parts[1] as an aicam camera name.
    """
    cam = _Cam()
    client = _Client({"peach_tree": cam})
    msg = mock.Mock()
    msg.topic = "frigate/peach_tree/motion"
    msg.payload = b"ON"
    main.on_message(client, None, msg)
    assert cam.motion_pending is True


def test_on_connect_subscribes_only_when_frigate_cameras_exist():
    with mock.patch.object(main, "publish_discovery"):
        client = _Client({"peach_tree": _Cam()})
        client._reconnect_deadline = object()
        main.on_connect(client, None, {}, 0)
        assert "frigate/+/motion" in client.subscribed

        bare = _Client({})
        bare._reconnect_deadline = object()
        main.on_connect(bare, None, {}, 0)
        assert "frigate/+/motion" not in bare.subscribed


def test_repeat_motion_while_pending_is_idempotent():
    """Frigate re-publishes ON during a long episode; that must not stack up."""
    cam = _Cam()
    client = _Client({"peach_tree": cam})
    for _ in range(5):
        main._handle_frigate_motion(client, "peach_tree", b"ON")
    assert cam.motion_pending is True
