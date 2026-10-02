"""Tests for creating Frigate events from aicam's animal alerts."""

from configparser import ConfigParser
from unittest import mock

import frigate_events


class _Cam:
    def __init__(self, uri="http://frigate.home:1984/api/frame.jpeg?src=tree_line&w=1088",
                 name="tree line"):
        self.blueiris_uri = uri
        self.name = name


def _cfg(**frigate):
    c = ConfigParser()
    c.read_dict({"frigate": dict({"url": "http://frigate.home:5000"}, **frigate)})
    return c


def _p(label, prob=0.81, priority=0, **extra):
    p = {"tagName": label, "probability": prob, "priority": priority,
         "boundingBox": {"left": 0.589, "top": 0.225, "width": 0.019, "height": 0.029}}
    p.update(extra)
    return p


def test_the_raccoon_frigate_never_saw_becomes_an_event():
    """2026-10-01 19:31, tree line: aicam alerted, Frigate had nothing."""
    events = frigate_events.events_for(_Cam(), [_p("raccoon")], _cfg())
    assert len(events) == 1
    camera, label, body = events[0]
    assert (camera, label) == ("tree_line", "raccoon")
    assert body["sub_label"] == "aicam" and body["duration"] == 60
    assert body["draw"]["boxes"][0]["box"] == [0.589, 0.225, 0.019, 0.029]
    assert body["draw"]["boxes"][0]["score"] == 81


def test_what_notify_ignored_or_excluded_is_not_recorded():
    preds = [_p("rabbit", priority=-4), _p("fox", ignore="rock"),
             _p("coyote", departed=True)]
    assert frigate_events.events_for(_Cam(), preds, _cfg()) == []


def test_people_and_vehicles_are_left_to_frigate():
    preds = [_p("person"), _p("vehicle"), _p("package")]
    assert frigate_events.events_for(_Cam(), preds, _cfg()) == []


def test_a_road_label_is_recorded_as_the_animal():
    events = frigate_events.events_for(_Cam(), [_p("dog_road")], _cfg())
    assert events[0][1] == "dog"


def test_one_event_per_label_with_every_box():
    preds = [_p("deer", 0.7), _p("deer", 0.9), _p("rabbit", 0.6)]
    events = frigate_events.events_for(_Cam(), preds, _cfg())
    assert [e[1] for e in events] == ["deer", "rabbit"]
    assert events[0][2]["score"] == 0.9 and len(events[0][2]["draw"]["boxes"]) == 2


def test_labels_and_duration_are_configurable():
    cfg = _cfg(**{"event-labels": "coyote", "event-duration": "90"})
    events = frigate_events.events_for(_Cam(), [_p("raccoon"), _p("coyote")], cfg)
    assert [e[1] for e in events] == ["coyote"] and events[0][2]["duration"] == 90


def test_nothing_without_frigate_or_a_frigate_camera():
    c = ConfigParser()
    assert frigate_events.events_for(_Cam(), [_p("raccoon")], c) == []
    assert frigate_events.events_for(_Cam(uri=None, name="mailbox"),
                                     [_p("raccoon")], _cfg()) == []


def test_mark_posts_to_the_create_endpoint():
    with mock.patch.object(frigate_events.requests, "post") as post, \
         mock.patch.object(frigate_events.threading, "Thread",
                           side_effect=lambda target, args, daemon: mock.Mock(
                               start=lambda: target(*args))):
        assert frigate_events.mark(_Cam(), [_p("raccoon")], _cfg()) == 1
    url = post.call_args[0][0]
    assert url == "http://frigate.home:5000/api/events/tree_line/raccoon/create"


def test_a_frigate_failure_is_only_a_log_line():
    with mock.patch.object(frigate_events.requests, "post",
                           side_effect=frigate_events.requests.ConnectionError("down")):
        frigate_events._post("http://frigate.home:5000", "tree_line", "raccoon", {})
