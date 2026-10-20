#!/usr/bin/env python3
import configparser
from unittest.mock import MagicMock, patch

import pytest

from homeassistant import HomeAssistant


config = configparser.ConfigParser()
config.read("config-test.txt")


class MockHAAPI:
    """Stateful mock of the Home Assistant REST API."""

    def __init__(self):
        self.states = {
            "input_boolean.vehicle_detector": "on",
            "input_boolean.person_detector": "on",
            "binary_sensor.konnected_198e05_zone_4": "off",
            "binary_sensor.konnected_198e05_zone_5": "off",
            "input_boolean.night_mode": "off",
            "input_boolean.vacation_mode": "off",
            "binary_sensor.is_dark": "off",
            "group.egge": "home",
        }

    def get(self, url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/api/"):
            resp.json.return_value = {"message": "API running."}
        elif url.endswith("/api/states"):
            resp.json.return_value = [
                {"entity_id": k, "attributes": {"friendly_name": k}, "state": v}
                for k, v in self.states.items()
            ]
        elif "/api/states/" in url:
            entity = url.split("/api/states/")[-1]
            state = self.states.get(entity, "unknown")
            resp.json.return_value = {"state": state}
        return resp

    def post(self, url, json=None, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"ok"
        if json and "entity_id" in json:
            entity = json["entity_id"]
            if "input_boolean/turn_on" in url:
                self.states[entity] = "on"
            elif "input_boolean/turn_off" in url:
                self.states[entity] = "off"
            elif entity == "script.pause_person_detector":
                self.states["input_boolean.person_detector"] = "off"
        return resp


@pytest.fixture
def ha():
    mock_api = MockHAAPI()
    with patch("homeassistant.requests") as mock_requests:
        mock_requests.get = mock_api.get
        mock_requests.post = mock_api.post
        mock_requests.exceptions = __import__("requests").exceptions
        yield HomeAssistant(config["homeassistant"])


def test_should_notify_vehicle(ha):
    orig = ha.should_notify_vehicle()
    assert type(orig) == bool
    ha.set_notify_vehicle(False)
    assert ha.should_notify_vehicle() is False
    ha.set_notify_vehicle(True)
    assert ha.should_notify_vehicle() is True
    ha.set_notify_vehicle(orig)


def test_should_notify_person(ha):
    assert type(ha.should_notify_person()) == bool
    ha.suppress_notify_person()
    assert ha.should_notify_person() is False


def test_door_contacts(ha):
    assert ha.get_door_left() is not None
    assert ha.get_door_right() is not None


def test_mode(ha):
    assert ha.mode() in ["home", "night"]


def test_dark(ha):
    assert ha.is_dark() in [True, False]


def test_vacation(ha):
    assert ha.vacation_mode() in [True, False]


def test_is_before_six(ha):
    result = ha.is_time_after_midnight_and_before_six()
    assert result in [True, False]


def main():
    mock_api = MockHAAPI()
    with patch("homeassistant.requests") as mock_requests:
        mock_requests.get = mock_api.get
        mock_requests.post = mock_api.post
        mock_requests.exceptions = __import__("requests").exceptions
        ha = HomeAssistant(config["homeassistant"])
        print("HomeAssistant Status:")
        print("-" * 30)
        print(f"Notify Vehicle: {ha.should_notify_vehicle()}")
        print(f"Notify Person: {ha.should_notify_person()}")
        print(f"Door Left: {ha.get_door_left()}")
        print(f"Door Right: {ha.get_door_right()}")
        print(f"Mode: {ha.mode()}")
        print(f"Is Dark: {ha.is_dark()}")
        print(f"Vacation Mode: {ha.vacation_mode()}")
        print(
            f"Time between midnight and 6AM: {ha.is_time_after_midnight_and_before_six()}"
        )


if __name__ == "__main__":
    main()


def _ha_with(responses):
    """A HomeAssistant whose state reads answer from a list, in order."""
    ha = HomeAssistant.__new__(HomeAssistant)
    ha.api = "http://ha/api/"
    ha.headers = {}
    ha.cache = {}
    replies = iter(responses)

    def get(url, **kwargs):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        resp = MagicMock()
        resp.json.return_value = r
        return resp
    return ha, get


def test_entity_not_found_does_not_replace_the_last_good_state():
    """2026-10-20 09:24: Home Assistant starting up answered "Entity not found"."""
    ha, get = _ha_with([{"state": "on"}, {"message": "Entity not found."}])
    with patch("homeassistant.requests.get", side_effect=get):
        assert ha.get_state("binary_sensor.is_dark") is True
        assert ha.get_state("binary_sensor.is_dark") is True
    assert ha.cache["binary_sensor.is_dark"] == {"state": "on"}


def test_is_dark_with_nothing_known_falls_back_quietly_and_retries_soon():
    import requests as rq
    ha, get = _ha_with([rq.exceptions.ConnectionError("down"), {"state": "on"}])
    with patch("homeassistant.requests.get", side_effect=get) as g:
        assert ha.is_dark() is False
        assert ha.is_dark() is False      # within the retry window: not asked
        assert g.call_count == 1
        ha._is_dark_cached = (False, ha._is_dark_cached[1]
                              - __import__("datetime").timedelta(seconds=16))
        assert ha.is_dark() is True
