import datetime
import logging
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

log = logging.getLogger(__name__)


class HomeAssistant:
    """interface with HomeAssistant"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.config['token']}"
        }
        self.api: str = self.config.get("api", "http://homeassistant.home:8123/api/")
        try:
            response: requests.Response = requests.get(self.api, headers=self.headers, timeout=10)
            message: str = response.json().get("message", "")
            log.debug(f"Home Assistant {message}")
            if message != "API running.":
                log.warning(f"Unexpected HA response: {message}")
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            log.warning(f"Failed to connect to Home Assistant: {e}")
        self.names: Dict[str, Any] = {}
        try:
            response = requests.get(urljoin(self.api, "states"), headers=self.headers, timeout=10)
            for entity in response.json():
                if "friendly_name" in entity["attributes"]:
                    self.names[entity["attributes"]["friendly_name"]] = entity
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            log.warning(f"Failed to load HA states: {e}")
        self.last_house_cleaners_arrived: Optional[datetime.datetime] = None
        self.cache: Dict[str, Any] = {}

    def set_scene(self, scene: str) -> requests.Response:
        r = requests.post(
            f"{self.api}services/scene/turn_on",
            json={"entity_id": f"scene.{scene}"},
            headers=self.headers,
        )
        log.info(f"Turned on scene {scene}={r}")
        return r

    def open_garage_door(self) -> None:
        raise NotImplementedError

    def deer_alert(self, location: str) -> None:
        json = {
            "entity_id": "script.deer",
            "variables": {"location": location},
        }
        r = requests.post(
            f"{self.api}services/script/turn_on", json=json, headers=self.headers
        )
        log.info(f"Deer alert={r}")

    def get_device(self, name: str) -> Any:
        try:
            return self.devices[name.lower()]
        except KeyError:
            raise KeyError('No such device "{}"'.format(name))

    def get_state(self, entity: str) -> bool:
        try:
            response = requests.get(
                f"{self.api}states/{entity}", headers=self.headers
            ).json()
            self.cache[entity] = response
        except requests.exceptions.ConnectionError:
            if entity not in self.cache:
                raise RuntimeError(
                    f"Failed to fetch {self.api}states/{entity} and no cached response available"
                )
            log.warning(
                f"Failed to fetch {self.api}states/{entity}. Using cached response"
            )
            response = self.cache[entity]
        if "state" in response:
            return response["state"] == "on"
        log.debug(response)
        raise RuntimeError(f"Invalid response from entity {entity}={response}")

    def set_input_boolean(self, switch: str, state: bool) -> requests.Response:
        json = {
            "entity_id": switch,
        }
        if state:
            response = requests.post(
                f"{self.api}services/input_boolean/turn_on",
                json=json,
                headers=self.headers,
            )
        else:
            response = requests.post(
                f"{self.api}services/input_boolean/turn_off",
                json=json,
                headers=self.headers,
            )
        if response.status_code != 200:
            log.warning(f"Set input_boolean {switch} to {state}={response.content}")
        return response

    def set_notify_vehicle(self, state: bool) -> requests.Response:
        return self.set_input_boolean("input_boolean.vehicle_detector", state)

    def get_door_left(self) -> bool:
        return self.get_state("binary_sensor.konnected_198e05_zone_4")

    def get_door_right(self) -> bool:
        return self.get_state("binary_sensor.konnected_198e05_zone_5")

    def get_presence(self, person: str) -> bool:
        response = requests.get(
            f"{self.api}states/{person}", headers=self.headers
        ).json()
        return response["state"] == "home"

    def should_notify_vehicle(self) -> bool:
        return self.get_state("input_boolean.vehicle_detector")

    def should_notify_person(self) -> bool:
        return self.get_state("input_boolean.person_detector")

    def vacation_mode(self) -> bool:
        return self.get_state("input_boolean.vacation_mode")

    def echo_speaks(self, message: str) -> Any:
        if self.get_presence("group.egge"):
            log.info("Speaking {}".format(message))
            json = {"message": message, "data": {"type": "tts"}}
            r = requests.post(
                f"{self.api}services/notify/alexa_media_kitchen_ecobee4",
                json=json,
                headers=self.headers,
            )
            return r.content.decode("utf-8")
        else:
            log.info("Not speaking {}".format(message))
            return True

    def mode(self) -> str:
        if self.get_state("input_boolean.night_mode"):
            return "night"
        elif self.get_presence("group.egge"):
            return "home"
        else:
            return "away"

    def is_dog_inside(self) -> bool:
        entity = "sensor.rufus_status"
        try:
            response = requests.get(
                f"{self.api}states/{entity}", headers=self.headers
            ).json()
            self.cache[entity] = response
        except requests.exceptions.ConnectionError:
            if entity not in self.cache:
                raise RuntimeError(
                    f"Failed to fetch {entity} and no cached response available"
                )
            log.warning(f"Failed to fetch {entity}. Using cached response")
            response = self.cache[entity]
        return response.get("state") == "inside"

    def is_dark(self) -> bool:
        return self.get_state("binary_sensor.is_dark")

    def is_time_after_midnight_and_before_six(self) -> bool:
        current_time = datetime.datetime.now().time()
        midnight = datetime.time(0, 0)
        six_am = datetime.time(6, 0)

        if midnight <= current_time < six_am:
            return True
        else:
            return False

    def suppress_notify_person(self) -> str:
        log.debug("Keep person notify suppressed")
        r = requests.post(
            f"{self.api}services/script/turn_on",
            json={"entity_id": "script.pause_person_detector"},
            headers=self.headers,
        )
        return r.content.decode("utf-8")

    def house_cleaners_arrived(self) -> None:
        if (
            self.last_house_cleaners_arrived is None
            or datetime.datetime.now() - self.last_house_cleaners_arrived
            > datetime.timedelta(days=1)
        ):
            r = requests.post(
                f"{self.api}services/script/turn_on",
                json={"entity_id": "script.house_cleaners_arrive"},
                headers=self.headers,
            )
            log.info(f"Run script house cleaners arrive={r}")
            self.last_house_cleaners_arrived = datetime.datetime.now()
        else:
            log.info(f"House cleaners last arrrived {self.last_house_cleaners_arrived}")
