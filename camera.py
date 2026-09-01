import hashlib
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

from utils import cleanup

logger = logging.getLogger(__name__)

# One Blue Iris host serves frames for every camera, and captures run
# concurrently from a thread pool, so share a session with a pool sized for it.
_bi_session = None


def _blueiris_session():
    global _bi_session
    if _bi_session is None:
        _bi_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=20)
        _bi_session.mount("http://", adapter)
    return _bi_session


class Camera:
    def __init__(self, config, excludes, mqtt_client, blueiris_url=None):
        self.name = config["name"]
        self.ha_name = self.name.replace(" ", "_")
        self.config = config
        self.objects = set()
        self.prev_predictions = {}
        self.is_file = False
        self.counts = {}
        self.last_show_count = -1
        self.vehicle_check = config.getboolean("vehicle_check", False)
        self.excludes = excludes
        self.capture_async = config.getboolean("async", False)
        self.error = None
        self.image = None
        self.image_hash = 0
        self.source = None
        self.prior_image = None
        self.prior_time = datetime.fromtimestamp(0)
        self.prior_priority = -4
        self.age = 0
        self.fails = 0
        self.skip = 0
        self.ftp_path = config.get("ftp-path", None)
        self.interval = config.getint("interval", 30)
        self.session = None
        self.mqtt = set(config.get("mqtt", "").split(","))
        self.mqtt_client = mqtt_client
        if blueiris_url:
            bi_name = config.get("blueiris-name", self._default_blueiris_name())
            # w/h pin the output size: /image otherwise follows whatever
            # streaming profile is active, which can dip below the 608x608
            # model input. Blue Iris fits within the box preserving aspect.
            self.blueiris_uri = f"{blueiris_url}/image/{bi_name}?q=85&w=1920&h=1080"
        else:
            self.blueiris_uri = None
        road_line_raw = config.get("road_line", None)
        if road_line_raw == "all":
            self.road_line = "all"
        elif road_line_raw:
            self.road_line = [tuple(float(v) for v in p.split(":")) for p in road_line_raw.split(",")]
        else:
            self.road_line = None

    def _default_blueiris_name(self):
        """Derive Blue Iris short name from the camera URI hostname."""
        host = urlparse(self.config["uri"]).hostname or ""
        # strip trailing -cam.home → e.g. "front-entry-cam.home" → "front-entry"
        return host.replace("-cam.home", "").replace(".home", "")

    def road_y_at(self, x):
        if self.road_line is None or self.road_line == "all":
            return None
        points = self.road_line
        if x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return points[-1][1]

    def publish(self, topic, payload, **kwargs):
        """Publish to MQTT, tolerating short disconnects."""
        try:
            self.mqtt_client.publish(topic, payload, **kwargs)
        except Exception:
            logger.warning("MQTT publish failed for %s (topic=%s), will retry on reconnect", self.name, topic)

    def poll(self):
        # logger.debug('read ftp {}'.format(self.name))
        if self.ftp_path:
            img = None
            try:
                files = sorted(
                    Path(self.ftp_path).glob("**/*.jpg"), key=os.path.getmtime
                )
                if len(files) == 0:
                    cleanup(self.ftp_path)
                    return None
            except OSError as e:
                logger.error(f"Error scanning {self.ftp_path}: {e}\n{e.args}")
                return None
            good_files = []
            for f in files:
                if datetime.fromtimestamp(
                    os.path.getmtime(f)
                ) < datetime.now() - timedelta(minutes=5):
                    logger.warning(f"Skipping old file {f}")
                    os.remove(f)
                    continue
                else:
                    good_files.append(f)
            if len(good_files) == 0:
                return None
            f = good_files[0]
            # requires SUID on fuser
            # sudo chmod u+s /bin/fuser
            completedProc = subprocess.run(["/bin/fuser", str(f)])
            if completedProc.returncode == 0:
                print(f"{f} is open for writing")
                return None
            img = cv2.imread(str(f))
            os.remove(f)
            if img is not None and len(img) > 0:
                h = hashlib.md5(img.tobytes()).hexdigest()
                if self.image_hash == h:
                    self.error = "dup"
                    return None
                self.image = img
                self.image_hash = h
                self.source = f
                self.resize()
                return self
            else:
                self.error = "bad file"
        return None

    def capture(self):
        self.image = None
        self.resized = None
        self.resized2 = None
        if self.skip > 0:
            self.error = "skip={}".format(self.skip)
            self.skip -= 1
            return self
        if "file" in self.config:
            self.is_file = True
            self.image = cv2.imread(self.config["file"])
            self.source = self.config["file"]
            if self.image is not None:
                self.resize()
        else:
            # Blue Iris first: it already decodes every camera's stream for
            # recording and serves the latest frame in well under a second.
            # snapshot.cgi makes the camera encode a full-res JPEG on demand —
            # seconds of camera CPU per poll, and the busy Dahua cams drop
            # pings or time out outright under that load.
            if self.blueiris_uri and self._capture_blueiris():
                return self
            try:
                with self._get_session().get(
                    self.config["uri"], timeout=20, stream=True
                ) as resp:
                    resp.raise_for_status()
                    bytes = np.asarray(bytearray(resp.raw.read()), dtype="uint8")
                    if len(bytes) == 0:
                        self.error = "empty"
                        return self
                    self.image = cv2.imdecode(bytes, cv2.IMREAD_UNCHANGED)
                    self.image_hash = hashlib.md5(self.image.tobytes()).hexdigest()
                    self.source = self.config["uri"]
                    self.resize()
                    self.error = None
                    self.fails = 0
            except Exception:
                self.error = sys.exc_info()[0]
                logger.exception(f"Error with {self.name}:{self.error}")
                if self.image is None:
                    self.image_hash = 0
                    self.source = None
                    self.resized = None
                    self.skip = 2 ** self.fails
                    self.fails += 1
                    if self.skip > 3:
                        self.reboot()
        return self

    def _capture_blueiris(self):
        """Grab the latest decoded frame from Blue Iris.

        Returns True when a frame was handled (including a duplicate, which
        marks a frozen stream); False means the caller should fall back to a
        direct camera snapshot.
        """
        try:
            resp = _blueiris_session().get(self.blueiris_uri, timeout=10)
            resp.raise_for_status()
            bytes = np.asarray(bytearray(resp.content), dtype="uint8")
            if len(bytes) == 0:
                logger.warning(f"Blue Iris returned empty image for {self.name}")
                return False
            image = cv2.imdecode(bytes, cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.warning(f"Blue Iris returned undecodable image for {self.name}")
                return False
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            if image_hash == self.image_hash:
                # The camera streams continuously, so an identical frame means
                # the stream into Blue Iris is frozen; let the caller try the
                # camera directly.
                logger.warning(f"Blue Iris frame for {self.name} is unchanged (frozen stream?)")
                return False
            self.image = image
            self.image_hash = image_hash
            self.source = self.blueiris_uri
            self.resize()
            self.error = None
            self.fails = 0
            return True
        except Exception:
            logger.warning(f"Blue Iris capture failed for {self.name}, falling back to camera: {sys.exc_info()[1]}")
            return False

    def _get_session(self):
        if self.session is None:
            self.session = requests.Session()
            if "user" in self.config:
                self.session.auth = HTTPDigestAuth(
                    self.config["user"], self.config["password"]
                )
        return self.session

    def reboot(self):
        # "http://treeline-cam.home/cgi-bin/magicBox.cgi?action=reboot"
        url = (
            urlparse(self.config["uri"])
            ._replace(path="/cgi-bin/magicBox.cgi", query="action=reboot")
            .geturl()
        )
        logger.info(f"Rebooting {self.name}: {url}")
        try:
            r = self._get_session().get(url, timeout=10)
            r.raise_for_status()
        except Exception:
            logger.exception("Failed to reboot %s", self.name)

    def resize(self):
        if self.image is None:
            return
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        sum = np.sum(hsv[:, :, 0])
        if sum == 0:
            self.resized2 = cv2.resize(
                cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), (608, 608)
            )
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.resized = cv2.resize(self.image, (608, 608))
        else:
            resized = cv2.resize(self.image, (608, 608))
            self.resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.resized2 = self.resized
