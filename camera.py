import hashlib
import logging
import os
import subprocess
import sys
import time
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
# concurrently from a thread pool. Built fully at import time so pool threads
# never race a lazy init or see the session before the adapter is mounted.
_bi_adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=32)
_bi_session = requests.Session()
_bi_session.mount("http://", _bi_adapter)
_bi_session.mount("https://", _bi_adapter)

# Circuit breaker for a shared snapshot host: without it, an outage makes all
# cameras each burn the 10s timeout before their direct fallback, every cycle.
# Races on these are benign (worst case one extra probe).
#
# Keyed by host:port, NOT global. Cameras now draw from two independent
# services -- Blue Iris and Frigate's go2rtc on another machine -- and a single
# shared counter meant powering Blue Iris down tripped the breaker for the
# go2rtc cameras too, blacking out all 15 cameras when only 10 had lost their
# source.
_breaker_fails = {}
_breaker_until = {}
BI_BREAKER_FAILURES = 3
BI_BREAKER_SECONDS = 60.0


def _breaker_key(uri):
    """Which host's breaker a request belongs to."""
    if not uri:
        return ""
    return urlparse(uri).netloc
# Consecutive identical frames before declaring the stream into BI frozen.
# The cameras stamp an OSD clock into every frame, so identical bytes should
# mean frozen — the margin covers a truly static re-encode.
BI_FROZEN_THRESHOLD = 3
# Routine polling only feeds a 608x608 model input, so fetching a full 4MB
# frame every cycle for every camera was pure waste -- at 15 cameras it put
# ~20MB/s through Blue Iris and pushed its CPU to 51-87%. w=1088 keeps both
# dimensions above the model input while costing ~10-13x less
# (peach tree 3.86MB -> 302KB). Full resolution is fetched lazily, only when a
# frame is actually going to be cropped for a notification or for ALPR.
BI_DETECT_PARAMS = "q=85&w=1088"
BI_FULL_PARAMS = "q=100&s=100"
# Model input geometry, as (width, height). The ipcams color/grey models are
# square; the vehicle/packages model need not be -- a 1088x608 model matches the
# cameras' 16:9 framing instead of squashing it, and measured better at the
# operational thresholds. set_model_input_sizes() points these at whatever the
# loaded models actually declare, so a model swap does not need a code change.
IPCAMS_INPUT_SIZE = (608, 608)
VEHICLE_INPUT_SIZE = (608, 608)


def set_model_input_sizes(ipcams_model, vehicle_model):
    """Point resize() at the geometry the loaded models actually want."""
    global IPCAMS_INPUT_SIZE, VEHICLE_INPUT_SIZE
    IPCAMS_INPUT_SIZE = (ipcams_model.model_width, ipcams_model.model_height)
    VEHICLE_INPUT_SIZE = (vehicle_model.model_width, vehicle_model.model_height)
# Tolerate this many consecutive failures before backing off at all. Blue Iris
# drops roughly one request in ten under normal load, and doubling from the
# first failure turned that into 65% of camera samples being skipped -- 6 real
# failures on one camera produced 15 skipped cycles. A transient blip should
# cost the next cycle's retry, nothing more; the exponential curve is for a
# camera that is actually gone.
BACKOFF_AFTER_FAILURES = 3


def backoff_cycles(fails):
    """Cycles to skip after `fails` consecutive capture failures."""
    if fails <= BACKOFF_AFTER_FAILURES:
        return 0
    return 2 ** min(fails - BACKOFF_AFTER_FAILURES - 1, 6)
# In blueiris-only mode there is no direct snapshot to escalate to, and Blue
# Iris legitimately repeats frames for cameras it is limit-decoding, so a short
# run means "idle", not "broken". Only a run this long is worth warning about.
# ~5 minutes at a 3s interval.
BI_STUCK_THRESHOLD = 100


class Camera:
    def __init__(
        self, config, excludes, mqtt_client, blueiris_url=None, blueiris_only=False
    ):
        # Set on hosts with no route to the cameras themselves (the detector
        # now runs in the OpenClaw subnet, which is firewalled off from the
        # IPCAMS VLAN). Blue Iris is the only way in, so the direct snapshot
        # fallback can only ever burn its connect timeout.
        self.blueiris_only = blueiris_only
        self.name = config["name"]
        self.ha_name = self.name.replace(" ", "_")
        self.config = config
        self.objects = set()
        # tagName -> when that class was last accepted on this camera. Backs the
        # durable low-confidence hold in detect.py; self.objects only ever
        # covered a single frame.
        self.recent_objects = {}
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
        # BI-only freshness state: self.image_hash is shared with the direct
        # and FTP paths, and comparing across sources de-arms the check.
        self.bi_hash = None
        self.bi_dups = 0
        self.mqtt = set(config.get("mqtt", "").split(","))
        self.mqtt_client = mqtt_client
        # A camera can name its own snapshot source, which takes precedence
        # over Blue Iris. Used to pull from Frigate's bundled go2rtc, which
        # holds the main stream open and so returns a consistently sharp frame:
        # Blue Iris swings ~100x in sharpness between polls depending on
        # whether it currently has the main stream decoded or only the
        # substream, which it then upscales.
        self._full_image = None
        snapshot_url = config.get("snapshot-url", None)
        if snapshot_url:
            self.blueiris_uri = snapshot_url
            self.full_uri = config.get("full-snapshot-url", snapshot_url)
        elif blueiris_url:
            bi_name = config.get("blueiris-name", self._default_blueiris_name())
            # q=100&s=100: notify.py crops self.image at native resolution
            # for push notifications and plate recognition, so size here is
            # not just about the 608x608 model input. Without s=100 BI caps
            # /image at ~1920 wide even when it has the 4K main stream
            # decoded. Caveat: when BI is "limit decoding" a camera it only
            # has the substream (e.g. 856x480), and s=100 cannot upscale —
            # the frame is whatever BI is currently decoding.
            base_uri = f"{blueiris_url}/image/{bi_name}"
            self.blueiris_uri = f"{base_uri}?{BI_DETECT_PARAMS}"
            self.full_uri = f"{base_uri}?{BI_FULL_PARAMS}"
        else:
            self.blueiris_uri = None
            self.full_uri = None
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

    def full_image(self):
        """Full-resolution frame, fetched lazily and cached for this cycle.

        notify.py crops at native resolution for push images and alpr.py for
        plate reads, so those need the real frame -- but detection does not,
        and it runs every cycle on every camera. Falls back to the detection
        frame rather than losing the event.
        """
        if self._full_image is not None:
            return self._full_image
        if not self.full_uri or self.full_uri == self.blueiris_uri:
            return self.image
        try:
            resp = _bi_session.get(self.full_uri, timeout=15)
            resp.raise_for_status()
            image = cv2.imdecode(
                np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            if image is None:
                raise ValueError("undecodable")
            self._full_image = image
            return image
        except Exception:
            logger.warning(
                "Full-resolution fetch failed for %s, cropping the detection frame: %s",
                self.name,
                sys.exc_info()[1],
            )
            return self.image

    def capture(self):
        self._full_image = None
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
            if self.blueiris_only:
                if self.bi_dups >= BI_FROZEN_THRESHOLD:
                    # Blue Iris limit-decodes idle cameras, so a repeated frame
                    # is normal here and only means there is nothing new to
                    # look at. With no direct path to escalate to, backing off
                    # would sample a quiet camera less and less often for no
                    # reason — just wait for the next frame.
                    self.error = "dup"
                    return self
                # A real Blue Iris failure. Back off exactly as the direct path
                # would, so a camera it cannot serve stops being retried every
                # cycle, but skip the request that can only time out.
                self.error = "no-bi"
                self.image_hash = 0
                self.source = None
                self.resized = None
                self.fails += 1
                self.skip = backoff_cycles(self.fails)
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
                    # Capped and delayed: uncapped, a camera down for hours
                    # earned ~20-minute blackouts (mailbox hit skip=700 on
                    # 2026-09-01). Backing off from the very first failure was
                    # just as bad in the other direction -- see
                    # BACKOFF_AFTER_FAILURES.
                    self.fails += 1
                    self.skip = backoff_cycles(self.fails)
                    if self.skip > 3:
                        self.reboot()
        return self

    def _capture_blueiris(self):
        """Grab the latest decoded frame from Blue Iris.

        Returns True only when self.image now holds a usable BI frame. False
        on any failure — including a stream frozen for BI_FROZEN_THRESHOLD
        cycles — means the caller should try the camera directly.
        """
        key = _breaker_key(self.blueiris_uri)
        if time.monotonic() < _breaker_until.get(key, 0.0):
            return False
        try:
            resp = _bi_session.get(self.blueiris_uri, timeout=10)
            resp.raise_for_status()
            content = resp.content
            if len(content) == 0:
                logger.warning(f"Blue Iris returned empty image for {self.name}")
                return False
        except Exception as e:
            # Only a transport failure means the *host* is down. An HTTP error
            # status means the host answered and is therefore up -- it just
            # cannot serve that one camera, typically because go2rtc has not
            # finished warming that stream.
            #
            # This distinction became critical once every camera moved behind a
            # single go2rtc: with all 14 sharing one host key, three HTTP 500s
            # from one unready stream tripped the breaker for the entire fleet
            # and blacked out detection for 60s at a time, repeatedly.
            if isinstance(e, requests.exceptions.HTTPError):
                logger.warning(
                    "Snapshot for %s failed: %s (host is up; not tripping the breaker)",
                    self.name,
                    e,
                )
                return False
            fails = _breaker_fails.get(key, 0) + 1
            _breaker_fails[key] = fails
            if fails >= BI_BREAKER_FAILURES:
                _breaker_until[key] = time.monotonic() + BI_BREAKER_SECONDS
                logger.warning(
                    "Snapshot host %s unreachable (%d straight failures) — skipping it for %ds",
                    key,
                    fails,
                    int(BI_BREAKER_SECONDS),
                )
            else:
                logger.warning(
                    f"Blue Iris capture failed for {self.name}, falling back to camera: {e}"
                )
            return False
        _breaker_fails[key] = 0
        # Hash the raw response before decoding: cheap, and it lets a frozen
        # stream short-circuit ahead of the decode. Tracked in a BI-only
        # field — the direct and FTP paths write self.image_hash, and sharing
        # it would de-arm this check after every fallback.
        content_hash = hashlib.md5(content).hexdigest()
        if content_hash == self.bi_hash:
            self.bi_dups += 1
            if self.bi_dups >= BI_FROZEN_THRESHOLD:
                if self.blueiris_only:
                    # Blue Iris limit-decodes idle cameras, so a short run of
                    # repeats is routine and there is nothing to escalate to.
                    # Only a run long enough to mean a genuinely stuck stream
                    # is worth a warning — at WARNING this fired ~1200x/day
                    # into the syslog sink describing quiet cameras.
                    if self.bi_dups == BI_STUCK_THRESHOLD:
                        logger.warning(
                            "Blue Iris frame for %s unchanged %d cycles — stuck stream?",
                            self.name,
                            self.bi_dups,
                        )
                    elif self.bi_dups == BI_FROZEN_THRESHOLD:
                        logger.debug(
                            "Blue Iris frame for %s unchanged %d cycles — waiting for a new frame",
                            self.name,
                            self.bi_dups,
                        )
                elif self.bi_dups == BI_FROZEN_THRESHOLD:
                    logger.warning(
                        f"Blue Iris frame for {self.name} unchanged {self.bi_dups} cycles (frozen stream?) — using direct snapshots"
                    )
                return False
        else:
            self.bi_hash = content_hash
            self.bi_dups = 0
        try:
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.warning(f"Blue Iris returned undecodable image for {self.name}")
                return False
            self.image = image
            self.source = self.blueiris_uri
            self.resize()
        except Exception:
            # Leave no partial state behind: the caller's direct-path
            # escalation counts failures via `self.image is None`.
            self.image = None
            self.resized = None
            self.resized2 = None
            logger.warning(
                f"Blue Iris frame for {self.name} failed to process: {sys.exc_info()[1]}"
            )
            return False
        self.error = None
        self.fails = 0
        return True

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
                cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), VEHICLE_INPUT_SIZE
            )
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.resized = cv2.resize(self.image, IPCAMS_INPUT_SIZE)
        else:
            resized = cv2.resize(self.image, IPCAMS_INPUT_SIZE)
            self.resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            if VEHICLE_INPUT_SIZE == IPCAMS_INPUT_SIZE:
                self.resized2 = self.resized
            else:
                self.resized2 = cv2.cvtColor(
                    cv2.resize(self.image, VEHICLE_INPUT_SIZE), cv2.COLOR_BGR2RGB
                )
