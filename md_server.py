#!/usr/bin/env python3
"""MegaDetector as a local HTTP service, for mdconfirm.py.

Runs under its own venv (~/md-venv: megadetector + torch), not aicam's, and
keeps the model loaded: loading MDv1000-redwood takes about 20 s, a frame then
takes under a second on the M4's GPU.

    ~/md-venv/bin/python md_server.py [--port 5060] [--model MDV1000-REDWOOD]

    POST /detect   body: a JPEG frame
                   -> {"detections": [{"category": "animal", "conf": 0.91,
                                       "bbox": [x, y, w, h]}, ...]}
                   bbox is normalised to the frame, top-left origin -- the
                   same layout as aicam's boundingBox.
    GET  /health   -> {"ok": true, "model": ...}

Listens on 127.0.0.1 only. One frame at a time: aicam asks a few dozen times
a night, and a second inference in parallel would only contend for the GPU
that aicam's own detector is using.
"""
import argparse
import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger("md_server")

CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}
# Low on purpose: the caller applies its own bar, and seeing what was just
# under it is what makes a log line worth reading.
FLOOR = 0.1


class Detector(object):
    def __init__(self, model):
        from megadetector.detection.run_detector import load_detector
        self.model = model
        self.detector = load_detector(model)
        self.lock = threading.Lock()

    def detect(self, jpeg):
        from PIL import Image
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        with self.lock:
            result = self.detector.generate_detections_one_image(
                image, "frame", detection_threshold=FLOOR)
        out = []
        for d in result.get("detections") or []:
            out.append({"category": CATEGORIES.get(str(d["category"]), str(d["category"])),
                        "conf": round(float(d["conf"]), 3),
                        "bbox": [round(float(v), 4) for v in d["bbox"]]})
        return out


def handler_for(detector):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body):
            blob = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"ok": True, "model": detector.model})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/detect":
                self._reply(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                self._reply(400, {"error": "empty body"})
                return
            try:
                self._reply(200, {"detections": detector.detect(self.rfile.read(length))})
            except Exception as e:
                logger.exception("detect failed")
                self._reply(500, {"error": str(e)})

        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=5060)
    parser.add_argument("--model", default="MDV1000-REDWOOD")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    detector = Detector(args.model)
    logger.info("loaded %s; listening on 127.0.0.1:%d", args.model, args.port)
    HTTPServer(("127.0.0.1", args.port), handler_for(detector)).serve_forever()


if __name__ == "__main__":
    main()
