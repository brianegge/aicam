#!/usr/bin/env python3
"""Standalone HTTP server that uploads flagged detection images to Roboflow.

Listens for POST /upload requests from Home Assistant and uploads the
referenced image to the Roboflow REST API for model retraining.

Python 3.6 compatible (runs on Jetson Nano).
"""
import argparse
import base64
import datetime
import html
import json
import logging
import os
import sys
from configparser import ConfigParser
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
except ImportError:
    pass
try:
    from urllib.parse import urlparse, parse_qs, urlencode
except ImportError:
    from urlparse import urlparse, parse_qs
    from urllib import urlencode

from excludes import slug as _slug

logger = logging.getLogger("aicam-review")


class Config(object):
    def __init__(self, config):
        section = config["roboflow"]
        self.api_key = section["api-key"]
        self.delete_after_upload = section.getboolean("delete-after-upload", True)
        self.save_path = config["detector"]["save-path"]
        self.review_dir = os.path.join(self.save_path, "review")
        # Candidate exclusions from an "all false" tap. A subdirectory, because
        # excludes.load_dir globs *.yaml and does not recurse: nothing here is
        # loaded until a human audits it and moves it up.
        self.pending_dir = os.path.join(
            config["detector"].get("excludes-dir", "excludes"), "pending")
        # Build project routing: {class_name: [project_id, ...]}
        # Config keys like: project.ipcams2 = cat,dog,person
        self.projects = {}  # project_id -> set of classes
        for key, value in section.items():
            if key.startswith("project."):
                project_id = key[len("project."):]
                classes = set(c.strip() for c in value.split(","))
                self.projects[project_id] = classes
        if not self.projects:
            raise ValueError("No project.* keys found in [roboflow] config")

    def projects_for_tags(self, tags):
        """Return list of project IDs that cover any of the given tags."""
        matched = []
        for project_id, classes in self.projects.items():
            if tags & classes:
                matched.append(project_id)
        return matched


def upload_name(cam, detection_tags, filename):
    """A name that says where the frame came from and what was claimed.

    Review files are content hashes, and passing that straight through as the
    Roboflow name threw away the only provenance the dataset had: 212 of 1045
    images in packages-vehicles2 and 539 of 5260 in ipcams2 arrived as bare
    hashes, so no one could tell which camera they came from or what the
    detector had said. The hash is kept as a suffix to stay unique.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    cam_part = (cam or "unknown").replace(" ", "_").replace("/", "_")
    tag_part = "_".join(sorted(t for t in (detection_tags or set()) if t)) or "none"
    return "%s-%s-%s" % (cam_part, tag_part, stem)


def upload_image(api_key, project_id, name, image_bytes, split="train", tags=None):
    """POST one image. Returns the Roboflow image id, or None."""
    query = {"api_key": api_key, "name": name, "split": split}
    if tags:
        query["tag"] = ",".join(sorted(tags))
    url = "https://api.roboflow.com/dataset/%s/upload?%s" % (project_id, urlencode(query))
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    req = Request(url, data=encoded.encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    body = urlopen(req, timeout=30).read().decode("utf-8")
    try:
        return json.loads(body).get("id")
    except ValueError:
        return None


_VOC_HEAD = ("<annotation><folder></folder><filename>%s.jpg</filename>"
             "<size><width>%d</width><height>%d</height><depth>3</depth></size>"
             "<segmented>0</segmented>")
_VOC_OBJECT = ("<object><name>%s</name><pose>Unspecified</pose><truncated>0</truncated>"
               "<difficult>0</difficult><bndbox><xmin>%d</xmin><ymin>%d</ymin>"
               "<xmax>%d</xmax><ymax>%d</ymax></bndbox></object>")


def build_voc(name, width, height, boxes=()):
    """A Pascal VOC annotation. With no boxes it is a background example.

    boxes are (label, left, top, w, h) with the geometry normalised 0-1, the
    same convention the detector uses. VOC wants absolute pixels, so they are
    converted here rather than at every call site.
    """
    parts = [_VOC_HEAD % (name, width, height)]
    for label, left, top, w, h in boxes:
        parts.append(_VOC_OBJECT % (
            label,
            max(0, round(left * width)), max(0, round(top * height)),
            min(width, round((left + w) * width)), min(height, round((top + h) * height))))
    parts.append("</annotation>")
    return "".join(parts)


def annotate(api_key, project_id, image_id, name, width, height, boxes=()):
    """Attach a Pascal VOC annotation. No boxes means a background example."""
    url = "https://api.roboflow.com/dataset/%s/annotate/%s?%s" % (
        project_id, image_id,
        urlencode({"api_key": api_key, "name": name + ".xml"}))
    body = build_voc(name, width, height, boxes).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "text/xml")
    return urlopen(req, timeout=30).read().decode("utf-8")


def annotate_null(api_key, project_id, image_id, name, width, height):
    """Mark an uploaded image as a background example.

    Uploading alone is not enough: an image with no annotation record is a
    pending chore, while one annotated with zero boxes is training data that
    actively suppresses whatever the detector thought it saw.

    The annotation is Pascal VOC with no <object> elements. The obvious
    encoding -- an empty YOLO .txt, which is exactly how a background image is
    represented on disk -- is rejected by the upload API with
    InvalidAnnotationFormat, as are an empty body, a bare newline and an empty
    CreateML array. VOC is the one format that expresses "this image, no
    objects" in a way the parser accepts. Verified against the live API on
    2026-09-13; the result reads back as {"count": 0, "classes": {}}, the same
    shape as an image marked null by hand in the web UI.
    """
    return annotate(api_key, project_id, image_id, name, width, height)


_config = None

# What a tap on the phone can say about a frame.
#   flag    -- upload it unannotated, label it by hand later (the original)
#   correct -- the boxes in the alert were right; upload them as the annotation
#   false   -- nothing in the frame was real; upload it as a background example
VERDICTS = ("flag", "correct", "false")

_MISSING = ("", "unknown", "none", "None")


def split_verdict(filename):
    """('abc.jpg|correct') -> ('abc.jpg', 'correct'); a bare name means flag.

    The verdict arrives glued to the file name because the Home Assistant
    webhook forwards four fixed query fields to the rest_command and its
    payload is YAML on that box -- see notify.verdict_url. A `v=` of its own is
    still honoured for callers that can send one.
    """
    name = os.path.basename(filename or "")
    if "|" not in name:
        return name, "flag"
    name, _, verdict = name.partition("|")
    verdict = verdict.strip().lower()
    if verdict not in VERDICTS:
        logger.warning("unknown verdict %r on %s, treating as flag", verdict, name)
        return name, "flag"
    return name, verdict


def read_sidecar(review_dir, filename):
    """The detector's own account of the frame, written by notify.write_sidecar."""
    path = os.path.join(review_dir, os.path.splitext(filename)[0] + ".json")
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, OSError, ValueError) as e:
        logger.warning("no sidecar for %s: %s", filename, e)
        return {}


def _or(value, fallback):
    return fallback if value is None or str(value).strip() in _MISSING else value


def pending_exclusion(pending_dir, cam, filename, box, score, model, image_bytes):
    """Park a candidate exclusion where nothing loads it.

    `excludes.load_dir` globs `excludes/*.yaml` and does not recurse, so a file
    here is inert until a human moves it up a directory. That gap is the point.
    A real exclusion in this repo is not one frame's box -- deck-person-22-69
    is the median of 21 archive hits, paired with the highest-scoring one so
    `recheck_excludes.py` can ever answer STILL NEEDED. A tap on a phone cannot
    produce that, and a too-generous box silently suppresses real detections
    for as long as nobody notices.
    """
    stem = "%s-%s-%02d-%02d" % (
        _slug(cam), _slug(box.get("label") or "object"),
        int((box["left"] + box["width"] / 2.0) * 100 + 0.5),
        int((box["top"] + box["height"] / 2.0) * 100 + 0.5))
    try:
        os.makedirs(pending_dir)
    except OSError:
        pass
    doc = (
        "# CANDIDATE -- not loaded from here. Flagged \"all false\" from a phone on\n"
        "# %s, from a single frame and nothing else.\n"
        "#\n"
        "# Before moving this up into excludes/: audit the box against the capture\n"
        "# archive by IoU, take the median of the hits rather than this one box,\n"
        "# and pair it with the highest-scoring hit -- a pair whose own frame\n"
        "# cannot reach the class threshold can never report STILL NEEDED.\n"
        "# Then say in the comment what the object is and why the model cannot\n"
        "# learn it. If it can learn it, the \"all false\" upload already did that\n"
        "# and this file is not needed at all.\n"
        "camera: %s\n"
        "label: %s\n"
        "comment: flagged from a phone, unaudited\n"
        "box:\n"
        "  left: %.8f\n"
        "  top: %.8f\n"
        "  width: %.8f\n"
        "  height: %.8f\n"
        "center:\n"
        "  x: %.3f\n"
        "  y: %.3f\n"
        "created: %s\n"
        "score: %.5f\n"
        "model: %s\n"
        "source_review: %s\n"
    ) % (
        datetime.date.today().isoformat(),
        cam, box.get("label") or "object",
        box["left"], box["top"], box["width"], box["height"],
        box["left"] + box["width"] / 2.0, box["top"] + box["height"] / 2.0,
        datetime.date.today().isoformat(), score or 0.0, model, filename,
    )
    with open(os.path.join(pending_dir, stem + ".yaml"), "w") as f:
        f.write(doc)
    # The frame travels with the geometry: recheck_excludes.py replays the
    # paired jpg, and delete-after-upload is about to remove the original.
    with open(os.path.join(pending_dir, stem + ".jpg"), "wb") as f:
        f.write(image_bytes)
    return stem


def _do_upload(filename, model, cam, detection_tags, verdict=None):
    """Upload a review image to Roboflow. Returns (status_code, result_dict)."""
    filename, from_name = split_verdict(filename)
    verdict = _or(verdict, None) or from_name
    if verdict not in VERDICTS:
        verdict = "flag"
    filepath = os.path.join(_config.review_dir, filename)

    if not os.path.isfile(filepath):
        return (404, {"error": "file not found: %s" % filename})

    sidecar = read_sidecar(_config.review_dir, filename)
    cam = _or(cam, None) or sidecar.get("cam") or "unknown"
    model = _or(model, None) or sidecar.get("model") or "unknown"
    detection_tags = set(t for t in (detection_tags or ()) if t) or set(sidecar.get("tags") or ())

    target_projects = _config.projects_for_tags(detection_tags)
    if not target_projects:
        logger.warning("No project matches tags %s on %s (%s), skipping upload",
                       detection_tags, filename, verdict)
        return (400, {"error": "no project matches tags: %s" % ",".join(sorted(detection_tags))})

    try:
        with open(filepath, "rb") as f:
            image_data = f.read()
    except IOError as e:
        return (500, {"error": "failed to read file: %s" % e})

    boxes = [(b["label"], b["left"], b["top"], b["width"], b["height"])
             for b in sidecar.get("boxes") or ()]
    width, height = sidecar.get("width"), sidecar.get("height")
    if verdict == "correct" and not (boxes and width and height):
        # Annotating "correct" with no geometry would file the frame as a
        # background example -- the opposite verdict. Fall back to the one
        # that asks a human instead of asserting the wrong thing.
        logger.warning("%s: no boxes in the sidecar, downgrading correct to flag", filename)
        verdict = "flag"
    if verdict == "false" and not (width and height):
        logger.warning("%s: no frame size in the sidecar, downgrading false to flag", filename)
        verdict = "flag"

    name = upload_name(cam, detection_tags, filename)
    upload_tags = [cam.replace(" ", "_"), model.replace(" ", "_")]
    if verdict != "flag":
        upload_tags.append("verdict-%s" % verdict)
    uploaded = []
    annotated = []

    for project_id in target_projects:
        try:
            image_id = upload_image(_config.api_key, project_id, name, image_data,
                                    tags=upload_tags)
        except (HTTPError, URLError) as e:
            error_body = ""
            if hasattr(e, "read"):
                error_body = e.read().decode("utf-8", errors="replace")
            logger.error("Roboflow upload to %s failed for %s: %s %s",
                         project_id, filename, e, error_body)
            continue
        logger.info("Uploaded %s to %s as %s (%s)", filename, project_id, image_id, verdict)
        uploaded.append(project_id)
        if verdict == "flag" or not image_id:
            continue
        try:
            if verdict == "correct":
                annotate(_config.api_key, project_id, image_id, name, width, height, boxes)
            else:
                annotate_null(_config.api_key, project_id, image_id, name, width, height)
            annotated.append(project_id)
        except (HTTPError, URLError) as e:
            # The image is in the dataset either way; an un-annotated one is a
            # chore, not a wrong label, so this is a warning and not a failure.
            logger.warning("Annotating %s in %s failed: %s", name, project_id, e)

    if not uploaded:
        return (502, {"error": "all uploads failed"})

    pending = []
    if verdict == "false":
        for box in sidecar.get("boxes") or ():
            try:
                pending.append(pending_exclusion(
                    _config.pending_dir, cam, filename, box,
                    box.get("probability"), model, image_data))
            except Exception:
                logger.exception("could not write a candidate exclusion for %s", filename)

    if _config.delete_after_upload:
        try:
            os.remove(filepath)
            logger.info("Deleted review file %s", filepath)
        except OSError as e:
            logger.warning("Failed to delete %s: %s", filepath, e)
        try:
            os.remove(os.path.join(_config.review_dir,
                                   os.path.splitext(filename)[0] + ".json"))
        except OSError:
            pass

    result = {"status": "uploaded", "file": filename, "projects": uploaded,
              "verdict": verdict}
    if annotated:
        result["annotated"] = annotated
    if pending:
        result["pending_exclusions"] = pending
    return (200, result)


_TITLES = {
    "flag": "Image Flagged for Review",
    "correct": "Detections Confirmed",
    "false": "Detections Marked False",
}


class UploadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._respond(200, {"status": "ok"})
            return

        params = parse_qs(parsed.query)
        filename = params.get("file", [None])[0]
        if not filename:
            self._respond(404, {"error": "not found"})
            return

        if parsed.path == "/image":
            # The frame, for review.html. Only ever reachable from the home
            # network -- this host is on the isolated OpenClaw subnet.
            self._serve_image(filename)
            return

        model = params.get("model", ["unknown"])[0]
        cam = params.get("cam", ["unknown"])[0]
        tags_str = params.get("tags", [""])[0]
        detection_tags = set(tags_str.split(",")) if tags_str else set()
        verdict = params.get("v", [None])[0]

        code, result = _do_upload(filename, model, cam, detection_tags, verdict)

        if code == 200:
            title = _TITLES[result["verdict"]]
            body = "<p>Uploaded <b>%s</b> from <b>%s</b> to %s.</p>" % (
                html.escape(os.path.basename(result["file"])),
                html.escape(cam.replace("_", " ")),
                html.escape(", ".join(result["projects"])),
            )
            if result["verdict"] == "correct":
                body += "<p>Annotated with the boxes from the alert.</p>"
            elif result["verdict"] == "false":
                body += "<p>Annotated with no boxes, so it trains as background.</p>"
                if result.get("pending_exclusions"):
                    body += "<p>Candidate exclusion%s parked for review: %s</p>" % (
                        "" if len(result["pending_exclusions"]) == 1 else "s",
                        html.escape(", ".join(result["pending_exclusions"])))
        else:
            title = "Upload Failed"
            body = "<p>%s</p>" % html.escape(result.get("error", "unknown error"))

        doc = (
            "<!DOCTYPE html><html><head>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>%s</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:480px;"
            "margin:40px auto;padding:0 16px;text-align:center}"
            ".ok{color:#16a34a}.err{color:#dc2626}</style>"
            "</head><body>"
            "<h2 class='%s'>%s</h2>%s"
            "</body></html>"
        ) % (html.escape(title), "ok" if code == 200 else "err", html.escape(title), body)

        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(doc.encode("utf-8"))

    def _serve_image(self, filename):
        name = split_verdict(filename)[0]
        if not name.endswith(".jpg"):
            self._respond(404, {"error": "not found"})
            return
        try:
            with open(os.path.join(_config.review_dir, name), "rb") as f:
                blob = f.read()
        except (IOError, OSError):
            self._respond(404, {"error": "file not found: %s" % name})
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_POST(self):
        if self.path != "/upload":
            self._respond(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._respond(400, {"error": "empty body"})
            return

        try:
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (json.JSONDecodeError, ValueError) as e:
            self._respond(400, {"error": "invalid json: %s" % e})
            return

        filename = body.get("file")
        model = body.get("model", "unknown")
        cam = body.get("cam", "unknown")
        detection_tags = set(body.get("tags", "").split(",")) if body.get("tags") else set()
        verdict = body.get("v")

        if not filename:
            self._respond(400, {"error": "missing 'file' field"})
            return

        code, result = _do_upload(filename, model, cam, detection_tags, verdict)
        self._respond(code, result)

    def _respond(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def log_message(self, format, *args):
        logger.info(format, *args)


def main():
    global _config

    parser = argparse.ArgumentParser(description="Roboflow review image upload server")
    parser.add_argument("--port", type=int, default=5050, help="HTTP listen port")
    parser.add_argument("--config", default="config.txt", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = ConfigParser()
    config.read(args.config)

    if "roboflow" not in config:
        logger.error("Missing [roboflow] section in %s", args.config)
        sys.exit(1)
    if "detector" not in config:
        logger.error("Missing [detector] section in %s", args.config)
        sys.exit(1)

    _config = Config(config)
    os.makedirs(_config.review_dir, exist_ok=True)

    server = HTTPServer(("", args.port), UploadHandler)
    logger.info("Listening on port %d, review dir: %s", args.port, _config.review_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    logger.info("Server stopped")


if __name__ == "__main__":
    main()
