#!/usr/bin/env python3
"""Standalone HTTP server that uploads flagged detection images to Roboflow.

Listens for POST /upload requests from Home Assistant and uploads the
referenced image to the Roboflow REST API for model retraining.

Python 3.6 compatible (runs on Jetson Nano).
"""
import argparse
import base64
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

logger = logging.getLogger("aicam-review")


class Config(object):
    def __init__(self, config):
        section = config["roboflow"]
        self.api_key = section["api-key"]
        self.delete_after_upload = section.getboolean("delete-after-upload", True)
        self.save_path = config["detector"]["save-path"]
        self.review_dir = os.path.join(self.save_path, "review")
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


def annotate_null(api_key, project_id, image_id, name):
    """Attach an empty YOLO annotation, which is what makes it a negative.

    Uploading alone is not enough. An image with no annotation sits in
    Roboflow's unannotated bucket and teaches the model nothing; an image
    annotated with zero boxes is a background example and actively suppresses
    whatever the detector thought it saw there.
    """
    url = "https://api.roboflow.com/dataset/%s/annotate/%s?%s" % (
        project_id, image_id,
        urlencode({"api_key": api_key, "name": name + ".txt"}))
    req = Request(url, data=b"", method="POST")
    req.add_header("Content-Type", "text/plain")
    return urlopen(req, timeout=30).read().decode("utf-8")


_config = None


def _do_upload(filename, model, cam, detection_tags):
    """Upload a review image to Roboflow. Returns (status_code, result_dict)."""
    filename = os.path.basename(filename)
    filepath = os.path.join(_config.review_dir, filename)

    if not os.path.isfile(filepath):
        return (404, {"error": "file not found: %s" % filename})

    target_projects = _config.projects_for_tags(detection_tags)
    if not target_projects:
        logger.warning("No project matches tags %s, skipping upload", detection_tags)
        return (400, {"error": "no project matches tags: %s" % ",".join(sorted(detection_tags))})

    try:
        with open(filepath, "rb") as f:
            image_data = f.read()
    except IOError as e:
        return (500, {"error": "failed to read file: %s" % e})

    encoded = base64.b64encode(image_data).decode("utf-8")
    name = upload_name(cam, detection_tags, filename)
    upload_tags = "%s,%s" % (cam.replace(" ", "_"), model.replace(" ", "_"))
    uploaded = []

    for project_id in target_projects:
        query = urlencode({
            "api_key": _config.api_key,
            "name": name,
            "split": "train",
            "tag": upload_tags,
        })
        url = "https://api.roboflow.com/dataset/%s/upload?%s" % (project_id, query)
        try:
            req = Request(url, data=encoded.encode("utf-8"), method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            resp = urlopen(req, timeout=30)
            resp_body = resp.read().decode("utf-8")
            logger.info("Uploaded %s to %s: %s", filename, project_id, resp_body)
            uploaded.append(project_id)
        except (HTTPError, URLError) as e:
            error_body = ""
            if hasattr(e, "read"):
                error_body = e.read().decode("utf-8", errors="replace")
            logger.error("Roboflow upload to %s failed for %s: %s %s", project_id, filename, e, error_body)

    if not uploaded:
        return (502, {"error": "all uploads failed"})

    if _config.delete_after_upload:
        try:
            os.remove(filepath)
            logger.info("Deleted review file %s", filepath)
        except OSError as e:
            logger.warning("Failed to delete %s: %s", filepath, e)

    return (200, {"status": "uploaded", "file": filename, "projects": uploaded})


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

        model = params.get("model", ["unknown"])[0]
        cam = params.get("cam", ["unknown"])[0]
        tags_str = params.get("tags", [""])[0]
        detection_tags = set(tags_str.split(",")) if tags_str else set()

        code, result = _do_upload(filename, model, cam, detection_tags)

        if code == 200:
            title = "Image Flagged for Review"
            body = "<p>Uploaded <b>%s</b> from <b>%s</b> to %s.</p>" % (
                html.escape(filename),
                html.escape(cam.replace("_", " ")),
                html.escape(", ".join(result["projects"])),
            )
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

        if not filename:
            self._respond(400, {"error": "missing 'file' field"})
            return

        code, result = _do_upload(filename, model, cam, detection_tags)
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
