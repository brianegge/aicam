#!/usr/bin/env python3
"""Standalone HTTP server that uploads flagged detection images to Roboflow.

Listens for POST /upload requests from Home Assistant and uploads the
referenced image to the Roboflow REST API for model retraining.

Python 3.6 compatible (runs on Jetson Nano).
"""
import argparse
import base64
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
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urlparse import urlparse, parse_qs

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
    name = os.path.splitext(filename)[0]
    upload_tags = "%s,%s" % (cam.replace(" ", "_"), model.replace(" ", "_"))
    uploaded = []

    for project_id in target_projects:
        url = "https://api.roboflow.com/dataset/%s/upload?api_key=%s&name=%s&split=train&tag=%s" % (
            project_id, _config.api_key, name, upload_tags
        )
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
                filename, cam.replace("_", " "),
                ", ".join(result["projects"]),
            )
        else:
            title = "Upload Failed"
            body = "<p>%s</p>" % result.get("error", "unknown error")

        html = (
            "<!DOCTYPE html><html><head>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>%s</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:480px;"
            "margin:40px auto;padding:0 16px;text-align:center}"
            ".ok{color:#16a34a}.err{color:#dc2626}</style>"
            "</head><body>"
            "<h2 class='%s'>%s</h2>%s"
            "</body></html>"
        ) % ("ok" if code == 200 else "err", title, title, body)

        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

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
