"""Tests for the review-upload server's verdicts.

A tap on a phone is the only human judgement this system ever gets about a
frame, and the three verdicts disagree about what the frame means: "flag"
asserts nothing, "correct" writes the detector's own boxes into the dataset as
truth, and "false" writes the frame in with no boxes at all. Getting one of
those wrong teaches the model the opposite of what the person meant, so the
routing between them is worth testing even though the uploads themselves are
somebody else's HTTP.
"""

import json
import os
from configparser import ConfigParser
from unittest import mock

import pytest

import roboflow_upload


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _rewrite_boxes(review_dir, boxes):
    """Replace the sidecar's boxes, keeping everything else."""
    path = str(review_dir / "abc123.json")
    doc = json.load(open(path))
    doc["boxes"] = boxes
    with open(path, "w") as f:
        json.dump(doc, f)


@pytest.fixture
def review(tmp_path):
    """A configured server with one frame and its sidecar on disk."""
    save = tmp_path / "data"
    (save / "review").mkdir(parents=True)
    (save / "review" / "abc123.jpg").write_bytes(b"\xff\xd8jpegbytes")
    (save / "review" / "abc123.json").write_text(json.dumps({
        "file": "abc123.jpg", "cam": "peach tree", "model": "ipcams",
        "tags": ["deer"], "width": 2688, "height": 1520,
        # A rock's worth of frame: 0.06% of it. The size guard exists because
        # a tap may only silence something this small.
        "boxes": [{"label": "deer", "left": 0.1, "top": 0.2,
                   "width": 0.02, "height": 0.03, "probability": 0.72}],
    }))
    cfg = ConfigParser()
    cfg.read_dict({
        "detector": {"save-path": str(save),
                     "excludes-dir": str(tmp_path / "excludes"),
                     "excludes-auto-dir": str(tmp_path / "excludes-auto")},
        "color-model": {"onnx": "/models/ipcams_v32.onnx"},
        "vehicle-model": {"onnx": "/models/packages_v11.onnx"},
        "roboflow": {"api-key": "k", "delete-after-upload": "false",
                     "project.ipcams2": "deer,person", "project.pv2": "package,vehicle"},
    })
    config_path = tmp_path / "config.txt"
    with open(str(config_path), "w") as f:
        cfg.write(f)
    conf = roboflow_upload.Config(cfg)
    conf.config_path = str(config_path)
    with mock.patch.object(roboflow_upload, "_config", conf):
        yield save / "review"


@pytest.fixture
def api():
    """The three Roboflow calls, stubbed. upload_image hands back an id."""
    with mock.patch.object(roboflow_upload, "upload_image", return_value="img1") as up, \
         mock.patch.object(roboflow_upload, "annotate") as ann, \
         mock.patch.object(roboflow_upload, "annotate_null") as null:
        yield {"upload": up, "annotate": ann, "null": null}


class TestSplitVerdict:
    def test_a_bare_name_is_a_flag(self):
        """What the MQTT button in main.py has always sent."""
        assert roboflow_upload.split_verdict("abc123.jpg") == ("abc123.jpg", "flag")

    def test_the_verdict_rides_on_the_name(self):
        assert roboflow_upload.split_verdict("abc123.jpg|correct") == ("abc123.jpg", "correct")
        assert roboflow_upload.split_verdict("abc123.jpg|false") == ("abc123.jpg", "false")

    def test_a_verdict_nobody_defined_is_not_guessed_at(self):
        """Better an unannotated upload than an invented label."""
        assert roboflow_upload.split_verdict("abc123.jpg|sure") == ("abc123.jpg", "flag")

    def test_a_path_is_stripped_before_the_split(self):
        assert roboflow_upload.split_verdict("/tmp/abc123.jpg|false")[0] == "abc123.jpg"


class TestVerdicts:
    def test_flag_uploads_without_saying_anything_about_the_frame(self, review, api):
        code, result = roboflow_upload._do_upload("abc123.jpg", "ipcams", "peach_tree", {"deer"})
        assert code == 200 and result["verdict"] == "flag"
        assert api["upload"].called
        assert not api["annotate"].called and not api["null"].called

    def test_correct_annotates_with_the_boxes_from_the_alert(self, review, api):
        code, result = roboflow_upload._do_upload("abc123.jpg|correct", "", "", set())
        assert code == 200 and result["verdict"] == "correct"
        _, _, _, _, width, height, boxes = api["annotate"].call_args[0]
        assert (width, height) == (2688, 1520)
        assert boxes == [("deer", 0.1, 0.2, 0.02, 0.03)]
        assert not api["null"].called

    def test_false_annotates_with_no_boxes_at_all(self, review, api):
        code, result = roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        assert code == 200 and result["verdict"] == "false"
        assert api["null"].called and not api["annotate"].called

    def test_a_verdict_tags_the_image_so_the_dataset_can_be_filtered(self, review, api):
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        assert "verdict-false" in api["upload"].call_args[1]["tags"]
        api["upload"].reset_mock()
        roboflow_upload._do_upload("abc123.jpg", "", "", set())
        assert not any(t.startswith("verdict-") for t in api["upload"].call_args[1]["tags"])

    def test_correct_without_geometry_does_not_become_a_background_example(self, review, api):
        """The downgrade that matters: no boxes must not read as "no objects"."""
        os.remove(str(review / "abc123.json"))
        code, result = roboflow_upload._do_upload("abc123.jpg|correct", "m", "c", {"deer"})
        assert code == 200 and result["verdict"] == "flag"
        assert not api["annotate"].called and not api["null"].called

    def test_the_sidecar_supplies_what_home_assistant_leaves_empty(self, review, api):
        """The webhook templates render to '' when a query field is absent."""
        code, result = roboflow_upload._do_upload("abc123.jpg", "", "unknown", set())
        assert code == 200
        # Routed to the project that covers deer, named for the camera.
        assert result["projects"] == ["ipcams2"]
        assert api["upload"].call_args[0][2].startswith("peach_tree-deer-")

    def test_a_frame_that_is_gone_is_not_an_upload(self, review, api):
        code, result = roboflow_upload._do_upload("nope.jpg|false", "", "", set())
        assert code == 404 and not api["upload"].called


class TestSilencing:
    """"All false" has to stop the false positive, not just file a note.

    A rock called a rabbit goes on being a rabbit every three seconds until
    something suppresses it, so the tap writes a live exclusion. What keeps
    that from being reckless is that it expires on its own: the file names the
    models it was written against, and excludes.load_dir drops it the moment
    that set changes.
    """

    def test_false_silences_the_spot_now(self, review, api):
        code, result = roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        auto = roboflow_upload._config.auto_dir
        assert result["silenced"] == ["peach_tree-deer-11-22"]
        assert sorted(os.listdir(auto)) == ["peach_tree-deer-11-22.jpg",
                                            "peach_tree-deer-11-22.yaml"]

    def test_the_exclusion_names_the_models_it_was_written_against(self, review, api):
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        doc = _load_yaml(os.path.join(roboflow_upload._config.auto_dir,
                                      "peach_tree-deer-11-22.yaml"))
        assert doc["provisional"] is True
        assert sorted(doc["models"]) == ["ipcams_v32.onnx", "packages_v11.onnx"]
        assert doc["camera"] == "peach tree" and doc["label"] == "deer"
        assert doc["box"]["left"] == 0.1

    def test_a_live_exclusion_is_loaded_and_survives_its_own_models(self, review, api):
        import excludes
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        loaded = excludes.load_dir(roboflow_upload._config.auto_dir,
                                   ["ipcams_v32.onnx", "packages_v11.onnx"])
        assert loaded["peach tree"]["deer"][0]["left"] == 0.1

    def test_a_retrain_puts_the_blind_spot_back_on_trial(self, review, api):
        """The whole expiry rule: a new model set means it stops applying."""
        import excludes
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        assert excludes.load_dir(roboflow_upload._config.auto_dir,
                                 ["ipcams_v33.onnx", "packages_v11.onnx"]) == {}

    def test_the_frame_is_paired_with_the_geometry(self, review, api):
        """recheck_excludes.py replays the jpg; delete-after-upload eats the original."""
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        assert open(os.path.join(roboflow_upload._config.auto_dir,
                                 "peach_tree-deer-11-22.jpg"), "rb").read() \
            == b"\xff\xd8jpegbytes"

    def test_nothing_is_silenced_by_the_other_verdicts(self, review, api):
        roboflow_upload._do_upload("abc123.jpg|correct", "", "", set())
        roboflow_upload._do_upload("abc123.jpg", "", "", set())
        assert not os.path.isdir(roboflow_upload._config.auto_dir)

    def test_the_same_spot_twice_is_one_exclusion(self, review, api):
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        yamls = [f for f in os.listdir(roboflow_upload._config.auto_dir)
                 if f.endswith(".yaml")]
        assert yamls == ["peach_tree-deer-11-22.yaml"]


class TestSilencingGuards:
    """One tap is one frame of evidence, and an exclusion is a blind spot."""

    def test_a_box_too_big_to_silence_is_held_back(self, review, api):
        _rewrite_boxes(review, [{"label": "deer", "left": 0.1, "top": 0.1,
                                 "width": 0.5, "height": 0.5, "probability": 0.6}])
        code, result = roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        assert not result.get("silenced")
        assert result["pending_exclusions"] == ["peach_tree-deer-35-35"]
        assert not os.path.isdir(roboflow_upload._config.auto_dir) or \
            [f for f in os.listdir(roboflow_upload._config.auto_dir)
             if f.endswith(".yaml")] == []
        doc = open(os.path.join(roboflow_upload._config.pending_dir,
                                "peach_tree-deer-35-35.yaml")).read()
        assert "25.0% of the frame" in doc and "provisional" not in doc

    def test_a_held_back_candidate_suppresses_nothing(self, review, api):
        import excludes
        _rewrite_boxes(review, [{"label": "deer", "left": 0.1, "top": 0.1,
                                 "width": 0.5, "height": 0.5, "probability": 0.6}])
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        # pending/ is a subdirectory and load_dir does not recurse.
        assert excludes.load_dir(roboflow_upload._config.auto_dir, ["ipcams_v32.onnx"]) == {}

    def test_a_camera_may_only_collect_so_many(self, review, api):
        roboflow_upload._config.max_exclusions_per_camera = 1
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        _rewrite_boxes(review, [{"label": "deer", "left": 0.8, "top": 0.8,
                                 "width": 0.05, "height": 0.05, "probability": 0.6}])
        code, result = roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        assert result["pending_exclusions"] == ["peach_tree-deer-83-83"]
        doc = open(os.path.join(roboflow_upload._config.pending_dir,
                                "peach_tree-deer-83-83.yaml")).read()
        assert "already has 1 silenced spots" in doc


class TestCleanup:
    def test_the_sidecar_is_deleted_with_the_frame(self, review, api, tmp_path):
        roboflow_upload._config.delete_after_upload = True
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        assert os.listdir(str(review)) == []


class TestServeImage:
    def test_a_verdict_suffix_does_not_reach_the_filesystem(self, review):
        """review.html passes the plain name, but the links do not."""
        assert roboflow_upload.split_verdict("abc123.jpg|false")[0] == "abc123.jpg"

    def test_a_traversal_is_a_basename(self):
        assert roboflow_upload.split_verdict("../../etc/passwd.jpg")[0] == "passwd.jpg"


class TestTapConfirmation:
    """The tap's only answer.

    A Home Assistant webhook automation returns an empty body no matter what it
    does -- checked on 2026.9.2 with an automation that does nothing but stop
    with a literal response -- so the `stop`/`response_variable` pair in
    aicam_roboflow_upload_webhook never told anyone anything and the phone has
    always shown a blank page. This is the confirmation instead.
    """

    def test_a_silenced_spot_is_named(self, review, api):
        msg = roboflow_upload.tap_message(200, {
            "verdict": "false", "projects": ["ipcams2"],
            "silenced": ["peach_tree-rabbit-11-22"]}, "abc123.jpg")
        assert "silenced peach_tree-rabbit-11-22 until the models change" in msg

    def test_a_guard_says_so_rather_than_claiming_silence(self, review, api):
        """The failure worth hearing about: it uploaded but nothing went quiet."""
        msg = roboflow_upload.tap_message(200, {
            "verdict": "false", "projects": ["ipcams2"],
            "pending_exclusions": ["peach_tree-deer-35-35"]}, "abc123.jpg")
        assert "held back peach_tree-deer-35-35, not silenced" in msg
        assert "silenced peach" not in msg

    def test_an_error_carries_the_reason(self, review, api):
        msg = roboflow_upload.tap_message(404, {"error": "file not found: abc123.jpg"},
                                          "abc123.jpg")
        assert msg == "abc123.jpg: file not found: abc123.jpg"

    def test_a_confirmation_is_quiet(self, review, api):
        roboflow_upload._config.pushover = ("tok", "usr")
        with mock.patch.object(roboflow_upload, "urlopen") as sent:
            roboflow_upload.announce(200, {"verdict": "flag", "projects": ["ipcams2"]},
                                     "abc123.jpg")
        body = sent.call_args[0][0].data.decode()
        assert "priority=-1" in body and "token=tok" in body

    def test_pushover_being_down_does_not_fail_the_tap(self, review, api):
        roboflow_upload._config.pushover = ("tok", "usr")
        with mock.patch.object(roboflow_upload, "urlopen", side_effect=OSError("nope")):
            roboflow_upload.announce(200, {"verdict": "flag", "projects": []}, "a.jpg")

    def test_nothing_is_sent_when_it_is_not_configured(self, review, api):
        roboflow_upload._config.pushover = None
        with mock.patch.object(roboflow_upload, "urlopen") as sent:
            roboflow_upload.announce(200, {"verdict": "flag", "projects": []}, "a.jpg")
        assert not sent.called
