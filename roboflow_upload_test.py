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


@pytest.fixture
def review(tmp_path):
    """A configured server with one frame and its sidecar on disk."""
    save = tmp_path / "data"
    (save / "review").mkdir(parents=True)
    (save / "review" / "abc123.jpg").write_bytes(b"\xff\xd8jpegbytes")
    (save / "review" / "abc123.json").write_text(json.dumps({
        "file": "abc123.jpg", "cam": "peach tree", "model": "ipcams",
        "tags": ["deer"], "width": 2688, "height": 1520,
        "boxes": [{"label": "deer", "left": 0.1, "top": 0.2,
                   "width": 0.3, "height": 0.4, "probability": 0.72}],
    }))
    cfg = ConfigParser()
    cfg.read_dict({
        "detector": {"save-path": str(save), "excludes-dir": str(tmp_path / "excludes")},
        "roboflow": {"api-key": "k", "delete-after-upload": "false",
                     "project.ipcams2": "deer,person", "project.pv2": "package,vehicle"},
    })
    with mock.patch.object(roboflow_upload, "_config", roboflow_upload.Config(cfg)):
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
        assert boxes == [("deer", 0.1, 0.2, 0.3, 0.4)]
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


class TestPendingExclusions:
    def test_false_parks_a_candidate_exclusion_no_one_loads(self, review, api):
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        pending = os.path.join(roboflow_upload._config.pending_dir)
        assert sorted(os.listdir(pending)) == ["peach_tree-deer-25-40.jpg",
                                               "peach_tree-deer-25-40.yaml"]
        doc = open(os.path.join(pending, "peach_tree-deer-25-40.yaml")).read()
        assert "camera: peach tree" in doc and "label: deer" in doc
        assert "left: 0.10000000" in doc
        assert "CANDIDATE" in doc

    def test_the_frame_is_paired_with_the_geometry(self, review, api):
        """recheck_excludes.py replays the jpg; delete-after-upload eats the original."""
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        pending = roboflow_upload._config.pending_dir
        assert open(os.path.join(pending, "peach_tree-deer-25-40.jpg"), "rb").read() \
            == b"\xff\xd8jpegbytes"

    def test_nothing_is_parked_for_the_other_verdicts(self, review, api):
        roboflow_upload._do_upload("abc123.jpg|correct", "", "", set())
        roboflow_upload._do_upload("abc123.jpg", "", "", set())
        assert not os.path.isdir(roboflow_upload._config.pending_dir)

    def test_a_candidate_is_not_loaded_as_an_exclusion(self, review, api):
        """The whole reason it goes in a subdirectory: load_dir does not recurse."""
        import excludes
        roboflow_upload._do_upload("abc123.jpg|false", "", "", set())
        assert excludes.load_dir(os.path.dirname(roboflow_upload._config.pending_dir)) == {}


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
