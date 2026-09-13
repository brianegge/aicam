"""Tests for turning suppressed detections into training negatives.

The case that matters most is the one that must never upload: a frame holding
a real animal alongside a suppressed false positive. Getting that wrong
teaches the detector that the animal is scenery.
"""
import configparser
import json
from unittest import mock

import pytest

import autolabel


def config(tmp_path, **over):
    c = configparser.ConfigParser()
    c["detector"] = {"save-path": str(tmp_path)}
    c["roboflow"] = dict({
        "api-key": "rf-test",
        "auto-null-uploads": "true",
        "auto-null-per-camera-daily": "5",
        "project.ipcams2": "cat,coyote,deer,dog,fox,person,rabbit,raccoon",
        "project.packages-vehicles2": "package,vehicle",
    }, **{k.replace("_", "-"): v for k, v in over.items()})
    return c


def pred(tag="deer", ignore="verified: nothing", left=0.5, top=0.4, w=0.05, h=0.06):
    p = {"tagName": tag,
         "boundingBox": {"left": left, "top": top, "width": w, "height": h},
         "probability": 0.7}
    if ignore is not None:
        p["ignore"] = ignore
    return p


# --- what counts as a negative -----------------------------------------------

def test_a_frame_where_everything_was_suppressed_is_a_negative():
    assert autolabel.is_negative([pred()])


def test_a_real_animal_alongside_a_false_positive_is_not():
    """Uploading this would teach the detector that raccoons are scenery."""
    assert not autolabel.is_negative(
        [pred("rabbit", "mulch bed rock with an IR shadow"),
         pred("raccoon", ignore=None, left=0.2)])


def test_an_empty_frame_is_not_worth_uploading():
    """The dataset has plenty; what is wanted is a frame the detector got wrong."""
    assert not autolabel.is_negative([])


def test_a_road_suppression_is_not_a_statement_about_the_pixels():
    """'ignore: road' means out of area, not that there is no vehicle there."""
    assert not autolabel.is_negative([pred("vehicle", "road")])


def test_a_neighbour_suppression_is_not_either():
    assert not autolabel.is_negative([pred("person", "neighbor")])


def test_an_exclusion_counts_as_trainable():
    """Training these out is the entire point."""
    assert autolabel.is_negative([pred("rabbit", "static iou 0.91")])


# --- uploading ---------------------------------------------------------------

@pytest.fixture
def uploads():
    with mock.patch.object(autolabel.roboflow_upload, "upload_image",
                           return_value="img123") as up, \
         mock.patch.object(autolabel.roboflow_upload, "annotate_null") as ann:
        yield up, ann


def run(tmp_path, preds, cam="peach tree", cfg=None):
    return autolabel.maybe_upload_negative(
        cam, b"jpeg-bytes", preds, cfg or config(tmp_path))


def test_it_uploads_and_then_annotates_as_null(tmp_path, uploads):
    up, ann = uploads
    name = run(tmp_path, [pred("deer")])
    assert name and up.called and ann.called
    # Uploading without annotating leaves it in the unannotated bucket, doing
    # nothing -- which is the state the 2222 flagged images are already in.
    assert ann.call_args[0][2] == "img123"


def test_the_name_carries_the_camera_and_class(tmp_path, uploads):
    up, _ = uploads
    name = run(tmp_path, [pred("deer")])
    assert name.startswith("peach_tree-deer-")


def test_it_routes_to_the_project_that_owns_the_class(tmp_path, uploads):
    up, _ = uploads
    run(tmp_path, [pred("vehicle", "static")], cam="driveway")
    assert up.call_args[0][1] == "packages-vehicles2"


def test_a_real_animal_in_frame_blocks_the_upload(tmp_path, uploads):
    up, _ = uploads
    assert run(tmp_path, [pred("rabbit", "static"),
                          pred("raccoon", ignore=None, left=0.1)]) is None
    assert not up.called


def test_it_is_off_unless_configured(tmp_path, uploads):
    up, _ = uploads
    cfg = config(tmp_path, auto_null_uploads="false")
    assert run(tmp_path, [pred("deer")], cfg=cfg) is None
    assert not up.called


# --- rate limiting -----------------------------------------------------------

def test_the_same_object_is_not_uploaded_twice(tmp_path, uploads):
    """The peach tree rock produced 96 detections over three days."""
    up, _ = uploads
    for _ in range(10):
        run(tmp_path, [pred("raccoon", "mulch bed rock")])
    assert up.call_count == 1


def test_a_different_object_on_the_same_camera_is_uploaded(tmp_path, uploads):
    up, _ = uploads
    run(tmp_path, [pred("raccoon", "mulch bed rock", left=0.01, top=0.81)])
    run(tmp_path, [pred("rabbit", "lawn stake", left=0.83, top=0.55)])
    assert up.call_count == 2


def test_a_camera_cannot_exceed_its_daily_allowance(tmp_path, uploads):
    up, _ = uploads
    for i in range(12):
        run(tmp_path, [pred("deer", "static", left=0.05 * i, top=0.05 * i)])
    assert up.call_count == 5


def test_cameras_have_separate_allowances(tmp_path, uploads):
    up, _ = uploads
    for i in range(8):
        run(tmp_path, [pred("deer", "static", left=0.05 * i)], cam="peach tree")
    for i in range(8):
        run(tmp_path, [pred("deer", "static", left=0.05 * i)], cam="tree line")
    assert up.call_count == 10


def test_the_allowance_resets_on_a_new_day(tmp_path, uploads):
    up, _ = uploads
    for i in range(6):
        run(tmp_path, [pred("deer", "static", left=0.05 * i)])
    state = json.loads((tmp_path / "autolabel-state.json").read_text())
    state["date"] = "2020-01-01"
    (tmp_path / "autolabel-state.json").write_text(json.dumps(state))
    run(tmp_path, [pred("deer", "static", left=0.9)])
    assert up.call_count == 6


def test_a_failed_upload_is_not_counted_against_the_allowance(tmp_path):
    with mock.patch.object(autolabel.roboflow_upload, "upload_image",
                           side_effect=RuntimeError("network")):
        assert run(tmp_path, [pred("deer")]) is None
    assert not (tmp_path / "autolabel-state.json").exists()


def test_an_upload_with_no_id_does_not_get_annotated(tmp_path):
    with mock.patch.object(autolabel.roboflow_upload, "upload_image",
                           return_value=None), \
         mock.patch.object(autolabel.roboflow_upload, "annotate_null") as ann:
        assert run(tmp_path, [pred("deer")]) is None
    assert not ann.called


def test_a_free_text_exclusion_comment_is_trainable():
    """Exclusion comments are whatever the author wrote; all of them count."""
    for comment in ("pale stone at the lawn edge under IR",
                    "small stake or marker in the lawn",
                    "bare shrub branch tip under IR",
                    "dark foliage and shadow at the treeline"):
        assert autolabel.is_negative([pred("rabbit", comment)]), comment


def test_a_comment_that_merely_mentions_a_road_still_counts():
    """Matched exactly, so only detect.py's literal 'road' is excluded."""
    assert autolabel.is_negative([pred("deer", "stone at the road edge")])
