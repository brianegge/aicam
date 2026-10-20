"""Tests for replaying an exclusion through the current model.

Written after the 2026-09-19 audit reported RETIRE for all five wildcard
exclusions -- including one whose own file records 96 hits over 3 days --
because the matcher compared tagName against the literal "*".
"""

from recheck_excludes import matches

BOX = {"left": 0.10, "top": 0.10, "width": 0.10, "height": 0.10}


def pred(tag, prob=0.9, box=None):
    return {"tagName": tag, "probability": prob,
            "boundingBox": box or dict(BOX)}


def test_a_named_exclusion_matches_only_that_class():
    preds = [pred("deer"), pred("rabbit")]
    assert [p["tagName"] for p in matches(preds, "deer", BOX)] == ["deer"]


def test_a_wildcard_matches_whatever_the_detector_called_it():
    """The peach_tree stake was called rabbit 44, raccoon 42, fox 10 over one
    archive. A single-class entry only covers the name used that day."""
    preds = [pred("rabbit"), pred("raccoon"), pred("fox")]
    assert len(matches(preds, "*", BOX)) == 3


def test_a_wildcard_still_respects_geometry():
    """It suppresses a spot, not a camera."""
    far = {"left": 0.80, "top": 0.80, "width": 0.05, "height": 0.05}
    assert matches([pred("deer", box=far)], "*", BOX) == []


def test_a_wildcard_does_not_report_retire_when_the_model_still_fires():
    """The regression itself: tagName != "*" excluded every prediction, so
    `here` came back empty and the verdict was RETIRE at score 0.00."""
    assert matches([pred("cat", 0.81)], "*", BOX)


import datetime
import os
import textwrap

import yaml

from recheck_excludes import restamp, retire_pair

TAPPED = textwrap.dedent("""\
    # PROVISIONAL -- written by an "all false" tap.
    camera: play
    label: "*"
    box: {left: 0.0, top: 0.88, width: 0.22, height: 0.11}
    provisional: true
    models:
      - ipcams_v33.onnx
      - packages_v13.onnx
    source_review: 5b2faefa.jpg
    """)


def test_restamp_records_the_recheck_and_keeps_the_header(tmp_path):
    p = tmp_path / "play-person-11-94.yaml"
    p.write_text(TAPPED)
    restamp(str(p), ["ipcams_v34.onnx", "packages_v13.onnx"])
    text = p.read_text()
    assert text.startswith("# PROVISIONAL")
    doc = yaml.safe_load(text)
    assert doc["models"] == ["ipcams_v34.onnx", "packages_v13.onnx"]
    assert doc["source_review"] == "5b2faefa.jpg" and doc["provisional"] is True


def test_a_restamped_exclusion_is_current_again(tmp_path):
    import excludes
    p = tmp_path / "play-person-11-94.yaml"
    p.write_text(TAPPED)
    restamp(str(p), ["ipcams_v34.onnx", "packages_v13.onnx"])
    assert not excludes._stale(yaml.safe_load(p.read_text()),
                               ["ipcams_v34.onnx", "packages_v13.onnx"], None)


def test_retire_keeps_the_evidence_where_nothing_loads_it(tmp_path):
    import excludes
    (tmp_path / "play-person-11-94.yaml").write_text(TAPPED)
    (tmp_path / "play-person-11-94.jpg").write_bytes(b"jpeg")
    dest = retire_pair(str(tmp_path / "play-person-11-94.yaml"),
                       datetime.date(2026, 11, 2))
    assert os.path.basename(dest) == "retired-20261102"
    assert sorted(os.listdir(dest)) == ["play-person-11-94.jpg", "play-person-11-94.yaml"]
    assert excludes.load_dir(str(tmp_path), ["ipcams_v34.onnx"]) == {}
