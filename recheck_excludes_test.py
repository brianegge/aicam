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
