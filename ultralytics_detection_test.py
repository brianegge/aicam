"""Decoding tests for the Ultralytics output layout.

Built on synthetic tensors so they assert the layout contract -- boxes as
cx/cy/w/h in input pixels, class score as the confidence -- without needing a
trained model checked in.
"""

import numpy as np
import pytest

pytest.importorskip("PIL")

from ultralytics_detection import (
    CONF_THRESHOLD,
    ONNXRuntimeUltralyticsObjectDetection,
)

LABELS = ["package", "vehicle"]
SIZE = 608


class _Decoder(ONNXRuntimeUltralyticsObjectDetection):
    """Just the decoding half -- no onnxruntime session."""

    def __init__(self):
        self.labels = LABELS
        self.model_width = SIZE
        self.model_height = SIZE
        self.channels = 3
        self.max_detections = 20
        self.conf_threshold = CONF_THRESHOLD
        self.nms_threshold = 0.6


def _raw(boxes):
    """boxes: list of (cx, cy, w, h, package_score, vehicle_score) in pixels."""
    out = np.zeros((1, 4 + len(LABELS), len(boxes)), dtype=np.float32)
    for i, b in enumerate(boxes):
        out[0, :, i] = b
    return [out]


def test_decodes_centre_box_to_normalised_corner():
    # a 152x152 box centred at (304, 304) is the middle quarter of the frame
    preds = _Decoder().postprocess(_raw([(304, 304, 152, 152, 0.0, 0.9)]))
    assert len(preds) == 1
    p = preds[0]
    assert p["tagName"] == "vehicle"
    assert p["probability"] == pytest.approx(0.9, abs=1e-6)
    b = p["boundingBox"]
    assert b["left"] == pytest.approx(0.375, abs=1e-6)
    assert b["top"] == pytest.approx(0.375, abs=1e-6)
    assert b["width"] == pytest.approx(0.25, abs=1e-6)
    assert b["height"] == pytest.approx(0.25, abs=1e-6)


def test_class_score_selects_the_label():
    preds = _Decoder().postprocess(_raw([(304, 304, 100, 100, 0.8, 0.1)]))
    assert [p["tagName"] for p in preds] == ["package"]


def test_low_confidence_is_dropped():
    assert _Decoder().postprocess(_raw([(304, 304, 100, 100, 0.0, 0.05)])) == []


def test_floor_is_low_enough_to_surface_a_held_object():
    """The parked car at peach tree read 0.155 in a weak frame. Below the old
    0.4 floor nothing was emitted, so detect.py's hold had nothing to hold."""
    preds = _Decoder().postprocess(_raw([(304, 304, 100, 100, 0.0, 0.155)]))
    assert [p["tagName"] for p in preds] == ["vehicle"]


def test_floor_stays_below_the_new_object_thresholds():
    """The floor must not become the reporting threshold -- detect.py still
    requires 0.70 by day and 0.90 while dark for a new object."""
    assert CONF_THRESHOLD < 0.70


def test_overlapping_boxes_of_one_class_are_suppressed():
    dup = (304, 304, 152, 152, 0.0, 0.9)
    nearly = (306, 306, 152, 152, 0.0, 0.8)
    preds = _Decoder().postprocess(_raw([dup, nearly]))
    assert len(preds) == 1
    assert preds[0]["probability"] == pytest.approx(0.9, abs=1e-6)


def test_different_classes_are_not_suppressed_against_each_other():
    """A package sitting on a vehicle must not delete one of them."""
    preds = _Decoder().postprocess(
        _raw([(304, 304, 152, 152, 0.0, 0.9), (304, 304, 152, 152, 0.85, 0.0)])
    )
    assert sorted(p["tagName"] for p in preds) == ["package", "vehicle"]


def test_boxes_are_clipped_to_the_frame():
    # centred on the top-left corner, so half the box is outside
    preds = _Decoder().postprocess(_raw([(0, 0, 200, 200, 0.0, 0.9)]))
    b = preds[0]["boundingBox"]
    assert b["left"] == 0.0 and b["top"] == 0.0
    assert 0.0 < b["width"] <= 1.0 and 0.0 < b["height"] <= 1.0


def test_results_are_ordered_by_confidence():
    preds = _Decoder().postprocess(
        _raw([(100, 100, 50, 50, 0.0, 0.5), (400, 400, 50, 50, 0.0, 0.95)])
    )
    assert [p["probability"] for p in preds] == sorted(
        [p["probability"] for p in preds], reverse=True
    )


def test_wrong_class_count_is_caught():
    """Pointing the config at a model trained on other classes must not
    silently mislabel every detection."""
    d = _Decoder()
    three_class = np.zeros((1, 4 + 3, 1), dtype=np.float32)
    with pytest.raises(AssertionError):
        d.postprocess([three_class])
