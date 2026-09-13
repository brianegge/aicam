"""Tests for the vision-model second opinion.

The cases that matter most are the ones asserting the alert still goes out:
every failure of this feature has to fall through to notifying.
"""
import configparser
import io
import time
from unittest import mock

import numpy as np
import pytest
from PIL import Image

import verify


def config(**over):
    c = configparser.ConfigParser()
    c["verify"] = dict({
        "enabled": "true",
        "api-key": "sk-test",
        "model": "test/model",
        "classes": "deer,fox,rabbit,raccoon,coyote,cat",
        "min-confidence": "0.6",
    }, **{k.replace("_", "-"): v for k, v in over.items()})
    return c


class FakeCam:
    def __init__(self, name="tree line"):
        self.name = name
        self.prev_predictions = {}


def pred(tag="fox", left=0.5, top=0.4, width=0.05, height=0.06, score=0.8):
    box = {"left": left, "top": top, "width": width, "height": height}
    return {"tagName": tag, "boundingBox": box, "probability": score,
            "center": {"x": left + width / 2, "y": top + height / 2}}


def frame(w=3840, h=2160):
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def clear_cache():
    verify._suppressed.clear()
    yield
    verify._suppressed.clear()


def verdict(label, confidence=0.95, note="a squirrel"):
    return {"label": label, "confidence": confidence, "note": note, "cost": 0.0002}


def run(preds, cfg=None, **ask):
    cam = FakeCam()
    with mock.patch.object(verify, "ask", **ask) as asked:
        verify.verify_predictions(cam, frame(), preds, cfg or config())
    return asked


# --- suppressing and keeping -------------------------------------------------

def test_a_squirrel_called_a_fox_is_suppressed():
    p = pred("fox")
    run([p], return_value=verdict("squirrel"))
    assert p["ignore"] == "verified: squirrel"


def test_a_confirmed_fox_still_alerts():
    p = pred("fox")
    run([p], return_value=verdict("fox"))
    assert "ignore" not in p


def test_scenery_is_suppressed():
    """'nothing' covers the twig, the stone and the paint patch."""
    p = pred("rabbit")
    run([p], return_value=verdict("nothing", note="bare twig"))
    assert p["ignore"] == "verified: nothing"


def test_a_different_real_animal_still_alerts():
    """The question is whether it is worth waking up for, not who was right."""
    p = pred("fox")
    run([p], return_value=verdict("coyote"))
    assert "ignore" not in p


def test_the_verdict_is_kept_on_the_prediction():
    p = pred("fox")
    run([p], return_value=verdict("squirrel"))
    assert p["verified"]["label"] == "squirrel"


# --- failing open ------------------------------------------------------------

def test_an_unsure_model_does_not_silence_the_alert():
    p = pred("fox")
    run([p], return_value=verdict("squirrel", confidence=0.4))
    assert "ignore" not in p


def test_an_exception_does_not_silence_the_alert():
    p = pred("fox")
    run([p], side_effect=RuntimeError("timeout"))
    assert "ignore" not in p


def test_an_unparseable_verdict_does_not_silence_the_alert():
    p = pred("fox")
    run([p], return_value={"label": "chupacabra", "confidence": 1.0})
    assert "ignore" not in p


def test_no_verdict_at_all_does_not_silence_the_alert():
    p = pred("fox")
    run([p], return_value=None)
    assert "ignore" not in p


def test_a_missing_key_disables_the_feature_silently():
    p = pred("fox")
    asked = run([p], cfg=config(api_key=""), return_value=verdict("squirrel"))
    assert "ignore" not in p and not asked.called


def test_disabled_means_disabled():
    p = pred("fox")
    asked = run([p], cfg=config(enabled="false"), return_value=verdict("squirrel"))
    assert "ignore" not in p and not asked.called


def test_no_verify_section_is_not_an_error():
    p = pred("fox")
    verify.verify_predictions(FakeCam(), frame(), [p], configparser.ConfigParser())
    assert "ignore" not in p


# --- what gets asked ---------------------------------------------------------

def test_only_the_configured_classes_are_asked_about():
    person, fox = pred("person"), pred("fox", left=0.1)
    asked = run([person, fox], return_value=verdict("fox"))
    assert asked.call_count == 1
    assert asked.call_args[0][1] == "fox"


def test_an_already_excluded_detection_is_not_asked_about():
    """excludes/ is free and instant; it should run first and win."""
    p = pred("rabbit")
    p["ignore"] = "static"
    asked = run([p], return_value=verdict("nothing"))
    assert not asked.called


def test_a_held_detection_is_not_asked_about():
    p = pred("fox")
    p["hold_only"] = True
    asked = run([p], return_value=verdict("squirrel"))
    assert not asked.called


def test_an_object_already_being_tracked_is_not_asked_again():
    """Once per arrival, not once per sweep."""
    cam = FakeCam()
    p = pred("fox")
    cam.prev_predictions["fox"] = [{"boundingBox": dict(p["boundingBox"])}]
    with mock.patch.object(verify, "ask", return_value=verdict("squirrel")) as asked:
        verify.verify_predictions(cam, frame(), [p], config())
    assert not asked.called


# --- the cache ---------------------------------------------------------------

def test_the_same_scenery_is_only_paid_for_once():
    """The garage stone re-acquired seventeen times in one evening."""
    cam = FakeCam()
    with mock.patch.object(verify, "ask", return_value=verdict("nothing")) as asked:
        for _ in range(17):
            p = pred("rabbit")
            verify.verify_predictions(cam, frame(), [p], config())
            assert p["ignore"] == "verified: nothing"
    assert asked.call_count == 1


def test_something_somewhere_else_is_asked_about_separately():
    cam = FakeCam()
    with mock.patch.object(verify, "ask", return_value=verdict("nothing")) as asked:
        verify.verify_predictions(cam, frame(), [pred("rabbit", left=0.1)], config())
        verify.verify_predictions(cam, frame(), [pred("rabbit", left=0.7)], config())
    assert asked.call_count == 2


def test_a_confirmation_is_not_cached():
    """A real animal moves, so the next frame is a different question."""
    cam = FakeCam()
    with mock.patch.object(verify, "ask", return_value=verdict("fox")) as asked:
        for _ in range(3):
            verify.verify_predictions(cam, frame(), [pred("fox")], config())
    assert asked.call_count == 3


def test_the_cache_expires():
    cam = FakeCam()
    with mock.patch.object(verify, "ask", return_value=verdict("nothing")) as asked:
        verify.verify_predictions(cam, frame(), [pred("rabbit")],
                                  config(cache_minutes="0"))
        time.sleep(0.01)
        verify.verify_predictions(cam, frame(), [pred("rabbit")],
                                  config(cache_minutes="0"))
    assert asked.call_count == 2


def test_cameras_do_not_share_a_cache():
    a, b = FakeCam("tree line"), FakeCam("garage-l")
    with mock.patch.object(verify, "ask", return_value=verdict("nothing")) as asked:
        verify.verify_predictions(a, frame(), [pred("rabbit")], config())
        verify.verify_predictions(b, frame(), [pred("rabbit")], config())
    assert asked.call_count == 2


# --- the crop ----------------------------------------------------------------

def opened(jpeg):
    return Image.open(io.BytesIO(jpeg))


def test_the_crop_carries_context_around_the_box():
    box = {"left": 0.5, "top": 0.5, "width": 0.01, "height": 0.01}
    jpeg = verify._crop(frame(), box, context=3.0, max_edge=768)
    im = opened(jpeg)
    # 1% of 3840 is ~38px; three times that is ~115, scaled up to the long edge.
    assert max(im.size) == 768


def test_the_crop_is_marked_so_the_model_knows_which_thing():
    box = {"left": 0.4, "top": 0.4, "width": 0.2, "height": 0.2}
    im = opened(verify._crop(frame(), box, context=3.0, max_edge=400))
    px = list(im.convert("RGB").get_flattened_data())
    reds = [c for c in px if c[0] > 150 and c[1] < 90]
    assert reds, "no red box drawn on the crop"


def test_the_crop_does_not_run_off_the_frame():
    """Animals at the frame edge are common; PIL would silently pad."""
    box = {"left": 0.0, "top": 0.0, "width": 0.02, "height": 0.02}
    im = opened(verify._crop(frame(640, 480), box, context=8.0, max_edge=256))
    assert im.size[0] > 0 and im.size[1] > 0


def test_a_tiny_box_is_enlarged_rather_than_sent_at_46_pixels():
    box = {"left": 0.5, "top": 0.5, "width": 0.012, "height": 0.02}
    im = opened(verify._crop(frame(), box, context=3.0, max_edge=768))
    assert max(im.size) == 768


def test_a_pil_frame_works_as_well_as_an_array():
    box = {"left": 0.4, "top": 0.4, "width": 0.1, "height": 0.1}
    pil = Image.new("RGB", (1920, 1080))
    assert verify._crop(pil, box, context=2.0, max_edge=300)


# --- not overriding a confident detector -------------------------------------

def test_a_confident_detection_is_still_second_guessed():
    """Six of seven confirmed deer false positives scored 0.92-0.97."""
    p = pred("deer", score=0.97)
    asked = run([p], return_value=verdict("nothing", note="tree trunk and ground"))
    assert asked.called and p["ignore"] == "verified: nothing"


def test_a_ceiling_can_be_configured_for_those_who_want_it():
    p = pred("deer", score=0.92)
    asked = run([p], cfg=config(max_score="0.90"), return_value=verdict("nothing"))
    assert "ignore" not in p and not asked.called


def test_nothing_needs_more_confidence_than_a_positive_identification():
    """Absence of evidence from a crop the model may have failed to read."""
    p = pred("deer", score=0.7)
    run([p], return_value=verdict("nothing", confidence=0.85))
    assert "ignore" not in p


def test_a_positive_identification_at_the_same_confidence_does_suppress():
    p = pred("fox", score=0.7)
    run([p], return_value=verdict("squirrel", confidence=0.85))
    assert p["ignore"] == "verified: squirrel"
