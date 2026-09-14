"""Tests for the vision-model second opinion.

The cases that matter most are the ones asserting the alert still goes out:
every failure of this feature has to fall through to notifying.
"""
import configparser
import io
import time
from unittest import mock

import numpy as np
import requests
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


# --- saying why, when it could not verify ------------------------------------

class FakeResponse:
    def __init__(self, status): self.status_code = status


def http_error(status):
    e = requests.HTTPError("%d" % status)
    e.response = FakeResponse(status)
    return e


def test_running_out_of_credit_is_named_on_the_alert():
    """The 2026-09-13 outage read as ordinary 3am wildlife alerts."""
    p = pred("raccoon")
    run([p], side_effect=http_error(402))
    assert "ignore" not in p
    assert verify.unverified_note([p]) == " (unverified: no OpenRouter credit)"


def test_a_rejected_key_is_named():
    p = pred("fox")
    run([p], side_effect=http_error(401))
    assert p["unverified"] == "OpenRouter rejected the key"


def test_a_timeout_is_named():
    p = pred("fox")
    run([p], side_effect=requests.Timeout("too slow"))
    assert p["unverified"] == "model timed out"


def test_an_unusable_verdict_is_named():
    p = pred("fox")
    run([p], return_value={"label": "chupacabra", "confidence": 1.0})
    assert p["unverified"] == "model gave no usable answer"


def test_an_unexpected_error_still_produces_a_note():
    p = pred("fox")
    run([p], side_effect=RuntimeError("something odd"))
    assert p["unverified"] == "verification failed"


def test_a_successful_verdict_leaves_no_note():
    p = pred("fox")
    run([p], return_value=verdict("fox"))
    assert verify.unverified_note([p]) == ""


def test_the_note_does_not_repeat_one_reason_per_detection():
    a, b = pred("fox"), pred("deer", left=0.1)
    run([a, b], side_effect=http_error(402))
    assert verify.unverified_note([a, b]) == " (unverified: no OpenRouter credit)"


def test_distinct_reasons_are_both_reported():
    a, b = pred("fox"), pred("deer", left=0.1)
    a["unverified"] = "model timed out"
    b["unverified"] = "no OpenRouter credit"
    assert verify.unverified_note([a, b]) == (
        " (unverified: model timed out, no OpenRouter credit)")


# --- the breaker -------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_breaker():
    verify._breaker.update(until=0.0, reason="")
    yield
    verify._breaker.update(until=0.0, reason="")


def test_a_dead_key_is_only_discovered_once():
    """Twenty-two round trips to be told the same thing, one per detection."""
    cam = FakeCam()
    with mock.patch.object(verify, "ask", side_effect=http_error(402)) as asked:
        for i in range(6):
            p = pred("fox", left=0.1 * i)
            verify.verify_predictions(cam, frame(), [p], config())
            assert p["unverified"] == "no OpenRouter credit"
    assert asked.call_count == 1


def test_a_timeout_does_not_open_the_breaker():
    """One slow call says nothing about the next."""
    cam = FakeCam()
    with mock.patch.object(verify, "ask", side_effect=requests.Timeout()) as asked:
        for i in range(3):
            verify.verify_predictions(cam, frame(), [pred("fox", left=0.1 * i)], config())
    assert asked.call_count == 3


def test_the_breaker_reopens_after_its_window():
    cam = FakeCam()
    with mock.patch.object(verify, "ask", side_effect=http_error(402)) as asked:
        verify.verify_predictions(cam, frame(), [pred("fox")], config())
        verify._breaker["until"] = 0.0          # window elapsed
        verify.verify_predictions(cam, frame(), [pred("fox", left=0.3)], config())
    assert asked.call_count == 2


def test_an_animal_big_enough_to_matter_is_not_in_the_other_bucket():
    """"other" is suppressed, so any animal missing from LABELS is dropped."""
    for animal in ("bear", "coyote", "deer", "fox"):
        assert animal in verify.LABELS, animal
    assert "other" not in ("bear", "coyote")


def test_a_bear_alerts():
    p = pred("dog", score=0.7)
    run([p], return_value=verdict("bear", confidence=0.95))
    assert "ignore" not in p


# --- person is not symmetric with the other classes --------------------------

def test_a_dog_called_a_person_is_suppressed():
    """The deck dog produced person 0.64 and 0.83 on 2026-09-14."""
    p = pred("person", score=0.83)
    run([p], cfg=config(classes="deer,fox,rabbit,raccoon,coyote,cat,dog,person"),
        return_value=verdict("dog", confidence=0.99, note="dog on wooden deck"))
    assert p["ignore"] == "verified: dog"


def test_an_unmistakable_nothing_silences_a_person():
    """deck 17:05: person 0.57 over a chair handle, "nothing" at 0.99."""
    p = pred("person", score=0.57)
    run([p], cfg=config(classes="person"),
        return_value=verdict("nothing", confidence=0.99, note="plastic equipment handle"))
    assert p["ignore"] == "verified: nothing"


def test_a_merely_probable_nothing_still_alerts_a_person():
    """Most person "nothing" verdicts land at 0.70-0.80 and are not enough.

    The same 0.72 would silence a deer. A person is held higher because the
    cost of being wrong is not symmetric.
    """
    p = pred("person", score=0.8)
    run([p], cfg=config(classes="person"),
        return_value=verdict("nothing", confidence=0.72, note="only shadow and pavement"))
    assert "ignore" not in p


def test_that_same_confidence_does_silence_a_deer():
    p = pred("deer", score=0.8)
    run([p], return_value=verdict("nothing", confidence=0.92))
    assert p["ignore"] == "verified: nothing"


def test_other_never_silences_a_person():
    p = pred("person", score=0.7)
    run([p], cfg=config(classes="person"), return_value=verdict("other", confidence=0.99))
    assert "ignore" not in p


def test_a_confirmed_person_alerts():
    p = pred("person", score=0.7)
    run([p], cfg=config(classes="person"), return_value=verdict("person", confidence=0.99))
    assert "ignore" not in p


def test_a_balance_bike_called_a_person_is_suppressed():
    """The gate already called one a vehicle rather than a dog."""
    p = pred("person", score=0.7)
    run([p], cfg=config(classes="person"),
        return_value=verdict("vehicle", confidence=0.9, note="toddler balance bike"))
    assert p["ignore"] == "verified: vehicle"


def test_nothing_still_silences_a_deer():
    """The restriction applies to person, not to everything."""
    p = pred("deer", score=0.7)
    run([p], return_value=verdict("nothing", confidence=0.99))
    assert p["ignore"] == "verified: nothing"


def test_the_person_rule_is_configurable():
    p = pred("person", score=0.7)
    run([p], cfg=config(classes="person", suppress_person="nothing"),
        return_value=verdict("nothing", confidence=0.99))
    assert p["ignore"] == "verified: nothing"


# --- correcting the label rather than silencing the alert --------------------

def test_relabelling_renames_rather_than_silencing_when_enabled():
    """garage-l, 2026-09-14: deer 0.77 -> "dog 0.98, dog walking on pavement".

    Opt-in: enabling it by default overrode a correct detector within the
    hour. Both verdicts go in the alert text instead -- see
    detect.label_with_confidence.
    """
    p = pred("deer", score=0.77)
    run([p], cfg=config(relabel="true"),
        return_value=verdict("dog", confidence=0.98, note="dog walking on pavement"))
    assert "ignore" not in p
    assert p["tagName"] == "dog"
    assert p["relabelled_from"] == "deer"


def test_suppression_takes_precedence_over_relabelling():
    """squirrel is not an aicam class, so it silences rather than renames."""
    p = pred("fox", score=0.8)
    run([p], return_value=verdict("squirrel", confidence=0.95))
    assert p["ignore"] == "verified: squirrel"
    assert p["tagName"] == "fox"


def test_a_confirmed_class_is_left_alone():
    p = pred("deer", score=0.8)
    run([p], return_value=verdict("deer", confidence=0.99))
    assert p["tagName"] == "deer" and "relabelled_from" not in p


def test_an_unsure_verdict_does_not_relabel():
    """Renaming on a guess is worse than the detector's own guess."""
    p = pred("deer", score=0.8)
    run([p], cfg=config(relabel="true"), return_value=verdict("dog", confidence=0.70))
    assert p["tagName"] == "deer" and "relabelled_from" not in p


def test_a_verdict_aicam_cannot_report_does_not_relabel():
    """'other' is not a class; leave the detector's word and alert."""
    p = pred("deer", score=0.8)
    run([p], cfg=config(suppress="squirrel,bird,nothing", relabel="true"),
        return_value=verdict("other", confidence=0.99))
    assert p["tagName"] == "deer" and "ignore" not in p


def test_relabelling_is_off_unless_asked_for():
    """Default is to report both verdicts, not to prefer the model's."""
    p = pred("deer", score=0.77)
    run([p], return_value=verdict("dog", confidence=0.98))
    assert p["tagName"] == "deer" and "ignore" not in p
    assert p["verified"]["label"] == "dog"


def test_a_person_misread_as_deer_is_relabelled_not_lost():
    """The direction that matters most: never quieter than the detector was."""
    p = pred("deer", score=0.6)
    run([p], cfg=config(relabel="true"), return_value=verdict("person", confidence=0.97))
    assert "ignore" not in p and p["tagName"] == "person"


# --- a wild animal beside a person is a pet ---------------------------------

def test_a_deer_beside_a_person_is_the_dog():
    """west_lawn 16:29: detector deer 0.75, model dog 0.98, child in frame."""
    deer = pred("deer", score=0.75)
    person = pred("person", score=0.67, left=0.556)
    run([deer, person], return_value=verdict("dog", confidence=0.98))
    assert deer["tagName"] == "dog"
    assert deer["relabelled_from"] == "deer"
    assert deer["prior"] == "person in frame"
    assert "ignore" not in deer


def test_a_dog_the_model_calls_a_deer_stays_a_dog_beside_a_person():
    """play 15:19: detector dog 0.79, model deer 0.93, child in frame.

    The prior decides it without trusting either model over the other --
    which is the point, because each was right once today and wrong once.
    """
    dog = pred("dog", score=0.79)
    person = pred("person", score=0.87, left=0.100)
    run([dog, person], cfg=config(classes="dog,deer"),
        return_value=verdict("deer", confidence=0.93))
    assert dog["tagName"] == "dog" and "ignore" not in dog
    assert dog["prior"] == "person in frame"


def test_no_person_means_the_prior_does_not_apply():
    """A deer alone in the garden is just a deer."""
    deer = pred("deer", score=0.75)
    run([deer], return_value=verdict("deer", confidence=0.95))
    assert deer["tagName"] == "deer" and "prior" not in deer


def test_a_low_confidence_person_does_not_trigger_it():
    deer = pred("deer", score=0.75)
    person = pred("person", score=0.30, left=0.556)
    run([deer, person], return_value=verdict("deer", confidence=0.95))
    assert deer["tagName"] == "deer" and "prior" not in deer


def test_an_ignored_person_does_not_count():
    """A person suppressed as road traffic is not in the garden."""
    deer = pred("deer", score=0.75)
    person = pred("person", score=0.9, left=0.556)
    person["ignore"] = "road"
    run([deer, person], return_value=verdict("deer", confidence=0.95))
    assert deer["tagName"] == "deer" and "prior" not in deer


def test_the_prior_does_not_touch_unrelated_pairs():
    """rabbit vs nothing is not about people; normal handling applies."""
    rabbit = pred("rabbit", score=0.6)
    person = pred("person", score=0.9, left=0.556)
    run([rabbit, person], return_value=verdict("nothing", confidence=0.99))
    assert rabbit["ignore"] == "verified: nothing"


def test_the_prior_can_be_turned_off():
    deer = pred("deer", score=0.75)
    person = pred("person", score=0.87, left=0.556)
    run([deer, person], cfg=config(people_imply_pets="false"),
        return_value=verdict("dog", confidence=0.98))
    assert deer["tagName"] == "deer"
