"""Tests for the per-file exclusion geometries."""
import json
import os
import textwrap

import pytest

import excludes


def write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def test_stem_encodes_camera_label_and_centre():
    # the real peach tree false positive
    box = {"left": 0.18505859, "top": 0.85449219,
           "width": 0.08007812, "height": 0.09326172}
    assert excludes.exclusion_stem("peach tree", "deer", box) == "peach_tree-deer-23-90"


def test_stem_rounds_halves_up_not_to_even():
    """round() would give 22 here; a filename scheme should not do that."""
    box = {"left": 0.20, "top": 0.40, "width": 0.05, "height": 0.05}
    assert excludes.exclusion_stem("x", "y", box) == "x-y-23-43"


def test_stem_pads_small_percentages():
    box = {"left": 0.0, "top": 0.0, "width": 0.06, "height": 0.04}
    assert excludes.exclusion_stem("deck", "cat", box) == "deck-cat-03-02"


def test_load_dir_reads_geometry_and_comment(tmp_path):
    write(tmp_path, "peach_tree-deer-23-90.yaml", """
        camera: peach tree
        label: deer
        comment: mulch bed
        box: {left: 0.1, top: 0.2, width: 0.3, height: 0.4}
    """)
    got = excludes.load_dir(str(tmp_path))
    assert got["peach tree"]["deer"][0]["left"] == 0.1
    assert got["peach tree"]["deer"][0]["comment"] == "mulch bed"


def test_disabled_exclusion_is_ignored(tmp_path):
    """Retiring an exclusion should not require deleting the evidence."""
    write(tmp_path, "a.yaml", """
        camera: deck
        label: cat
        disabled: true
        box: {left: 0.1, top: 0.1, width: 0.1, height: 0.1}
    """)
    assert excludes.load_dir(str(tmp_path)) == {}


def test_malformed_file_does_not_lose_the_others(tmp_path):
    write(tmp_path, "bad.yaml", "camera: deck\nlabel: cat\n")   # no box
    write(tmp_path, "good.yaml", """
        camera: deck
        label: dog
        box: {left: 0.1, top: 0.1, width: 0.1, height: 0.1}
    """)
    got = excludes.load_dir(str(tmp_path))
    assert list(got) == ["deck"] and "dog" in got["deck"]


def test_missing_directory_is_not_an_error(tmp_path):
    assert excludes.load_dir(str(tmp_path / "nope")) == {}


def test_merge_keeps_both_sources_and_does_not_mutate(tmp_path):
    base = {"deck": {"person": [{"left": 0.0, "top": 0.0, "width": 0.1, "height": 0.1}]}}
    extra = {"deck": {"person": [{"left": 0.5, "top": 0.5, "width": 0.1, "height": 0.1}]},
             "shed": {"deer": [{"left": 0.2, "top": 0.2, "width": 0.1, "height": 0.1}]}}
    out = excludes.merge(base, extra)
    assert len(out["deck"]["person"]) == 2
    assert "shed" in out
    assert len(base["deck"]["person"]) == 1, "merge must not mutate its input"


def test_load_merges_json_and_directory(tmp_path):
    j = tmp_path / "excludes.json"
    j.write_text(json.dumps({"deck": {"person": [
        {"left": 0.0, "top": 0.0, "width": 0.1, "height": 0.1}]}}))
    d = tmp_path / "excl"
    d.mkdir()
    write(d, "shed-deer-50-50.yaml", """
        camera: shed
        label: deer
        box: {left: 0.45, top: 0.45, width: 0.1, height: 0.1}
    """)
    out = excludes.load(str(j), str(d))
    assert "deck" in out and "shed" in out
