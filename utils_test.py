#!/usr/bin/env python3
from PIL import ImageColor


def test_colors():
    for color in ["grey", "red"]:
        rgba = ImageColor.getrgb(color) + (128,)
        assert rgba is not None
