from codeproject import is_plausible_plate


def test_accepts_real_plates():
    # Read off the box truck by three cameras on 2026-09-02.
    for plate in ("BW41507", "B41507", "41507", "8W41507", "AB 1234", "1-ABC23"):
        assert is_plausible_plate(plate), plate


def test_rejects_badge_lettering():
    """ALPR scored the MITSUBISHI FUSO grille badge 0.957 as a plate."""
    assert not is_plausible_plate("ITSUBISHIFUS")
    assert not is_plausible_plate("MITSUBISHI")


def test_rejects_letters_only():
    """A bumper reflection came back as 'REC' — plates carry a digit."""
    assert not is_plausible_plate("REC")
    assert not is_plausible_plate("FUSO")


def test_rejects_empty_and_junk():
    assert not is_plausible_plate("")
    assert not is_plausible_plate("1")
    assert not is_plausible_plate("!!@@##")
