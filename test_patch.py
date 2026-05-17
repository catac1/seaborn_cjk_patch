"""
Tests for seaborn_cjk.

Run with:  pytest tests/
"""
import pytest
import matplotlib as mpl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sans_serif():
    return list(mpl.rcParams.get("font.sans-serif", []))


# ---------------------------------------------------------------------------
# detect / list
# ---------------------------------------------------------------------------

def test_list_cjk_fonts_returns_list():
    from seaborn_cjk import list_cjk_fonts
    result = list_cjk_fonts()
    assert isinstance(result, list)
    # Every entry should be a string font name
    for name in result:
        assert isinstance(name, str)


def test_detect_cjk_font_returns_str_or_none():
    from seaborn_cjk import detect_cjk_font
    result = detect_cjk_font()
    assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# patch / unpatch
# ---------------------------------------------------------------------------

def test_patch_returns_none_or_str():
    from seaborn_cjk import patch, unpatch
    result = patch()
    unpatch()
    assert result is None or isinstance(result, str)


def test_patch_prepends_cjk_font():
    from seaborn_cjk import list_cjk_fonts, patch, unpatch
    fonts = list_cjk_fonts()
    if not fonts:
        pytest.skip("No CJK font available on this system")

    patch()
    first = _get_sans_serif()[0]
    unpatch()
    assert first in fonts


def test_patch_sets_unicode_minus_false():
    from seaborn_cjk import list_cjk_fonts, patch, unpatch
    if not list_cjk_fonts():
        pytest.skip("No CJK font available on this system")

    patch()
    val = mpl.rcParams.get("axes.unicode_minus")
    unpatch()
    assert val is False


def test_unpatch_restores_rcparams():
    from seaborn_cjk import list_cjk_fonts, patch, unpatch
    if not list_cjk_fonts():
        pytest.skip("No CJK font available on this system")

    original_sans = _get_sans_serif()
    original_minus = mpl.rcParams.get("axes.unicode_minus")

    patch()
    unpatch()

    assert _get_sans_serif() == original_sans
    assert mpl.rcParams.get("axes.unicode_minus") == original_minus


def test_patch_invalid_font_raises():
    from seaborn_cjk import patch, unpatch
    with pytest.raises(ValueError, match="not available"):
        patch("ThisFontDefinitelyDoesNotExist_XYZ_12345")
    unpatch()  # no-op, but safe


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------

def test_cjk_context_restores_on_exit():
    from seaborn_cjk import cjk_context, list_cjk_fonts
    if not list_cjk_fonts():
        pytest.skip("No CJK font available on this system")

    original_sans = _get_sans_serif()

    with cjk_context():
        pass  # patch applied inside

    assert _get_sans_serif() == original_sans


def test_cjk_context_restores_on_exception():
    from seaborn_cjk import cjk_context, list_cjk_fonts
    if not list_cjk_fonts():
        pytest.skip("No CJK font available on this system")

    original_sans = _get_sans_serif()

    with pytest.raises(RuntimeError):
        with cjk_context():
            raise RuntimeError("deliberate error")

    assert _get_sans_serif() == original_sans


# ---------------------------------------------------------------------------
# Seaborn monkey-patch
# ---------------------------------------------------------------------------

def test_seaborn_set_theme_keeps_cjk_font():
    """After sns.set_theme() the CJK font should still be first."""
    pytest.importorskip("seaborn")
    from seaborn_cjk import list_cjk_fonts, patch, unpatch
    if not list_cjk_fonts():
        pytest.skip("No CJK font available on this system")

    import seaborn as sns

    font = patch()
    sns.set_theme()   # normally wipes out our font
    first = _get_sans_serif()[0]
    unpatch()

    assert first == font, (
        f"Expected '{font}' to be first in font.sans-serif after sns.set_theme(), "
        f"but got '{first}'"
    )


def test_seaborn_set_style_keeps_cjk_font():
    pytest.importorskip("seaborn")
    from seaborn_cjk import list_cjk_fonts, patch, unpatch
    if not list_cjk_fonts():
        pytest.skip("No CJK font available on this system")

    import seaborn as sns

    font = patch()
    sns.set_style("whitegrid")
    first = _get_sans_serif()[0]
    unpatch()

    assert first == font


# ---------------------------------------------------------------------------
# Localized name resolution
# ---------------------------------------------------------------------------

def test_list_cjk_fonts_localized_returns_list():
    from seaborn_cjk import list_cjk_fonts
    eng = list_cjk_fonts(localized=False)
    loc = list_cjk_fonts(localized=True)
    assert isinstance(loc, list)
    assert len(loc) == len(eng), "localized and english lists should have the same length"


def test_list_cjk_fonts_detailed_structure():
    from seaborn_cjk import list_cjk_fonts_detailed
    for entry in list_cjk_fonts_detailed():
        assert "english" in entry
        assert "localized" in entry
        assert isinstance(entry["localized"], dict)


def test_patch_with_localized_name_resolves_to_english():
    """patch('文泉驿正黑') should resolve and apply 'WenQuanYi Zen Hei'."""
    from seaborn_cjk import list_cjk_fonts_detailed, patch, unpatch
    # Find a font that actually has a localized name on this system
    details = list_cjk_fonts_detailed()
    target = next((d for d in details if d["localized"]), None)
    if target is None:
        pytest.skip("No fonts with localized names available on this system")

    # Pick any localized name variant
    loc_name = next(iter(target["localized"].values()))
    result = patch(loc_name)
    unpatch()

    assert result == target["english"], (
        f"patch({loc_name!r}) should return english name {target['english']!r}, got {result!r}"
    )


def test_patch_localized_name_actually_sets_rcparams():
    from seaborn_cjk import list_cjk_fonts_detailed, patch, unpatch
    import matplotlib as mpl
    details = list_cjk_fonts_detailed()
    target = next((d for d in details if d["localized"]), None)
    if target is None:
        pytest.skip("No fonts with localized names available on this system")

    loc_name = next(iter(target["localized"].values()))
    patch(loc_name)
    first = mpl.rcParams["font.sans-serif"][0]
    unpatch()

    assert first == target["english"]


def test_patch_unknown_localized_name_raises():
    from seaborn_cjk import patch, unpatch
    with pytest.raises(ValueError, match="not available"):
        patch("없는글꼴_XYZZY_does_not_exist")
    unpatch()
