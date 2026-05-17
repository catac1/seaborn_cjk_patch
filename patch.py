"""
Core patching logic for seaborn_cjk.

Strategy
--------
Two layers of defence are applied:

1. **rcParams patch** – prepend the best available CJK font to
   ``font.sans-serif`` and set ``axes.unicode_minus = False``.

2. **Seaborn monkey-patch** – wrap ``seaborn.set_theme``, ``seaborn.set_style``,
   and ``seaborn.set`` so that every call to those functions re-applies layer 1
   afterwards.  This handles the common case where user code calls
   ``sns.set_theme()`` *after* importing seaborn_cjk.

Localized font names
--------------------
Font files embed localized family names in their SFNT name table (name_id=1,
platform=3/Windows/Unicode).  This module scans that table so users can
refer to fonts by their native script name — e.g. "굴림" instead of "Gulim",
"文泉驿正黑" instead of "WenQuanYi Zen Hei" — and the right font file will
still be selected.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib import ft2font

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows LCID -> BCP-47-style tag for CJK languages we care about
# ---------------------------------------------------------------------------
_CJK_LANG_IDS: Dict[int, str] = {
    0x0404: "zh-TW",   # Traditional Chinese - Taiwan
    0x0804: "zh-CN",   # Simplified Chinese - PRC
    0x0C04: "zh-HK",   # Chinese - Hong Kong
    0x1004: "zh-SG",   # Chinese - Singapore
    0x1404: "zh-MO",   # Chinese - Macao
    0x0411: "ja",      # Japanese
    0x0412: "ko",      # Korean
}

# ---------------------------------------------------------------------------
# Ranked list of well-known CJK font *English* names.
# ---------------------------------------------------------------------------
_CJK_CANDIDATES: List[str] = [
    # Noto (cross-platform, open source) - most reliable
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    "Noto Serif CJK SC",
    "Noto Serif CJK TC",
    "Noto Serif CJK JP",
    # Windows
    "Microsoft YaHei",       # simplified chinese
    "Microsoft JhengHei",    # traditional chinese
    "MS Gothic",             # japanese
    "MS Mincho",             # japanese serif
    "Meiryo",                # japanese
    "Malgun Gothic",         # korean
    "Gulim",                 # korean
    "Batang",                # korean serif
    "Dotum",                 # korean
    "SimHei",
    "SimSun",
    "FangSong",
    "KaiTi",
    # macOS
    "PingFang SC",
    "PingFang TC",
    "Hiragino Sans",
    "Hiragino Kaku Gothic Pro",
    "Hiragino Mincho Pro",
    "STHeiti",
    "STSong",
    "STFangsong",
    "Apple LiGothic",
    # Linux common
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
    "Source Han Sans SC",
    "Source Han Serif SC",
    "IPAGothic",
    "IPAPGothic",
    # Broad-coverage fallback
    "Arial Unicode MS",
]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_original_rcparams: dict = {}
_patched_seaborn: bool = False
_active_cjk_font: Optional[str] = None

# Lazily built indexes
_localized_to_english: Optional[Dict[str, str]] = None
_english_to_localized: Optional[Dict[str, Dict[str, str]]] = None


# ---------------------------------------------------------------------------
# SFNT / localized-name scanning
# ---------------------------------------------------------------------------

def _build_localized_index() -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Scan every font known to matplotlib's FontManager and extract localized
    family names from the SFNT name table.

    Returns
    -------
    localized_to_english : dict
        Maps each localized name to the English family name matplotlib uses.
        e.g. {"굴림": "Gulim", "文泉驿正黑": "WenQuanYi Zen Hei"}.

    english_to_localized : dict
        Maps each English name to {lang_tag: localized_name}.
        e.g. {"Gulim": {"ko": "굴림"}}.
    """
    loc_to_eng: Dict[str, str] = {}
    eng_to_loc: Dict[str, Dict[str, str]] = {}
    seen_paths: set = set()

    for entry in fm.fontManager.ttflist:
        if entry.fname in seen_paths:
            continue
        seen_paths.add(entry.fname)
        eng_name = entry.name

        try:
            font_obj = ft2font.FT2Font(entry.fname)
            sfnt = font_obj.get_sfnt()
        except Exception:
            continue

        # Collect name_id=1 (family) and name_id=16 (preferred family) per lang.
        # name_id=16 gives the clean base family name without weight suffixes
        # (e.g. "Noto Sans CJK JP" not "Noto Sans CJK JP Thin"), so we
        # prefer it when available.
        per_lang: Dict[int, Dict[int, str]] = {}  # lang -> {name_id -> decoded}
        for (plat, _enc, lang, name_id), raw in sfnt.items():
            if plat != 3 or name_id not in (1, 16) or lang not in _CJK_LANG_IDS:
                continue
            try:
                decoded = raw.decode("utf-16-be").strip()
                if decoded:
                    per_lang.setdefault(lang, {})[name_id] = decoded
            except Exception:
                continue

        for lang, names in per_lang.items():
            # Prefer name_id=16; fall back to name_id=1
            loc_name = names.get(16) or names.get(1)
            if not loc_name or loc_name == eng_name:
                continue
            lang_tag = _CJK_LANG_IDS[lang]
            loc_to_eng[loc_name] = eng_name
            eng_to_loc.setdefault(eng_name, {})[lang_tag] = loc_name

    return loc_to_eng, eng_to_loc


def _ensure_index() -> None:
    global _localized_to_english, _english_to_localized
    if _localized_to_english is None:
        _localized_to_english, _english_to_localized = _build_localized_index()


def _resolve_font_name(name: str) -> str:
    """
    Given either an English or localized font name, return the English name
    that matplotlib's font registry understands.

    Raises ValueError if the name cannot be resolved.
    """
    _ensure_index()

    # 1. Already a known English name in the registry?
    available_english = {f.name for f in fm.fontManager.ttflist}
    if name in available_english:
        return name

    # 2. A known localized name?
    if name in _localized_to_english:
        resolved = _localized_to_english[name]
        logger.debug("seaborn_cjk: resolved localized name %r -> %r", name, resolved)
        return resolved

    raise ValueError(
        f"Font '{name}' is not available in matplotlib's font registry "
        f"(checked both English and localized names).\n"
        f"Available CJK fonts: {list_cjk_fonts() or ['(none found)']}\n"
        f"Tip: call seaborn_cjk.list_cjk_fonts(localized=True) to see native names."
    )


# ---------------------------------------------------------------------------
# Public font discovery API
# ---------------------------------------------------------------------------

def list_cjk_fonts(localized: bool = False) -> List[str]:
    """
    Return all CJK fonts found in matplotlib's font registry.

    Parameters
    ----------
    localized : bool, default False
        If True, return the localized (native-script) name where one exists,
        falling back to the English name. If False, always return English names.

    Examples
    --------
    >>> seaborn_cjk.list_cjk_fonts()
    ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'IPAGothic']

    >>> seaborn_cjk.list_cjk_fonts(localized=True)
    ['Noto Sans CJK SC', '文泉驛正黑', 'IPAゴシック']
    """
    _ensure_index()
    available = {f.name for f in fm.fontManager.ttflist}
    found_english = [n for n in _CJK_CANDIDATES if n in available]

    if not localized:
        return found_english

    lang_priority = ["ja", "ko", "zh-CN", "zh-TW", "zh-HK", "zh-SG", "zh-MO"]
    result = []
    for eng in found_english:
        loc_variants = _english_to_localized.get(eng, {})
        chosen = next((loc_variants[l] for l in lang_priority if l in loc_variants), None)
        result.append(chosen if chosen else eng)
    return result


def list_cjk_fonts_detailed() -> List[Dict]:
    """
    Return a detailed list of available CJK fonts with all their names.

    Each entry is a dict with:
      - ``english``:   name matplotlib uses internally
      - ``localized``: dict of {lang_tag: localized_name}

    Example
    -------
    >>> seaborn_cjk.list_cjk_fonts_detailed()
    [
      {'english': 'IPAGothic',        'localized': {'ja': 'IPAゴシック'}},
      {'english': 'WenQuanYi Zen Hei','localized': {'zh-CN': '文泉驿正黑', ...}},
    ]
    """
    _ensure_index()
    available = {f.name for f in fm.fontManager.ttflist}
    return [
        {"english": name, "localized": _english_to_localized.get(name, {})}
        for name in _CJK_CANDIDATES
        if name in available
    ]


def detect_cjk_font() -> Optional[str]:
    """Return the English name of the best available CJK font, or None."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            logger.debug("seaborn_cjk: detected CJK font '%s'", name)
            return name
    logger.warning(
        "seaborn_cjk: no known CJK font found. "
        "Install one (e.g. `apt install fonts-noto-cjk`) "
        "or pass a font name explicitly to patch()."
    )
    return None


# ---------------------------------------------------------------------------
# rcParams helpers
# ---------------------------------------------------------------------------

def _apply_rcparams(english_font_name: str) -> None:
    current = list(mpl.rcParams.get("font.sans-serif", []))
    cleaned = [f for f in current if f != english_font_name]
    mpl.rcParams["font.sans-serif"] = [english_font_name] + cleaned
    mpl.rcParams["axes.unicode_minus"] = False
    logger.debug("seaborn_cjk: font.sans-serif[0] = %r", english_font_name)


def _save_rcparams() -> None:
    global _original_rcparams
    _original_rcparams = {
        "font.sans-serif": list(mpl.rcParams.get("font.sans-serif", [])),
        "axes.unicode_minus": mpl.rcParams.get("axes.unicode_minus", True),
    }


def _restore_rcparams() -> None:
    for key, val in _original_rcparams.items():
        mpl.rcParams[key] = val


# ---------------------------------------------------------------------------
# Seaborn monkey-patching
# ---------------------------------------------------------------------------

def _patch_seaborn(english_font_name: str) -> None:
    global _patched_seaborn
    try:
        import seaborn as sns
    except ImportError:
        logger.debug("seaborn not installed, skipping monkey-patch")
        return

    if _patched_seaborn:
        return

    for fn_name in ("set_theme", "set_style", "set"):
        original_fn = getattr(sns, fn_name, None)
        if original_fn is None or getattr(original_fn, "_seaborn_cjk_patched", False):
            continue

        @functools.wraps(original_fn)
        def _wrapper(*args, _orig=original_fn, _font=english_font_name, **kwargs):
            result = _orig(*args, **kwargs)
            _apply_rcparams(_font)
            return result

        _wrapper._seaborn_cjk_patched = True
        setattr(sns, fn_name, _wrapper)
        logger.debug("seaborn_cjk: patched sns.%s", fn_name)

    _patched_seaborn = True


def _unpatch_seaborn() -> None:
    global _patched_seaborn
    try:
        import seaborn as sns
    except ImportError:
        return

    for fn_name in ("set_theme", "set_style", "set"):
        fn = getattr(sns, fn_name, None)
        if fn is not None and getattr(fn, "_seaborn_cjk_patched", False):
            setattr(sns, fn_name, fn.__wrapped__)
            logger.debug("seaborn_cjk: un-patched sns.%s", fn_name)

    _patched_seaborn = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def patch(font: Optional[str] = None) -> Optional[str]:
    """
    Apply the CJK font patch globally.

    Parameters
    ----------
    font : str, optional
        Font name — accepts **both English and localized (native-script) names**.

        Examples of equivalent pairs:
          - ``"Gulim"``          / ``"굴림"``       (Korean)
          - ``"WenQuanYi Zen Hei"`` / ``"文泉驿正黑"``  (Simplified Chinese)
          - ``"IPAGothic"``      / ``"IPAゴシック"`` (Japanese)

        If omitted, the best available CJK font is auto-detected.

    Returns
    -------
    str or None
        The **English** name that was applied (what matplotlib uses internally),
        or None if no CJK font was found.

    Raises
    ------
    ValueError
        If *font* is specified but cannot be resolved.
    """
    global _active_cjk_font

    english_name = _resolve_font_name(font) if font is not None else detect_cjk_font()
    if english_name is None:
        return None

    _save_rcparams()
    _apply_rcparams(english_name)
    _patch_seaborn(english_name)
    _active_cjk_font = english_name

    logger.info("seaborn_cjk: patched — using font '%s'", english_name)
    return english_name


def unpatch() -> None:
    """Remove the CJK patch and restore the original rcParams."""
    global _active_cjk_font
    _restore_rcparams()
    _unpatch_seaborn()
    _active_cjk_font = None
    logger.info("seaborn_cjk: un-patched")


@contextlib.contextmanager
def cjk_context(font: Optional[str] = None):
    """
    Context manager for a temporary CJK patch. Accepts English or localized names.

    >>> with seaborn_cjk.cjk_context("IPAゴシック"):
    ...     sns.lineplot(data=df)
    """
    chosen = patch(font)
    try:
        yield chosen
    finally:
        unpatch()
