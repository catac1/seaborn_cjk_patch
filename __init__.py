"""
seaborn_cjk
===========
Automatically patches seaborn and matplotlib so that CJK (Chinese, Japanese,
Korean) characters render correctly in legends, titles, and all other text
elements — even after seaborn's set_theme() / set_style() override the font
list with Arial-first defaults.

Usage
-----
    import seaborn_cjk          # patch at import time with auto-detected font
    seaborn_cjk.patch()         # explicit call, same effect
    seaborn_cjk.patch("Noto Sans CJK JP")  # force a specific font
    seaborn_cjk.unpatch()       # restore original rcParams

    # context manager
    with seaborn_cjk.cjk_context("SimHei"):
        sns.lineplot(...)
"""

from .patch import patch, unpatch, cjk_context, detect_cjk_font, list_cjk_fonts

__version__ = "0.1.0"
__all__ = ["patch", "unpatch", "cjk_context", "detect_cjk_font", "list_cjk_fonts"]
