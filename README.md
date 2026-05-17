# seaborn-cjk

Automatically patch **seaborn** and **matplotlib** so that CJK (Chinese,
Japanese, Korean) characters render correctly in legends, titles, tick labels,
and all other text — even after `sns.set_theme()` resets the font list.

## The problem

Seaborn's `set_theme()` / `set_style()` hardcode `font.sans-serif` to:

```
['Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
```

Arial has no CJK glyphs, so matplotlib falls back to a placeholder glyph and
emits warnings like:

```
findfont: Font family 'Arial' not found. Falling back to DejaVu Sans.
glyph missing from current font — fallback to Arial
```

## The fix

`seaborn-cjk` does two things:

1. **Prepends the best available CJK font** to `font.sans-serif` and sets
   `axes.unicode_minus = False`.
2. **Monkey-patches `sns.set_theme`, `sns.set_style`, and `sns.set`** so that
   every call to those functions re-applies the fix automatically.

## Installation

```bash
pip install seaborn-cjk
```

To install with seaborn as an explicit dependency:

```bash
pip install "seaborn-cjk[seaborn]"
```

You still need a CJK font installed on your system. If none is present,
install one — e.g. on Ubuntu/Debian:

```bash
sudo apt install fonts-noto-cjk
```

## Usage

### Auto-patch at import time

```python
import seaborn_cjk          # detects best CJK font and patches immediately
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()              # safe — patch survives this call

df = ...
sns.lineplot(data=df)
plt.legend(["中文", "日本語", "한국어"])
plt.show()
```

### Explicit patch with font selection

```python
import seaborn_cjk

# Auto-detect
seaborn_cjk.patch()

# Force a specific font
seaborn_cjk.patch("Noto Sans CJK JP")

# See what's available
print(seaborn_cjk.list_cjk_fonts())
# ['Noto Sans CJK SC', 'Microsoft YaHei', ...]
```

### Context manager (temporary patch)

```python
import seaborn_cjk
import seaborn as sns

with seaborn_cjk.cjk_context():
    sns.lineplot(data=df)
    plt.legend(["图例一", "图例二"])
    plt.savefig("cjk_plot.png")
# original rcParams restored here
```

### Restoring original state

```python
seaborn_cjk.unpatch()
```

## How it works

| Layer | What it does |
|---|---|
| **rcParams patch** | Prepends CJK font to `font.sans-serif`; sets `axes.unicode_minus=False` |
| **Seaborn wrap** | `functools.wraps` around `sns.set_theme/set_style/set` — re-applies rcParams after each call |
| **Font detection** | Walks a ranked list of ~30 known CJK fonts against `fm.fontManager.ttflist` |

Requires matplotlib ≥ 3.6 (which introduced per-glyph font fallback).

## License

MIT
