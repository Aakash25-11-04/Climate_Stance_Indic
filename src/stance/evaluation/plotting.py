"""Shared matplotlib styling: one validated categorical palette (fixed order), thin marks,
recessive grid, text in ink tokens (never series colours)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (dataviz skill reference instance), fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LABEL_COLOURS = {"Favour": SERIES[0], "Against": SERIES[1], "None": SERIES[2]}
LANG_COLOURS = {"hi": SERIES[3], "bn": SERIES[6]}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e2"


def _indic_fonts() -> list[str]:
    """Font fallback chain covering Devanagari and Bengali (Nirmala UI ships with Windows;
    Noto fonts are the usual Linux / Colab choice). Only installed fonts are returned."""
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    wanted = ["DejaVu Sans", "Nirmala UI", "Noto Sans Devanagari", "Noto Sans Bengali", "Mangal", "Vrinda", "Lohit Devanagari"]
    return [f for f in wanted if f in have] or ["DejaVu Sans"]


def style() -> None:
    plt.rcParams["font.family"] = _indic_fonts()
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.labelsize": 9.5, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 9.5, "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
        "patch.linewidth": 0, "lines.linewidth": 2, "lines.markersize": 7,
    })


def bars(ax, labels, values, colours, horizontal=False, value_fmt="{:.0f}", label_top=True):
    """Thin bars with a 2px surface gap, rounded look approximated by edge colour."""
    if horizontal:
        b = ax.barh(labels, values, color=colours, edgecolor=SURFACE, linewidth=2, height=0.7)
        if label_top:
            for rect, v in zip(b, values):
                ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, " " + value_fmt.format(v), va="center", ha="left", color=INK2, fontsize=8.5)
        ax.grid(axis="y", visible=False)
    else:
        b = ax.bar(labels, values, color=colours, edgecolor=SURFACE, linewidth=2, width=0.7)
        if label_top:
            for rect, v in zip(b, values):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), value_fmt.format(v), va="bottom", ha="center", color=INK2, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    return b
