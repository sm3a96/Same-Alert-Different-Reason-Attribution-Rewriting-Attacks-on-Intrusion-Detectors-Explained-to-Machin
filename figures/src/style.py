"""Shared figure style — import this in every plotting script so all paper figures
look like one paper. Vector PDF, sized for the venue's text
block, colorblind-safe Okabe-Ito palette.
"""
from __future__ import annotations

# Okabe-Ito colorblind-safe palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

# Widths in inches. acmart/acmsmall (TOPS) is single-column with \textwidth = 395.82pt
# = 5.476in, measured with a probe document, not assumed. A figure built at any other
# width gets scaled by \includegraphics and its fonts land off-spec: the old
# COL_DOUBLE=7.0 printed 8pt text at 6.3pt, and COL_SINGLE stretched to \linewidth
# printed it at 12.9pt. Full-width figures use COL_DOUBLE and are included at
# \linewidth; smaller ones use COL_SINGLE and are included at their natural size.
COL_SINGLE = 3.4
COL_DOUBLE = 5.47


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,           # embed TrueType (no Type-3) for camera-ready
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.4,
    })


def savefig(fig, path: str) -> None:
    fig.savefig(path)
