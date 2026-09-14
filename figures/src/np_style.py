"""Figure style for the paper figures (IEEEtran journal, two columns).

Widths are measured, not assumed: IEEEtran `lettersize,journal` sets \\columnwidth to
252pt = 3.5in and \\textwidth to 516pt = 7.16in. A figure built at these widths and
included at its natural size keeps its 8pt text at 8pt on the page.
"""
from __future__ import annotations

from style import PALETTE  # noqa: F401  (Okabe-Ito, validated)
from style import apply_style as _apply_base


def apply_style() -> None:
    """The shared style, plus one typeface for every paper figure.

    Liberation Sans carries Arial's metrics, the face IEEE figures usually use, and is 15%
    narrower than DejaVu; the motivating-example cards do not fit at 8 pt without it
    (measured 2026-09-10). DejaVu stays as the fallback for glyphs Liberation lacks: the
    pick marker, tick and cross.
    """
    import matplotlib as mpl

    _apply_base()
    mpl.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]

COL = 3.5
TEXT = 7.16

DATASETS = ["fiveg_nidd", "ciciomt2024", "ciciot2023"]
DATASET_LABEL = {"fiveg_nidd": "5G-NIDD", "ciciomt2024": "CICIoMT2024", "ciciot2023": "CICIoT2023"}
DATASET_MARKER = {"fiveg_nidd": "o", "ciciomt2024": "s", "ciciot2023": "^"}

ATTACKS = ["A1_displacement", "A1_misdirection", "A3_scaffolding"]
ATTACK_LABEL = {"A1_displacement": "cause displacement",
                "A1_misdirection": "rank promotion",
                "A3_scaffolding": "explainer scaffolding"}
# Fixed hue per attack, identical across every figure in the paper.
ATTACK_COLOR = {"A1_displacement": PALETTE[1], "A1_misdirection": PALETTE[0],
                "A3_scaffolding": PALETTE[2]}
