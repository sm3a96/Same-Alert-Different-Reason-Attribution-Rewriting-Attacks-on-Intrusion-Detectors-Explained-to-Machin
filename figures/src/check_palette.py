"""Colourblind-safety check for the figure palette.

The dataviz guidance is that this is computable, so it should be computed rather than
asserted — but its validator is Node and this box has no Node. This is the same check
in Python: simulate dichromatic vision, then measure perceptual distance in OKLab.

  python figures/src/check_palette.py

Thresholds follow the guidance: adjacent-pair separation >= 8 under every CVD type
(6-8 is a floor that needs a second encoding channel), and >= 15 for normal vision,
which is a hard failure below that because full-colour readers cannot separate the
pair either. Distances are OKLab Euclidean x100.

Viénot-Brettel-Mollon (1999) LMS projection for protanopia and deuteranopia;
Brettel et al. (1997) two-plane construction for tritanopia.

every constant here is a published colour-space conversion
matrix (Hunt-Pointer-Estevez RGB->LMS, the Viénot dichromat projections, and Ottosson's
OKLab coefficients). None of them is a measurement from this project, so they cannot drift
from the data; changing one would be a bug in the colour maths, not stale prose.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

# sRGB <-> linear
def _lin(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# Hunt-Pointer-Estevez normalised to D65, as used by Viénot et al.
_RGB2LMS = np.array([[0.31399022, 0.63951294, 0.04649755],
                     [0.15537241, 0.75789446, 0.08670142],
                     [0.01775239, 0.10944209, 0.87256922]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)

# Dichromat projections in LMS (Viénot-Brettel-Mollon 1999).
_PROJ = {
    "protan": np.array([[0.0, 1.05118294, -0.05116099], [0, 1, 0], [0, 0, 1]]),
    "deutan": np.array([[1, 0, 0], [0.9513092, 0.0, 0.04866992], [0, 0, 1]]),
    "tritan": np.array([[1, 0, 0], [0, 1, 0], [-0.86744736, 1.86727089, 0.0]]),
}


def simulate(rgb: np.ndarray, kind: str) -> np.ndarray:
    lms = _RGB2LMS @ _lin(rgb)
    return np.clip(_srgb(_LMS2RGB @ (_PROJ[kind] @ lms)), 0, 1)


# linear sRGB -> OKLab (Björn Ottosson)
_LMS_OK = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]])
_OKLAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab(rgb: np.ndarray) -> np.ndarray:
    return _OKLAB @ np.cbrt(_LMS_OK @ _lin(rgb))


def delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(oklab(a) - oklab(b)) * 100)


def check(palette: list[str], adjacent_only: bool = True) -> bool:
    rgb = [hex_to_rgb(h) for h in palette]
    pairs = (list(zip(range(len(rgb) - 1), range(1, len(rgb)))) if adjacent_only
             else list(itertools.combinations(range(len(rgb)), 2)))
    ok = True
    print(f"{'pair':>9s} {'normal':>8s} {'protan':>8s} {'deutan':>8s} {'tritan':>8s}   verdict")
    for i, j in pairs:
        normal = delta(rgb[i], rgb[j])
        cvd = {k: delta(simulate(rgb[i], k), simulate(rgb[j], k)) for k in _PROJ}
        worst = min(cvd.values())
        if normal < 15:
            verdict, ok = "FAIL normal-vision floor", False
        elif worst < 6:
            verdict, ok = "FAIL cvd", False
        elif worst < 8:
            verdict = "floor - needs a 2nd encoding channel"
        else:
            verdict = "pass"
        print(f"  {i}-{j:<5d} {normal:8.1f} {cvd['protan']:8.1f} {cvd['deutan']:8.1f} "
              f"{cvd['tritan']:8.1f}   {verdict}")
    return ok


if __name__ == "__main__":
    from style import PALETTE

    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(PALETTE)
    pal = PALETTE[:n]
    print(f"palette: {pal}\n")
    print("adjacent pairs (the order hues are actually assigned in):")
    ok_adj = check(pal, adjacent_only=True)
    print("\nall pairs:")
    ok_all = check(pal, adjacent_only=False)
    print(f"\nadjacent: {'PASS' if ok_adj else 'FAIL'}   all-pairs: {'PASS' if ok_all else 'FAIL'}")
    sys.exit(0 if ok_adj else 1)
