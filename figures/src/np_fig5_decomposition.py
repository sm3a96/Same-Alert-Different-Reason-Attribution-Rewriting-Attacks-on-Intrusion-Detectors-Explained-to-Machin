"""Fig. 5 (paper) -- the top-1 decomposition, one panel per attack.

Three bars per group: Delta_paper (Eq. dpaper), Delta_inf (Eq. dinf) and Delta_rel
(Eq. drel), pooled and per corpus, with 95% intervals clustered by seed. By construction
Delta_paper = Delta_inf + Delta_rel in every group.

Reads results/reference_decomposition/summary/decomposition.json (the same file the
decomposition table is written from). Writes figures/out/fig5_decomposition.pdf.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from np_style import (ATTACK_LABEL, ATTACKS, DATASET_LABEL, DATASETS, PALETTE, TEXT,  # noqa: E402
                      apply_style)

DEC = REPO / "results" / "reference_decomposition" / "summary" / "decomposition.json"
OUT = REPO / "figures" / "out" / "fig5_decomposition.pdf"

TERMS = ["paper", "infidelity", "reliance"]
TERM_LABEL = {"paper": r"$\Delta_{\mathrm{paper}}$ (vs $S(x)$)",
              "infidelity": r"$\Delta_{\mathrm{inf}}$ (infidelity)",
              "reliance": r"$\Delta_{\mathrm{rel}}$ (reliance shift)"}
TERM_COLOR = {"paper": "0.55", "infidelity": PALETTE[3], "reliance": PALETTE[4]}


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    dec = json.loads(DEC.read_text())
    scopes = ["pooled"] + DATASETS
    fig, axes = plt.subplots(1, len(ATTACKS), figsize=(TEXT, 2.4), sharey=True)
    w = 0.26
    for ax, atk in zip(axes, ATTACKS):
        ax.axhline(0, color="0.3", lw=0.8, zorder=1)
        for gi, scope in enumerate(scopes):
            for ti, term in enumerate(TERMS):
                m = dec[atk][scope][term]
                x = gi + (ti - 1) * w
                ax.bar(x, m["mean"], width=w * 0.92, color=TERM_COLOR[term], edgecolor="white",
                       linewidth=0.6, zorder=2)
                ax.errorbar(x, m["mean"], yerr=[[m["mean"] - m["ci_lo"]], [m["ci_hi"] - m["mean"]]],
                            fmt="none", ecolor="0.25", elinewidth=0.8, capsize=1.5, zorder=3)
                print(f"  {ATTACK_LABEL[atk]:22s} {scope:12s} {term:11s} {m['mean']:+.3f} "
                      f"[{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]")
        ax.set_xticks(range(len(scopes)))
        ax.set_xticklabels(["pooled", "5G-NIDD", "CICIoMT\n2024", "CICIoT\n2023"], fontsize=7.5)
        ax.set_title(ATTACK_LABEL[atk], fontsize=8.5, pad=3)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("change in $P[\\,t(\\cdot) \\in S\\,]$")
    axes[0].set_ylim(-0.62, 0.62)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TERM_COLOR[t], ec="white", label=TERM_LABEL[t])
               for t in TERMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.04), handlelength=1.2, handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(top=0.9, bottom=0.26, left=0.06, right=0.995, wspace=0.06)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
