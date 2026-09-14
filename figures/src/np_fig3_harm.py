"""Fig. 3 (paper) -- reader accuracy per condition, per attack, both readers pooled.

Bars per attack group, competent cells only (clean accuracy >= 0.5, fixed before the run):

  clean                 the reader's accuracy on the honest explanation, against S(x)
  attacked, vs S(x)     the attacked decision scored against the clean causal set, which
                        is what a study that fixed its reference before the attack reports
  attacked, vs S(x')    the same decision scored against the causal set of the flow shown,
                        the paper's reference (Definition 2)
  ranking withheld      the same ten features in corpus order with the scores removed

Every bar is a mean over cells of both readers (a cell is one reader, corpus, class and
seed), with a 95% interval clustered by seed. The two readers agree to within 0.01 on
every clean bar and are shown pooled; per-reader values are printed on stdout. The rescoring
against S(x') reuses the per-instance causal sets written by
scripts/run_reference_decomposition.py; no reader is re-run.

Reads results/decision_utility/raw/decisions.json, summary/per_cell.json and
results/reference_decomposition/raw/instances.json. Writes figures/out/fig3_harm.pdf.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from np_style import (ATTACK_LABEL, ATTACKS, COL, PALETTE, apply_style)  # noqa: E402
from avert.eval.decision_utility import competent_cell_set  # noqa: E402
from avert.eval.metrics import seed_clustered_ci  # noqa: E402

DU = REPO / "results" / "decision_utility"
INST = REPO / "results" / "reference_decomposition" / "raw" / "instances.json"
OUT = REPO / "figures" / "out" / "fig3_harm.pdf"
FLOOR = 0.5

CONDS = ["clean", "attacked_x", "attacked_a", "withheld"]
COND_LABEL = {"clean": "clean", "attacked_x": "attacked, vs $S(x)$",
              "attacked_a": "attacked, vs $S(x')$", "withheld": "ranking withheld"}
COND_COLOR = {"clean": "0.55", "attacked_x": PALETTE[5], "attacked_a": PALETTE[1],
              "withheld": "0.82"}
COND_HATCH = {"clean": "", "attacked_x": "", "attacked_a": "", "withheld": "////"}


def load():
    rows = json.loads((DU / "raw" / "decisions.json").read_text())
    cells = json.loads((DU / "summary" / "per_cell.json").read_text())
    comp = competent_cell_set(cells, FLOOR)
    inst = json.loads(INST.read_text())
    usable = {(r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]): r
              for r in inst if r["shown_matches_cache"] and r["prediction_preserved"]}
    S_a = {k: set(r["S_a"]) for k, r in usable.items()}
    # Per (judge, attack, condition): list of per-cell accuracies with their seeds.
    per_cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["variant"] != "default" or r["order"] != "ranked":
            continue
        k = (r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"])
        if k not in S_a:
            continue
        cell = (r["judge"], r["dataset"], r["class"], r["attack"], r["seed"])
        if cell not in comp:
            continue
        if r["condition"] == "clean":
            per_cell[(r["judge"], r["attack"], "clean")][cell].append(bool(r["det_correct"]))
        elif r["condition"] == "attacked":
            per_cell[(r["judge"], r["attack"], "attacked_x")][cell].append(bool(r["det_correct"]))
            per_cell[(r["judge"], r["attack"], "attacked_a")][cell].append(r["det_choice"] in S_a[k])
        elif r["condition"] == "no_explanation":
            per_cell[(r["judge"], r["attack"], "withheld")][cell].append(bool(r["det_correct"]))
    # Pooled over readers: every cell of both readers, clustered by seed.
    pooled = defaultdict(dict)
    for (j, atk, cond), cells_ in per_cell.items():
        for cell, v in cells_.items():
            pooled[("pooled", atk, cond)][cell] = v
    per_cell.update(pooled)
    stats = {}
    for key, cells_ in per_cell.items():
        acc = np.array([np.mean(v) for v in cells_.values()])
        seeds = np.array([c[4] for c in cells_])
        stats[key] = seed_clustered_ci(acc, seeds) + (len(cells_),)
    return stats


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    stats = load()
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    w = 0.2
    for gi, atk in enumerate(ATTACKS):
        for ci_, cond in enumerate(CONDS):
            mu, hw, n = stats[("pooled", atk, cond)]
            x = gi + (ci_ - 1.5) * w
            ax.bar(x, mu, width=w * 0.92, color=COND_COLOR[cond], hatch=COND_HATCH[cond],
                   edgecolor="white", linewidth=0.6, zorder=2)
            ax.errorbar(x, mu, yerr=hw, fmt="none", ecolor="0.25", elinewidth=0.8,
                        capsize=1.5, zorder=3)
            ax.text(x, 0.02, f"{mu:.2f}", ha="center", va="bottom", fontsize=7, rotation=90,
                    color="white" if cond != "withheld" else "0.2", zorder=4)
    ax.set_xticks(range(len(ATTACKS)))
    ax.set_xticklabels([ATTACK_LABEL[a].replace(" ", "\n") for a in ATTACKS], fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("reader accuracy")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COND_COLOR[c], hatch=COND_HATCH[c],
                             ec="white", label=COND_LABEL[c]) for c in CONDS]
    ax.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7,
              bbox_to_anchor=(0.5, 1.22), handlelength=1.2, handletextpad=0.4,
              columnspacing=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    for k in sorted(stats):
        mu, hw, n = stats[k]
        print(f"  {k[0]:9s} {k[1]:16s} {k[2]:11s} {mu:.3f} +/- {hw:.3f}  ({n} cells)")


if __name__ == "__main__":
    main()
