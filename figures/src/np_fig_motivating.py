"""Motivating-examples figure (paper Section 2): three real flows, one card per flow.

Each column is one flow from the 5G-NIDD / HTTP flood / seed-0 cell: the card title carries
the attack, the flow id and what a prediction monitor sees (class and probability, clean
and attacked). The clean card sits above the attacked card. Each row is one of the five
shown features in the order the reader saw them: rank (the reader's pick carries a marker
and a bold name), feature name, value, the attribution score the reader was shown, and the
erasure delta as a bar with its value printed after it. The erasure delta is the fall in the
detector's HTTP-flood probability when that feature alone is erased toward benign, measured
on the component that decided. A blue bar is at or above tau = 0.05 (a causal feature); a
delta below it is printed in grey. A value in the attack colour is one the adversary
changed; in card (c) the artifact changed and not the flow, so its attacked title is in the
same colour. The strip under each column says what the reader did.

Layout is in points on one axes, and every column position is computed from measured text
widths; the script refuses to write a figure in which any two texts overlap or any text
leaves its card. Reads results/motivating_examples/raw/examples.json, written by
scripts/export_motivating_examples.py, which asserts every list against the case cache the
readers saw. Writes figures/out/fig_motivating.pdf.

every constant here is a layout parameter (points, gaps,
font sizes) or tau, which the caption states and the registry holds; none is a measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from np_style import ATTACK_COLOR, PALETTE, TEXT, apply_style  # noqa: E402

SRC = REPO / "results" / "motivating_examples" / "raw" / "examples.json"
OUT = REPO / "figures" / "out" / "fig_motivating.pdf"

TAU = 0.05
TOP = 5
CAUSAL, SUB_BAR, SUB_INK, INK, MUTED = PALETTE[0], "0.82", "0.45", "0.1", "0.35"
CHANGED = ATTACK_COLOR["A1_displacement"]        # anything the adversary changed: values in (a), (b); the artifact in (c)
SCORE_CLEAN = "0.45"                             # score bars on the clean cards
FS, FS_NAME, FS_TITLE = 8, 8.5, 8.5
PITCH = 12            # points per row
W_PT = TEXT * 72      # \textwidth in points
GUTTER = 4            # between columns
TITLE = {"harm": "(a) cause displacement, s305325",
         "reliance_shift": "(b) cause displacement, s316302",
         "scaffolding": "(c) explainer scaffolding, s129551"}
STORY = {"harm": "sMeanPktSz carried nothing. AckDat did.",
         "reliance_shift": "AckDat is causal now, and reported honestly.",
         "scaffolding": "Same list. The surrogate decided, on TcpRtt."}


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v:.2e}".replace("e+0", "e")
    if a >= 1e4:
        return f"{v / 1e3:.1f}k"
    if a >= 100 or v == int(v):
        return f"{v:.0f}" if v == int(v) else f"{v:.1f}"
    return f"{v:.3g}"


def _pick(e, cond):
    q, p = e["reader"]["Qwen3-8B"][cond], e["reader"]["phi-4"][cond]
    key = "in_S_x" if cond == "clean" else "in_S_a"
    if q["choice"] != p["choice"]:
        raise RuntimeError("readers disagree on this flow; the figure assumes they agree")
    return q["choice"], bool(q[key])


class Measure:
    """Text width in points, from the renderer, for the layout arithmetic."""

    def __init__(self, fig):
        self.fig = fig
        self.r = fig.canvas.get_renderer()

    def __call__(self, s, size, weight="normal"):
        t = self.fig.text(0, 0, s, fontsize=size, fontweight=weight)
        w = t.get_window_extent(self.r).width * 72 / self.fig.dpi
        t.remove()
        return w


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ex = json.loads(SRC.read_text())
    assert len(ex) == 3

    # Vertical plan of one column, in points from the top.
    y_title1, y_title2 = 6, 17
    blocks = {"clean": 34, "attacked": 130}          # block label centre line; one blank row between blocks
    head_dy, row0_dy = 12, 24                        # column headers, first row
    y_verdict = blocks["attacked"] + row0_dy + TOP * PITCH + 2
    verdict_h = 30
    y_legend = y_verdict + verdict_h + 14
    H_PT = y_legend + 8 + PITCH   # room for a second legend row; trimmed below if the legend fits one row
    fig = plt.figure(figsize=(TEXT, H_PT / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(H_PT, 0)                             # y grows downward, like the plan
    ax.axis("off")
    m = Measure(fig)

    card_w = (W_PT - 2 * GUTTER) / 3
    # Column widths from the widest string in the whole figure, so every card aligns.
    names = [d["feature"] for e in ex for c in ("shown_clean", "shown_attacked") for d in e[c][:TOP]]
    vals = [_fmt(d["value"]) for e in ex for c in ("shown_clean", "shown_attacked") for d in e[c][:TOP]]
    scores = [f"{d['score']:+.1f}" for e in ex for c in ("shown_clean", "shown_attacked") for d in e[c][:TOP]]
    w_mark = m("▶", FS, "bold")
    w_name = max(m(n, FS_NAME, "bold") for n in names)
    w_val = max(m(v, FS, "bold") for v in vals)
    w_score = max(m(s, FS) for s in scores)
    w_dlab = m("0.00", FS)
    g = 3
    # Score bar: a 28 pt budget from a zero line, positive right and negative left, scaled
    # per card to that card's largest |score|. The negative side only needs the largest
    # negative fraction in the figure, so the column is 14 pt plus that, not the full 28.
    sbar_half = 14
    neg_frac = 0.0
    for e in ex:
        pm = max(abs(d["score"]) for c in ("shown_clean", "shown_attacked") for d in e[c][:TOP])
        neg_frac = max(neg_frac, max(-d["score"] for c in ("shown_clean", "shown_attacked") for d in e[c][:TOP]) / pm)
    neg_ext = sbar_half * max(neg_frac, 0.0)
    x_name = w_mark + 2
    x_val = x_name + w_name + w_val + 2             # right edge; column is its widest string + 2 pt
    x_zero = x_val + g + neg_ext                     # score bar zero line
    x_score = x_zero + sbar_half + 2 + w_score       # score number, right edge
    x_bar = x_score + g                              # delta bar starts here
    bar_max = card_w - x_bar - 2 - w_dlab - 1
    assert bar_max >= 16, f"delta bar too short: {bar_max:.1f} pt"
    bar_max = min(bar_max, 34)
    print(f"card {card_w:.1f} pt: name {w_name:.1f}, value {w_val:.1f}, score {w_score:.1f}, "
          f"score bar ±{sbar_half} pt (negative side used {neg_ext:.1f}), delta bar max {bar_max:.1f} pt")

    def text(x, y, s, size=FS, weight="normal", color=INK, ha="left"):
        return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va="center")

    orange = {}
    for ci, e in enumerate(ex):
        x0 = ci * (card_w + GUTTER)
        text(x0, y_title1, TITLE[e["label"]], FS_TITLE, "bold")
        text(x0, y_title2, f"HTTP flood {e['p_clean']:.2f} → {e['p_attacked']:.2f}, no alarm", FS, color=MUTED)
        clean_vals = e["clean_value_of_attacked_shown"]      # from the full clean vector, not the clean shown list
        phi_max = max(abs(d["score"]) for c in ("shown_clean", "shown_attacked") for d in e[c][:TOP])
        for cond, y_lab in blocks.items():
            sbar_col = SCORE_CLEAN if cond == "clean" else ATTACK_COLOR[e["attack"]]
            shown = e["shown_clean" if cond == "clean" else "shown_attacked"][:TOP]
            deltas = e["delta_x" if cond == "clean" else "delta_a"]
            S = set(e["S_x"] if cond == "clean" else e["S_a"])
            pick, _ = _pick(e, cond)
            if cond == "clean":
                label, lcol = "clean flow", INK
            elif e["attack"] == "A3_scaffolding":
                label, lcol = "same flow, scaffolded model", CHANGED
            else:
                label, lcol = "attacked flow", INK
            text(x0, y_lab, label, FS_NAME, "bold", lcol)
            yh = y_lab + head_dy
            text(x0 + x_name, yh, "feature", color=MUTED)
            text(x0 + x_val, yh, "value", color=MUTED, ha="right")
            text(x0 + x_score, yh, "score", color=MUTED, ha="right")
            text(x0 + x_bar, yh, "delta", color=MUTED)
            ax.plot([x0 + x_zero, x0 + x_zero], [y_lab + row0_dy - 5, y_lab + row0_dy + (TOP - 1) * PITCH + 5],
                    color="0.6", lw=0.5, solid_capstyle="butt", zorder=1)
            for i, d in enumerate(shown):
                y = y_lab + row0_dy + i * PITCH
                n = d["feature"]
                is_pick = n == pick
                if is_pick:
                    text(x0, y, "▶", FS, "bold", INK)
                text(x0 + x_name, y, n, FS_NAME, "bold" if is_pick else "normal")
                changed = (cond == "attacked" and n in clean_vals
                           and abs(d["value"] - clean_vals[n]) > 1e-9 * max(1.0, abs(clean_vals[n])))
                text(x0 + x_val, y, _fmt(d["value"]), FS, "bold" if changed else "normal",
                     CHANGED if changed else MUTED, ha="right")
                if changed:
                    orange.setdefault(e["label"], []).append(n)
                sl = sbar_half * d["score"] / phi_max
                ax.add_patch(Rectangle((x0 + x_zero + min(sl, 0.0), y - 3.4), abs(sl), 6.8, color=sbar_col, lw=0))
                text(x0 + x_score, y, f"{d['score']:+.1f}", FS, color=INK, ha="right")
                v = max(deltas[n], 0.0)
                causal = n in S
                L = bar_max * min(v, 1.0)
                if round(v, 2) >= 0.01:            # no hairline stub for a delta that prints as 0.00
                    ax.add_patch(Rectangle((x0 + x_bar, y - 3.4), L, 6.8, color=CAUSAL if causal else SUB_BAR, lw=0))
                text(x0 + x_bar + L + 2, y, f"{v:.2f}", FS, color=INK if causal else SUB_INK)
        # Verdict strip.
        c_pick, c_ok = _pick(e, "clean")
        a_pick, a_ok = _pick(e, "attacked")
        ax.add_patch(FancyBboxPatch((x0 + 1, y_verdict), card_w - 2, verdict_h, boxstyle="round,pad=1.5",
                                    fc="0.95", ec="0.7", lw=0.6, mutation_aspect=1))
        text(x0 + 5, y_verdict + 9, f"reader picks {c_pick} {'✓' if c_ok else '✗'} → {a_pick} {'✓' if a_ok else '✗'}",
             FS, "bold")
        text(x0 + 5, y_verdict + 21, STORY[e["label"]], FS)
        for line in [STORY[e["label"]]]:
            for tok in line.replace(",", " ").replace(".", " ").split():
                if tok in e["delta_a"]:
                    assert tok in {d["feature"] for d in e["shown_attacked"][:TOP]}, (e["label"], tok)
        print(f"{e['label']:15s} {e['sample_id']} p {e['p_clean']:.3f}->{e['p_attacked']:.3f} "
              f"S(x)={e['S_x']} S(x')={e['S_a']} reader {c_pick}->{a_pick}")

    print("values in the attack colour:", orange)

    # Legend: five entries, drawn by hand; a row that would not fit the width is split in two.
    items = [(SCORE_CLEAN, "score, clean"), (ATTACK_COLOR["A1_displacement"], "score, cause displacement"),
             (ATTACK_COLOR["A3_scaffolding"], "score, explainer scaffolding"),
             (CAUSAL, "delta ≥ 0.05 (causal)"), (SUB_BAR, "delta < 0.05")]
    widths = [m(s, FS) + 16 for _, s in items]
    gap = 12
    rows_ = [items] if sum(widths) + gap * (len(items) - 1) <= W_PT else [items[:3], items[3:]]
    legend_texts = []
    for ri, row in enumerate(rows_):
        ws = [m(s, FS) + 16 for _, s in row]
        x = (W_PT - sum(ws) - gap * (len(row) - 1)) / 2
        yl = y_legend + ri * PITCH
        for (col, s), w in zip(row, ws):
            ax.add_patch(Rectangle((x, yl - 3.4), 10, 6.8, color=col, lw=0))
            legend_texts.append(text(x + 13, yl, s, FS))
            x += w + gap
    if len(rows_) == 1:                              # give back the reserved row
        H_PT -= PITCH
        fig.set_size_inches(TEXT, H_PT / 72)
        ax.set_ylim(H_PT, 0)

    # Refuse to write a figure with an overlap or a text outside its card.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = 72 / fig.dpi
    problems = []
    boxes = []
    for t in ax.texts:
        bb = t.get_window_extent(r)
        xl, xr, yl, yr = bb.x0 * k, bb.x1 * k, bb.y0 * k, bb.y1 * k
        boxes.append((xl, xr, yl, yr, t.get_text()))
        if xl < -0.5 or xr > W_PT + 0.5:
            problems.append(("outside page", t.get_text(), round(xr, 1)))
        ci = min(int(xl // (card_w + GUTTER)), 2)
        if ci < 2 and xr > ci * (card_w + GUTTER) + card_w + 0.5 and t not in legend_texts:
            problems.append(("crosses gutter", t.get_text(), round(xr, 1)))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a[0] < b[1] - 0.3 and b[0] < a[1] - 0.3 and a[2] < b[3] - 0.3 and b[2] < a[3] - 0.3:
                problems.append(("overlap", a[4], b[4]))
    if problems:
        raise RuntimeError(f"layout faults: {problems}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.0)
    print(f"wrote {OUT}  ({H_PT / 72:.2f} in tall)")


if __name__ == "__main__":
    main()
