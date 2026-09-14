"""Motivating examples for the paper's Section 2: three real flows from one C2 cell.

Rebuilds the 5G-NIDD / HTTPFlood / seed-0 cell exactly as run_reference_decomposition.py
does, picks three flows that the cached reader decisions already contain, and records for
each the quantities the figure and the section text quote:

  * the predicted-class probability on the clean flow and on the attacked flow;
  * the erasure delta of every feature on x (under f) and on x' (under the component h
    that decided: the surrogate when the scaffold's router diverted the flow);
  * the top-10 shown features with values and attribution scores, clean and attacked;
  * the causal sets S(x) and S(x'), and both readers' choices from the cached decisions.

Every rebuilt shown list and causal set is asserted equal to the 13 August case cache and
to results/reference_decomposition/raw/instances.json, so the figure shows the case the
readers actually saw. A mismatch aborts; nothing is silently substituted.

  python scripts/export_motivating_examples.py
"""
from __future__ import annotations

import json
import logging

import numpy as np

from avert.benchmark import XIntBench
from avert.benchmark.ground_truth import causal_features
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.decision_utility import CASE_CACHE
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment

log = logging.getLogger("motivating")

DATASET, CLASS, SEED = "fiveg_nidd", "HTTPFlood", 0
N_CLASSES, N_TEST, N_SHOWN, TOP_K, TAU, MIN_K = 3, 40, 10, 5, 0.05, 3
DECISIONS = RESULTS_ROOT / "decision_utility" / "raw" / "decisions.json"
INSTANCES = RESULTS_ROOT / "reference_decomposition" / "raw" / "instances.json"

# (example label, attack, flow). Chosen on 2026-09-10 from the cached decisions: the first
# is a flow both readers get right clean and wrong attacked against S(x'); the second is
# the Fig. 1 flow, where the attacked pick is causal on x'; the third is a scaffolded flow
# the router diverted, so the shown list is unchanged and the decider is not.
EXAMPLES = [("harm", "A1_displacement", "s305325"),
            ("reliance_shift", "A1_displacement", "s316302"),
            ("scaffolding", "A3_scaffolding", "s129551")]


def _shown(names, feats, attributions):
    order = list(np.argsort(-np.abs(attributions))[:N_SHOWN])
    return [{"feature": names[i], "value": float(feats[i]), "score": float(attributions[i])}
            for i in order]


def build(ctx) -> RunResult:
    cache = [r for r in json.loads(CASE_CACHE.read_text())["cases"]
             if r["dataset"] == DATASET and r["class"] == CLASS and r["seed"] == SEED]
    cached = {(r["sample_id"], r["attack"]): r for r in cache}
    inst = {(r["sample_id"], r["attack"]): r for r in json.loads(INSTANCES.read_text())
            if r["dataset"] == DATASET and r["class"] == CLASS and r["seed"] == SEED}
    decisions = [r for r in json.loads(DECISIONS.read_text())
                 if r["dataset"] == DATASET and r["class"] == CLASS and r["seed"] == SEED
                 and r["variant"] == "default" and r["order"] == "ranked"]

    data = load_dataset(DATASET, load_config(f"datasets/{DATASET}"))
    classes, _ = target_classes(data, N_CLASSES, n_cal=100, n_test=N_TEST, seed=0,
                                min_count=2000, seeds=[0, 1, 2, 3, 4])
    target = [c for c in classes if data.label_names[c] == CLASS][0]
    bench = XIntBench(data, n_cal=100, n_test=N_TEST, top_k=TOP_K, perm_samples=10,
                      stability_n=30, seed=SEED, scaffold_contamination=0.3)
    st = bench._prepare(target=target)
    names, ref = st.names, st.benign_ref
    by_id = {s.sample_id: s for s in st.test}

    # Walk every test flow and every attack in the order rebuild_cell does, because the
    # attack searches draw from one RNG stream per cell: skipping flows changes the draws
    # and the rebuilt attacked case no longer matches the cache (seen 2026-09-10).
    wanted = {(sid, atk): label for label, atk, sid in EXAMPLES}
    found = {}
    for s in st.test:
        sid = s.sample_id
        x = s.features
        S_x, d_x = causal_features(s, st.detector, ref, TAU, MIN_K)
        pred = int(st.detector.predict(x.reshape(1, -1))[0])
        p_x = float(st.detector.predict_proba(x.reshape(1, -1))[0, pred])
        e_clean = st.explainer.explain(st.detector, x, names, pred, sid, TOP_K)
        for atk in ("A1_displacement", "A1_misdirection", "A3_scaffolding"):
            a, det = bench.attacked(s, atk, st, list(S_x))
            if (sid, atk) not in wanted:
                continue
            label = wanted[(sid, atk)]
            pred_a = int(det.predict(a.features.reshape(1, -1))[0])
            p_a = float(det.predict_proba(a.features.reshape(1, -1))[0, pred_a])
            e_atk = st.explainer.explain(det, a.features, names, pred_a, sid, TOP_K)
            if atk == "A3_scaffolding":
                routed = bool(det.ood.is_probe(a.features.reshape(1, -1))[0])
                h = det.surrogate if routed else det.real
                p_real = float(det.real.predict_proba(a.features.reshape(1, -1))[0, pred_a])
                p_surr = float(det.surrogate.predict_proba(a.features.reshape(1, -1))[0, pred_a])
            else:
                routed, h, p_real, p_surr = False, det, None, None
            S_a, d_a = causal_features(a, h, ref, TAU, MIN_K)
            found[(sid, atk)] = dict(label=label, s=s, a=a, x=x, S_x=S_x, d_x=d_x, S_a=S_a, d_a=d_a, pred=pred,
                                     pred_a=pred_a, p_x=p_x, p_a=p_a, routed=routed, p_real=p_real, p_surr=p_surr,
                                     e_clean=e_clean, e_atk=e_atk)
    assert set(found) == set(wanted), set(wanted) - set(found)

    out = []
    for label, atk, sid in EXAMPLES:
        f = found[(sid, atk)]
        x, a, S_x, d_x, S_a, d_a = f["x"], f["a"], f["S_x"], f["d_x"], f["S_a"], f["d_a"]
        pred, pred_a, p_x, p_a, routed = f["pred"], f["pred_a"], f["p_x"], f["p_a"], f["routed"]
        p_real, p_surr, e_clean, e_atk = f["p_real"], f["p_surr"], f["e_clean"], f["e_atk"]

        shown_clean = _shown(names, x, e_clean.attributions)
        shown_atk = _shown(names, a.features, e_atk.attributions)
        c = cached[(sid, atk)]
        i = inst[(sid, atk)]
        # The figure must show the case the readers saw. Assert, do not flag.
        assert [d["feature"] for d in shown_clean] == c["cases"]["clean"]["candidates"], (sid, atk, "clean shown")
        assert np.allclose([d["score"] for d in shown_clean], c["cases"]["clean"]["importances"]), (sid, atk)
        assert [d["feature"] for d in shown_atk] == c["cases"]["attacked"]["candidates"], (sid, atk, "attacked shown")
        assert np.allclose([d["value"] for d in shown_atk], c["cases"]["attacked"]["values"]), (sid, atk)
        assert sorted(names[j] for j in S_x) == sorted(i["S_x"]), (sid, atk, "S_x")
        assert sorted(names[j] for j in S_a) == sorted(i["S_a"]), (sid, atk, "S_a")
        assert routed == i["routed_to_surrogate"], (sid, atk, "routing")
        assert pred == pred_a and data.label_names[pred] == CLASS, (sid, atk, "class")

        picks = {}
        for r in decisions:
            if r["sample_id"] == sid and r["attack"] == atk and r["condition"] in ("clean", "attacked"):
                picks.setdefault(r["judge"], {})[r["condition"]] = {
                    "choice": r["det_choice"], "rank": int(r["det_choice_idx"]) + 1,
                    "in_S_x": r["det_choice"] in i["S_x"], "in_S_a": r["det_choice"] in i["S_a"]}
        assert set(picks) == {"Qwen3-8B", "phi-4"} and all(set(v) == {"clean", "attacked"} for v in picks.values())

        out.append({
            "label": label, "attack": atk, "sample_id": sid, "dataset": DATASET, "class": CLASS, "seed": SEED,
            "p_clean": p_x, "p_attacked": p_a, "routed_to_surrogate": routed,
            "p_attacked_real_model": p_real, "p_attacked_surrogate": p_surr,
            "S_x": [names[j] for j in S_x], "S_a": [names[j] for j in S_a],
            "delta_x": {names[j]: float(d_x[j]) for j in range(len(names))},
            "delta_a": {names[j]: float(d_a[j]) for j in range(len(names))},
            "shown_clean": shown_clean, "shown_attacked": shown_atk,
            # The clean value of every feature on the attacked shown list, from the full clean
            # vector x, so the figure can colour what the adversary changed even where the
            # feature was not on the clean shown list.
            "clean_value_of_attacked_shown": {d["feature"]: float(x[names.index(d["feature"])]) for d in shown_atk},
            "top5_jaccard": 1.0 - len({d["feature"] for d in shown_clean[:TOP_K]} & {d["feature"] for d in shown_atk[:TOP_K]})
                              / len({d["feature"] for d in shown_clean[:TOP_K]} | {d["feature"] for d in shown_atk[:TOP_K]}),
            "reader": picks,
        })
        log.info("%s %s %s: p %.3f -> %.3f, S(x)=%s, S(x')=%s, routed=%s", label, atk, sid, p_x, p_a,
                 out[-1]["S_x"], out[-1]["S_a"], routed)

    (ctx.run_dir / "raw" / "examples.json").write_text(json.dumps(out, indent=1))
    lines = ["# Motivating examples (5G-NIDD, HTTPFlood, seed 0)", "",
             "| Example | Flow | Attack | p(clean) | p(attacked) | S(x) | S(x') | J top-5 | Qwen clean -> attacked | phi-4 clean -> attacked |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in out:
        q, p = e["reader"]["Qwen3-8B"], e["reader"]["phi-4"]
        lines.append(f"| {e['label']} | {e['sample_id']} | {e['attack']} | {e['p_clean']:.3f} | {e['p_attacked']:.3f} | "
                     f"{', '.join(e['S_x'])} | {', '.join(e['S_a'])} | {e['top5_jaccard']:.2f} | "
                     f"{q['clean']['choice']} -> {q['attacked']['choice']} | {p['clean']['choice']} -> {p['attacked']['choice']} |")
    (ctx.run_dir / "summary" / "examples.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return RunResult(summary={e["label"]: {"sample_id": e["sample_id"], "p_clean": e["p_clean"],
                                           "p_attacked": e["p_attacked"], "S_x": e["S_x"], "S_a": e["S_a"]}
                              for e in out},
                     floats=[{"float_id": "Fig_motivating", "kind": "figure", "paper_section": "2",
                              "path": str(ctx.run_dir / "raw" / "examples.json"),
                              "claim": "three real flows: harm, reliance shift, scaffolding"}])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment("motivating_examples", build, config={"dataset": DATASET, "class": CLASS, "seed": SEED,
                                                         "examples": EXAMPLES}, seed=SEED)


if __name__ == "__main__":
    main()
