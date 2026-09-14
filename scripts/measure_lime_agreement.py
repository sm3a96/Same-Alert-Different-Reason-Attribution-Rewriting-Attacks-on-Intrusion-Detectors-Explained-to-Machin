"""How much does LIME agree with TreeSHAP on clean flows?

The plan says LIME was tried and dropped because its Spearman correlation against TreeSHAP on
clean data is about -0.14 -- noise, so putting it in the cross-method consensus signal would
have injected disagreement unrelated to any attack. That number had no artifact anywhere in
the repo and no code path either, which makes it a claim resting on a console session someone
once ran. Either it regenerates or it comes out of the paper.

This measures it: fit the deployed detector, take clean test flows, attribute each with
TreeSHAP and with LIME, and correlate |phi| over the shared feature space. Reported per
dataset with a seed-clustered interval, because three seeds of one dataset are not three
independent observations.

  python scripts/measure_lime_agreement.py --datasets fiveg_nidd ciciot2023 ciciomt2024

Writes results/_logs/lime_agreement.json.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from avert.attribution.real import TreeSHAPAttributor
from avert.benchmark import XIntBench
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.metrics import seed_clustered_ci

REPO = Path(__file__).resolve().parents[1]
LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}


def lime_vector(explainer, detector, x, n_features):
    """|phi| over the full feature space, from a LIME explanation of the predicted class."""
    cls = int(detector.predict(x.reshape(1, -1))[0])
    exp = explainer.explain_instance(x, detector.predict_proba, labels=(cls,),
                                     num_features=n_features, num_samples=500)
    # `local_exp[label]` is a dict on some paths and a list of pairs on others -- a known
    # LIME wart, and indexing without checking is how it bites.
    raw = exp.local_exp[cls]
    pairs = raw.items() if hasattr(raw, "items") else raw
    v = np.zeros(n_features)
    for j, w in pairs:
        v[int(j)] = abs(float(w))
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["fiveg_nidd", "ciciot2023", "ciciomt2024"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-test", type=int, default=40)
    a = ap.parse_args()

    from lime.lime_tabular import LimeTabularExplainer

    out = {}
    for ds in a.datasets:
        data = load_dataset(ds, load_config(f"datasets/{ds}"))
        corrs, seeds_of = [], []
        for seed in a.seeds:
            bench = XIntBench(data, n_cal=150, n_test=a.n_test, top_k=5, seed=seed)
            st = bench._prepare(explainer=TreeSHAPAttributor())
            det, shap, names = st.detector, st.explainer, list(st.names)
            lime_expl = LimeTabularExplainer(
                np.asarray(st.X[st.tr]), feature_names=names,
                discretize_continuous=False, random_state=seed)
            for fs in st.test:
                x = np.asarray(fs.features, dtype=float)
                e = shap.explain(det, x, names, int(det.predict(x.reshape(1, -1))[0]),
                                 "s", top_k=len(names))
                sv = np.abs(np.asarray(e.attributions, dtype=float))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lv = lime_vector(lime_expl, det, x, len(names))
                r = spearmanr(sv, lv).correlation
                if r == r:
                    corrs.append(float(r))
                    seeds_of.append(seed)
        if not corrs:
            continue
        mu, hw = seed_clustered_ci(corrs, seeds_of)
        out[ds] = {"mean_spearman": mu, "halfwidth": hw, "n": len(corrs),
                   "seeds": sorted(set(seeds_of))}
        print(f"  {LABEL.get(ds, ds):13s} Spearman(|phi_SHAP|, |phi_LIME|) = "
              f"{mu:+.3f} +/- {hw:.3f}   n={len(corrs)} flows over {len(set(seeds_of))} seeds")

    if out:
        vals = [v["mean_spearman"] for v in out.values()]
        print(f"\n  across datasets: {min(vals):+.3f} to {max(vals):+.3f}")
        dest = REPO / "results" / "_logs" / "lime_agreement.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1))
        print(f"  saved {dest}")


if __name__ == "__main__":
    main()
