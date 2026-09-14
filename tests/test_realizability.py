"""The realizable set: declared relations hold, projection restores them, attacks respect them.

These are the tests that make C3's central negative mean something. Before 2026-08-09 the A1
attacks perturbed every feature independently inside a per-feature box, and a defender checking
the extractor's own arithmetic separated clean from attacked flows at AUROC 1.000 on 5G-NIDD. The
claim "no statistical signal detects the displacement attack" was therefore about the signals that
had been tried, not about the attack. Everything below exists so that cannot recur silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from avert.benchmark.perturbation import FeatureBounds, RealizableBounds
from avert.benchmark.realizability import SCHEMAS, schema_for
from avert.signals.feature_consistency import FeatureConsistencySignal
from avert.types import Explanation, FlowSample

FIVEG = ["Dur", "RunTime", "Mean", "Sum", "Min", "Max", "TotPkts", "SrcPkts", "DstPkts",
         "TotBytes", "SrcBytes", "DstBytes", "sMeanPktSz", "dMeanPktSz", "Load", "SrcLoad",
         "DstLoad", "Loss", "SrcLoss", "DstLoss", "Rate", "SrcRate", "SynAck", "AckDat",
         "TcpRtt", "sTtl"]


def _consistent_fiveg(n: int, seed: int = 0) -> np.ndarray:
    """Build rows that satisfy every declared Argus relation, the way an extractor would."""
    rng = np.random.default_rng(seed)
    src_pkts = rng.integers(1, 500, n).astype(float)
    dst_pkts = rng.integers(1, 500, n).astype(float)
    src_bytes = src_pkts * rng.uniform(40, 1500, n)
    dst_bytes = dst_pkts * rng.uniform(40, 1500, n)
    dur = rng.uniform(0.01, 60.0, n)
    src_load, dst_load = rng.uniform(0, 1e6, n), rng.uniform(0, 1e6, n)
    src_loss, dst_loss = rng.integers(0, 5, n).astype(float), rng.integers(0, 5, n).astype(float)
    syn, ack = rng.uniform(0, 0.5, n), rng.uniform(0, 0.5, n)
    cols = {
        "Dur": dur, "RunTime": dur, "Mean": dur, "Sum": dur, "Min": dur, "Max": dur,
        "SrcPkts": src_pkts, "DstPkts": dst_pkts, "TotPkts": src_pkts + dst_pkts,
        "SrcBytes": src_bytes, "DstBytes": dst_bytes, "TotBytes": src_bytes + dst_bytes,
        "sMeanPktSz": src_bytes / src_pkts, "dMeanPktSz": dst_bytes / dst_pkts,
        "SrcLoad": src_load, "DstLoad": dst_load, "Load": src_load + dst_load,
        "SrcLoss": src_loss, "DstLoss": dst_loss, "Loss": src_loss + dst_loss,
        "Rate": (src_pkts + dst_pkts - 1) / dur, "SrcRate": (src_pkts - 1) / dur,
        "SynAck": syn, "AckDat": ack, "TcpRtt": syn + ack,
        "sTtl": rng.integers(1, 255, n).astype(float),
    }
    return np.stack([cols[c] for c in FIVEG], axis=1)


# ---- the declarations themselves --------------------------------------------------

def test_every_schema_binds_and_orders_without_cycles():
    for name, schema in SCHEMAS.items():
        bound = schema.bind([i.target for i in schema.identities]
                            + [c for i in schema.identities for c in i.inputs])
        produced: set[str] = set()
        for ident in bound.identities:
            assert all(dep in produced or dep not in {j.target for j in bound.identities}
                       for dep in ident.inputs), f"{name}: {ident.expr} consumes a later target"
            produced.add(ident.target)


def test_binding_drops_constraints_whose_features_are_absent():
    """CICIoMT2024 loses Drate to the constant-column filter, so Rate = Srate + Drate must not
    silently apply there with a missing column."""
    bound = schema_for("ciciot2023").bind(["Rate", "Srate", "Min", "AVG", "Max"])
    assert [i.expr for i in bound.identities] == []   # Drate absent -> identity dropped
    assert any("Min <= AVG <= Max" in o.expr for o in bound.orderings)


def test_consistent_rows_have_zero_residual():
    bound = schema_for("fiveg_nidd").bind(FIVEG)
    assert np.max(bound.residual(_consistent_fiveg(200))) < 1e-6


def test_fiveg_marks_the_expected_features_derived():
    bound = schema_for("fiveg_nidd").bind(FIVEG)
    assert set(bound.derived) == {
        "TotPkts", "TotBytes", "Loss", "TcpRtt", "Load", "sMeanPktSz", "dMeanPktSz",
        "Rate", "SrcRate", "RunTime", "Sum", "Mean", "Min", "Max"}
    assert "SrcPkts" in bound.free and "Dur" in bound.free


# ---- projection ------------------------------------------------------------------

def test_box_projection_leaves_flows_unrealizable():
    """The regression this whole module exists for. A per-feature box does not restore the
    extractor's arithmetic, and the residual it leaves is what a defender reads."""
    X = _consistent_fiveg(400)
    box = FeatureBounds(X)
    rng = np.random.default_rng(0)
    perturbed = np.stack([box.project(x + rng.normal(0, 0.05 * X.std(axis=0))) for x in X])
    residual = schema_for("fiveg_nidd").bind(FIVEG).residual(perturbed)
    assert np.mean(residual > 1e-3) > 0.9, "box projection should break the identities"


def test_realizable_projection_restores_every_relation():
    X = _consistent_fiveg(400)
    rb = RealizableBounds(X, FIVEG, "fiveg_nidd")
    rng = np.random.default_rng(0)
    perturbed = np.stack([rb.project(x + rng.normal(0, 0.05 * X.std(axis=0))) for x in X])
    assert np.max(rb.residual(perturbed)) < 1e-6


def test_projection_is_idempotent():
    X = _consistent_fiveg(100)
    rb = RealizableBounds(X, FIVEG, "fiveg_nidd")
    rng = np.random.default_rng(1)
    once = np.stack([rb.project(x + rng.normal(0, 0.1 * X.std(axis=0))) for x in X])
    twice = np.stack([rb.project(x) for x in once])
    assert np.allclose(once, twice, rtol=1e-9, atol=1e-9)


def test_derived_features_are_not_in_free_idx():
    X = _consistent_fiveg(50)
    rb = RealizableBounds(X, FIVEG, "fiveg_nidd")
    derived = set(rb.derived_names)
    assert not (set(np.array(FIVEG)[rb.free_idx]) & derived)
    assert len(rb.free_idx) == len(FIVEG) - len(derived)


def test_cic_ordering_repair_sorts_the_triple():
    names = ["Min", "AVG", "Max", "Std", "Header_Length", "Rate", "Srate"]
    X = np.array([[1.0, 2.0, 3.0, 0.5, 20.0, 5.0, 5.0]] * 8)
    rb = RealizableBounds(X, names, "ciciomt2024")
    broken = np.array([9.0, 2.0, 3.0, 0.5, -4.0, 5.0, 5.0])   # Min > AVG, negative header
    fixed = rb.project(broken)
    assert fixed[0] <= fixed[1] <= fixed[2]
    assert fixed[4] >= 0
    assert rb.residual(fixed.reshape(1, -1))[0] < 1e-9


def test_unknown_dataset_degrades_visibly_not_silently():
    X = _consistent_fiveg(20)
    rb = RealizableBounds(X, FIVEG, "synthetic")
    assert rb.derived_names == []
    assert len(rb.free_idx) == len(FIVEG)
    assert np.max(rb.residual(X)) == 0.0


# ---- the signal ------------------------------------------------------------------

def _expl():
    return Explanation(sample_id="t", attributions=np.zeros(len(FIVEG)), feature_names=FIVEG,
                       method="toy", predicted_class=1)


def _flow(x):
    return FlowSample(features=np.asarray(x, dtype=float), feature_names=FIVEG,
                      dataset="fiveg_nidd", true_label=1, sample_id="t")


def test_consistency_signal_fires_on_unrealizable_and_not_on_realizable():
    X = _consistent_fiveg(200)
    sig = FeatureConsistencySignal("fiveg_nidd", FIVEG)
    expl = _expl()

    clean = sig.score(_flow(X[0]), expl, None)
    assert clean.score < 1e-6 and clean.evidence["realizable"]

    broken = X[0].copy()
    broken[FIVEG.index("TotBytes")] *= 1.5          # break TotBytes = SrcBytes + DstBytes
    dirty = sig.score(_flow(broken), expl, None)
    assert dirty.score > 1e-3 and not dirty.evidence["realizable"]


def test_consistency_signal_is_blind_to_a_realizable_attack_by_construction():
    """The claim the paper makes: once the attack respects the extractor, input validation has
    nothing to see. Blindness here is a property of the threat model, not a tuning failure."""
    X = _consistent_fiveg(300)
    rb = RealizableBounds(X, FIVEG, "fiveg_nidd")
    rng = np.random.default_rng(3)
    mask = np.zeros(len(FIVEG))
    mask[rb.free_idx] = 1.0
    attacked = np.stack([rb.project(x + mask * rng.normal(0, 0.05 * X.std(axis=0))) for x in X])
    assert np.max(rb.residual(attacked)) < 1e-6

    sig = FeatureConsistencySignal("fiveg_nidd", FIVEG)
    expl = _expl()
    scores_clean = [sig.score(_flow(x), expl, None).score for x in X[:50]]
    scores_att = [sig.score(_flow(x), expl, None).score for x in attacked[:50]]
    assert max(scores_clean) < 1e-6 and max(scores_att) < 1e-6


def test_consistency_signal_does_not_rank_extractor_rounding():
    """The failure this test exists for, reproduced: sub-tolerance rounding is not evidence.

    Real captures do not satisfy the declared arithmetic to the last bit -- 5G-NIDD carries
    residuals around 1e-7 from the extractor's own float arithmetic -- while the projection
    writes exact values. Ranked raw, that orders the two arms almost perfectly and the matrix
    reported AUROC 0.022 for a signal that had detected nothing: every value in both arms sat
    three orders of magnitude below the tolerance the signal calls realizable. Fused, it was
    worse than useless, since z-scoring a 1e-7 spread turned rounding into z=+3.5 on clean
    flows and pushed the fused score the wrong way.

    So: below tolerance the score is exactly zero and the signal abstains, and no ordering
    survives for an AUROC to find.
    """
    from sklearn.metrics import roc_auc_score

    X = _consistent_fiveg(120)
    rng = np.random.default_rng(11)
    tol = FeatureConsistencySignal.TOL
    # Clean rows as an extractor emits them: correct to ~1e-7, not to the bit.
    rounded = X * (1.0 + rng.normal(0, 1e-7, X.shape))
    sig = FeatureConsistencySignal("fiveg_nidd", FIVEG)
    expl = _expl()

    clean = [sig.score(_flow(x), expl, None) for x in rounded]
    exact = [sig.score(_flow(x), expl, None) for x in X]
    assert 0 < max(RealizableBounds(X, FIVEG, "fiveg_nidd").residual(rounded)) < tol, \
        "the fixture must sit in the sub-tolerance band this test is about"

    assert all(s.score == 0.0 and s.abstained for s in clean + exact)
    y = [0] * len(clean) + [1] * len(exact)
    assert roc_auc_score(y, [s.score for s in clean + exact]) == 0.5

    # And it still fires when the violation is one a monitor could act on.
    broken = X[0].copy()
    broken[FIVEG.index("TotPkts")] *= 1.5
    fired = sig.score(_flow(broken), expl, None)
    assert fired.score > tol and not fired.abstained


@pytest.mark.parametrize("dataset", sorted(SCHEMAS))
def test_declared_schemas_are_documented(dataset):
    """Every schema states what it does not model. A silent schema invites over-reading."""
    s = SCHEMAS[dataset]
    assert s.extractor and (s.identities or s.orderings)
    assert s.notes, f"{dataset}: declare what was tested and rejected"
