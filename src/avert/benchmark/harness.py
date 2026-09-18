"""XInt-Bench harness — the reusable, "test your own pipeline" interface (the product).

A vendor or researcher supplies a fitted detector and the explainer they show to analysts;
XIntBench runs the explanation-integrity attack suite against that pipeline and returns a
structured vulnerability report: for each attack, how well it corrupts the shown
explanation (while preserving the prediction) and how well the monitor's integrity signals
catch it. This is what makes the work adoptable independently of any single result.

    report = XIntBench(data).evaluate_pipeline(detector, explainer)
    print(report.summary())

Attacks (against the stable ExplanationAttack contract):
    A1_misdirection  — promote non-causal features out of the analyst's view (locally faithful)
    A1_displacement     — tiny perturbation that makes the attribution hypersensitive
    A3_scaffolding   — Slack-style model that fools perturbation-based explainers
Monitor signals reported per attack: cross-method consensus, certified stability (honestly,
including where they fail), and their conformal fusion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from avert.attribution.real import PermutationAttributor, TreeSHAPAttributor
from avert.benchmark.attacks import AttributionEvasion, DisplacementAttack
from avert.benchmark.ground_truth import causal_features
from avert.benchmark.perturbation import bounds_for
from avert.data.grouping import grouped_split, grouping_for
from avert.benchmark.scaffold import make_scaffolded_detector
from avert.data.datasets import LoadedData
from avert.detectors.base import Detector
from avert.detectors.real import XGBoostDetector
from avert.fusion.conformal import ConformalFuser, SignalAggregator
from avert.signals.certified_stability import CertifiedStabilitySignal
from avert.signals.cross_method_consensus import CrossMethodConsensusSignal
from avert.signals.feature_consistency import FeatureConsistencySignal
from avert.signals.erasure_faithfulness import ErasureFaithfulnessSignal
from avert.signals.pasa import PASASignal
from avert.types import FlowSample

ALL_ATTACKS = ("A1_misdirection", "A1_displacement", "A3_scaffolding")


@dataclass
class AttackResult:
    attack: str
    n: int
    prediction_preserved: float          # attack validity (should be ~1.0)
    explanation_corruption: float        # mean drop in GT-causal top-k overlap (clean - attacked)
    monitor_auroc: float                 # best channel INCLUDING the fused row, clean vs attacked;
                                         # the paper's "best single check" is eval.metrics.best_single_auroc
    per_signal_auroc: dict = field(default_factory=dict)
    fused_detection: float = 0.0         # fused violation rate at alpha
    false_alarm: float = 0.0


@dataclass
class _Setup:
    """Everything `evaluate_pipeline` and `decision_cases` must agree on for a given
    (dataset, target, seed): the fitted detector, the flows, and the attack objects."""
    detector: Detector
    explainer: object
    X: np.ndarray
    y: np.ndarray
    names: list[str]
    tr: np.ndarray
    benign_ref: np.ndarray
    target: int
    target_class: str
    cal: list[FlowSample]
    test: list[FlowSample]
    evader: AttributionEvasion
    frag: DisplacementAttack
    scaffold: Detector


@dataclass
class PipelineReport:
    dataset: str
    target_class: str
    n_features: int
    alpha: float
    results: list[AttackResult]

    def to_dict(self):
        return {**{k: v for k, v in self.__dict__.items() if k != "results"},
                "results": [asdict(r) for r in self.results]}

    def summary(self) -> str:
        lines = [f"XInt-Bench pipeline report — {self.dataset} (target: {self.target_class}, "
                 f"{self.n_features} features, alpha={self.alpha})", "-" * 72,
                 f"{'attack':17s} {'valid':>6s} {'corrupt':>8s} {'AUROC':>6s} {'detect':>7s} {'FA':>5s}"]
        for r in self.results:
            lines.append(f"{r.attack:17s} {r.prediction_preserved:6.2f} {r.explanation_corruption:8.2f} "
                         f"{r.monitor_auroc:6.2f} {r.fused_detection:7.2f} {r.false_alarm:5.2f}")
        lines.append("-" * 72)
        lines.append("valid=prediction preserved; corrupt=top-k Jaccard distance (NOT the replaced fraction); "
                     "AUROC=best signal (clean vs attacked); detect=fused@alpha; FA=false-alarm")
        return "\n".join(lines)


def classes_supported_by_split(data, n_cal: int, n_test: int, seed: int = 0,
                               min_count: int = 2000) -> tuple[list[int], list[dict]]:
    """Classes with enough flows on EVERY side of the session-aware split, most common first.

    Picking targets by global frequency was fine under a random split, where every class is
    spread evenly across all three sides. Under a grouped split it is not: a class confined to a
    few captures can be absent from the calibration groups entirely, and the cell then dies with
    "0 calibration flows". Twelve of forty-five matrix cells failed that way on the first
    session-aware run.

    Selecting on what the split can actually support is the honest fix, and the classes it
    excludes are reported rather than dropped quietly -- a target that cannot be evaluated under
    a sound protocol is a limitation of the corpus, and hiding it would misrepresent coverage.
    """
    from avert.data.grouping import grouped_split, grouping_for

    grouping = data.grouping or grouping_for(data.name, len(data.X), None)
    split = grouped_split(grouping, seed=seed)
    bi = data.benign_index()
    y = data.y
    classes, counts = np.unique(y, return_counts=True)

    keep, rejected = [], []
    for c, cnt in sorted(zip(classes.tolist(), counts.tolist()), key=lambda t: -t[1]):
        if c == bi or cnt < min_count:
            continue
        n_c = int((y[split.calibration] == c).sum())
        n_t = int((y[split.test] == c).sum())
        n_r = int((y[split.train] == c).sum())
        # Headroom over the requirement: the detector must also get the flow right, so a class
        # that only just clears the bar will not survive that filter either.
        if n_c >= 2 * n_cal and n_t >= 2 * n_test and n_r >= 100:
            keep.append(int(c))
        else:
            # Name the side that failed. A blanket "too rare in the held-out captures" is
            # misleading for a class with 173,036 calibration flows and none in the test
            # partition, and these exclusions are reported in the paper.
            short = [name for name, have, need in
                     (("train", n_r, 100), ("calibration", n_c, 2 * n_cal),
                      ("test", n_t, 2 * n_test)) if have < need]
            rejected.append({"class": int(c), "total": int(cnt), "in_train": n_r,
                             "in_calibration": n_c, "in_test": n_t,
                             "short_on": short,
                             "reason": f"too few flows on the {' and '.join(short)} side "
                                       f"of the session-aware split"})
    return keep, rejected


def target_classes(data, n: int, n_cal: int = 100, n_test: int = 40, seed: int = 0,
                   min_count: int = 2000, seeds=None) -> tuple[list[int], list[dict]]:
    """The shared class-selection rule for C2 and C3.

    ONE definition, deliberately. `run_matrix.py` and `run_decision_utility.py` each carried
    their own copy with a comment promising they matched, which is how two rules that must agree
    stop agreeing -- and "C2 and C3 describe the same instances" is a project gate.
    Changing the rule in one place now changes it in both, or in neither.

    `n` caps the number of targets and MUST be applied here. Deduplicating the two copies of this
    rule on 2026-08-10 dropped the cap that both originals had: `n` was accepted and discarded,
    so CICIoT2023 ran 14 targets instead of 3 and the matrix reached 84 cells against a design of
    45. 5G-NIDD hid it, because only three of its classes survive the split anyway.

    `seeds` must be EVERY seed the caller will run, not just the first. The grouped split is
    seeded, so which captures land in calibration changes with the seed, and a class comfortably
    supported at seed 0 can have zero calibration flows at seed 2. Selecting at seed 0 alone cost
    four CICIoMT2024 cells on 2026-08-11 and left that corpus aggregating over 11 cells while the
    others had 15. A target is only usable if it survives at every seed, because the seed is
    supposed to be the only thing varying across replicates.
    """
    seed_list = [seed] if seeds is None else list(seeds)
    per_seed, rejected = [], []
    for s in seed_list:
        keep_s, rej_s = classes_supported_by_split(data, n_cal, n_test, seed=s,
                                                   min_count=min_count)
        per_seed.append(keep_s)
        rejected += [{**r, "seed": s} for r in rej_s]

    # Intersection, in the order of the first seed so the most common classes still come first.
    survives_all = set(per_seed[0]).intersection(*(set(k) for k in per_seed[1:]))
    keep = [c for c in per_seed[0] if c in survives_all]
    dropped_by_seed = [c for c in per_seed[0] if c not in survives_all]
    if dropped_by_seed:
        rejected.append({"class": dropped_by_seed, "reason":
                         "supported at some seeds but not all; a target must survive every seed "
                         "or the replicates are not comparable"})
    return keep[:n], rejected


class XIntBench:
    def __init__(self, data: LoadedData, n_train=60000, n_cal=120, n_test=70,
                 top_k=5, alpha=0.05, perm_samples=15, stability_n=60, seed=0,
                 scaffold_contamination=0.3):
        self.data, self.n_train, self.n_cal, self.n_test = data, n_train, n_cal, n_test
        self.top_k, self.alpha, self.seed = top_k, alpha, seed
        self.perm_samples, self.stability_n = perm_samples, stability_n
        # The scaffold's OOD detector routes this fraction of CLEAN training traffic to the
        # surrogate. 0.3 is the value every published number used; the reference-decomposition
        # sensitivity arm lowers it, because a careful attacker would.
        self.scaffold_contamination = scaffold_contamination

    def _prepare(self, detector: Detector | None = None, explainer=None,
                 target: int | None = None) -> "_Setup":
        """Detector, splits, and attack instances — the part every experiment shares.

        Both `evaluate_pipeline` and `decision_cases` go through here so that, for a
        given (dataset, target, seed), they operate on the *same* flows and the same
        attacked variants. The detection numbers and the agent-decision numbers in the
        paper therefore describe one set of instances, not two coincidental ones.
        """
        d, rng = self.data, np.random.default_rng(self.seed)
        X, y, names = d.X, d.y, d.feature_names
        bi = d.benign_index()

        # Split by capture session, never by row. Two defects closed at once, both of which
        # were live until 2026-08-10 and both of which invalidate every number computed before.
        #
        #   1. A random permutation put flows from one capture on both sides, so any feature
        #      correlated with session became a shortcut. Measured: detector accuracy 0.916 on a
        #      random split against 0.860 grouped, on 5G-NIDD.
        #   2. Worse and simpler -- calibration and test flows were drawn from the WHOLE dataset,
        #      training rows included. Measured on 5G-NIDD: 10 of the 100 calibration flows were
        #      training rows. The conformal false-alarm guarantee needs calibration data
        #      exchangeable with the test stream and disjoint from training; it was neither, and
        #      the plan claimed it was both.
        #
        # `grouped_split` asserts no group straddles a side, so train, calibration and test flows
        # now come from disjoint captures by construction rather than by hope.
        grouping = d.grouping or grouping_for(d.name, len(X), None)
        split = grouped_split(grouping, seed=self.seed)
        # Subsample ACROSS the training groups, never the first n_train rows: the split's
        # indices are row-ordered, so a head slice would train on the earliest captures only and
        # reintroduce a temporal bias inside the training side.
        tr = (np.sort(rng.choice(split.train, self.n_train, replace=False))
              if len(split.train) > self.n_train else split.train)

        detector = detector or XGBoostDetector(n_estimators=150).fit(X[tr], y[tr])
        explainer = explainer or TreeSHAPAttributor()
        bounds = bounds_for(X[tr], names, d.name)
        benign_ref = (X[np.intersect1d(np.where(y == bi)[0], tr)].mean(axis=0)
                      if bi is not None and len(np.intersect1d(np.where(y == bi)[0], tr))
                      else X[tr].mean(axis=0))
        std = X[tr].std(axis=0)

        classes, counts = np.unique(y, return_counts=True)
        if target is None:
            target = max([c for c in classes if c != bi], key=lambda c: counts[list(classes).index(c)])

        def draw(side: np.ndarray, k: int) -> list[int]:
            """Flows of the target class, from one side of the grouped split, that the detector
            already gets right -- an attack on an alert the detector never raised is not an
            attack on an explanation anyone would read."""
            cand = side[y[side] == target]
            rng.shuffle(cand)
            cand = cand[:8000]
            if len(cand) == 0:
                return []
            # One batched predict, not 8000 single-row calls. Per-call overhead dominates at
            # this shape and turned a two-and-a-half-minute cell into an eleven-minute one.
            keep = detector.predict(X[cand]) == target
            return [int(i) for i in cand[keep][:k]]

        cal_i, test_i = draw(split.calibration, self.n_cal), draw(split.test, self.n_test)
        if len(cal_i) < self.n_cal or len(test_i) < self.n_test:
            raise ValueError(
                f"{d.name}/class {target}: grouped split yields {len(cal_i)} calibration and "
                f"{len(test_i)} test flows, needed {self.n_cal} and {self.n_test}. The class is "
                f"too rare inside the held-out captures -- pick a commoner target, widen the "
                f"test fraction, or drop this cell and say so.")
        self._split_info = {"grouping": grouping.kind, "n_groups": grouping.n_groups,
                            **split.n_groups_used, "n_train_rows": int(len(tr))}

        def mk(i):
            return FlowSample(features=X[i], feature_names=names, dataset=d.name,
                              true_label=target, sample_id=f"s{i}")

        return _Setup(
            detector=detector, explainer=explainer, X=X, y=y, names=names, tr=tr,
            benign_ref=benign_ref, target=target,
            target_class=str(d.label_names[target]) if d.label_names else str(target),
            cal=[mk(i) for i in cal_i], test=[mk(i) for i in test_i],
            evader=AttributionEvasion(bounds, explainer, top_k=self.top_k, n_iter=30,
                                      restarts=2, seed=self.seed),
            frag=DisplacementAttack(explainer, std, bounds, eps=0.15, n_iter=80, restarts=2,
                                 top_k=self.top_k, seed=self.seed),
            scaffold=make_scaffolded_detector(X[tr], y[tr], n_corrupt=12,
                                              contamination=self.scaffold_contamination,
                                              seed=self.seed)[0],
        )

    def attacked(self, s: FlowSample, attack: str, st: "_Setup", causal_idx: list[int]):
        """Apply one attack to one sample. Returns (attacked sample, detector to explain with).

        A3 leaves the flow untouched and swaps in the scaffolded model, which is the
        whole point of that attack: the traffic is innocent, the artifact is not.
        """
        if attack == "A1_misdirection":
            return st.evader.apply(s, st.detector, causal_idx=causal_idx), st.detector
        if attack == "A1_displacement":
            return st.frag.apply(s, st.detector), st.detector
        return s, st.scaffold

    def decision_cases(self, attacks, detector: Detector | None = None, explainer=None,
                       target: int | None = None, n_shown: int = 10) -> list[dict]:
        """Matched decision cases for the C2 agent-harm eval: clean, attacked,
        re-derived, and ranking-withheld.

        The `repaired` key is a RE-DERIVATION arm and nothing more: the same explainer,
        on the same attacked flow. It therefore returns the same explanation and the same
        decision -- an identity, measured at 3,600 of 3,600 paired decisions under each of
        the three attacks. This docstring used to claim it re-derived with the exact
        attributor on the trusted flow and removed A3 outright; the code never did that,
        and describing a mechanism the code does not implement is how a wrong result gets
        believed.

        Trusted-channel re-derivation conditioned on a diagnosed
        violation -- is NOT called from here and no repair result is claimed anywhere in
        the paper. The arm is kept because dropping it would change the cached case ids,
        and reported as a design note.
        """
        from avert.eval.decision_utility import build_cases

        if isinstance(attacks, str):
            attacks = [attacks]
        st = self._prepare(detector, explainer, target)
        names = st.names
        out = []
        # The detector fit and the clean explanation are shared across attack classes.
        # Recomputing them per attack tripled the cost of the whole C2 run for nothing.
        for s in st.test:
            causal, _ = causal_features(s, st.detector, st.benign_ref, 0.05, 3)
            pred_clean = int(st.detector.predict(s.features.reshape(1, -1))[0])
            clean_e = st.explainer.explain(st.detector, s.features, names, pred_clean,
                                           s.sample_id, self.top_k)
            for attack in attacks:
                a, det = self.attacked(s, attack, st, list(causal))
                pred_atk = int(det.predict(a.features.reshape(1, -1))[0])
                atk_e = st.explainer.explain(det, a.features, names, pred_atk,
                                             s.sample_id, self.top_k)
                repaired_e = st.explainer.explain(st.detector, a.features, names, pred_atk,
                                                  s.sample_id, self.top_k)
                out.append({
                    "sample_id": s.sample_id, "attack": attack,
                    "cases": build_cases(
                        sample_id=s.sample_id, dataset=self.data.name,
                        predicted_class=st.target_class, attack=attack,
                        feature_names=names, feature_values=s.features,
                        clean_attr=clean_e.attributions, attacked_attr=atk_e.attributions,
                        attacked_values=a.features, repaired_attr=repaired_e.attributions,
                        causal=[names[j] for j in causal], n_shown=n_shown,
                    ),
                    "prediction_preserved": bool(pred_atk == pred_clean),
                })
        return out

    def evaluate_pipeline(self, detector: Detector | None = None, explainer=None,
                          attacks=ALL_ATTACKS, target: int | None = None) -> PipelineReport:
        d = self.data
        st = self._prepare(detector, explainer, target)
        detector, explainer, names = st.detector, st.explainer, st.names
        X, tr, benign_ref, target = st.X, st.tr, st.benign_ref, st.target
        cal, test = st.cal, st.test

        # Monitor signals + conformal calibration on clean within-class traffic.
        permutation = PermutationAttributor(X[tr], n_samples=self.perm_samples, seed=self.seed)
        signals = [
            CrossMethodConsensusSignal([explainer, permutation], top_k=self.top_k),
            # The cheapest defender, included so the displacement negative is measured against it
            # rather than around it. Blind by construction once the attack is realizable --
            # which is the claim, and it has to be shown rather than asserted.
            FeatureConsistencySignal(d.name, names),
            CertifiedStabilitySignal(explainer, n=self.stability_n, sigma=0.1, top_k=self.top_k, seed=self.seed),
            # Panel additions of September 2026. The three
            # signals above are unchanged and keep their own RNGs, so their per-cell numbers
            # must reproduce the August matrix bit for bit -- that is the check on this edit.
            CertifiedStabilitySignal(explainer, n=self.stability_n, sigma=0.1, top_k=self.top_k,
                                     seed=self.seed, bounds=st.frag.bounds),
            PASASignal(explainer, mode="range", top_k=self.top_k),
            PASASignal(explainer, mode="std", top_k=self.top_k),
            ErasureFaithfulnessSignal(benign_ref, top_k=self.top_k),
        ]
        cal_e = [explainer.explain(detector, s.features, names, target, s.sample_id, self.top_k) for s in cal]
        for sig in signals:
            sig.calibrate(cal, cal_e, detector)
        clean_scores = {sig.name: np.array([sig.score(s, e, detector).score for s, e in zip(cal, cal_e)])
                        for sig in signals}
        agg = SignalAggregator().fit(clean_scores)
        fuser = ConformalFuser()
        fuser.calibrate(np.array([agg.aggregate({sig.name: sig.score(s, e, detector) for sig in signals})
                                  for s, e in zip(cal, cal_e)]))

        def score(sample, det):
            e = explainer.explain(det, sample.features, names,
                                  int(det.predict(sample.features.reshape(1, -1))[0]), sample.sample_id, self.top_k)
            sc = {sig.name: sig.score(sample, e, det) for sig in signals}
            return {k: v.score for k, v in sc.items()}, agg.aggregate(sc), e

        def jaccard_distance(a: set, b: set) -> float:
            return 1.0 - (len(a & b) / len(a | b)) if (a | b) else 0.0

        # Reference clean scores, ground truth, and the clean shown-explanation top-k.
        clean_sig, clean_fused, gt, clean_topk = [], [], [], []
        for s in test:
            per, fu, e = score(s, detector)
            clean_sig.append(per)
            clean_fused.append(fu)
            causal, _ = causal_features(s, detector, benign_ref, 0.05, 3)
            gt.append(set(causal))
            clean_topk.append(set(e.top_features()))
        false_alarm = float(np.mean([fuser.is_violation(f, self.alpha) for f in clean_fused]))

        results = []
        for attack in attacks:
            atk_sig, atk_fused, corrupt, valid = [], [], [], []
            for idx, s in enumerate(test):
                causal_set = gt[idx]
                a, det = self.attacked(s, attack, st, list(causal_set))
                c0 = int(detector.predict(s.features.reshape(1, -1))[0])
                valid.append(int(det.predict(a.features.reshape(1, -1))[0]) == c0)
                per, fu, e = score(a, det)
                atk_sig.append(per)
                atk_fused.append(fu)
                # corruption = fraction of the shown top-k the attack altered (always defined)
                corrupt.append(jaccard_distance(clean_topk[idx], set(e.top_features())))

            per_auroc = {}
            for sig in signals:
                cs = [r[sig.name] for r in clean_sig]
                as_ = [r[sig.name] for r in atk_sig]
                per_auroc[sig.name.value] = round(float(roc_auc_score([0] * len(cs) + [1] * len(as_), cs + as_)), 3)
            fused_auroc = round(float(roc_auc_score(
                [0] * len(clean_fused) + [1] * len(atk_fused), clean_fused + atk_fused)), 3)
            per_auroc["FUSED"] = fused_auroc

            results.append(AttackResult(
                attack=attack, n=len(test),
                prediction_preserved=round(float(np.mean(valid)), 3),
                explanation_corruption=round(float(np.mean(corrupt)), 2),
                monitor_auroc=round(float(max(per_auroc.values())), 3),
                per_signal_auroc=per_auroc,
                fused_detection=round(float(np.mean([fuser.is_violation(f, self.alpha) for f in atk_fused])), 3),
                false_alarm=round(false_alarm, 3),
            ))

        return PipelineReport(dataset=d.name, target_class=st.target_class,
                              n_features=X.shape[1], alpha=self.alpha, results=results)
