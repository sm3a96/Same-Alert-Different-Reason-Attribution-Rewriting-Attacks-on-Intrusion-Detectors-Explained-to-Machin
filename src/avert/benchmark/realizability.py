"""Which feature vectors a flow extractor can actually emit.

An adversary sends packets. A NIDS does not see packets, it sees the feature vector its
extractor computes from them, so the only vectors an attack can reach are the ones in the image
of that extractor. Perturbing each feature inside its own observed [min, max] does not respect
that image, because flow features are not independent: on 5G-NIDD's Argus schema `TotPkts` is
*defined* as `SrcPkts + DstPkts` and `sMeanPktSz` as `SrcBytes / SrcPkts`. Both hold on 100% of
clean records, by construction of the extractor rather than by statistical association, and
independent per-feature perturbation destroys them.

This is the problem-space/feature-space distinction of Pierazzi et al. (IEEE S&P 2020,
doi:10.1109/SP40000.2020.00073) made concrete for flow features. The cost of ignoring it is not
only realism. Measured on this repo's own projection before the fix, a defender checking six
arithmetic identities separated clean from perturbed flows at AUROC 1.000 on 5G-NIDD and 0.96 on
CICIoT2023 -- so "no statistical signal detects the displacement attack" was being measured against
a defender who had never been asked.

The model here is deliberately small and entirely checkable:

  free features     the adversary sets these directly, inside the observed box
  derived features  the extractor computes these from the free ones; the adversary cannot
                    choose them independently, so we recompute them after every perturbation
  orderings         inequalities the extractor guarantees (Min <= AVG <= Max); repaired by
                    reordering rather than by recomputation

Every identity in `SCHEMAS` was discovered empirically and is re-validated against clean traffic
by `scripts/validate_feature_identities.py`, which `make gate` runs. Nothing is declared here on
the strength of a feature's name: a relation enters only if it holds on at least 99.9% of clean
rows at a 1e-3 relative tolerance, and the validator fails the build if a declared relation
stops holding.

What this does NOT model, stated so nobody reads more into it: reachability of a TCP flag
combination, inter-packet timing feasibility, and cross-flow consistency. An attacked flow from
this module satisfies every per-feature range, every integrality constraint, and every
arithmetic identity its extractor imposes. It is not thereby guaranteed to be a flow some real
host would have produced.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# A relation counts as declared only if it holds this often on clean traffic, at RTOL.
HOLD_THRESHOLD = 0.999
RTOL = 1e-3

Cols = Mapping[str, np.ndarray]


def relative_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Relative difference with an absolute floor, so near-zero targets do not blow up."""
    return np.abs(a - b) / np.maximum(np.abs(b), 1.0)


@dataclass(frozen=True)
class Identity:
    """An exact relation `target = fn(inputs)` imposed by the extractor.

    `guard` marks the rows on which the relation is defined at all -- `sMeanPktSz` means nothing
    when `SrcPkts` is zero. Guarded-out rows keep their original target value and are excluded
    from both the hold rate and the residual.
    """

    target: str
    inputs: tuple[str, ...]
    expr: str
    fn: Callable[[Cols], np.ndarray]
    guard: Callable[[Cols], np.ndarray] | None = None

    def defined(self, cols: Cols, n: int) -> np.ndarray:
        return np.ones(n, dtype=bool) if self.guard is None else np.asarray(self.guard(cols))


@dataclass(frozen=True)
class Ordering:
    """An inequality the extractor guarantees, repaired by reordering or clipping."""

    expr: str
    features: tuple[str, ...]
    violation: Callable[[Cols], np.ndarray]
    repair: Callable[[Cols], dict[str, np.ndarray]]


@dataclass(frozen=True)
class FeatureSchema:
    """The realizable set for one dataset's extractor."""

    dataset: str
    extractor: str
    identities: tuple[Identity, ...] = ()
    orderings: tuple[Ordering, ...] = ()
    notes: str = ""

    def bind(self, feature_names: Sequence[str]) -> BoundSchema:
        """Restrict to the constraints whose features all survived loading."""
        present = set(feature_names)
        ids = tuple(i for i in self.identities
                    if i.target in present and all(c in present for c in i.inputs))
        ords = tuple(o for o in self.orderings if all(c in present for c in o.features))
        return BoundSchema(self, list(feature_names), _topological(ids), ords)


@dataclass
class BoundSchema:
    """A schema bound to a concrete feature ordering, ready to project or validate."""

    schema: FeatureSchema
    feature_names: list[str]
    identities: tuple[Identity, ...]
    orderings: tuple[Ordering, ...]
    _index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self._index = {n: i for i, n in enumerate(self.feature_names)}

    # -- structure -------------------------------------------------------------------

    @property
    def derived(self) -> list[str]:
        return [i.target for i in self.identities]

    @property
    def free(self) -> list[str]:
        d = set(self.derived)
        return [n for n in self.feature_names if n not in d]

    def _cols(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {n: X[:, j] for n, j in self._index.items()}

    # -- projection ------------------------------------------------------------------

    def repair(self, X: np.ndarray) -> np.ndarray:
        """Return a copy of `X` on the realizable set: orderings first, then derived features.

        Orderings run first because an identity may consume a feature an ordering moves, and
        identities run in topological order because a derived feature may feed another one --
        `Rate = (TotPkts - 1) / Dur` needs the repaired `TotPkts`, not the perturbed one.
        """
        X = np.array(X, dtype=np.float64, copy=True)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        for o in self.orderings:
            for name, values in o.repair(self._cols(X)).items():
                X[:, self._index[name]] = values

        for ident in self.identities:
            cols = self._cols(X)
            defined = ident.defined(cols, len(X))
            if not defined.any():
                continue
            new = np.asarray(ident.fn(cols), dtype=np.float64)
            col = X[:, self._index[ident.target]]
            X[:, self._index[ident.target]] = np.where(defined & np.isfinite(new), new, col)

        return X[0] if squeeze else X

    # -- measurement -----------------------------------------------------------------

    def residual(self, X: np.ndarray) -> np.ndarray:
        """Worst relative violation per row. Zero means the row is on the realizable set.

        This doubles as the nonconformity score of the feature-consistency integrity signal --
        the cheapest defender there is, and the one C3 must be measured against.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        worst = np.zeros(len(X))
        cols = self._cols(X)
        for ident in self.identities:
            defined = ident.defined(cols, len(X))
            if not defined.any():
                continue
            got = np.asarray(ident.fn(cols), dtype=np.float64)
            err = relative_error(cols[ident.target], got)
            worst = np.maximum(worst, np.where(defined & np.isfinite(err), err, 0.0))
        for o in self.orderings:
            v = np.asarray(o.violation(cols), dtype=np.float64)
            worst = np.maximum(worst, np.where(np.isfinite(v), np.maximum(v, 0.0), 0.0))
        return worst

    def validate(self, X: np.ndarray, threshold: float = HOLD_THRESHOLD) -> dict:
        """Per-constraint hold rate on clean data. Used by the gate, not by the attack."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        cols = self._cols(X)
        out: dict[str, object] = {"dataset": self.schema.dataset,
                                  "extractor": self.schema.extractor,
                                  "rows": int(len(X)), "rtol": RTOL,
                                  "threshold": threshold, "constraints": []}
        ok = True
        for ident in self.identities:
            defined = ident.defined(cols, len(X))
            n = int(defined.sum())
            if n == 0:
                hold = float("nan")
            else:
                err = relative_error(cols[ident.target], np.asarray(ident.fn(cols)))
                hold = float(np.mean(err[defined] < RTOL))
            passed = bool(n > 0 and hold >= threshold)
            ok &= passed
            out["constraints"].append(
                {"kind": "identity", "expr": ident.expr, "defined_rows": n,
                 "hold_rate": None if n == 0 else round(hold, 6), "passed": passed})
        for o in self.orderings:
            hold = float(np.mean(np.asarray(o.violation(cols)) <= RTOL))
            passed = bool(hold >= threshold)
            ok &= passed
            out["constraints"].append(
                {"kind": "ordering", "expr": o.expr, "defined_rows": int(len(X)),
                 "hold_rate": round(hold, 6), "passed": passed})
        out["all_passed"] = bool(ok)
        out["derived_features"] = self.derived
        out["n_free"] = len(self.free)
        return out


def _topological(identities: tuple[Identity, ...]) -> tuple[Identity, ...]:
    """Order identities so every derived input is computed before it is consumed."""
    produced = {i.target: i for i in identities}
    ordered: list[Identity] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def visit(target: str) -> None:
        if target in seen or target not in produced:
            return
        if target in visiting:
            raise ValueError(f"cyclic feature identities through {target!r}")
        visiting.add(target)
        for dep in produced[target].inputs:
            visit(dep)
        visiting.discard(target)
        seen.add(target)
        ordered.append(produced[target])

    for i in identities:
        visit(i.target)
    return tuple(ordered)


# ====================================================================================
# Declared schemas. Every relation below was validated on clean traffic on 2026-08-09;
# `scripts/validate_feature_identities.py` re-checks them and the gate fails if one stops
# holding. Hold rates from that run are in the comments.
# ====================================================================================


def _sum(target: str, a: str, b: str) -> Identity:
    return Identity(target, (a, b), f"{target} = {a} + {b}",
                    lambda c, a=a, b=b: c[a] + c[b])


def _ratio(target: str, num: str, den: str) -> Identity:
    return Identity(
        target, (num, den), f"{target} = {num} / {den}",
        lambda c, n=num, d=den: c[n] / np.where(c[d] > 0, c[d], np.nan),
        guard=lambda c, d=den: c[d] > 0)


def _equal(target: str, src: str) -> Identity:
    return Identity(target, (src,), f"{target} = {src}", lambda c, s=src: c[s])


def _rate(target: str, pkts: str, dur: str) -> Identity:
    return Identity(
        target, (pkts, dur), f"{target} = ({pkts} - 1) / {dur}",
        lambda c, p=pkts, d=dur: (c[p] - 1) / np.where(c[d] > 0, c[d], np.nan),
        guard=lambda c, d=dur: c[d] > 0)


def _sorted_triple(lo: str, mid: str, hi: str) -> Ordering:
    """Enforce lo <= mid <= hi by sorting the three values, which is the nearest realizable
    assignment that preserves the multiset the extractor would have produced."""

    def violation(c: Cols) -> np.ndarray:
        return np.maximum(np.maximum(c[lo] - c[mid], c[mid] - c[hi]), c[lo] - c[hi])

    def repair(c: Cols) -> dict[str, np.ndarray]:
        stacked = np.sort(np.stack([c[lo], c[mid], c[hi]], axis=1), axis=1)
        return {lo: stacked[:, 0], mid: stacked[:, 1], hi: stacked[:, 2]}

    return Ordering(f"{lo} <= {mid} <= {hi}", (lo, mid, hi), violation, repair)


def _at_most(target: str, bound_expr: str, features: tuple[str, ...],
             bound: Callable[[Cols], np.ndarray]) -> Ordering:
    def violation(c: Cols) -> np.ndarray:
        return c[target] - bound(c)

    def repair(c: Cols) -> dict[str, np.ndarray]:
        return {target: np.minimum(c[target], bound(c))}

    return Ordering(f"{target} <= {bound_expr}", features, violation, repair)


def _non_negative(target: str) -> Ordering:
    return Ordering(f"{target} >= 0", (target,),
                    lambda c, t=target: -c[t],
                    lambda c, t=target: {t: np.maximum(c[t], 0.0)})


# ---- 5G-NIDD, Argus records -------------------------------------------------------
# Fourteen exact relations, all at 100.000% on 200k clean rows except SrcRate (99.993%).
# `Min`, `Mean`, `Max`, `Sum` and `RunTime` all equal `Dur` because these are single-record
# flows: the aggregation fields collapse onto the duration. Together they leave 29 of 43
# features free, so the realizable set is far smaller than the box it replaces.
FIVEG_NIDD = FeatureSchema(
    dataset="fiveg_nidd",
    extractor="argus",
    identities=(
        _sum("TotPkts", "SrcPkts", "DstPkts"),           # 100.000%
        _sum("TotBytes", "SrcBytes", "DstBytes"),        # 100.000%
        _sum("Loss", "SrcLoss", "DstLoss"),              # 100.000%
        _sum("TcpRtt", "SynAck", "AckDat"),              # 100.000%
        _sum("Load", "SrcLoad", "DstLoad"),              # 100.000%
        _ratio("sMeanPktSz", "SrcBytes", "SrcPkts"),     # 100.000% where SrcPkts > 0
        _ratio("dMeanPktSz", "DstBytes", "DstPkts"),     # 100.000% where DstPkts > 0
        _rate("Rate", "TotPkts", "Dur"),                 # 100.000% where Dur > 0
        _rate("SrcRate", "SrcPkts", "Dur"),              #  99.993% where Dur > 0
        _equal("RunTime", "Dur"),                        # 100.000%
        _equal("Sum", "Dur"),                            # 100.000%
        _equal("Mean", "Dur"),                           # 100.000%
        _equal("Min", "Dur"),                            # 100.000%
        _equal("Max", "Dur"),                            # 100.000%
    ),
    notes="DstRate does not follow (DstPkts-1)/Dur (9.4%) and pLoss misses the threshold "
          "(99.784%); both are left free rather than declared on a relation that does not hold.",
)

# ---- CICFlowMeter-family aggregates (CICIoT2023, CICIoMT2024) ----------------------
# Far less constrained than Argus, and that asymmetry is itself worth reporting: the same
# attack is much harder to make realizable on 5G-NIDD than on the CIC corpora. `Drate` is
# constant on CICIoMT2024 and dropped by the loader, so `bind` reduces Rate = Srate + Drate
# to nothing there and the Rate = Srate form covers it.
_CIC_ORDERINGS = (
    _sorted_triple("Min", "AVG", "Max"),                                    # 100.000%
    _at_most("Std", "Max - Min", ("Std", "Max", "Min"),
             lambda c: c["Max"] - c["Min"]),                                # 100.000%
    _non_negative("Header_Length"),                                         # 100.000%
)

CICIOT2023 = FeatureSchema(
    dataset="ciciot2023",
    extractor="cicflowmeter",
    identities=(_sum("Rate", "Srate", "Drate"),),                           #  99.992%
    orderings=_CIC_ORDERINGS,
    notes="Variance = Std^2 holds on only 67% and Magnitue = sqrt(2*AVG) on 90%; neither is "
          "declared. Tot size, Number and Weight follow no relation we could establish.",
)

CICIOMT2024 = FeatureSchema(
    dataset="ciciomt2024",
    extractor="cicflowmeter",
    identities=(_equal("Rate", "Srate"),),                                  # 100.000%
    orderings=_CIC_ORDERINGS,
    notes="Drate is constant here and dropped by the loader, so Rate collapses onto Srate.",
)

SCHEMAS: dict[str, FeatureSchema] = {
    "fiveg_nidd": FIVEG_NIDD,
    "ciciot2023": CICIOT2023,
    "ciciomt2024": CICIOMT2024,
}


def schema_for(dataset: str) -> FeatureSchema:
    """The declared schema, or an empty one for datasets we have not characterised.

    An empty schema is not a silent pass: it makes every feature free, `residual` returns zero
    everywhere, and the validator records that no relation was declared. Synthetic data used by
    the walking skeleton has no extractor and legitimately lands here.
    """
    return SCHEMAS.get(dataset, FeatureSchema(dataset=dataset, extractor="unknown",
                                              notes="no identities characterised"))
