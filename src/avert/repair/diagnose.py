"""Diagnosis: localise the compromised signal (Plan Section 3.7).

Leave-one-signal-out: recompute the fused score with each signal removed; the
signal whose removal most reduces the fused nonconformity is the one driving the
violation. That localisation maps back to the attack taxonomy:
  stability collapse                      -> A1 / A3
  knowledge inconsistency, stable attr    -> A2
  explanation drift without input drift   -> A4
"""
from __future__ import annotations

from avert.fusion.conformal import SignalAggregator
from avert.types import AttackClass, SignalName, SignalScore

_SIGNAL_TO_ATTACK = {
    SignalName.CERTIFIED_STABILITY: AttackClass.A1_EVASION,      # (or A3)
    SignalName.AGENTIC_VERIFIER: AttackClass.A2_POISONING,
    SignalName.CROSS_METHOD_CONSENSUS: AttackClass.A1_EVASION,
    SignalName.TEMPORAL: AttackClass.A4_DRIFT_MASQUERADE,
}


def diagnose(
    scores: dict[SignalName, SignalScore],
    aggregator: SignalAggregator,
) -> tuple[SignalName | None, AttackClass | None]:
    full = aggregator.aggregate(scores)
    best_sig, best_drop = None, 0.0
    for sig in scores:
        held = {s: v for s, v in scores.items() if s != sig}
        drop = full - aggregator.aggregate(held)
        if drop > best_drop:
            best_drop, best_sig = drop, sig
    if best_sig is None:
        return None, None
    return best_sig, _SIGNAL_TO_ATTACK.get(best_sig)
