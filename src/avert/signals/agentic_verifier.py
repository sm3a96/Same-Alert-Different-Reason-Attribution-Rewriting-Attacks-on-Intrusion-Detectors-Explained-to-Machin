"""Signal 2: knowledge-grounded agentic verifier (Plan Section 3.2).

A tool-augmented verifier checks whether the top-ranked features are consistent
with domain knowledge (MITRE ATT&CK, protocol/RFC summaries, dataset feature docs)
for the predicted class. It follows the Agent-as-a-Judge paradigm and resolves its
verdict by a short prosecutor-vs-defender debate adjudicated against retrieved
evidence. It VERIFIES, never generates; ungrounded verdicts abstain; disagreement
with the attribution is treated as integrity evidence, never suppressed.

Hardening: the prompt receives ONLY canonicalised numeric features, schema-
constrained tool outputs, and curated KB text — never raw payloads — which closes
the indirect-prompt-injection channel (OWASP LLM01).

Serving: local open-weights 7-14B model (Qwen3 / Gemma 3 / Phi-4 class, selected
empirically in Phase 2) via vLLM with schema-constrained decoding. Confidence is
calibrated by conformal prediction on clean traffic, so no human labels are needed.

This is a STUB. Build order: KB index -> retrieval tools ->
single-pass verifier -> debate -> conformal calibration. The score returned here is
the conformal nonconformity of the (calibrated) debate verdict.
"""
from __future__ import annotations

from avert.signals.base import IntegritySignal
from avert.types import SignalName


class AgenticVerifierSignal(IntegritySignal):
    name = SignalName.AGENTIC_VERIFIER

    def __init__(self, model_name: str = "to-select-in-phase-2", knowledge_base=None):
        self.model_name = model_name
        self.kb = knowledge_base

    def calibrate(self, clean_samples, clean_explanations, detector):
        raise NotImplementedError("Phase 2 P2-3: conformal-calibrate verdict confidence on clean traffic")

    def score(self, sample, explanation, detector):
        raise NotImplementedError("Phase 2 P2-3: retrieve -> debate -> grounded verdict -> nonconformity")
