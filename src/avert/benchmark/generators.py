"""Attack-traffic generators for XInt-Bench (Plan Section 4, construction).

Because we choose which mechanisms drive each attack instance, the ground-truth
causal feature set would be known by construction. NOTE: unimplemented -- no result in
the paper comes from this module; ground truth is detector-relative and interventional
(see benchmark/ground_truth.py). Each generator exposes the parameters
it sets explicitly (e.g. a SYN-flood generator's packet rate, source entropy, flag
distribution) and records which features those parameters drive, which becomes the
ground-truth attribution label.

Future work: implement generators in a contained testbed, emit pcaps,
run them through extraction.extract_flow_features, and record causal features.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TrafficGenerator(ABC):
    attack_name: str

    @abstractmethod
    def generate(self, params: dict, out_pcap: str) -> dict:
        """Emit traffic; return the ground-truth causal feature record."""


class SynFloodGenerator(TrafficGenerator):
    attack_name = "syn_flood"

    def generate(self, params, out_pcap):
        raise NotImplementedError(
            "Phase 1 P1-4: parameterised SYN-flood (packet rate, source entropy, flag dist); "
            "record causal features for ground truth."
        )
