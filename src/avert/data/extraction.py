"""Flow-feature extraction (Plan Section 4, construction).

CRITICAL for benchmark validity: XInt-Bench generated traffic must pass through the
SAME extractor used to build the target dataset, or the synthetic flows live off the
detector's training manifold and the monitor could exploit the artifact instead of
the attack. CICFlowMeter for the CIC corpora; each dataset's native pipeline
otherwise.

Future work: wrap the extractor (CICFlowMeter or nfstream) so a raw
pcap from the generator yields the dataset's exact feature schema.
"""
from __future__ import annotations


def extract_flow_features(pcap_path: str, schema: str):
    raise NotImplementedError(
        "Phase 1 P1-4: wrap CICFlowMeter/nfstream; output must match the target "
        "dataset's feature schema exactly."
    )
