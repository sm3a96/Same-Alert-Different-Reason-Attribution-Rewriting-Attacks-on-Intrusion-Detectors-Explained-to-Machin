"""Configuration loading. YAML files in configs/ are the single source of truth
for every experiment, so a run is fully described by its config plus its seed.

Layered: a config may declare `defaults: [default, datasets/ciciot2023]` to merge
base files before its own keys. Keep experiments declarative; no magic constants
buried in code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name: str, root: Path | None = None) -> dict[str, Any]:
    """Load a config by name (without .yaml), merging any `defaults` it lists."""
    root = root or CONFIG_ROOT
    path = (root / name).with_suffix(".yaml")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    merged: dict[str, Any] = {}
    for dep in cfg.pop("defaults", []) or []:
        merged = _deep_merge(merged, load_config(dep, root))
    return _deep_merge(merged, cfg)
