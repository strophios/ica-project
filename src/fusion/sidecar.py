# pattern: Imperative Shell
# Reason: reads/writes the .fusion.json sidecar. Pure fusion logic lives in combiner.py.

from __future__ import annotations

import json
from pathlib import Path

from src.fusion.combiner import FusionConfig


def fusion_path_for_weights(weights_path: Path | str) -> Path:
    """`*.weights.h5` -> `*.fusion.json` (fallback: append suffix).

    Mirrors calibration_path_for_weights convention for sidecar path derivation.
    """
    p = Path(weights_path)
    name = p.name
    if name.endswith(".weights.h5"):
        return p.with_name(name[: -len(".weights.h5")] + ".fusion.json")
    return p.with_name(name + ".fusion.json")


def save_fusion(cfg: FusionConfig, path: Path | str) -> None:
    """Save FusionConfig to a JSON sidecar.

    Args:
        cfg: FusionConfig to persist
        path: output path (typically derived via fusion_path_for_weights)
    """
    payload = {
        "gate_threshold": cfg.gate_threshold,
        "combine": cfg.combine,
        "coefs": cfg.coefs,
        "score_space": cfg.score_space,
        "includes_us": cfg.includes_us,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_fusion(path: Path | str) -> FusionConfig:
    """Load FusionConfig from a JSON sidecar.

    Args:
        path: sidecar path (typically a .fusion.json file)

    Returns:
        Reconstructed FusionConfig

    Raises:
        ValueError: if payload has invalid/unknown field values
        KeyError: if required fields are missing
        json.JSONDecodeError: if JSON is malformed
    """
    d = json.loads(Path(path).read_text())

    # Validate required fields upfront
    required = {"gate_threshold", "combine", "coefs", "score_space", "includes_us"}
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    return FusionConfig(
        gate_threshold=d["gate_threshold"],
        combine=d["combine"],
        coefs=d["coefs"],
        score_space=d["score_space"],
        includes_us=d["includes_us"],
    )
