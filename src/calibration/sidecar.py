# pattern: Imperative Shell
# Reason: reads/writes the .calibration.json sidecar. Pure calibration math
# lives in calibrator.py / report.py.

from __future__ import annotations

import json
from pathlib import Path

from src.calibration.calibrator import PlattCalibrator


def calibration_path_for_weights(weights_path: Path | str) -> Path:
    """`*.weights.h5` -> `*.calibration.json` (fallback: append suffix)."""
    p = Path(weights_path)
    name = p.name
    if name.endswith(".weights.h5"):
        return p.with_name(name[: -len(".weights.h5")] + ".calibration.json")
    return p.with_name(name + ".calibration.json")


def save_calibration(cal: PlattCalibrator, path: Path | str) -> None:
    payload = {"method": cal.method, "A": cal.A, "B": cal.B,
               "fit_population": cal.fit_population, "n": cal.n}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path | str) -> PlattCalibrator:
    d = json.loads(Path(path).read_text())
    method = d.get("method", "platt")
    if method != "platt":
        raise ValueError(f"unsupported calibration method: {method!r}")
    return PlattCalibrator(A=d["A"], B=d["B"], fit_population=d["fit_population"],
                           n=d["n"], method=method)
