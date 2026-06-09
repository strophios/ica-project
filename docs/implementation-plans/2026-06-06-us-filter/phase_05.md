# US/not-US Pre-Filter — Phase 5 Implementation Plan

**Goal:** A reusable, method- and head-agnostic calibration seam with a Platt implementation and quality metrics (ECE / Brier / reliability).

**Architecture:** Pure calibration math (fit, transform, report) in Functional-Core modules; sidecar JSON I/O in an Imperative-Shell module so the core stays I/O-free (FCIS: shell depends on core, never the reverse). Platt = `σ(A·logit + B)`, `A`/`B` from a 1-D logistic regression fit on the natural-balance validation split. Calibrated probabilities are calibrated *to a distribution*; `fit_population` records which.

**Tech Stack:** Python, numpy, scikit-learn (`LogisticRegression`), pytest + hypothesis.

**Scope:** Phase 5 of 8.

**Codebase verified:** 2026-06-09 (`pyproject.toml` deps; `src/diagnostics/` FCIS-header + sidecar conventions).

---

## Acceptance Criteria Coverage

This phase implements and tests **us-filter.AC4**:

### us-filter.AC4: Scores are calibrated and the calibrator is reusable
- **us-filter.AC4.1 Success:** `PlattCalibrator` fits on natural-balance val; `transform` maps logits to a monotonic [0,1] probability.
- **us-filter.AC4.2 Success:** `calibration_report` computes ECE / Brier / reliability on the test split and the pre-1986 hand-labeled slice.
- **us-filter.AC4.3 Success:** calibration params persist to `.calibration.json` and reload to an identical transform.
- **us-filter.AC4.4 Guard:** `fit_population` is recorded; fitting on rebalanced batches is documented as disallowed.
- **us-filter.AC4.5 Edge:** post-calibration ECE ≤ pre-calibration ECE on the evaluation set.

(AC4.2's application to the *pre-1986 hand-labeled slice* happens in Phase 6; here it is exercised on synthetic/held-out arrays.)

---

## Findings

- Deps available: `scikit-learn>=1.7.2`, `scipy>=1.16.3`, `numpy`, `pandas`.
- FCIS headers: first source line is `# pattern: <Functional Core | Imperative Shell | Mixed (unavoidable)>` plus a short Reason (see `src/diagnostics/distribution_metrics.py:1-6`).
- Module is built and tested on synthetic `(logits, labels)` arrays — no trained model required at build time. hypothesis is used for invariants (monotonicity, round-trip), mirroring `tests/test_diagnostics_trackers.py`.
- Design decision (carried in): Platt over temperature scaling (the `B` intercept absorbs class-balance skew) and over isotonic (needs 1000+/class, unstable under imbalance). Fit on the **natural-balance val split only**, never rebalanced batches.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: `src/calibration/calibrator.py` — Calibrator ABC + PlattCalibrator

**Verifies:** us-filter.AC4.1, AC4.4.

**Files:**
- Create: `src/calibration/calibrator.py`
- Create: `tests/test_calibration_platt.py`

**Implementation** (`# pattern: Functional Core`):

```python
# pattern: Functional Core
# Reason: pure calibration math — fitting params from arrays and mapping logits
# to probabilities are referentially transparent. Sidecar I/O lives in sidecar.py.

from __future__ import annotations
import abc
import dataclasses

import numpy as np
from sklearn.linear_model import LogisticRegression


def platt_fit(logits, labels):
    """Fit sigmoid(A*z + B) by 1-D logistic regression of label ~ logit.
    Returns (A, B). `logits`/`labels` are 1-D arrays over a held-out split at
    its NATURAL class balance (rebalanced batches are disallowed — they skew B).
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels).astype(int).reshape(-1)
    lr = LogisticRegression(penalty=None)
    lr.fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_transform(logits, A, B):
    """Map logits to calibrated probabilities sigmoid(A*z + B) in [0,1]."""
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(A * z + B)))


class Calibrator(abc.ABC):
    """Method-agnostic calibration interface. fit/transform are the pure core;
    persistence is provided by companion functions in calibration.sidecar
    (save_calibration / load_calibration) so the core stays I/O-free."""

    @classmethod
    @abc.abstractmethod
    def fit(cls, logits, labels, *, fit_population: str) -> "Calibrator": ...

    @abc.abstractmethod
    def transform(self, logits): ...


@dataclasses.dataclass(frozen=True)
class PlattCalibrator(Calibrator):
    A: float
    B: float
    fit_population: str          # e.g. "ldc_val_natural_balance" — what B is calibrated TO
    n: int                       # number of samples fit on
    method: str = "platt"

    @classmethod
    def fit(cls, logits, labels, *, fit_population: str) -> "PlattCalibrator":
        A, B = platt_fit(logits, labels)
        return cls(A=A, B=B, fit_population=fit_population, n=int(len(labels)))

    def transform(self, logits):
        return platt_transform(logits, self.A, self.B)
```

**Testing** (`tests/test_calibration_platt.py`, hypothesis where natural):
- **AC4.1**: fit on a synthetic natural-balance set; `transform` outputs all ∈ [0,1]; monotonic non-decreasing in the logit (property: for sorted logits, transformed probs are sorted — given `A>0`; the fit on separable-ish data yields `A>0`).
- **AC4.4**: `fit` requires the keyword-only `fit_population`; the value is stored on the calibrator; the docstring documents that rebalanced batches are disallowed.
- Determinism: same arrays → same `(A,B)`.

**Verification:** `uv run pytest tests/test_calibration_platt.py` → pass.
**Commit:** `feat(us-filter): Platt calibrator (pure core)`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `src/calibration/report.py` — calibration quality metrics

**Verifies:** us-filter.AC4.2, AC4.5.

**Files:**
- Create: `src/calibration/report.py`
- Create: `tests/test_calibration_report.py`

**Implementation** (`# pattern: Functional Core`):

```python
# pattern: Functional Core
# Reason: ECE / Brier / reliability are pure functions of (probs, labels).

from __future__ import annotations
import numpy as np


def calibration_report(probs, labels, n_bins: int = 15) -> dict:
    """Compute calibration quality on probabilities in [0,1].

    Returns {"ece", "brier", "reliability"} where reliability is a list of
    (mean_confidence, mean_accuracy, count) per non-empty equal-width bin.
    ECE = sum over bins of (count/N) * |mean_accuracy - mean_confidence|.
    Brier = mean((p - y)^2).
    """
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).astype(np.float64).reshape(-1)
    n = len(p)
    brier = float(np.mean((p - y) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]; clip so p==1.0 lands in the last bin.
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    reliability = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        reliability.append((conf, acc, cnt))
        ece += (cnt / n) * abs(acc - conf)
    return {"ece": float(ece), "brier": brier, "reliability": reliability}
```

**Testing** (`tests/test_calibration_report.py`):
- **AC4.2**: on a perfectly-calibrated synthetic set (probs equal observed bin frequencies) → ECE ≈ 0, low Brier; reliability has the expected bins; structure/keys correct.
- **AC4.5**: construct miscalibrated logits (e.g. overconfident); compare ECE of raw `sigmoid(logits)` vs `PlattCalibrator.fit(...).transform(logits)` on a held-out eval split → calibrated ECE ≤ raw ECE.
- Edge: all-one-class labels; single bin; `n_bins=1`.

**Verification:** `uv run pytest tests/test_calibration_report.py` → pass.
**Commit:** `feat(us-filter): calibration report (ECE/Brier/reliability)`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `src/calibration/sidecar.py` — persistence (Imperative Shell)

**Verifies:** us-filter.AC4.3.

**Files:**
- Create: `src/calibration/sidecar.py`
- Create: `tests/test_calibration_sidecar.py`

**Implementation** (`# pattern: Imperative Shell`):

```python
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
    if d.get("method", "platt") != "platt":
        raise ValueError(f"unsupported calibration method: {d.get('method')!r}")
    return PlattCalibrator(A=d["A"], B=d["B"], fit_population=d["fit_population"],
                           n=d["n"], method=d["method"])
```

**Testing** (`tests/test_calibration_sidecar.py`):
- **AC4.3**: fit a calibrator, `save_calibration` to `tmp_path`, `load_calibration`, assert the reloaded `transform` is fp-identical to the original on a logit grid (`np.linspace(-8, 8, 100)`); payload contains `{method, A, B, fit_population, n}`.
- `calibration_path_for_weights` maps `foo.weights.h5` → `foo.calibration.json`.
- `method != "platt"` raises.

**Verification:** `uv run pytest tests/test_calibration_sidecar.py` → pass.
**Commit:** `feat(us-filter): calibration sidecar persistence`
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

---

## Phase 5 Done When

- `PlattCalibrator.fit` fits on natural-balance arrays and `transform` is monotonic in [0,1]; `fit_population` recorded; rebalanced-batch fitting documented as disallowed.
- `calibration_report` computes ECE / Brier / reliability; Platt reduces ECE on a miscalibrated eval set.
- `save_calibration`/`load_calibration` round-trip to an identical transform via `.calibration.json`.
- The interface is method- and head-agnostic — no multi-head or PU machinery built (the convergence seam for later calibration work).

Covers **us-filter.AC4**.
