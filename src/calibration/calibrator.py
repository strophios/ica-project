# pattern: Functional Core
# Reason: pure calibration math — fitting params from arrays and mapping logits
# to probabilities are referentially transparent. Sidecar I/O lives in sidecar.py.

from __future__ import annotations

import abc
import dataclasses

import numpy as np
from sklearn.linear_model import LogisticRegression


def platt_fit(logits, labels, sample_weight=None):
    """Fit sigmoid(A*z + B) by 1-D logistic regression of label ~ logit.
    Returns (A, B). `logits`/`labels` are 1-D arrays over a held-out split at
    its NATURAL class balance (rebalanced batches are disallowed — they skew B).

    `sample_weight` (optional): per-row weights. Use to calibrate a score-stratified
    gold set to corpus proportions via inverse-probability weights (the gold set is
    not natural-balance, so the weights stand in for it). None → unweighted fit.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(labels).astype(int).reshape(-1)
    lr = LogisticRegression(penalty=None)
    lr.fit(z, y, sample_weight=sample_weight)
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
    def fit(cls, logits, labels, *, fit_population: str,
            sample_weight=None) -> "PlattCalibrator":
        A, B = platt_fit(logits, labels, sample_weight=sample_weight)
        return cls(A=A, B=B, fit_population=fit_population, n=int(len(labels)))

    def transform(self, logits):
        return platt_transform(logits, self.A, self.B)
