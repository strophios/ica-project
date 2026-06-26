# pattern: Functional Core
# Reason: pure combination math — elementwise product and logistic regression fitting/transformation
# are referentially transparent. Sidecar I/O lives in sidecar.py.

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression


def combine_and(
    p_cca: np.ndarray | list[float],
    p_rel: np.ndarray | list[float],
) -> np.ndarray:
    """Calibrated-AND combiner: elementwise product of two probability arrays.

    Args:
        p_cca: probability array from CCA head, shape (n,) or (n, 1)
        p_rel: probability array from relevance head, shape (n,) or (n, 1)

    Returns:
        Combined probabilities as elementwise product, shape (n,)

    Properties:
        - Monotone increasing in both arguments (p_cca * p_rel increases when either increases)
        - Commutative (order of arguments doesn't matter)
        - Idempotent with itself (combine_and(p, p) == p * p, not p)
    """
    p_cca = np.asarray(p_cca, dtype=np.float64).ravel()
    p_rel = np.asarray(p_rel, dtype=np.float64).ravel()

    if len(p_cca) != len(p_rel):
        raise ValueError(
            f"probability array lengths must match: {len(p_cca)} vs {len(p_rel)}"
        )

    return p_cca * p_rel


def fit_logistic_combiner(
    scores: np.ndarray | list,
    labels: np.ndarray | list,
    random_state: int = 42,
) -> LogisticRegression:
    """Fit logistic regression combiner over ≤3 features.

    Expects scores array with columns: z_cca, z_rel, [optional z_us].
    Fits a LogisticRegression(penalty=None) on the scores → labels mapping.

    Args:
        scores: feature matrix, shape (n, 2) or (n, 3). Columns expected:
                [z_cca, z_rel] or [z_cca, z_rel, z_us].
        labels: binary labels, shape (n,)
        random_state: fixed seed for determinism (default 42)

    Returns:
        Fitted LogisticRegression model (coef_ and intercept_ populated)

    Raises:
        ValueError: if scores has <2 or >3 columns
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int).ravel()

    if scores.ndim != 2:
        raise ValueError(
            f"scores must be 2-D array, got shape {scores.shape}"
        )

    n_features = scores.shape[1]
    if n_features < 2 or n_features > 3:
        raise ValueError(
            f"scores must have 2 or 3 columns, got {n_features}"
        )

    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels length mismatch: {len(scores)} vs {len(labels)}"
        )

    lr = LogisticRegression(penalty=None, random_state=random_state)
    lr.fit(scores, labels)
    return lr


def apply_logistic_combiner(
    model_or_coefs: LogisticRegression | tuple[np.ndarray, float],
    scores: np.ndarray | list,
) -> np.ndarray:
    """Apply logistic regression combiner to scores, returning probabilities.

    Args:
        model_or_coefs: either a fitted LogisticRegression, or a (coef, intercept) tuple
        scores: feature matrix, shape (n, 2) or (n, 3)

    Returns:
        Calibrated probabilities in [0, 1], shape (n,)
    """
    scores = np.asarray(scores, dtype=np.float64)

    if isinstance(model_or_coefs, LogisticRegression):
        return model_or_coefs.predict_proba(scores)[:, 1]

    # Tuple (coef, intercept) case
    coef, intercept = model_or_coefs
    coef = np.asarray(coef, dtype=np.float64).ravel()

    if coef.shape[0] != scores.shape[1]:
        raise ValueError(
            f"coef shape mismatch: {coef.shape[0]} vs {scores.shape[1]}"
        )

    # Linear combination: z_cca * coef[0] + z_rel * coef[1] + (z_us * coef[2] if present) + intercept
    logits = scores @ coef + intercept
    # Sigmoid transformation
    return 1.0 / (1.0 + np.exp(-logits))


@dataclasses.dataclass(frozen=True)
class FusionConfig:
    """Configuration for fusion combiner (AND vs logistic regression).

    Attributes:
        gate_threshold: decision threshold for the US gate (τ_us), float in [0, 1]
        combine: which combiner to use, "product" (calibrated-AND) or "logreg" (≤3-param LR)
        coefs: logistic regression coefficients (required when combine=="logreg"), or None
        score_space: whether scores/logits are in "prob" or "logit" space (for schema clarity)
        includes_us: whether the US head is included in the combiner (affects feature count)
    """

    gate_threshold: float
    combine: Literal["product", "logreg"]
    coefs: list[float] | None
    score_space: Literal["prob", "logit"]
    includes_us: bool

    def __post_init__(self) -> None:
        """Validate configuration on construction."""
        # Check combine value
        if self.combine not in ("product", "logreg"):
            raise ValueError(
                f"combine must be 'product' or 'logreg', got {self.combine!r}"
            )

        # Validate coefs requirement
        if self.combine == "logreg":
            if self.coefs is None:
                raise ValueError(
                    "coefs is required when combine=='logreg'"
                )
            # Check coefficient count: 2 for CCA+rel, 3 for CCA+rel+US
            expected_count = 3 if self.includes_us else 2
            if len(self.coefs) != expected_count:
                raise ValueError(
                    f"coefs must have {expected_count} elements (includes_us={self.includes_us}), "
                    f"got {len(self.coefs)}"
                )

        if self.combine == "product" and self.coefs is not None:
            raise ValueError(
                "coefs must be None when combine=='product'"
            )

        # Validate gate_threshold
        if not (0.0 <= self.gate_threshold <= 1.0):
            raise ValueError(
                f"gate_threshold must be in [0, 1], got {self.gate_threshold}"
            )

        # Validate score_space
        if self.score_space not in ("prob", "logit"):
            raise ValueError(
                f"score_space must be 'prob' or 'logit', got {self.score_space!r}"
            )
