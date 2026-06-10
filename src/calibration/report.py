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
