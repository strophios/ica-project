# pattern: Functional Core

"""Constructs the Tier 5 DiagnosticBundle from config + model state.

Pure construction: parameters in, a bundle of (stateful) tracker objects
out. No I/O. One source of truth for what diagnostics a run instantiates,
mirroring src/cca_metrics.py:make_cca_metrics.
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, TypedDict

import keras

from src.cca_config import DiagnosticsConfig
from src.diagnostics.trackers import (
    BatchLabelBalanceTracker,
    GradientFiniteTracker,
    LossComponentTracker,
    PerGroupGradNormTracker,
)

if TYPE_CHECKING:
    from src.model_setup.heads import ClassificationHead

__all__ = ["DiagnosticBundle", "build_trackers"]

# FLPU loss-component intermediates (Phase 3 exposes these via
# FLPULoss.call(return_intermediates=True)). Aggregation is fixed to "mean"
# (design DoD only needs mean-tracked components; no config knob — YAGNI).
_FLPU_COMPONENT_KEYS = ("positive_risk", "negative_risk", "correction_triggered")
_FLPU_LOSS_COMPONENT_AGG = "mean"


class DiagnosticBundle(TypedDict):
    per_step: dict[str, list[keras.metrics.Metric]]
    periodic: list  # permanently []; forward-compat slot, no current consumer
                     # (Phase 5 prediction-distribution metrics ride the head
                     # metric path, not this bundle — see phase_05.md)


def build_trackers(
    config: DiagnosticsConfig,
    *,
    group_fn,
    heads: dict[str, "ClassificationHead"],
    trainable_variables,
) -> DiagnosticBundle:
    """Construct the per-step + periodic diagnostic bundle.

    Groups are derived by walking `trainable_variables` with `group_fn`
    (deterministic, sorted). Under the frozen-encoder default, only head
    variables are trainable, so no encoder grad-norm tracker is built — this
    is correct (see tier5-design.md).
    """
    per_step: dict[str, list[keras.metrics.Metric]] = {
        "gradient": [],
        "loss_component": [],
        "batch_target": [],
    }
    periodic: list = []  # permanently empty forward-compat slot (no consumer)

    groups = sorted({group_fn(v) for v in trainable_variables})

    if config.enable_gradient_norms:
        for group in groups:
            for agg in config.gradient_norm_aggregations:
                per_step["gradient"].append(PerGroupGradNormTracker(group, agg))

    if config.enable_overflow_proxy:
        per_step["gradient"].append(GradientFiniteTracker())

    return DiagnosticBundle(per_step=per_step, periodic=periodic)
