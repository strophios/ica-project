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


def _loss_exposes_intermediates(loss_fn) -> bool:
    """True iff loss_fn.call accepts a `return_intermediates` parameter.

    Phase-ordering: the real FLPULoss gains this parameter in Phase 3.
    Phase 2 only ever sees synthetic stand-in losses; the factory meets
    the real FLPULoss at Phase 6.
    """
    if loss_fn is None:
        return False
    call = getattr(loss_fn, "call", None)
    if call is None:
        return False
    try:
        sig = inspect.signature(call)
    except (ValueError, TypeError):
        return False
    return "return_intermediates" in sig.parameters


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

    if config.enable_batch_balance:
        for head_name in heads:
            per_step["batch_target"].append(BatchLabelBalanceTracker(head_name))

    if config.enable_loss_components:
        supporting = [
            name for name, head in heads.items()
            if _loss_exposes_intermediates(head.loss_fn)
        ]
        if heads and not supporting:
            raise ValueError(
                "DiagnosticsConfig.enable_loss_components is True but no "
                "head's loss exposes `return_intermediates`; loss-component "
                f"tracking would produce nothing. Heads: {sorted(heads)}."
            )
        for name in heads:
            if name not in supporting:
                warnings.warn(
                    f"Head {name!r} loss does not expose "
                    f"`return_intermediates`; skipping its loss-component "
                    f"trackers.",
                    stacklevel=2,
                )
                continue
            for key in _FLPU_COMPONENT_KEYS:
                per_step["loss_component"].append(
                    LossComponentTracker(name, key, _FLPU_LOSS_COMPONENT_AGG)
                )

    return DiagnosticBundle(per_step=per_step, periodic=periodic)
