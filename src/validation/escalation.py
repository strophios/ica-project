# pattern: Functional Core
"""Fine-tuning escalation knobs and decision logic (AC3.3).

Provides grouping logic for per-layer discriminative learning rates and
a decision function to evaluate whether to escalate from frozen-probe
to unfreezing/fine-tuning based on transfer learning performance.
"""

from __future__ import annotations


def top_n_group_fn(n_top: int, n_layers: int = 12):
    """Create a group function for per-layer LR scaling with top-N unfreezing.

    Groups variables into three categories for LayerLRModel's multiplier-scaling:
    - "encoder_top": the top n_top RoBERTa layers (unfrozen in escalation mode)
    - "encoder_frozen": all other RoBERTa layers (frozen)
    - "head": classification head (always trained)

    The top N layers are identified by checking if any layer name in the set
    {roberta_layer_{n_layers-1}, ..., roberta_layer_{n_layers-n_top}} appears
    in the variable's path.

    Args:
        n_top: Number of top layers to unfreeze (0 <= n_top <= n_layers).
        n_layers: Total number of RoBERTa layers (default 12 for RoBERTa-base).

    Returns:
        A group function (var -> str) for use with LayerLRModel(group_fn=...).
    """
    # Pre-compute the set of top-layer names to check
    # If n_layers=12, n_top=2: check for "roberta_layer_11" and "roberta_layer_10"
    top_indices = [n_layers - 1 - i for i in range(n_top)]
    top_layer_names = {f"roberta_layer_{idx}" for idx in top_indices}

    def group_of(var):
        """Group a variable based on whether it's in the top N layers."""
        path = var.path

        # Check if any top-layer name appears in the path
        if any(layer_name in path for layer_name in top_layer_names):
            return "encoder_top"

        # Check if roberta appears in the path (but not in top layers)
        if "roberta" in path:
            return "encoder_frozen"

        # Everything else (head vars)
        return "head"

    return group_of


def escalation_decision(
    baseline_f1: float, transfer_f1: float, margin: float = 0.1
) -> dict[str, bool | str]:
    """Decide whether to escalate from frozen probe to unfreezing.

    Escalates (returns True) when the transfer F1 drops more than `margin`
    below the baseline F1. This is used to decide if fine-tuning on the
    transfer task is necessary.

    Args:
        baseline_f1: F1 score of the frozen probe on in-distribution data.
        transfer_f1: F1 score of the frozen probe on transfer (pre-1986) data.
        margin: Acceptable performance gap (default 0.1). Escalate when
                baseline_f1 - transfer_f1 > margin.

    Returns:
        Dict with keys:
        - escalate (bool): whether to fine-tune
        - rationale (str): human-readable explanation
    """
    gap = baseline_f1 - transfer_f1
    escalate = gap > margin

    rationale = (
        f"transfer F1 {transfer_f1:.3f} vs baseline {baseline_f1:.3f} "
        f"(gap {gap:.3f} {'>' if escalate else '<='} margin {margin})"
    )

    return {
        "escalate": escalate,
        "rationale": rationale,
    }
