"""
Compare DEDPUL prior estimation when fed (a) raw logits — what the current
`run_prior_estimate.py` does — vs. (b) sigmoid-transformed probabilities
(what DEDPUL actually expects).

The current pipeline reverses logits via `lu_preds - 2*lu_preds = -lu_preds`,
which is the unlabeled-class logit, then passes that to DEDPUL with
`kde_mode="prob"`. DEDPUL expects values in [0, 1] (probabilities of being
unlabeled). Feeding it logits in (-∞, ∞) builds the KDE in the wrong space.

This script runs DEDPUL both ways on the cached L/U classifier predictions
and reports the two resulting class priors. If the fix matters, the prior
will change materially.

Run from project root:
    uv run python scripts/compare_dedpul_logit_vs_prob.py
"""

import os

import numpy as np

from src.prior_estimation.dedpul_em import estimate_diff, estimate_poster_em


# ---------------------------------------------------------------------------
# Configuration: where the cached L/U classifier predictions live.
# ---------------------------------------------------------------------------
LOCAL_PATH = os.path.expanduser(
    "~/immigration_project/00_ML_data_expansion/00_explorer"
)
LU_DIR = os.path.join(LOCAL_PATH, "cca_set", "lu")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def reverse_via_negation(preds):
    """The current (broken) reversal: matches `run_prior_estimate.py`.

    `lu_preds - 2*lu_preds == -lu_preds`. If `preds` are logits, this is
    the unlabeled-class logit. Still in (-∞, ∞), not in [0, 1]."""
    return preds - 2 * preds


def reverse_via_sigmoid(preds):
    """The fix: convert to probability of being unlabeled (in [0, 1])."""
    return 1.0 - sigmoid(preds)


def main():
    print(f"Loading L/U predictions from {LU_DIR}")
    lu_preds = np.load(os.path.join(LU_DIR, "lu_preds.npy"))
    lu_targets = np.load(os.path.join(LU_DIR, "lu_targets.npy"))

    # Flatten predictions to shape (N,) and targets likewise.
    lu_preds = lu_preds.reshape(-1)
    lu_targets = lu_targets.reshape(-1)

    print(f"  preds shape: {lu_preds.shape}")
    print(f"  targets shape: {lu_targets.shape}")
    print(
        f"  pred range: [{lu_preds.min():.3f}, {lu_preds.max():.3f}], "
        f"mean: {lu_preds.mean():.3f}"
    )
    print(
        f"  targets: 0 (unlabeled) count={int((lu_targets == 0).sum())}, "
        f"1 (labeled) count={int((lu_targets == 1).sum())}"
    )

    # Reverse targets for DEDPUL convention (0 = positive/labeled, 1 = unlabeled)
    lu_targets_rev = 1 - lu_targets

    # ----- Variant A: current (broken) reversal — pass raw negated logits -----
    print("\n=== Variant A: current pipeline (logits passed to DEDPUL) ===")
    preds_a = reverse_via_negation(lu_preds)
    print(
        f"  reversed pred range: [{preds_a.min():.3f}, {preds_a.max():.3f}], "
        f"mean: {preds_a.mean():.3f}"
    )
    diffs_a = estimate_diff(preds_a, lu_targets_rev, tune=True, kde_mode="prob")
    alpha_a, _ = estimate_poster_em(diffs_a, preds_a, lu_targets_rev)
    print(f"  DEDPUL α (share of negatives in unlabeled): {alpha_a:.4f}")
    print(f"  Implied π_pos = 1 - α: {1 - alpha_a:.4f}")

    # ----- Variant B: fixed reversal — pass probabilities -----
    print("\n=== Variant B: fixed (probabilities passed to DEDPUL) ===")
    preds_b = reverse_via_sigmoid(lu_preds)
    print(
        f"  reversed pred range: [{preds_b.min():.4f}, {preds_b.max():.4f}], "
        f"mean: {preds_b.mean():.4f}"
    )
    diffs_b = estimate_diff(preds_b, lu_targets_rev, tune=True, kde_mode="prob")
    alpha_b, _ = estimate_poster_em(diffs_b, preds_b, lu_targets_rev)
    print(f"  DEDPUL α (share of negatives in unlabeled): {alpha_b:.4f}")
    print(f"  Implied π_pos = 1 - α: {1 - alpha_b:.4f}")

    # ----- Comparison -----
    print("\n=== Comparison ===")
    print(f"  α: {alpha_a:.4f} (current) vs {alpha_b:.4f} (fixed)")
    print(f"  π_pos: {1 - alpha_a:.4f} (current) vs {1 - alpha_b:.4f} (fixed)")
    print(f"  Currently used in run_cca_classification.py: prior=0.0300")
    if abs((1 - alpha_a) - (1 - alpha_b)) > 0.01:
        print(
            "  → MATERIAL difference (>0.01 in π_pos). "
            "FLPU should be re-trained with the corrected prior."
        )
    else:
        print(
            "  → small difference (<0.01 in π_pos). "
            "Fix is conceptually correct but didn't change the prior much."
        )


if __name__ == "__main__":
    main()
