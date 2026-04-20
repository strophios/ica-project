"""
Attribution analysis for the DEDPUL prior estimate.

The first version of this script compared only two configurations — raw logits
passed to DEDPUL (the buggy original pipeline) vs sigmoid-converted probabilities
(the fix) — both with `kde_mode="prob"` and `tune=True`. That A/B lumped three
separable effects together:

  1. The sigmoid fix: probabilities in [0, 1] vs logits in (-∞, ∞).
  2. The kde_mode choice: "prob" (our ad-hoc default) vs "logit" (DEDPUL's
     actual default), which changes the space the KDE is built in.
  3. The bandwidth grid: `maximize_log_likelihood` searches bandwidths in
     [0.01, 0.4], calibrated for [0, 1]-valued inputs. For logit-scale
     inputs (range ≈ [-4, 4] here), that grid is far too narrow — the KDE
     ends up extremely spiky and the density ratio becomes noisy.

An adversarial review correctly flagged that the "material 0.04 → 0.02 shift"
conclusion from the original 2-variant comparison conflates these. This
expanded version runs four variants so we can attribute the shift to
specific causes and decide the right configuration for the production
pipeline.

Variants:
  A. (broken logits, mode="prob", tune=True)       — the original broken pipeline
  B. (broken logits, mode="prob", bandwidth=1.0)   — isolates the bandwidth effect
  C. (probabilities, mode="prob", tune=True)       — the "fix" as currently in code
  D. (probabilities, mode="logit", tune=True)      — DEDPUL's actual default

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


def run_variant(name, preds, targets, **kwargs):
    """Run DEDPUL with the given inputs and kwargs, return (alpha, pi_pos)."""
    diffs = estimate_diff(preds, targets, **kwargs)
    alpha, _ = estimate_poster_em(diffs, preds, targets)
    pi_pos = 1 - alpha
    print(f"  [{name}] α = {alpha:.4f}  →  π_pos = {pi_pos:.4f}")
    return alpha, pi_pos


def main():
    print(f"Loading L/U predictions from {LU_DIR}")
    lu_preds = np.load(os.path.join(LU_DIR, "lu_preds.npy")).reshape(-1)
    lu_targets = np.load(os.path.join(LU_DIR, "lu_targets.npy")).reshape(-1)

    print(
        f"  preds: shape={lu_preds.shape}  range=[{lu_preds.min():.3f}, "
        f"{lu_preds.max():.3f}]  mean={lu_preds.mean():.3f}"
    )
    print(
        f"  targets: 0 (unlabeled)={int((lu_targets == 0).sum())}, "
        f"1 (labeled)={int((lu_targets == 1).sum())}"
    )

    # DEDPUL convention: 0 = positive/labeled, 1 = unlabeled.
    lu_targets_rev = 1 - lu_targets

    # Pre-compute both preprocessing variants once.
    logits_reversed = reverse_via_negation(lu_preds)  # negated logit
    probs_reversed = reverse_via_sigmoid(lu_preds)  # 1 - σ(logit), in [0, 1]

    print(
        f"\n  logit-reversed range: [{logits_reversed.min():.3f}, "
        f"{logits_reversed.max():.3f}]"
    )
    print(
        f"  prob-reversed  range: [{probs_reversed.min():.4f}, "
        f"{probs_reversed.max():.4f}]"
    )

    results = {}

    # Variant A: what the buggy original pipeline did.
    print("\n=== Variant A: logits, mode='prob', tune=True (BROKEN ORIGINAL) ===")
    results["A"] = run_variant(
        "A",
        logits_reversed,
        lu_targets_rev,
        tune=True,
        kde_mode="prob",
    )

    # Variant B: logits but with bandwidth appropriate for the logit scale.
    # tune's grid [0.01, 0.4] is way too narrow for data in [-4, 4]. Using
    # a fixed bandwidth on the scale of the data (bw=1.0 ≈ 25% of the range).
    print("\n=== Variant B: logits, mode='prob', bw=1.0 (LOGIT-SCALE BANDWIDTH) ===")
    results["B"] = run_variant(
        "B",
        logits_reversed,
        lu_targets_rev,
        tune=False,
        bw_mix=1.0,
        bw_pos=1.0,
        kde_mode="prob",
    )

    # Variant C: the fix as currently shipped.
    print("\n=== Variant C: probs, mode='prob', tune=True (CURRENT FIX) ===")
    results["C"] = run_variant(
        "C",
        probs_reversed,
        lu_targets_rev,
        tune=True,
        kde_mode="prob",
    )

    # Variant D: probs with DEDPUL's actual default kde_mode="logit".
    # (DEDPUL internally does log(p/(1-p)) and fits the KDE in logit space.)
    print("\n=== Variant D: probs, mode='logit', tune=True (DEDPUL DEFAULT) ===")
    results["D"] = run_variant(
        "D",
        probs_reversed,
        lu_targets_rev,
        tune=True,
        kde_mode="logit",
    )

    # Attribution table.
    print("\n=== Attribution summary ===")
    print(f"  {'variant':<10} {'preproc':<12} {'mode':<8} {'bw':<10} {'π_pos':<8}")
    print(f"  {'-' * 50}")
    labels = {
        "A": ("logits",      "prob",  "tune"),
        "B": ("logits",      "prob",  "bw=1.0"),
        "C": ("probs",       "prob",  "tune"),
        "D": ("probs",       "logit", "tune"),
    }
    for key in ("A", "B", "C", "D"):
        preproc, mode, bw = labels[key]
        _, pi_pos = results[key]
        print(f"  {key:<10} {preproc:<12} {mode:<8} {bw:<10} {pi_pos:<8.4f}")

    print()
    print(f"  Currently baked into run_cca_classification.py: prior=0.0300")
    print(f"  Current committed fix (Variant C): π_pos = {results['C'][1]:.4f}")
    print()

    # Isolate the three effects.
    pi_A = results["A"][1]
    pi_B = results["B"][1]
    pi_C = results["C"][1]
    pi_D = results["D"][1]
    print("  Attribution of the A → D shift:")
    print(f"    A → B  (bandwidth  effect alone): {pi_B - pi_A:+.4f}  (logits+tune → logits+bw=1.0)")
    print(f"    A → C  (sigmoid    fix, keep mode='prob'):  {pi_C - pi_A:+.4f}")
    print(f"    C → D  (mode='prob' → mode='logit'):       {pi_D - pi_C:+.4f}")
    print(f"    A → D  (sigmoid + mode='logit'):           {pi_D - pi_A:+.4f}")


if __name__ == "__main__":
    main()
