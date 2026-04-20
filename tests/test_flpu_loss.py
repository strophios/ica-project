"""
Invariant tests for FLPULoss.

These tests target structural properties of the loss that should hold under
any reasonable variant of the implementation. They serve double duty as:

  1. Regression catch — when we change the alpha-handling default, port off
     TF-specific ops, swap reduction=None for "none", or do later refactor
     work, these tests should keep passing. Anything that breaks them is
     either a bug or a behavior change worth flagging.

  2. Spec for the eventual refactor — when FLPU gets pulled into a multi-head
     classifier, the new loss-application machinery must still satisfy these
     properties.

Design choices:

  - Tests use small, hand-constructable batches with known expected behavior
    rather than fitting on real data.
  - Tests assert *invariants* (loss is non-negative; order doesn't matter;
    correct predictions give low loss) rather than specific numerical values,
    so they survive minor implementation changes.
  - One test (`test_known_value_easy_batch`) does pin a specific numerical
    behavior — that an easy batch under the default config gives loss ≈ 0
    after the negative-risk clipping. This locks in the no-clawback math.
"""

import numpy as np
import pytest

# Import keras before our loss module so the backend gets initialized.
import keras  # noqa: F401

from src.loss_functions.loss import FLPULoss


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _batch(positive_logits, unlabeled_logits):
    """Build a (y_true, y_pred) pair from per-class logit lists.

    y_true is shape (n,), y_pred is shape (n, 1) — matching what the
    classifier head produces and what the preprocessor emits.
    """
    n_pos = len(positive_logits)
    n_unl = len(unlabeled_logits)
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_unl)]).astype("float32")
    y_pred = np.concatenate([positive_logits, unlabeled_logits]).astype("float32")
    y_pred = y_pred.reshape(-1, 1)
    return y_true, y_pred


def _scalar(loss_output):
    """Convert a Keras Loss return value (TF tensor) to a Python float."""
    return float(np.asarray(loss_output))


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

class TestConstruction:
    def test_valid_priors_construct(self):
        FLPULoss(prior=0.03)
        FLPULoss(prior=0.5)
        FLPULoss(prior=0.99)

    @pytest.mark.parametrize("bad_prior", [0.0, -0.1, 1.0, 1.5])
    def test_invalid_prior_raises(self, bad_prior):
        with pytest.raises(NotImplementedError):
            FLPULoss(prior=bad_prior)


# -----------------------------------------------------------------------------
# Output structure
# -----------------------------------------------------------------------------

class TestOutputStructure:
    def test_returns_scalar(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch([1.0, 2.0], [-1.0, -2.0])
        out = loss(y_true, y_pred)
        # Convert to numpy and verify it's a 0-d array (a scalar).
        arr = np.asarray(out)
        assert arr.ndim == 0, f"Expected scalar output, got shape {arr.shape}"

    def test_returns_finite(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch([0.5, 1.0], [-0.5, -1.0])
        out = _scalar(loss(y_true, y_pred))
        assert np.isfinite(out), f"Loss should be finite, got {out}"


# -----------------------------------------------------------------------------
# Easy vs adversarial batches
# -----------------------------------------------------------------------------

class TestEasyVsAdversarial:
    def test_easy_batch_low_loss(self):
        """Positives predicted very positive, unlabeleds very negative.
        Both nnPU terms should be small or get clipped."""
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch([10.0, 10.0, 10.0], [-10.0, -10.0, -10.0])
        out = _scalar(loss(y_true, y_pred))
        assert out < 0.1, f"Easy batch loss should be near 0, got {out}"

    def test_adversarial_batch_high_loss(self):
        """Positives predicted very negative, unlabeleds very positive.
        Both terms should be large."""
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0])
        out = _scalar(loss(y_true, y_pred))
        assert out > 1.0, f"Adversarial batch loss should be large, got {out}"

    def test_known_value_easy_batch_clips_to_zero(self):
        """Pin the no-clawback math: with default kiryo_clawback=False, an
        easy batch where positives-predicted-positive (low base loss) and
        unlabeleds-predicted-negative (low base loss for negative-risk first
        term, but large base loss for the bias-correction subtraction term)
        should make negative_risk go negative, get clipped to 0, leaving only
        the small positive_risk."""
        loss = FLPULoss(prior=0.5, kiryo_clawback=False)
        y_true, y_pred = _batch([10.0], [-10.0])
        out = _scalar(loss(y_true, y_pred))
        # positive_risk ≈ 0 (positives correctly classified)
        # negative_risk ≈ 0 - 0.5 * (large) → clipped to 0
        # Total ≈ 0
        assert out < 0.01, f"Expected near-zero loss, got {out}"


# -----------------------------------------------------------------------------
# Order invariance
# -----------------------------------------------------------------------------

class TestOrderInvariance:
    def test_loss_invariant_under_permutation(self):
        loss = FLPULoss(prior=0.1)
        y_true_a, y_pred_a = _batch([0.5, 1.0, 1.5], [-0.5, -1.0, -1.5])

        # Same samples, reversed order
        perm = np.array([5, 4, 3, 2, 1, 0])
        y_true_b = y_true_a[perm]
        y_pred_b = y_pred_a[perm]

        out_a = _scalar(loss(y_true_a, y_pred_a))
        out_b = _scalar(loss(y_true_b, y_pred_b))
        np.testing.assert_allclose(out_a, out_b, rtol=1e-5)


# -----------------------------------------------------------------------------
# Non-negativity (with default kiryo_clawback=False)
# -----------------------------------------------------------------------------

class TestNonNegativity:
    @pytest.mark.parametrize(
        "pos_logits,unl_logits",
        [
            ([1.0], [-1.0]),
            ([10.0, 10.0], [-10.0]),
            ([-10.0, -10.0], [10.0]),
            ([0.0, 0.5, 1.0], [0.0, -0.5, -1.0]),
        ],
    )
    def test_loss_is_non_negative_without_clawback(self, pos_logits, unl_logits):
        loss = FLPULoss(prior=0.1, kiryo_clawback=False)
        y_true, y_pred = _batch(pos_logits, unl_logits)
        out = _scalar(loss(y_true, y_pred))
        assert out >= 0.0, f"Loss must be non-negative, got {out}"


# -----------------------------------------------------------------------------
# Prior sensitivity
# -----------------------------------------------------------------------------

class TestPriorSensitivity:
    def test_different_priors_give_different_loss(self):
        """Sanity check that the prior actually affects the loss value.
        Uses an adversarial batch to make sure positive_risk is nonzero."""
        y_true, y_pred = _batch([-2.0, -2.0], [2.0, 2.0, 2.0])

        loss_low = FLPULoss(prior=0.05)
        loss_high = FLPULoss(prior=0.5)

        out_low = _scalar(loss_low(y_true, y_pred))
        out_high = _scalar(loss_high(y_true, y_pred))

        assert out_low != out_high, (
            f"Different priors should give different losses, "
            f"got {out_low} (prior=0.05) vs {out_high} (prior=0.5)"
        )
