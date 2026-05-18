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
  - Most tests assert *invariants* (loss is non-negative; order doesn't
    matter; correct predictions give low loss) rather than specific numerical
    values, so they survive minor implementation changes.
  - One test (`test_matches_numpy_reference`) pins the full numerical value
    against an independent numpy implementation of FLPU. This is the strongest
    test in the suite and catches algorithmic errors the structural tests miss
    (mask swaps, prior misplacement, denominator swaps, etc.).
  - Separate classes exercise the clawback branch (two sub-cases), the
    edge-case batches (all-positive / all-unlabeled), mixed_float16 dtype
    handling, and the focal-loss shape contract that FLPU's masking relies on.
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


def _numpy_reference_flpu(
    y_true,
    y_pred,
    prior,
    focal_gamma=2.0,
    nn_beta=0.0,
    nn_gamma=1.0,
    kiryo_clawback=False,
):
    """Reference implementation of FLPU computed in numpy.

    Used to cross-check the production FLPU against an independent computation
    of the same formula. If the production code has a mask swap, denominator
    swap, prior misplacement, or sign error in the bias-correction term,
    this reference disagrees.

    Mirrors the math in `src/loss_functions/loss.py` but constructs every
    intermediate in numpy from primitives, so a bug in the Keras-side
    implementation does not mask itself here.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.float64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.float64)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def focal(y, logit):
        """Focal loss without class balancing, per Keras's
        `BinaryFocalCrossentropy(apply_class_balancing=False, gamma=γ,
        from_logits=True, reduction='none')`."""
        p = sigmoid(logit)
        # p_t is the predicted probability of the true class
        p_t = np.where(y == 1, p, 1.0 - p)
        # Clip to avoid log(0) in case of extreme logits
        p_t = np.clip(p_t, 1e-12, 1.0 - 1e-12)
        return -((1.0 - p_t) ** focal_gamma) * np.log(p_t)

    positive = (y_true == 1).astype(np.float64)
    unlabeled = (y_true == 0).astype(np.float64)
    n_positive = max(positive.sum(), 1.0)
    n_unlabeled = max(unlabeled.sum(), 1.0)

    fl_as_labeled = focal(y_true, y_pred)
    fl_flipped = focal(1 - y_true, y_pred)

    y_positive = fl_as_labeled * positive
    y_unlabeled = fl_as_labeled * unlabeled
    y_positive_inv = fl_flipped * positive

    positive_risk = prior * y_positive.sum() / n_positive
    negative_risk = (
        y_unlabeled.sum() / n_unlabeled
        - prior * y_positive_inv.sum() / n_positive
    )

    if not kiryo_clawback:
        return float(positive_risk + max(0.0, negative_risk))
    if negative_risk < -nn_beta:
        return float(-nn_gamma * negative_risk)
    return float(positive_risk + negative_risk)


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

    def test_higher_prior_amplifies_adversarial_loss(self):
        """On an adversarial batch (positives misclassified as negative,
        unlabeled misclassified as positive), both the positive-risk term
        and the positive-as-negative bias-correction term grow with π_p.
        The positive_risk term dominates, so higher prior should give
        higher total loss. This is a stronger check than mere inequality:
        it verifies the direction of sensitivity is correct, which mere
        inequality would miss if the prior were accidentally applied to
        the wrong term."""
        y_true, y_pred = _batch(
            positive_logits=[-3.0, -3.0, -3.0],  # positives predicted very negative
            unlabeled_logits=[3.0, 3.0, 3.0],  # unlabeled predicted very positive
        )
        out_low = _scalar(FLPULoss(prior=0.05)(y_true, y_pred))
        out_mid = _scalar(FLPULoss(prior=0.25)(y_true, y_pred))
        out_high = _scalar(FLPULoss(prior=0.5)(y_true, y_pred))

        assert out_low < out_mid < out_high, (
            f"Expected monotonic increase with prior, got "
            f"π=0.05: {out_low}, π=0.25: {out_mid}, π=0.5: {out_high}"
        )


# -----------------------------------------------------------------------------
# Numerical correctness vs independent reference
# -----------------------------------------------------------------------------

class TestNumpyReference:
    """Cross-check the production FLPU against a numpy reference implementation
    computed from the formula independently. This is the strongest test in the
    suite — it catches mask swaps, denominator swaps, sign errors, and prior
    misplacements that pure-structural tests would miss."""

    def test_matches_numpy_reference_asymmetric_batch(self):
        """A multi-sample batch with asymmetric positive/unlabeled counts and
        logits spanning the sigmoid domain. Structural tests with 1-of-each
        batches cannot distinguish many plausible implementation bugs
        (e.g., swapping n_positive and n_unlabeled gives the same answer
        when both are 1). This test uses 3 positives and 4 unlabeled with
        varied logits so those degeneracies are broken."""
        loss = FLPULoss(prior=0.1, focal_gamma=2.0)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        actual = _scalar(loss(y_true, y_pred))
        expected = _numpy_reference_flpu(
            y_true,
            y_pred,
            prior=0.1,
            focal_gamma=2.0,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_matches_numpy_reference_gamma_zero(self):
        """With γ=0, focal loss reduces to plain BCE. Test that FLPU still
        matches the reference in this simpler case — a check that the
        reference and production agree on the easy case before trusting
        agreement on harder ones."""
        loss = FLPULoss(prior=0.2, focal_gamma=0.0)
        y_true, y_pred = _batch(
            positive_logits=[1.5, -0.5],
            unlabeled_logits=[0.3, -1.2, 0.8],
        )
        actual = _scalar(loss(y_true, y_pred))
        expected = _numpy_reference_flpu(
            y_true, y_pred, prior=0.2, focal_gamma=0.0
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)


# -----------------------------------------------------------------------------
# Kiryo clawback branch
# -----------------------------------------------------------------------------

class TestKiryoClawback:
    """The clawback branch (`kiryo_clawback=True`) has its own math: when
    `negative_risk < -nn_beta`, return `-nn_gamma * negative_risk` instead
    of clipping. It uses `ops.cond` with Python lambdas and was previously
    untested — this suite exercises both the in-bound and out-of-bound
    cases."""

    def test_in_bound_returns_positive_plus_negative(self):
        """When negative_risk ≥ -nn_beta (in-bound), clawback returns
        `positive_risk + negative_risk` directly (no clipping, no flip)."""
        loss_cb = FLPULoss(prior=0.1, kiryo_clawback=True)
        # Adversarial batch: positives misclassified, unlabeled misclassified.
        # negative_risk will be positive (R_u^- large, π * R_p^- small),
        # so we are in-bound.
        y_true, y_pred = _batch(
            positive_logits=[-3.0, -3.0],
            unlabeled_logits=[3.0, 3.0, 3.0],
        )
        out = _scalar(loss_cb(y_true, y_pred))
        expected = _numpy_reference_flpu(
            y_true, y_pred, prior=0.1, kiryo_clawback=True
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4)
        # Sanity: the in-bound case returns the raw sum, which should be
        # positive on an adversarial batch.
        assert out > 0

    def test_out_of_bound_returns_flipped_negative_risk(self):
        """When negative_risk < -nn_beta (out-of-bound), clawback returns
        `-nn_gamma * negative_risk`, which is positive. Uses an easy batch
        where negative_risk goes very negative (positives predicted
        correctly but still contribute large positive-as-negative loss)."""
        loss_cb = FLPULoss(prior=0.5, kiryo_clawback=True, nn_gamma=1.0)
        # Easy batch: positive at logit=+10, unlabeled at logit=-10.
        # negative_risk = ~0 - 0.5 × large ≈ very negative → clawback fires.
        y_true, y_pred = _batch(
            positive_logits=[10.0],
            unlabeled_logits=[-10.0],
        )
        out = _scalar(loss_cb(y_true, y_pred))
        expected = _numpy_reference_flpu(
            y_true, y_pred, prior=0.5, kiryo_clawback=True, nn_gamma=1.0
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4)
        # The clawback return value is -γ * negative_risk, so it is positive.
        assert out > 0

    def test_clawback_nn_gamma_scales_output(self):
        """In the out-of-bound branch, the return is `-nn_gamma *
        negative_risk`, so doubling nn_gamma should double the output."""
        y_true, y_pred = _batch([10.0], [-10.0])

        out_g1 = _scalar(FLPULoss(prior=0.5, kiryo_clawback=True, nn_gamma=1.0)(y_true, y_pred))
        out_g2 = _scalar(FLPULoss(prior=0.5, kiryo_clawback=True, nn_gamma=2.0)(y_true, y_pred))

        np.testing.assert_allclose(out_g2, 2.0 * out_g1, rtol=1e-4)


# -----------------------------------------------------------------------------
# Edge-case batches
# -----------------------------------------------------------------------------

class TestEdgeCaseBatches:
    """The `min_count=1.0` floor in FLPU exists specifically to handle
    batches that are missing one class entirely. These tests exercise that
    path and lock in the intended behavior."""

    def test_all_positive_batch(self):
        """No unlabeled samples. y_unlabeled = 0, y_positive_inv contributes
        nothing to negative_risk since there's no unlabeled-as-negative term,
        so negative_risk = -π_p × mean(y_positive_inv) ≤ 0. Under
        no-clawback clipping, total = positive_risk."""
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[1.0, 2.0, 3.0],
            unlabeled_logits=[],
        )
        actual = _scalar(loss(y_true, y_pred))
        expected = _numpy_reference_flpu(y_true, y_pred, prior=0.1)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)
        assert actual >= 0

    def test_all_unlabeled_batch(self):
        """No positive samples. y_positive = 0, y_positive_inv = 0. Total
        reduces to `max(0, mean(focal(y=0, x_u)))`, i.e., the mean
        unlabeled-as-negative loss."""
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[],
            unlabeled_logits=[-1.0, -2.0, -3.0],
        )
        actual = _scalar(loss(y_true, y_pred))
        expected = _numpy_reference_flpu(y_true, y_pred, prior=0.1)
        np.testing.assert_allclose(actual, expected, rtol=1e-4)
        assert actual >= 0


# -----------------------------------------------------------------------------
# Production configuration: mixed_float16 and the focal-loss shape contract
# -----------------------------------------------------------------------------

class TestProductionConfiguration:
    """FLPU is used in scripts with mixed_float16 dtype policy. The mask
    tensors in FLPU are explicitly cast to float32, which creates a dtype
    interaction with pn_loss under mixed precision. Also, FLPU's masking
    relies on `BinaryFocalCrossentropy(reduction='none')` producing per-
    sample output matching the batch dim; if a Keras upgrade ever silently
    changes that contract, the masking silently broadcasts and produces
    garbage. These tests are the tripwires for those regressions."""

    def test_focal_loss_produces_per_sample_output(self):
        """Lock in the shape contract FLPU's masking depends on. If this
        fails, Keras's reduction='none' semantics have changed and FLPU's
        math is silently wrong regardless of how clean the tests below
        look — a dedicated early-warning tripwire."""
        fl = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=False,
            gamma=2.0,
            from_logits=True,
            reduction="none",
        )
        y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        y_pred = np.array([[2.0], [-1.0], [0.5], [-0.5]], dtype=np.float32)
        out = fl(y_true, y_pred)
        out_arr = np.asarray(out)
        assert out_arr.shape == (4,), (
            f"Expected per-sample focal loss shape (4,), got {out_arr.shape}. "
            f"Keras's reduction='none' semantics may have changed; FLPU's "
            f"masking in src/loss_functions/loss.py will silently produce "
            f"wrong losses if this contract has broken."
        )

    def test_flpu_accepts_float16_predictions(self):
        """Simulates the mixed_float16 production configuration where the
        classifier head produces float16 logits. FLPU must still produce
        a finite loss. Does not flip the global dtype policy because that
        has side effects on other tests; just feeds float16 input."""
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[1.0, 2.0],
            unlabeled_logits=[-1.0, -2.0, -3.0],
        )
        y_pred_fp16 = y_pred.astype(np.float16)
        out = _scalar(loss(y_true, y_pred_fp16))
        assert np.isfinite(out), (
            f"FLPU produced non-finite loss under float16 inputs: {out}. "
            f"mixed_float16 training would immediately NaN."
        )


# -----------------------------------------------------------------------------
# return_intermediates parameter: optional exposure of loss components
# -----------------------------------------------------------------------------

class TestReturnIntermediates:
    """Tests for the optional return_intermediates parameter on FLPULoss.call.

    When False (default), call returns a scalar loss (backward compatible).
    When True, call returns (loss, components_dict) where components_dict
    holds the intermediate tensors. The loss scalar must be identical between
    paths (bit-identical by design, not coincidence)."""

    def test_default_path_returns_scalar(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        out = loss.call(y_true, y_pred)
        # scalar tensor, not a tuple
        assert not isinstance(out, tuple)
        assert float(out) == float(out)  # finite, indexable as scalar

    def test_flag_path_returns_scalar_and_dict(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        out = loss.call(y_true, y_pred, return_intermediates=True)
        assert isinstance(out, tuple) and len(out) == 2
        scalar, comps = out
        assert set(comps.keys()) == {
            "positive_risk", "negative_risk", "correction_triggered"
        }

    def test_loss_scalar_bit_identical_between_paths(self):
        # Design DoD: loss scalar is bit-identical with/without the flag.
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        scalar_only = loss.call(y_true, y_pred)
        scalar_with, _ = loss.call(y_true, y_pred, return_intermediates=True)
        assert float(scalar_only) == float(scalar_with)  # exact equality

    def test_loss_scalar_bit_identical_clawback_path(self):
        loss = FLPULoss(prior=0.1, kiryo_clawback=True)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        scalar_only = loss.call(y_true, y_pred)
        scalar_with, _ = loss.call(y_true, y_pred, return_intermediates=True)
        assert float(scalar_only) == float(scalar_with)

    def test_direct_call_equals_dunder_call(self):
        # Pins the equivalence that justifies the flag-on head path using
        # loss_fn.call(...) directly rather than loss_fn(...).
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        via_dunder = float(loss(y_true, y_pred))
        via_call = float(loss.call(y_true, y_pred))
        np.testing.assert_allclose(via_dunder, via_call, rtol=1e-6)

    def test_existing_dunder_call_path_unchanged(self):
        # Back-compat: the __call__ path (used by all existing tests) still
        # returns a finite scalar identical to pre-change behavior.
        loss = FLPULoss(prior=0.1, focal_gamma=2.0)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        expected = _numpy_reference_flpu(
            y_true, y_pred, prior=0.1, focal_gamma=2.0
        )
        np.testing.assert_allclose(_scalar(loss(y_true, y_pred)), expected, rtol=1e-4)


# -----------------------------------------------------------------------------
# Loss-component numerical correctness (contract + semantics)
# -----------------------------------------------------------------------------

class TestLossComponentCorrectness:
    """Verifies that the return_intermediates components match the loss scalar
    and exhibit the expected semantics (e.g., correction_triggered is binary
    and fires at the right threshold)."""

    def test_no_clawback_loss_equals_pos_plus_clamped_neg(self):
        # Ties components to the already-numpy-verified scalar without
        # reimplementing the component math.
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        scalar, c = loss.call(y_true, y_pred, return_intermediates=True)
        recombined = float(c["positive_risk"]) + max(float(c["negative_risk"]), 0.0)
        np.testing.assert_allclose(float(scalar), recombined, rtol=1e-5)

    def test_correction_triggered_is_binary(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0],
            unlabeled_logits=[-3.0, 0.0, 1.0],
        )
        _, c = loss.call(y_true, y_pred, return_intermediates=True)
        assert float(c["correction_triggered"]) in (0.0, 1.0)

    def test_correction_fires_iff_negative_risk_below_zero_no_clawback(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[5.0, 5.0],
            unlabeled_logits=[-5.0, -5.0, -5.0],
        )
        scalar, c = loss.call(y_true, y_pred, return_intermediates=True)
        neg = float(c["negative_risk"])
        fired = float(c["correction_triggered"])
        if neg < 0.0:
            assert fired == 1.0
            np.testing.assert_allclose(float(scalar), float(c["positive_risk"]), rtol=1e-5)
        else:
            assert fired == 0.0
            np.testing.assert_allclose(
                float(scalar),
                float(c["positive_risk"]) + neg,
                rtol=1e-5,
            )

    def test_clawback_path_correction_semantics(self):
        loss = FLPULoss(prior=0.1, kiryo_clawback=True, nn_beta=0.0, nn_gamma=1.0)
        y_true, y_pred = _batch(
            positive_logits=[5.0, 5.0],
            unlabeled_logits=[-5.0, -5.0, -5.0],
        )
        scalar, c = loss.call(y_true, y_pred, return_intermediates=True)
        neg = float(c["negative_risk"])
        fired = float(c["correction_triggered"])
        if neg < -0.0:
            assert fired == 1.0
            np.testing.assert_allclose(float(scalar), -1.0 * neg, rtol=1e-5)
        else:
            assert fired == 0.0
            np.testing.assert_allclose(
                float(scalar), float(c["positive_risk"]) + neg, rtol=1e-5
            )

    def test_positive_risk_non_negative(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, -1.0, 0.5],
            unlabeled_logits=[-3.0, 1.0],
        )
        _, c = loss.call(y_true, y_pred, return_intermediates=True)
        assert float(c["positive_risk"]) >= 0.0
