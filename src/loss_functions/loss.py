"""
Loss functions for the ICA classification pipeline.

Currently contains FLPULoss (focal-loss-flavored non-negative PU learning).
ALUM / virtual adversarial training will live here when implemented.
"""

import keras
from keras import ops


# The serialization decorator lets Keras save and reload models that are
# compiled with this loss without losing the loss configuration.
@keras.saving.register_keras_serializable()
class FLPULoss(keras.losses.Loss):
    """
    A non-negative PU learning loss built on focal cross-entropy.

    Implements the FLPU loss from Ji et al. 2023 (Eq. 14), which composes the
    focal-loss modulation (Lin et al. 2020) with the non-negative PU
    formulation of Kiryo et al. 2017:

        R(g) = π_p · R_p^+(g) + max(0, R_u^-(g) - π_p · R_p^-(g))

    where:
      - R_p^+(g) = mean focal loss over positives, treated as positive.
      - R_u^-(g) = mean focal loss over unlabeled samples, treated as negative.
      - R_p^-(g) = mean focal loss over positives, treated as negative
                   (the bias-correction term; both "negative-side" terms
                   come from the same expectation π_n·E_{p_n}[ℓ], which is
                   why they must share coefficients — see "Why no α" below).

    Parameters
    ----------
    prior : float
        Positive class prior π_p, in (0, 1).
    focal_gamma : float
        Focal modulation strength. γ=0 reduces to plain cross-entropy.
        Default 2.0 matches Lin et al. 2020 and Ji et al. 2023.
    nn_beta : float
        Lower bound on the negative risk before the Kiryo "clawback"
        engages. Only relevant when `kiryo_clawback=True`. Default 0.
    nn_gamma : float
        Step-scale factor for the Kiryo clawback. Only relevant when
        `kiryo_clawback=True`. Default 1.
    kiryo_clawback : bool
        If True, when the negative risk falls below -nn_beta, return
        -nn_gamma * negative_risk (a gradient-ascent step on the negative
        risk, attempting to recover from overfitting). If False, simply
        clip the negative risk at 0. Default False.

    Why no α (focal-loss class balancing)
    --------------------------------------
    Standard focal loss has an α knob (e.g., α=0.25 in Lin et al. 2020)
    that weights y=1 samples by α and y=0 samples by (1-α). Plugged into
    FLPU, this turns out to be a well-defined construction: both
    "negative-side" terms (R_u^-, the unlabeled risk, and R_p^-, the
    bias-correction) receive the same (1-α) coefficient (because Keras's
    focal loss applies the weight by label and both are y=0), so the
    underlying distributional identity is preserved. The resulting
    estimator is unbiased (up to the max(0, ·) clip) for a
    *cost-sensitive* nnPU risk with cost ratio α : (1-α). It is not
    "broken" — it is cost-sensitive PU learning.

    We nonetheless default to α=off, for three reasons:

    1. For the CCA head, we do not have a deliberate cost-sensitivity
       preference (both false positives and false negatives are roughly
       equally costly for the research goal of building a filterable
       candidate set).
    2. Lin 2020's canonical α=0.25 specifically down-weights the positive
       class. For a rare-positive PU problem, this is the wrong sign of
       adjustment.
    3. If we do want cost-sensitive PU later (perhaps on a different head),
       we should parameterize α_pos and α_neg directly as deliberate knobs
       rather than importing them from the Lin 2020 α, (1-α) convention.

    Ji et al. 2023's Eq. 14 writes a uniform α across all three terms,
    which under the default `nn_beta=0` factors out as a global scalar
    multiplier on the loss (redundant with the learning rate). This
    "factors out" property does NOT hold when `nn_beta > 0`, because the
    clawback threshold `R_u^- - π_p · R_p^- < -nn_beta` is not α-invariant
    — so if anyone ever turns on nn_beta>0, the Ji 2023 uniform-α reading
    needs a fresh look.

    See `docs/notes/pinned-questions.md` for the full discussion of how the
    different mechanisms (nnPU identity, focal γ, α as cost-sensitivity,
    Ratio Batch, ALUM) compose across four layers of the loss stack.
    """

    def __init__(
        self,
        prior,
        focal_gamma=2.0,
        nn_beta=0.0,
        nn_gamma=1.0,
        kiryo_clawback=False,
    ):
        super().__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.focal_gamma = focal_gamma
        self.nn_beta = nn_beta
        self.nn_gamma = nn_gamma
        self.kiryo_clawback = kiryo_clawback

        # Per-sample focal loss with no class-balance α (see docstring).
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=False,
            gamma=self.focal_gamma,
            from_logits=True,
            reduction="none",
        )

        # Label values and a numerical floor for the per-class sample counts
        # to avoid divide-by-zero when a batch is missing one class entirely.
        self.positive = 1
        self.unlabeled = 0
        self.min_count = 1.0

    def call(self, y_true, y_pred):
        # Boolean masks per sub-population, cast to float for elementwise
        # multiplication. Reshape to 1-D so the masks line up with the
        # per-sample focal loss output (which is also 1-D under reduction="none").
        positive = ops.cast(y_true == self.positive, dtype="float32")
        unlabeled = ops.cast(y_true == self.unlabeled, dtype="float32")
        positive = ops.reshape(positive, (-1,))
        unlabeled = ops.reshape(unlabeled, (-1,))

        # Sample counts, floored at 1 to avoid division by zero.
        n_positive = ops.maximum(ops.sum(positive), self.min_count)
        n_unlabeled = ops.maximum(ops.sum(unlabeled), self.min_count)

        # Per-sample focal loss over the whole batch; we zero out irrelevant
        # entries via the masks below. We evaluate over the full batch
        # (rather than slicing) to keep the graph static for autograph/jit.
        #
        # Shape expectation: `pn_loss` is 1-D of length batch_size, matching
        # `positive` and `unlabeled` after their reshape. This depends on
        # `BinaryFocalCrossentropy(reduction="none")` returning per-sample
        # losses. A Python-level assert here would not be a runtime guard:
        # under `tf.function` tracing, asserts run only at trace time, and
        # an asserted shape equality on tensors with partially-unknown dims
        # could silently pass. Instead, the shape invariant is guarded by
        # the test suite (tests/test_flpu_loss.py, TestOutputStructure and
        # the production-configuration test for mixed_float16). If a Keras
        # upgrade ever changes reduction="none" semantics, those tests are
        # the intended tripwire, not this comment.
        pn_loss = self.focal_loss(y_true, y_pred)

        # Three FLPU components.
        y_positive = pn_loss * positive  # positives, treated as positive
        y_unlabeled = pn_loss * unlabeled  # unlabeled, treated as negative
        y_positive_inv = (
            self.focal_loss(ops.abs(y_true - 1), y_pred) * positive
        )  # positives, treated as negative (bias correction)

        positive_risk = self.prior * ops.sum(y_positive) / n_positive
        negative_risk = (
            ops.sum(y_unlabeled) / n_unlabeled
            - self.prior * ops.sum(y_positive_inv) / n_positive
        )

        if not self.kiryo_clawback:
            # Standard nnPU: clip the negative risk at 0 to prevent the
            # bias-correction subtraction from going negative due to
            # overfitting on the positive samples.
            return positive_risk + ops.maximum(negative_risk, 0)

        # Kiryo "active recovery" branch: when the negative risk falls below
        # -nn_beta, take a gradient-ascent step on the (negated) negative
        # risk, scaled by nn_gamma, while ignoring the positive risk for
        # this step. Intent: actively claw back from overfitting.
        return ops.cond(
            pred=negative_risk < -self.nn_beta,
            true_fn=lambda: -self.nn_gamma * negative_risk,
            false_fn=lambda: positive_risk + negative_risk,
        )
