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
                   (the bias-correction term).

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
    Standard focal loss has an α knob (e.g., α=0.25 in Lin et al. 2020) that
    asymmetrically weights the y=1 vs y=0 sample contributions. Applied
    directly in FLPU, this asymmetry compounds with the π_p prior weighting
    in ways that don't have a clean theoretical justification: the same
    labeled-positive sample gets different α weights in its "positive" role
    (R_p^+) vs. its "positive-treated-as-negative" role (R_p^-), despite
    both terms being part of the same bias-corrected risk estimate.

    Ji et al. 2023's Eq. 14 writes a uniform α across all three terms, which
    factors out as a global scalar multiplier on the entire loss — redundant
    with the learning rate. We therefore drop α entirely. The class prior π_p
    is what does the class balancing in nnPU; the focal γ is what
    down-weights easy examples. They serve distinct purposes; α has no
    distinct purpose to serve here.

    See `docs/notes/pinned-questions.md` for the full discussion of how the
    different mechanisms (nnPU prior, focal modulation, ALUM) compose.
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
        # entries via the masks below. We evaluate over the full batch (rather
        # than slicing) to keep the graph static for autograph/jit.
        pn_loss = self.focal_loss(y_true, y_pred)
        assert pn_loss.shape == positive.shape, (
            f"Loss component output ({pn_loss.shape}) and masking tensor "
            f"({positive.shape}) must have the same shape."
        )

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
