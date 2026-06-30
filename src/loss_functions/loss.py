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
        nnpnu_eta=0.0,
    ):
        super().__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        if not 0.0 <= nnpnu_eta <= 1.0:
            raise ValueError(
                f"nnpnu_eta (PU<->PN mixing weight) must be in [0, 1], got {nnpnu_eta}"
            )
        if nnpnu_eta > 0.0 and kiryo_clawback:
            # The clawback's gradient-ASCENT recovery must be isolated to the PU
            # negative-risk term: ascending on the reliable-negative term would
            # push reliable negatives toward positive (un-learning them). Composing
            # the two needs a deliberate design pass, not a silent stack.
            raise NotImplementedError(
                "nnpnu_eta>0 with kiryo_clawback is not yet supported "
                "(see docs/notes/pinned-questions.md)."
            )
        self.prior = prior
        self.focal_gamma = focal_gamma
        self.nn_beta = nn_beta
        self.nn_gamma = nn_gamma
        self.kiryo_clawback = kiryo_clawback
        self.nnpnu_eta = nnpnu_eta

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
        self.reliable_negative = -1
        self.min_count = 1.0

    def call(self, y_true, y_pred, return_intermediates=False):
        # Boolean masks per sub-population, cast to float for elementwise
        # multiplication. Reshape to 1-D so the masks line up with the
        # per-sample focal loss output (which is also 1-D under reduction="none").
        # Three label values: positive (1), unlabeled (0), reliable negative (-1).
        positive = ops.reshape(ops.cast(y_true == self.positive, "float32"), (-1,))
        unlabeled = ops.reshape(ops.cast(y_true == self.unlabeled, "float32"), (-1,))
        reliable_neg = ops.reshape(
            ops.cast(y_true == self.reliable_negative, "float32"), (-1,)
        )

        # Sample counts, floored at 1 to avoid division by zero.
        n_positive = ops.maximum(ops.sum(positive), self.min_count)
        n_unlabeled = ops.maximum(ops.sum(unlabeled), self.min_count)
        n_reliable_neg = ops.maximum(ops.sum(reliable_neg), self.min_count)

        # We evaluate focal loss over the whole batch and zero out irrelevant
        # entries via the masks below (rather than slicing) to keep the graph
        # static for autograph/jit. Shape expectation: `focal_pos`/`focal_neg`
        # are 1-D of length batch_size, matching the masks after their reshape --
        # this depends on `BinaryFocalCrossentropy(reduction="none")` returning
        # per-sample losses, and is guarded by tests/test_flpu_loss.py (a
        # Python-level assert would only run at trace time), the intended
        # tripwire if a Keras upgrade ever changes reduction="none" semantics.
        #
        # Per-sample focal loss treating EVERY sample as positive / as negative,
        # then masked below. This decomposition is bit-identical to the prior
        # focal(y_true) / focal(1 - y_true) form at 0/1 labels, and lets the third
        # label value (-1, reliable negative) join without feeding -1 to the focal
        # loss (which expects 0/1).
        focal_pos = ops.reshape(self.focal_loss(ops.ones_like(y_pred), y_pred), (-1,))
        focal_neg = ops.reshape(self.focal_loss(ops.zeros_like(y_pred), y_pred), (-1,))

        positive_risk = self.prior * ops.sum(focal_pos * positive) / n_positive
        # PU estimate of (1 - pi) * R_n^-: the ONLY term that can go negative (the
        # bias-correction subtraction overfits the positives). nnPU clips it.
        pu_negative_risk = (
            ops.sum(focal_neg * unlabeled) / n_unlabeled
            - self.prior * ops.sum(focal_neg * positive) / n_positive
        )
        # PN estimate of the same (1 - pi) * R_n^- from reliable negatives: a mean
        # of non-negative focal losses, so always >= 0 (no clip/clawback needed).
        pn_negative_risk = (
            (1.0 - self.prior) * ops.sum(focal_neg * reliable_neg) / n_reliable_neg
        )

        eta = self.nnpnu_eta
        if not self.kiryo_clawback:
            # nnPNU: the clip applies to the PU term ONLY; the reliable-negative
            # term is an ordinary supervised loss added OUTSIDE the clip. `eta`
            # blends the PU and PN estimates of the same (1 - pi) * R_n^-, so
            # eta=0 reduces exactly to nnPU.
            negative_risk = (
                (1.0 - eta) * ops.maximum(pu_negative_risk, 0.0)
                + eta * pn_negative_risk
            )
            loss = positive_risk + negative_risk
            correction_triggered = ops.cast(pu_negative_risk < 0, "float32")
        else:
            # Kiryo "active recovery": gradient-ascent step on the (negated) PU
            # negative risk when it falls below -nn_beta, ignoring positive risk.
            # Guarded to eta==0 at __init__, so no reliable-negative term enters
            # here -- identical to the original nnPU clawback.
            loss = ops.cond(
                pred=pu_negative_risk < -self.nn_beta,
                true_fn=lambda: -self.nn_gamma * pu_negative_risk,
                false_fn=lambda: positive_risk + pu_negative_risk,
            )
            correction_triggered = ops.cast(
                pu_negative_risk < -self.nn_beta, "float32"
            )

        if return_intermediates:
            return loss, {
                "positive_risk": positive_risk,
                "negative_risk": pu_negative_risk,
                "correction_triggered": correction_triggered,
            }
        return loss
