"""
Loss functions, currently including both FLPU and ALUM.
"""

import numpy as np
import keras
from keras import layers
import keras_hub
import tensorflow as tf
import warnings

import src.model_setup.dapt_setup
from keras import ops
import os


# decorator added cause otherwise Keras won't save it
# when saving a model compiled with it,creating problems for loading
# (at least that's the theory; don't know for certain it works)
@keras.saving.register_keras_serializable()
class FLPULoss(keras.losses.Loss):
    """
    A non-negative PU learning implementation of the focal loss.

    positive class prior * mean loss of positive samples +
        max(0 | mean loss of unlabeled samples assuming they're negative -
            positive class prior * mean loss of positive samples *assuming they're negative*)

    Note: it appears that @Kiryo2017 allow some flexibility in the actual implementation of nnPU,
    parameterized by nn_beta between 0 and the max possible predicted value (I think) and nn_gamma,
    between 0 and 1. nn_beta sets the strictness of the bound: instead of a max(0, Ru), we ask whether
    Ru is >= -nn_beta. Thus, if nn_beta = 0, we have a strict non-negative bound.

    (nn_gamma is a scaling factor that discounts the contribution of Ru to the loss in cases where it
    falls outside the bound (so you take a step of size nn_gamma * step_size along the gradient, rather
    than just step_size).) - I actually think this is wrong, see below.

    If Ru is outside the bound, then the nnPU estimator tries to actively claw our way back from overfitting
    by taking a step in the *opposite direction* indicated by the Ru part of the loss and ignoring the positive
    piece entirely. nn_gamma scales the size of that step (as a fraction of a standard step). I'm not 100% sure
    that this is what's going on, since @Kiryo2017 is not super explicit about it, but it seems to be what they
    lay out and what they've implemented.

    I'm also not totally sure whether we'd want the explicit overfitting walk-back if we're using this in
    conjunction with ALUM. @Ji2023 certainly doesn't indicate doing it (that I can tell, anyways). The parameter
    `kiryo_clawback` determines whether we do the above mentioned "clawback" or just zero that part of the loss
    """

    def __init__(
        self,
        prior,
        focal_alpha=0.25,
        focal_gamma=2,
        nn_beta=0,
        nn_gamma=1,
        kiryo_clawback=False,
    ):
        super().__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.nn_beta = nn_beta
        self.nn_gamma = nn_gamma
        self.kiryo_clawback = kiryo_clawback
        if self.focal_alpha is not None:
            self.apply_class_balancing = True
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=self.apply_class_balancing,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            from_logits=True,
            reduction=None,  # pretty sure about this
        )
        # Some useful constants
        self.positive = 1  # value of positive labels
        self.unlabeled = 0  # value of unlabeled labels
        self.min_count = 1.0

    def call(self, y_true, y_pred):
        # For PU, we need to treat labeled and unlabeled cases differently
        # So we separate them by creating two boolean masks, then to integers
        # (now floats, since we were getting a type mismatch below)
        positive, unlabeled = y_true == self.positive, y_true == self.unlabeled
        positive, unlabeled = (
            ops.cast(positive, dtype="float32"),
            ops.cast(unlabeled, dtype="float32"),
        )
        # we cast to int (now float) so that we can elementwise set values of the loss to 0 (via multiplication)
        # we do this rather than boolean masking because I *think* this is maybe more efficient, in
        # terms of memory, assignment operations, etc. (I'm not actually sure about this, but I've seen
        # it done this way in other implementations (of similar things) that I've looked at).

        # We reshape these to match the output of self.focal loss, which outputs a 1-D tensor
        positive = tf.reshape(
            positive, shape=[-1]
        )  # theoretically could use tf.squeeze(), but it broke the shape comparison in the assert statement for
        # reasons related to the use fo TF/Keras symbolic tensors and pre-execution graphing/planning or whatever
        unlabeled = tf.reshape(unlabeled, shape=[-1])

        # Now count positive and negative samples; since we're using these for division, we set a minimum of 1
        n_positive, n_unlabeled = (
            ops.maximum(ops.sum(positive), self.min_count),
            ops.maximum(ops.sum(unlabeled), self.min_count),
        )
        # Now we calculate the losses for each subgroup (note: reduction = None for self.focal_loss has been set above)
        # We actually calculate for all inputs for each one, but then we zero out the out-of-group results
        pn_loss = self.focal_loss(
            y_true, y_pred
        )  # this contains both the y_positive and y_unlabeled components
        assert (
            pn_loss.shape == positive.shape
        ), (  # make sure the shapes match, otherwise the multiplication is silently wrong
            f"Loss component output ({pn_loss.shape}) and masking tensor (){positive.shape}) must have the same shape."
        )

        y_positive = (
            pn_loss * positive
        )  # error for positive samples w/r/t positive ground truth
        y_unlabeled = (
            pn_loss * unlabeled
        )  # error for unlabeled samples assuming negative ground truth
        y_positive_inv = (
            self.focal_loss(ops.abs(y_true - 1), y_pred) * positive
        )  # error for positive samples assuming negative ground truth

        positive_risk = self.prior * ops.sum(y_positive) / n_positive
        negative_risk = (
            ops.sum(y_unlabeled) / n_unlabeled
            - self.prior * ops.sum(y_positive_inv) / n_positive
        )
        if not self.kiryo_clawback:
            return positive_risk + ops.maximum(negative_risk, 0)
        else:  # note the use of tf.cond() instead of a standard Python if-statement ("using a symbolic tf.Tensor as a Python bool is not allowed")
            return tf.cond(
                pred=negative_risk < -self.nn_beta,
                true_fn=lambda: -self.nn_gamma * negative_risk,
                false_fn=lambda: positive_risk + negative_risk,
            )

        # if negative_risk < -self.nn_beta:
        #     if not self.kiryo_clawback:
        #         return positive_risk + 0
        #     else:
        #         return -self.nn_gamma * negative_risk
        #     # I'm pretty sure this is right, even if it seemed off at first glance
        #     # in particular, I think the idea is that putting the gamma here is equivalent to
        #     # to having it directly scale step size. Also, I'm not 100% on the reason that
        #     # the whole thing is negative, but that's correct to their paper and implementation.
        #     # I *think* that we make the whole thing negative (and omit the positive risk part
        #     # of the loss) because the point here is to try and compensate for the overfitting
        #     # that's happening and actually claw our way back out of it, I think.
        #
        # return positive_risk + negative_risk
