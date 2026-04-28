"""
Script to actually perform prior estimation. Currently assumes a trained LU classifier, but I'll probably want to change
that so that it actually runs the whole workflow.
"""

import keras
import polars as pl
import numpy as np

import src.config as config
import src.data_setup.data
import src.preproc.preprocessor
import src.prior_estimation.dedpul_em
import src.prior_estimation.dedpul_utils

# Platform-conditional dtype policy.
keras.config.set_dtype_policy(config.DTYPE_POLICY)

# Load the fitted LU Classifier
lu_classifier = keras.models.load_model(str(config.LU_CLASSIFIER_MODEL))

BATCH_SIZE = 256
SEQ_LENGTH = 128

if not config.LU_PREDS_DIR.is_dir():
    # Get predictions for all the training data. Standard-mode
    # preprocessor — returns (features_dict, targets_dict). The
    # targets_dict key is arbitrary here because we extract it
    # manually for DEDPUL rather than routing it through compile-
    # time loss; "label" is descriptive.
    preprocess = src.preproc.preprocessor.ClassifierPreprocessor(
        SEQ_LENGTH=SEQ_LENGTH,
        text_key="headline_with_lead",
        label_keys={"label": "cca_label"},
        endpoint_model=False,
    )
    ldc_data = src.data_setup.data.data_from_parquet(
        config.PROJECT_ROOT,
        "ldc_corpus",
        addl_columns=["cca", "cca_descriptor", "immig", "immig_descriptor"],
    )  # the function includes "ldc_corpus" as a default arg
    ldc_data = src.data_setup.data.create_classifier_data(
        ldc_data, separate_labels=False
    )
    prediction_set = pl.concat([ldc_data["train"], ldc_data["val"]])
    # prediction_set = ldc_data["val"]  # using just validation data for testing
    prediction_set = prediction_set.select("headline_with_lead", "cca_label").to_dict()
    features_dict, targets_dict = preprocess(prediction_set)
    lu_preds = lu_classifier.predict(
        features_dict, batch_size=BATCH_SIZE
    )  # doing this for just the validation set on my local machine takes ~18 minutes
    lu_targets = targets_dict["label"]
    config.LU_PREDS_DIR.mkdir()
    np.save(str(config.LU_PREDS_DIR / "lu_preds.npy"), lu_preds)
    lu_targets = np.asarray(lu_targets)
    np.save(str(config.LU_PREDS_DIR / "lu_targets.npy"), lu_targets)
else:
    lu_preds = np.load(str(config.LU_PREDS_DIR / "lu_preds.npy"))
    lu_targets = np.load(str(config.LU_PREDS_DIR / "lu_targets.npy"))

lu_preds = np.reshape(lu_preds, lu_preds.shape[0])
# DEDPUL wants the *probability of being unlabeled* (its convention is
# 0 = labeled positive, 1 = unlabeled), in [0, 1]. Our `lu_classifier`
# outputs raw logits for the labeled class (no final activation; loss
# uses from_logits=True). The correct conversion:
#   P(labeled | x)   = sigmoid(logit)
#   P(unlabeled | x) = 1 - sigmoid(logit)
#
# History: an earlier version of this file did `lu_preds - 2*lu_preds`,
# i.e. `-lu_preds`, which gives the unlabeled-class *logit*, not a
# probability. The first fix commit attributed the resulting ~0.04 → ~0.02
# shift in π_pos to "the sigmoid fix", but that attribution was wrong.
# An adversarial review and follow-up four-variant comparison
# (`scripts/compare_dedpul_logit_vs_prob.py`) showed that the shift was
# almost entirely driven by DEDPUL's `tune=True` bandwidth grid
# ([0.01, 0.4]) being calibrated for [0, 1]-valued inputs but being
# applied to logit-scale inputs — the mis-scaled bandwidth made the KDE
# spiky and inflated π_pos. Switching the inputs to probabilities
# happens to also put the bandwidth grid in the right regime, so the
# fix remediates both issues at once, but the primary effect was not
# "logits vs probs" — it was "bandwidth scale."
#
# Current best estimate on the cached L/U predictions: π_pos ≈ 0.02,
# robust across kde_mode="prob" vs "logit" and across manual-bandwidth
# vs tune=True.
lu_preds_rev = 1.0 - (1.0 / (1.0 + np.exp(-lu_preds)))
lu_targets_rev = 1 - lu_targets

# Feed those predictions into one of the DEDPUL algorithms
diffs = src.prior_estimation.dedpul_em.estimate_diff(
    lu_preds_rev, lu_targets_rev, tune=True, kde_mode="prob"
)
alpha, posterior = src.prior_estimation.dedpul_em.estimate_poster_em(
    diffs, lu_preds_rev, lu_targets_rev
)
# Current estimates (from the cached val+train L/U predictions; full
# attribution table in scripts/compare_dedpul_logit_vs_prob.py):
#
#   Variant                                   α        π_pos
#   ---------------------------------------   ------   ------
#   broken logits + tune (original bug)       0.9597   0.0403
#   broken logits + bw=1.0 (scale-matched)    0.9790   0.0210
#   probs + tune + kde_mode="prob" (current)  0.9800   0.0200
#   probs + tune + kde_mode="logit" (default) 0.9800   0.0200
#
# The shift from 0.0403 → 0.0200 was mostly about bandwidth scale, not
# about logits vs probabilities. Current code (probs + tune + "prob") is
# robust: it gives the same answer as DEDPUL's own default config.
#
# The value currently baked into run_cca_classification.py (prior=0.03) is
# stale by a factor of ~1.5; next CCA training run should use the corrected
# prior closer to 0.02.
