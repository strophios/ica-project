"""
Script to actually perform prior estimation. Currently assumes a trained LU classifier, but I'll probably want to change
that so that it actually runs the whole workflow.
"""

import keras
import tensorflow as tf
import polars as pl
import numpy as np
import src.data_setup.dapt_data
import src.preproc.preprocessor
import src.prior_estimation.dedpul_em
import src.prior_estimation.dedpul_utils

import os

# setting for automatic mixed precision
keras.config.set_dtype_policy(
    "mixed_float16"
)  # want to make sure this works on Explorer

# path_prefix = os.path.expanduser(
#     "~/immigration_project/00_ML_data_expansion/00_explorer"
# )
path_prefix = os.path.abspath("/projects/ahd")

# Load the fitted LU Classifier
lu_classifier = keras.models.load_model(f"{path_prefix}/lu_classifier.keras")

BATCH_SIZE = 256
SEQ_LENGTH = 128

if not os.path.isdir(f"{path_prefix}/cca_set/lu"):
    # Get predictions for all the training data
    preprocess = src.preproc.preprocessor.ClassifierPreprocessor(
        SEQ_LENGTH, text_key="headline_with_lead", label_key="cca_label"
    )
    ldc_data = src.data_setup.dapt_data.data_from_parquet(
        path_prefix,
        "ldc_corpus",
        addl_columns=["cca", "cca_descriptor", "immig", "immig_descriptor"],
    )  # the function includes "ldc_corpus" as a default arg
    ldc_data = src.data_setup.dapt_data.create_classifier_data(
        ldc_data, separate_labels=False
    )
    prediction_set = pl.concat([ldc_data["train"], ldc_data["val"]])
    # prediction_set = ldc_data["val"]  # using just validation data for testing
    prediction_set = prediction_set.select("headline_with_lead", "cca_label").to_dict()
    prediction_set = preprocess(prediction_set)
    lu_preds = lu_classifier.predict(
        prediction_set[0], batch_size=BATCH_SIZE
    )  # doing this for just the validation set on my local machine takes ~18 minutes
    lu_targets = prediction_set[1]
    os.mkdir(f"{path_prefix}/cca_set/lu")
    np.save(f"{path_prefix}/cca_set/lu/lu_preds.npy", lu_preds)
    lu_targets = lu_targets.to_numpy()
    np.save(f"{path_prefix}/cca_set/lu/lu_targets.npy", lu_targets)
else:
    lu_preds = np.load(f"{path_prefix}/cca_set/lu/lu_preds.npy")
    lu_targets = np.load(f"{path_prefix}/cca_set/lu/lu_targets.npy")

lu_preds = np.reshape(lu_preds, lu_preds.shape[0])
# DEDPUL wants the *probability of being unlabeled* (its convention is 0 = labeled
# positive, 1 = unlabeled), in [0, 1]. Our `lu_classifier` outputs raw logits for
# the labeled class (no final activation; loss uses from_logits=True). To convert:
#   P(labeled | x)   = sigmoid(logit)
#   P(unlabeled | x) = 1 - sigmoid(logit)
# An earlier version of this file did `lu_preds - 2*lu_preds == -lu_preds`, which
# is the unlabeled-class *logit*, not a probability — DEDPUL's KDE step then ran in
# the wrong space. See `scripts/compare_dedpul_logit_vs_prob.py` for the empirical
# comparison; switching to the sigmoid-then-subtract form changes π_pos from
# ~0.04 to ~0.02 on the cached L/U predictions, which is large enough to matter
# for downstream FLPU training.
lu_preds_rev = 1.0 - (1.0 / (1.0 + np.exp(-lu_preds)))
lu_targets_rev = 1 - lu_targets

# Feed those predictions into one of the DEDPUL algorithms
diffs = src.prior_estimation.dedpul_em.estimate_diff(
    lu_preds_rev, lu_targets_rev, tune=True, kde_mode="prob"
)
alpha, posterior = src.prior_estimation.dedpul_em.estimate_poster_em(
    diffs, lu_preds_rev, lu_targets_rev
)
# Current estimates (from the cached val+train L/U predictions):
#
# Pre-fix (logits passed directly to DEDPUL — incorrect):
#   tune=False → α=0.98     → π_pos ≈ 0.02
#   tune=True  → α=0.9597   → π_pos ≈ 0.04
#   midpoint:  → α≈0.97     → π_pos ≈ 0.03  (this is the value baked into
#                                            run_cca_classification.py)
#
# Post-fix (sigmoid-then-reverse — correct):
#   tune=True  → α=0.9800   → π_pos ≈ 0.02  (see scripts/compare_dedpul_logit_vs_prob.py)
#
# So the correct prior is closer to 0.02 than 0.03. When CCA training is
# rerun, FLPULoss(prior=...) should be updated accordingly.
