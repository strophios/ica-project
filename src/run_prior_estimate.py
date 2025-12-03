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

path_prefix = os.path.expanduser(
    "~/immigration_project/00_ML_data_expansion/00_explorer"
)
# path_prefix = os.path.abspath("/projects/ahd")

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
    # prediction_set = pl.concat([ldc_data["train"], ldc_data["val"]])
    prediction_set = ldc_data["val"]  # using just validation data for testing
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
# Note that DEDPUL wants the reverse of what everyone else does: they want to predict
# the probability of being *unlabeled*, rather than labeled, so we have to reverse our
# predictions and targets (i.e., 0 = labeled, 1 = unlabeled)
lu_preds_rev = lu_preds - (2 * lu_preds)
lu_targets_rev = lu_targets - (2 * lu_targets) + 1  # could also use np.absolute

# Feed those predictions into one of the DEDPUL algorithms
diffs = src.prior_estimation.dedpul_em.estimate_diff(
    lu_preds_rev, lu_targets_rev, tune=True, kde_mode="prob"
)
alpha, posterior = src.prior_estimation.dedpul_em.estimate_poster_em(
    diffs, lu_preds_rev, lu_targets_rev
)
# Current estimates, based solely on validation data:
# w/ tune = False: 0.98
# w/ tune = True: 0.9596673650517707
