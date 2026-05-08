"""
Evaluate a trained CCA classifier on the held-out test set, attaching
per-sample logit scores to the underlying polars dataframe for
qualitative review.

This is the cross-process companion to `run_cca_classification.py`:
the training script saves weights to disk; this script reconstructs
the model architecture from scratch (with a fresh `ClassificationHead`
instance), loads the weights by name, and runs predictions. Pattern 2
of the train-vs-inference split (see Tier 2 Piece 4b design doc) —
in-process Pattern A weight-sharing isn't available across script
boundaries, so this script doesn't share Layer instances with the
training script; it gets parity by matching head configuration so
weight names line up at load time.
"""

import keras
import numpy as np
import polars as pl
import tensorflow as tf

import src.config as config
import src.data_setup.data
from src.preproc.preprocessor import ClassifierPreprocessor
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_inference_model
from src.loss_functions.loss import FLPULoss

# Apply platform-conditional dtype policy (no-op equivalent at eval
# time, but kept symmetric with run_cca_classification.py).
keras.config.set_dtype_policy(config.DTYPE_POLICY)

# Seed for reproducibility (shouldn't matter much at eval time but
# keeps any incidental shuffling / sampling deterministic).
keras.utils.set_random_seed(200)

BATCH_SIZE = 256
SEQ_LENGTH = 128

# Class prior — must match the value used at training time so the
# head's reconstructed loss has the same configuration as the trained
# model (weights load by name; loss config doesn't strictly affect
# inference, but matching is good hygiene). Updated 4c: 0.03 → 0.02.
FLPU_PRIOR = 0.02


# -----------------------------------------------------------------------------
# Build inference model (Pattern 2: fresh head, weights loaded by name)
# -----------------------------------------------------------------------------
backbone = load_dapt_backbone(config.DAPT_BACKBONE_WEIGHTS)

# Fresh head instance with the same configuration the training script
# used. Variable names within the head are derived from the head's
# `name="cca"` plus sublayer names, so they line up with the saved
# weights from training. The metrics list is reproduced for
# parity, although it's not actually used at inference (inf model
# calls the head with targets=None, so update_state never fires).
cca_head = ClassificationHead(
    hidden_dim=backbone.hidden_dim,
    loss_fn=FLPULoss(prior=FLPU_PRIOR, kiryo_clawback=False),
    metrics=[
        keras.metrics.BinaryAccuracy(threshold=0.0),
        keras.metrics.Precision(thresholds=0.0, name="precision"),
        keras.metrics.Recall(thresholds=0.0, name="recall"),
        keras.metrics.AUC(curve="PR", from_logits=True, name="pr_auc"),
    ],
    name="cca",
)

cca_inference = build_inference_model(
    backbone=backbone,
    heads={"cca": cca_head},
    seq_length=SEQ_LENGTH,
)

# Load weights saved by the training script. Keras 3's `.weights.h5`
# save format keys variables by layer-class + positional index —
# matching is structural (layer types, ordering, weight shapes),
# not by user-given name. The fresh head here uses the same
# `ClassificationHead(name="cca", hidden_dim=...)` configuration as
# the training script, so the architecture aligns and weights load
# cleanly. `skip_mismatch=False` pins the load-strict discipline
# (Tier 3 Piece 2): a future architectural drift between training
# and eval (e.g., changing `hidden_dim`, adding a head-internal
# layer) raises `ValueError` rather than silently producing a model
# with partial-or-default weights.
cca_inference.load_weights(
    str(config.CCA_CLASSIFIER_WEIGHTS), skip_mismatch=False
)


# -----------------------------------------------------------------------------
# Build finite predict datasets
# -----------------------------------------------------------------------------
# Predict-only preprocessor: no targets emitted, just the model inputs
# the inference graph declares (`token_ids`, `padding_mask`).
predict_preprocess = ClassifierPreprocessor(
    SEQ_LENGTH=SEQ_LENGTH,
    text_key="headline_with_lead",
    label_keys={},
    endpoint_model=True,
)


def _finite_predict_dataset(saved_dataset_path):
    """Build a finite (non-repeated) tf.data pipeline for predict.
    Avoids the `steps=validation_steps` + `.repeat()` pitfall that
    produced duplicate predictions in the pre-Tier-2 code."""
    return (
        tf.data.Dataset.load(str(saved_dataset_path))
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(predict_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


test_pos_set = _finite_predict_dataset(config.CCA_SET_DIR / "test_pos.tf")
test_unl_set = _finite_predict_dataset(config.CCA_SET_DIR / "test_unl.tf")


# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
# Multi-head models return predict() output as a dict keyed by output
# name. Single-head models can return either a dict or a bare tensor
# depending on Keras version / output structure; handle both.
pos_scores = cca_inference.predict(test_pos_set, batch_size=BATCH_SIZE)
unl_scores = cca_inference.predict(test_unl_set, batch_size=BATCH_SIZE)

if isinstance(pos_scores, dict):
    pos_scores = pos_scores["cca"]
    unl_scores = unl_scores["cca"]


# -----------------------------------------------------------------------------
# Attach scores to dataframe for qualitative review
# -----------------------------------------------------------------------------
ldc_data = src.data_setup.data.data_from_parquet(
    config.PROJECT_ROOT,
    "ldc_corpus",
    addl_columns=["cca", "cca_descriptor", "immig", "immig_descriptor"],
)
ldc_data = src.data_setup.data.create_classifier_data(ldc_data, separate_labels=True)

pos_df = ldc_data["test"]["pos"]
unl_df = ldc_data["test"]["unl"]

# Predictions are now finite-dataset-sized (no `.repeat()` overrun),
# so the slicing-to-dataframe-length workaround the old script needed
# isn't required. Sanity-check the lengths align before we attach.
assert pos_scores.shape[0] == pos_df.shape[0], (
    f"pos_scores length ({pos_scores.shape[0]}) != pos_df length "
    f"({pos_df.shape[0]}). Likely cause: dataset construction or "
    f"predict pipeline produced a different number of samples than "
    f"the source dataframe — check batch / drop_remainder settings."
)
assert unl_scores.shape[0] == unl_df.shape[0], (
    f"unl_scores length ({unl_scores.shape[0]}) != unl_df length "
    f"({unl_df.shape[0]})."
)

pos_df = pos_df.with_columns(cca_logit=pos_scores.squeeze())
# Subset unl_df to match pos_df length for the qualitative review
# CSV (matches the prior script's "balanced top-K" behavior).
unl_df = unl_df[: pos_df.shape[0]].with_columns(
    cca_logit=unl_scores[: pos_df.shape[0]].squeeze()
)


# -----------------------------------------------------------------------------
# Output: bottom-200 positives by logit (most-confidently-negative
# positives — useful for hand-checking the indexer's CCA tags)
# -----------------------------------------------------------------------------
# Distributions of logit scores for sanity inspection.
print("pos_scores percentiles [1, 2, 3, 4, 5, 10, 20]:",
      np.percentile(pos_scores, [1, 2, 3, 4, 5, 10, 20]))  # LOG
print("unl_scores percentiles [50, 75, 80, 85, 90, 95, 98]:",
      np.percentile(unl_scores, [50, 75, 80, 85, 90, 95, 98]))  # LOG

pos_df.select("id", "headline", "lead_paragraph", "cca_logit").bottom_k(
    by="cca_logit", k=200
).write_csv(str(config.CCA_CLASSIFIER_DIR / "pos_top_200.csv"))
