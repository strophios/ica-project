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
import src.cca_config as cca_config
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


# -----------------------------------------------------------------------------
# Run configuration
# -----------------------------------------------------------------------------
# Load the RunConfig sidecar that the training script wrote
# alongside the weights file (Tier 3 Piece 3 — I4: train/eval
# coupling). Drives preprocessor + head + assembly construction
# below with the *exact* same values the training script used —
# eliminating the class of bugs where train and eval scripts
# silently disagree on `seq_length`, `text_key`, head config, etc.
#
# If the sidecar is missing (e.g., a weights file from before the
# Piece 3 sidecar discipline), `from_json` raises a clear error
# pointing at the CLI helper for ad-hoc sidecar creation:
# `python -m src.cca_config write_default <weights_path>`.

_sidecar_path = cca_config.config_path_for_weights(config.CCA_CLASSIFIER_WEIGHTS)
run_config = cca_config.RunConfig.from_json(_sidecar_path)
_cca_head_config = run_config.heads[0]

# Script-local: BATCH_SIZE is eval-only operational (loading
# throughput vs. memory), independent of train.
BATCH_SIZE = 256


# -----------------------------------------------------------------------------
# Build inference model (Pattern 2: fresh head, weights loaded by name)
# -----------------------------------------------------------------------------
backbone = load_dapt_backbone(run_config.backbone_weights_path)

# Defense-in-depth: catch backbone-vs-config hidden_dim drift before
# weight load. Piece 2's `skip_mismatch=False` is the load-time
# backstop; this fires earlier with a clearer message.
run_config.validate_against_backbone(backbone)

# Fresh head instance with the same configuration the training
# script used (driven by the same `RunConfig`). Variable shapes line
# up with the saved weights because architecture-shape comes from
# the same config — see `docs/notes/tier3-design.md` Piece 2 for
# the structural-vs-name-matching framing of weight loading.
#
# Metrics are intentionally NOT constructed here. Tier 3 closeout
# (addressing I3 from the adversarial review): empirically verified
# that metric state variables live on the metric instances under
# `head.metrics` but are NOT included in `head.weights`, which is
# what `save_weights` / `load_weights` operate on. So eval-side
# metrics are not load-bearing for shape match — and the inference
# graph never updates them anyway (head's `call` only updates
# metrics when targets are provided; the inference graph passes
# `targets=None`). The previous "for parity" reproduction here was
# unused machinery + a hand-coupling drift risk; both are gone now.
# When `evaluate()` is needed at eval time (e.g., a hand-labeled
# PN test set lands), construct a separate evaluate-mode model
# from the same RunConfig with metrics from `make_cca_metrics()`.
cca_head = ClassificationHead(
    hidden_dim=_cca_head_config.hidden_dim,
    loss_fn=FLPULoss(
        prior=_cca_head_config.loss.prior,
        kiryo_clawback=_cca_head_config.loss.kiryo_clawback,
    ),
    name=_cca_head_config.name,
)

cca_inference = build_inference_model(
    backbone=backbone,
    heads={_cca_head_config.name: cca_head},
    seq_length=run_config.seq_length,
)

# Load weights saved by the training script. Keras 3's `.weights.h5`
# save format keys variables by layer-class + positional index —
# matching is structural (layer types, ordering, weight shapes),
# not by user-given name. The fresh head here uses the same
# `ClassificationHead` configuration the training script used (same
# RunConfig), so the architecture aligns and weights load cleanly.
# `skip_mismatch=False` pins the load-strict discipline (Tier 3
# Piece 2): a future architectural drift between training and eval
# (e.g., changing `hidden_dim`, adding a head-internal layer)
# raises `ValueError` rather than silently producing a model with
# partial-or-default weights.
cca_inference.load_weights(
    str(config.CCA_CLASSIFIER_WEIGHTS), skip_mismatch=False
)


# -----------------------------------------------------------------------------
# Build finite predict datasets
# -----------------------------------------------------------------------------
# Layer-1 schema-aware validation against the cached test datasets,
# matching the discipline in run_cca_classification.py. Tier 3
# closeout (addressing I1). At eval time the labels columns aren't
# strictly required (predict mode emits no targets), but checking
# `run_config.text_key` ensures the dataset structure is what eval
# expects. We validate against test_pos as a representative; the
# saved tf.data datasets all have the same element_spec.
_test_pos_dataset = tf.data.Dataset.load(str(config.CCA_SET_DIR / "test_pos.tf"))
_dataset_columns = set(_test_pos_dataset.element_spec.keys())
if run_config.text_key not in _dataset_columns:
    raise ValueError(
        f"Cached test dataset at {config.CCA_SET_DIR} does not "
        f"contain text_key={run_config.text_key!r}. "
        f"Dataset columns: {sorted(_dataset_columns)}. The run "
        f"config sidecar at {_sidecar_path} declares this text_key, "
        f"but the cached dataset doesn't have it — likely a stale "
        f"cache vs. config mismatch."
    )

# Predict-only preprocessor: no targets emitted, just the model inputs
# the inference graph declares (`token_ids`, `padding_mask`).
predict_preprocess = ClassifierPreprocessor(
    SEQ_LENGTH=run_config.seq_length,
    text_key=run_config.text_key,
    label_keys={},
    endpoint_model=True,
    target_dtype=run_config.target_dtype,
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
