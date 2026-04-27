"""
Train the single-head CCA classifier on the cached LDC dataset.

This script integrates the Tier 2 abstractions:

  - `src.config` for platform-conditional paths and dtype policy.
  - `src.preproc.preprocessor.ClassifierPreprocessor` in endpoint
    mode, with multi-head-shaped `label_keys`.
  - `src.model_setup.heads.ClassificationHead` carrying its own
    `loss_fn` (FLPU) and `metrics` (per-head, name-prefixed).
  - `src.model_setup.backbone.load_dapt_backbone` for the
    DAPT-finetuned backbone.
  - `src.model_setup.assembly.build_endpoint_model` and
    `build_inference_model` to wire backbone + head into both a
    training model (with target inputs, head's add_loss handles
    loss, head's metric_objs handle metrics) and an inference model
    (no target inputs, for predict). The two models share the head
    and backbone Layer instances (Pattern A) — fit on the training
    model trains the inference model's weights by Python identity.

  - `src.model_setup.layer_lr_model.LayerLRModel` is the type
    returned by `build_endpoint_model`. With `freeze_encoder=True`
    and no `layer_multipliers` configured, it behaves identically
    to a plain `keras.Model` with a frozen backbone — but the
    forward-compatibility for discriminative LR / unfreezing is
    in place when we want it.
"""

import keras
import tensorflow as tf

import math
import datetime

import src.config as config
import src.data_setup.data
from src.preproc.preprocessor import ClassifierPreprocessor
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_endpoint_model, build_inference_model
from src.loss_functions.loss import FLPULoss

# Platform-conditional dtype policy (mixed_float16 on cluster CUDA;
# float32 locally — MPS mixed-precision support is patchy and there
# are no Tensor Cores to motivate it).
keras.config.set_dtype_policy(config.DTYPE_POLICY)

# Seed Python, NumPy, and the Keras backend RNG so training is
# reproducible. Matches the seed=200 used for the polars `.sample()`
# splits in `data_setup/data.py`.
keras.utils.set_random_seed(200)


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------

# Preprocessing params
# SEQ_LENGTH and BATCH_SIZE of 128 for local testing (see notes in the
# memo for rough assessment of how much truncation that causes); maybe
# bump SEQ_LENGTH back to 256 for Explorer? Not sure.
BATCH_SIZE = 256
SEQ_LENGTH = 128

# Training params
EPOCHS = 7

# Class prior estimate for FLPU. π_pos ≈ 0.02 — the corrected
# estimate from `run_prior_estimate.py` (after fixing the bandwidth-
# scale mis-calibration in DEDPUL; see the comment in
# `run_prior_estimate.py` for the four-variant attribution table).
# Was 0.03 in the pre-Tier-2 code; the bump down to 0.02 is one of
# the deferred empirical checks Tier 2 was tracking.
FLPU_PRIOR = 0.02


# -----------------------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------------------
# Load + split only if the cached tf.data datasets don't already exist
# on disk; otherwise skip straight to loading from disk. (Building the
# polars dataframe and turning it into tensor slices takes minutes on
# the full corpus, and the old code ran these unconditionally even
# when it was about to overwrite `ldc_data` with the cache load below.)
if not config.CCA_SET_DIR.is_dir():
    ldc_data = src.data_setup.data.data_from_parquet(
        config.PROJECT_ROOT,
        "ldc_corpus",
        addl_columns=["cca", "cca_descriptor", "immig", "immig_descriptor"],
    )
    ldc_data = src.data_setup.data.create_classifier_data(
        ldc_data, separate_labels=True
    )
    config.CCA_SET_DIR.mkdir()
    for split in ldc_data.keys():
        for pu in ldc_data[split].keys():
            ldc_data[split][pu] = tf.data.Dataset.from_tensor_slices(
                ldc_data[split][pu]
                .select(["headline_with_lead", "cca_label"])
                .to_dict()
            )
            ldc_data[split][pu].save(str(config.CCA_SET_DIR / f"{split}_{pu}.tf"))
else:
    ldc_data = {"train": {}, "val": {}, "test": {}}
    for split in ldc_data:
        for pu in ("pos", "unl"):
            ldc_data[split][pu] = tf.data.Dataset.load(
                str(config.CCA_SET_DIR / f"{split}_{pu}.tf")
            )


# -----------------------------------------------------------------------------
# Preprocessors
# -----------------------------------------------------------------------------
# Two preprocessor instances:
#
#  - `train_preprocess`: emits the full endpoint-mode batch including
#    `cca_targets`, used for fit/evaluate where the head's add_loss
#    needs the target tensor as a model input.
#  - `predict_preprocess`: same shape minus the `cca_targets` entry,
#    used when feeding the inference model (which has no target
#    Inputs in its graph and would otherwise have an unused dict key).
#
# Both use endpoint_model=True; the only difference is whether
# label_keys produces target columns. Empty label_keys + endpoint
# mode → output is just `{token_ids, padding_mask}`.
train_preprocess = ClassifierPreprocessor(
    SEQ_LENGTH=SEQ_LENGTH,
    text_key="headline_with_lead",
    label_keys={"cca_targets": "cca_label"},
    endpoint_model=True,
)
predict_preprocess = ClassifierPreprocessor(
    SEQ_LENGTH=SEQ_LENGTH,
    text_key="headline_with_lead",
    label_keys={},
    endpoint_model=True,
)


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
# Ratio Batch sampling: every training batch contains a known fraction
# of labeled positives. 1:9 for training (matching the original
# 9-unl-to-1-pos ratio); 1:1 for val/test where we want positives and
# unlabeled equally represented for metric stability.
shuffle_buffer = 100000

training_set = src.data_setup.data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    train_preprocess,
    data=[ldc_data["train"]["pos"], ldc_data["train"]["unl"]],
    weights=[1 / 10, 9 / 10],
)
validation_set = src.data_setup.data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    train_preprocess,
    data=[ldc_data["val"]["pos"], ldc_data["val"]["unl"]],
    weights=[0.5, 0.5],
)
test_set = src.data_setup.data.dataset_create(
    shuffle_buffer,
    BATCH_SIZE,
    train_preprocess,
    data=[ldc_data["test"]["pos"], ldc_data["test"]["unl"]],
    weights=[0.5, 0.5],
)

# Steps. Train: 18300 positives, 1026418 unlabeled.
# Val: 1017 positives, 57024 unlabeled.
# Test split is also 5% of source; positive count is approximately equal
# to val (same sampling logic). Computing exact test positive count
# from the saved tf.data dataset is possible but adds I/O; the val-
# derived approximation is fine for `steps=` purposes.
steps_per_epoch = math.floor(18300 / (BATCH_SIZE / 10))
validation_steps = math.floor(1017 / (BATCH_SIZE / 2))
test_steps = validation_steps  # see comment above


# -----------------------------------------------------------------------------
# Model assembly
# -----------------------------------------------------------------------------
# Backbone: DAPT-finetuned RoBERTa, weights loaded from the .h5 file
# produced by the DAPT phase.
backbone = load_dapt_backbone(config.DAPT_BACKBONE_WEIGHTS)

# Single-head classifier: FLPU loss handled internally via the head's
# add_loss path (endpoint mode). Per-head metrics handled internally
# via the head's metric_objs path (Tier 2 Piece 4c addition; symmetric
# with loss_fn). The head's name "cca" prefixes the metrics for
# disambiguation when more heads land later.
#
# With a ~2% class prior and 50/50-weighted validation batches,
# BinaryAccuracy alone is misleading (the model can score well by
# being very cautious about positives). Precision/Recall/PR-AUC
# capture the actual classification behavior under imbalance.
# Thresholds are 0.0 because outputs are logits (sigmoid(0)=0.5).
# F1 isn't included because keras.metrics.F1Score requires threshold
# in (0, 1] (probability output); compute it post-hoc from precision
# and recall.
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

# Pattern A: build train + inference models sharing the head and
# backbone Layer instances. Fitting the train model trains the
# inference model's weights by Python identity. The inference model
# is for predict() only when used this way — see the docstring on
# build_inference_model for the operational rule.
cca_classifier = build_endpoint_model(
    backbone=backbone,
    heads={"cca": cca_head},
    seq_length=SEQ_LENGTH,
    freeze_encoder=True,
)
cca_inference = build_inference_model(
    backbone=backbone,
    heads={"cca": cca_head},
    seq_length=SEQ_LENGTH,
)


# -----------------------------------------------------------------------------
# Optimizer and compile
# -----------------------------------------------------------------------------
# CosineDecay LR schedule with warmup. ChatGPT-suggested params from
# the pre-Tier-2 sweep; keep until we have a real LR sweep.
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=steps_per_epoch * 3,
    alpha=1e-1,
    warmup_target=1e-3,
    warmup_steps=steps_per_epoch / 4,
)

# AdamW + LossScaleOptimizer wrapping under mixed_float16 (cluster).
# Loss scaling protects against fp16 gradient underflow on small
# gradients, which is the standard practice for CUDA mixed precision.
# Locally (float32) it's unnecessary and just adds machinery, so
# we skip the wrap.
base_optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=5e-3,
)
if config.IS_CLUSTER:
    optimizer = keras.optimizers.LossScaleOptimizer(base_optimizer)
else:
    optimizer = base_optimizer

# Compile WITHOUT a loss or metrics argument — both live inside the
# head and propagate via model.losses / model.metrics. Keras handles
# the aggregation automatically. `jit_compile="auto"` lets Keras
# decide whether XLA compilation is beneficial.
cca_classifier.compile(optimizer=optimizer, jit_compile="auto")


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
config.CCA_CLASSIFIER_DIR.mkdir(exist_ok=True)
config.CCA_LOGS_DIR.mkdir(exist_ok=True)

_run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(config.CCA_CLASSIFIER_DIR / f"{_run_stamp}_checkpoint.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    ),
    keras.callbacks.TensorBoard(
        log_dir=str(config.CCA_LOGS_DIR / _run_stamp),
        histogram_freq=1,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=(500, 550),
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        verbose=1,
        start_from_epoch=2,
    ),
]


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
cca_classifier.fit(
    training_set,
    validation_data=validation_set,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks_list,
)

# Save weights only (LayerLRModel isn't fully serializable via the
# standard `.keras` save without registering custom Keras objects;
# weights load by name into a freshly-constructed model in the eval
# script — Pattern 2 for cross-process weight loading).
cca_classifier.save_weights(str(config.CCA_CLASSIFIER_WEIGHTS))


# -----------------------------------------------------------------------------
# Evaluate on test set
# -----------------------------------------------------------------------------
# Uses the training model (which has cca_targets as inputs and
# add_loss/add_metric machinery wired up).
test_results = cca_classifier.evaluate(
    test_set, steps=test_steps, return_dict=True
)
print(f"Test results: {test_results}")  # LOG


# -----------------------------------------------------------------------------
# Per-subset predictions for qualitative review
# -----------------------------------------------------------------------------
# Build *finite* (non-repeated) test datasets sized to the actual data
# rather than reusing dataset_create's repeat()-based pipeline. The
# old approach passed `steps=validation_steps` (which was wrong — it
# was sized from val positives, not test) to a repeated dataset,
# which produced duplicate predictions; downstream code worked around
# this by slicing to the real dataframe length. Building finite
# datasets here removes the workaround.
#
# `predict_preprocess` is used (no `cca_targets` in output) because
# the inference model's input signature doesn't include target tensors.
def _finite_predict_dataset(saved_dataset_path):
    return (
        tf.data.Dataset.load(str(saved_dataset_path))
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(predict_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


test_pos_finite = _finite_predict_dataset(config.CCA_SET_DIR / "test_pos.tf")
pos_scores = cca_inference.predict(test_pos_finite, batch_size=BATCH_SIZE)

test_unl_finite = _finite_predict_dataset(config.CCA_SET_DIR / "test_unl.tf")
unl_scores = cca_inference.predict(test_unl_finite, batch_size=BATCH_SIZE)

# pos_scores / unl_scores are dicts keyed by output name in the
# multi-head case; for our single-head model `cca_inference.predict`
# returns the output dict directly (key "cca").
print(f"pos_scores shape: {pos_scores['cca'].shape if isinstance(pos_scores, dict) else pos_scores.shape}")
print(f"unl_scores shape: {unl_scores['cca'].shape if isinstance(unl_scores, dict) else unl_scores.shape}")
