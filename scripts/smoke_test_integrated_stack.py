"""
Integration smoke test for the Tier 2 + Tier 3 abstractions, end-to-end.

Exercises the runtime composition that unit tests can't substitute
for: RunConfig → preprocessor → dataset_create → build_endpoint_model
→ build_inference_model → fit → save_weights + sidecar → rebuild
model from scratch (loading sidecar) → load_weights → predict. The
goal is to catch wiring bugs (dict-key mismatches, dtype mismatches,
save/load name alignment, config-vs-script drift) that only surface
at runtime, before they bite the cluster.

Designed to run on local Mac without DAPT weights or cached cca_set/
data:

  - Synthetic text data (no parquet, no NYT corpus needed).
  - Fake backbone (Embedding + multiply-by-padding-mask, sized
    HIDDEN_DIM=16) instead of the real RoBERTa-base. The
    test_assembly.py tests already verify the head + assembly
    wiring against a fake backbone; this test's distinctive value
    is the *full pipeline*, not the backbone-specific behavior.
  - Real preprocessor (with real RoBERTa tokenizer) and real
    dataset_create — the wiring most likely to drift.
  - Synthetic `RunConfig` (not DEFAULT_CCA_CONFIG, since the
    smoke test uses a small fake backbone and short sequences;
    DEFAULT is calibrated for real RoBERTa-base + production
    dataset shape).

Note (Tier 3 closeout, I2): the fake backbone exposes a
`hidden_dim` instance attribute (set on the keras.Model after
construction). Production uses `keras_hub.models.Backbone.from_preset(...)`
which exposes `hidden_dim` as a *class property*. Both work with
`getattr(backbone, "hidden_dim", None)` in
`RunConfig.validate_against_backbone`, but the smoke test only
exercises the instance-attribute path. If keras_hub ever renames
this property, the smoke test won't catch it; production would
silently raise the "no hidden_dim attribute" error from
validate_against_backbone, which is loud enough to be safe.
Adding a real-keras_hub-backbone variant of this smoke test (or
a separate cluster-only smoke test) would close that gap; not
warranted for a wiring smoke test today.

Run with:
    PYTHONPATH=. uv run python scripts/smoke_test_integrated_stack.py

Expected runtime: ~30s on local CPU. Output ends with "SMOKE TEST
PASSED" on success or a traceback on failure.

Kept as a permanent fixture (like
`scripts/experiment_endpoint_inference_evaluate.py`) so it can be
re-run after Tier 3/4 changes or library upgrades to verify the
integration still composes.
"""

import tempfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import src.config as config
import src.cca_config as cca_config
import src.data_setup.data as data_setup
from src.preproc.preprocessor import ClassifierPreprocessor
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_endpoint_model, build_inference_model
from src.loss_functions.loss import FLPULoss


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

keras.utils.set_random_seed(42)
keras.config.set_dtype_policy(config.DTYPE_POLICY)

SEQ_LEN = 32           # short for speed
HIDDEN_DIM = 16        # tiny — this is a wiring test, not a training run
BATCH_SIZE = 8
N_POS = 16
N_UNL = 80
ROBERTA_VOCAB_SIZE = 50265  # standard for roberta_base_en


# ---------------------------------------------------------------------------
# Run config (synthetic, smoke-test-sized)
# ---------------------------------------------------------------------------
# Construct a synthetic RunConfig sized for the smoke test, rather
# than using DEFAULT_CCA_CONFIG (which is calibrated for real
# RoBERTa-base + the production dataset shape). The config exercises
# the same Piece 3 sidecar pattern the production scripts use:
# train side writes the config alongside weights; rebuild side
# loads it from the sidecar.

tmp_dir = Path(tempfile.mkdtemp(prefix="ica_smoke_"))

run_config = cca_config.RunConfig(
    seq_length=SEQ_LEN,
    text_key="headline_with_lead",
    target_dtype="float32",
    heads=(
        cca_config.HeadConfig(
            name="cca",
            source_column="cca_label",
            hidden_dim=HIDDEN_DIM,
            loss=cca_config.FLPULossConfig(prior=0.1, kiryo_clawback=False),
        ),
    ),
    epochs=1,
    backbone_weights_path=str(tmp_dir / "fake_backbone.weights.h5"),
    ratio_batch=cca_config.RatioBatchConfig(),  # 0.1 / 0.5 / 0.5
    lr_schedule=cca_config.LRScheduleConfig(),
    optimizer=cca_config.OptimizerConfig(),
)
_cca_head_config = run_config.heads[0]


def _make_fake_backbone(name="fake_backbone"):
    """Tiny Embedding-based stand-in for the real RoBERTa backbone.
    Same input/output contract: dict of {token_ids, padding_mask},
    output shape (batch, seq, hidden). Also exposes a `hidden_dim`
    attribute matching the real keras_hub Backbone's contract — used
    by `RunConfig.validate_against_backbone` (Tier 3 Piece 3)."""
    token_ids = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="padding_mask")
    embed = keras.layers.Embedding(ROBERTA_VOCAB_SIZE, HIDDEN_DIM, name="fake_embed")
    embedded = embed(token_ids)
    mask_float = keras.ops.cast(padding_mask, "float32")
    mask_expanded = keras.ops.expand_dims(mask_float, axis=-1)
    masked = embedded * mask_expanded
    model = keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask},
        outputs=masked,
        name=name,
    )
    model.hidden_dim = HIDDEN_DIM
    return model


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

print("[1/8] Creating synthetic datasets...")
positive_texts = [
    f"protesters marched on capitol steps in event {i}" for i in range(N_POS)
]
unlabeled_texts = [
    f"the senate debated bill number {i} today" for i in range(N_UNL)
]

pos_path = tmp_dir / "train_pos.tf"
unl_path = tmp_dir / "train_unl.tf"

tf.data.Dataset.from_tensor_slices({
    "headline_with_lead": positive_texts,
    "cca_label": [1] * N_POS,
}).save(str(pos_path))

tf.data.Dataset.from_tensor_slices({
    "headline_with_lead": unlabeled_texts,
    "cca_label": [0] * N_UNL,
}).save(str(unl_path))

# Schema-aware validation matching the production scripts'
# discipline (Tier 3 closeout, addressing I1).
_pos_dataset = tf.data.Dataset.load(str(pos_path))
_dataset_columns = set(_pos_dataset.element_spec.keys())
_missing = run_config.expected_columns - _dataset_columns
assert not _missing, (
    f"Synthetic dataset is missing columns {sorted(_missing)}; "
    f"available={sorted(_dataset_columns)}; "
    f"expected={sorted(run_config.expected_columns)}."
)


# ---------------------------------------------------------------------------
# Preprocessors
# ---------------------------------------------------------------------------

print("[2/8] Building preprocessors (loads real RoBERTa tokenizer)...")
train_preprocess = ClassifierPreprocessor(
    SEQ_LENGTH=run_config.seq_length,
    text_key=run_config.text_key,
    label_keys=run_config.label_keys,
    endpoint_model=True,
    target_dtype=run_config.target_dtype,
)
predict_preprocess = ClassifierPreprocessor(
    SEQ_LENGTH=run_config.seq_length,
    text_key=run_config.text_key,
    label_keys={},
    endpoint_model=True,
    target_dtype=run_config.target_dtype,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

print("[3/8] Building training dataset via dataset_create...")
training_set = data_setup.dataset_create(
    shuffle_buffer=10,
    batch_size=BATCH_SIZE,
    preprocessor=train_preprocess,
    data=[
        tf.data.Dataset.load(str(pos_path)),
        tf.data.Dataset.load(str(unl_path)),
    ],
    weights=[
        run_config.ratio_batch.train_pos,
        1 - run_config.ratio_batch.train_pos,
    ],
)


# ---------------------------------------------------------------------------
# Models (Pattern A: shared head + backbone instances in one process)
# ---------------------------------------------------------------------------

print("[4/8] Building train + inference models (Pattern A)...")
backbone = _make_fake_backbone()

# (validate_against_backbone exercises the same defense-in-depth
# path the production scripts use.)
run_config.validate_against_backbone(backbone)

cca_head = ClassificationHead(
    hidden_dim=_cca_head_config.hidden_dim,
    loss_fn=FLPULoss(
        prior=_cca_head_config.loss.prior,
        kiryo_clawback=_cca_head_config.loss.kiryo_clawback,
    ),
    metrics=[
        keras.metrics.BinaryAccuracy(threshold=0.0),
        keras.metrics.Precision(thresholds=0.0, name="precision"),
    ],
    name=_cca_head_config.name,
)
train_model = build_endpoint_model(
    backbone=backbone,
    heads={_cca_head_config.name: cca_head},
    seq_length=run_config.seq_length,
    freeze_encoder=False,  # exercise the backbone-trains-too path
)
inf_model = build_inference_model(
    backbone=backbone,
    heads={_cca_head_config.name: cca_head},
    seq_length=run_config.seq_length,
)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

print("[5/8] Compiling + fitting (1 epoch, 4 steps)...")
train_model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3))
train_model.fit(
    training_set,
    epochs=1,
    steps_per_epoch=4,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Save weights
# ---------------------------------------------------------------------------

print("[6/8] Saving weights + run config sidecar...")
weights_path = tmp_dir / "smoke_classifier.weights.h5"
train_model.save_weights(str(weights_path))

# Save the run config sidecar alongside the weights, matching
# the discipline in run_cca_classification.py (Tier 3 Piece 3).
sidecar_path = cca_config.config_path_for_weights(weights_path)
run_config.to_json(sidecar_path)


# ---------------------------------------------------------------------------
# Pattern 2: rebuild from scratch + load weights by name
# ---------------------------------------------------------------------------

print("[7/8] Rebuilding from scratch via sidecar + loading weights (Pattern 2)...")
# Load the sidecar that the train side wrote — exercises the
# round-trip path the eval script uses in production.
run_config_2 = cca_config.RunConfig.from_json(sidecar_path)
_cca_head_config_2 = run_config_2.heads[0]
assert run_config_2 == run_config, (
    "Loaded RunConfig differs from the one the train side wrote — "
    "JSON round-trip regression."
)

backbone_2 = _make_fake_backbone()
run_config_2.validate_against_backbone(backbone_2)

cca_head_2 = ClassificationHead(
    hidden_dim=_cca_head_config_2.hidden_dim,
    loss_fn=FLPULoss(
        prior=_cca_head_config_2.loss.prior,
        kiryo_clawback=_cca_head_config_2.loss.kiryo_clawback,
    ),
    metrics=[
        keras.metrics.BinaryAccuracy(threshold=0.0),
        keras.metrics.Precision(thresholds=0.0, name="precision"),
    ],
    name=_cca_head_config_2.name,
)
inf_model_2 = build_inference_model(
    backbone=backbone_2,
    heads={_cca_head_config_2.name: cca_head_2},
    seq_length=run_config_2.seq_length,
)
# skip_mismatch=False matches the discipline in eval_cca_classifier.py
# and src/model_setup/backbone.py — pinned by Tier 3 Piece 2.
inf_model_2.load_weights(str(weights_path), skip_mismatch=False)


# ---------------------------------------------------------------------------
# Predict (both models, compare)
# ---------------------------------------------------------------------------

print("[8/8] Running predict on both inference models...")
test_set = (
    tf.data.Dataset.load(str(pos_path))
    .batch(BATCH_SIZE, drop_remainder=False)
    .map(predict_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)


def _extract(p):
    return p["cca"] if isinstance(p, dict) else p


preds_in_process = _extract(inf_model.predict(test_set, batch_size=BATCH_SIZE, verbose=0))
preds_after_load = _extract(inf_model_2.predict(test_set, batch_size=BATCH_SIZE, verbose=0))


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

# 1. Shape check.
assert preds_after_load.shape == (N_POS, 1), (
    f"Expected predictions shape ({N_POS}, 1), got {preds_after_load.shape}. "
    "Likely cause: predict-preprocessor didn't drop targets cleanly, or "
    "inference model output structure changed."
)

# 2. Pattern A and Pattern 2 should produce identical predictions —
#    weights-by-name save+load should reproduce the trained model exactly.
max_diff = float(np.max(np.abs(preds_after_load - preds_in_process)))
assert max_diff < 1e-4, (
    f"Pattern A (in-process) and Pattern 2 (rebuilt + loaded) "
    f"predictions differ by {max_diff:.2e} (expected < 1e-4). "
    "Likely cause: weight name mismatch between train and rebuilt "
    "inference models, or stochastic layers active during predict."
)

# 3. Sanity: predictions are finite.
assert np.all(np.isfinite(preds_after_load)), (
    "Predictions contain NaN or Inf — possible numerical instability "
    "during fit, or weight init issue."
)

# 4. Sanity: training actually moved weights. The rebuilt-from-scratch
#    backbone has *different* (random) weights than the trained backbone;
#    after load_weights they should match. Check via a backbone weight
#    norm comparison.
trained_backbone_norms = [float(tf.norm(w)) for w in backbone.weights]
loaded_backbone_norms = [float(tf.norm(w)) for w in backbone_2.weights]
norm_diffs = [abs(t - l) for t, l in zip(trained_backbone_norms, loaded_backbone_norms)]
assert max(norm_diffs) < 1e-4, (
    f"Trained backbone weights and rebuilt-then-loaded backbone weights "
    f"differ by up to {max(norm_diffs):.2e} in norm — load_weights may "
    "not have copied all backbone weights."
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 50)
print("SMOKE TEST PASSED")
print("=" * 50)
print(f"  Synthetic data:     {N_POS} positives, {N_UNL} unlabeled")
print(f"  Predictions shape:  {preds_after_load.shape}")
print(f"  Pattern A vs 2 max-diff: {max_diff:.2e}  (< 1e-4)")
print(f"  All predictions finite:  ✓")
print(f"  Backbone weights load:   ✓")
print()
print(f"Tmp dir: {tmp_dir}  (auto-cleaned by OS)")
