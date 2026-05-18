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

import dataclasses
import datetime
import math

import src.config as config
import src.cca_config as cca_config
from src.cca_metrics import make_cca_metrics
from src.diagnostics.distribution_metrics import make_distribution_metrics
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


def main(run_config=None, max_steps=None):
    """
    Train the CCA classifier.

    Args:
        run_config: Optional RunConfig instance. If None, uses DEFAULT_CCA_CONFIG.
        max_steps: Optional cap on steps_per_epoch. If provided, steps_per_epoch
                   is capped at this value.
    """
    # Run configuration
    # -----------------------------------------------------------------------
    # All architectural and research-dimension parameters come from a
    # `RunConfig` instance (Tier 3 Piece 3 — I4: train/eval coupling).
    # The config drives preprocessor / head / assembly / optimizer
    # construction, and gets serialized to a JSON sidecar alongside the
    # saved weights at the end of training. The eval script
    # (`eval_cca_classifier.py`) reads that sidecar and constructs its
    # inference model with the exact same values, eliminating the
    # class of bugs where train and eval scripts silently disagree on
    # `seq_length`, `text_key`, head config, prior, etc.
    #
    # To run an experimental variant, replace `DEFAULT_CCA_CONFIG` with
    # a `dataclasses.replace`-derived alternative — e.g.,
    #
    #     run_config = dataclasses.replace(
    #         cca_config.DEFAULT_CCA_CONFIG, epochs=10,
    #     )
    #
    # See `docs/notes/tier3-design.md` Piece 3 for the design framing.

    if run_config is None:
        run_config = cca_config.DEFAULT_CCA_CONFIG

    # The single (currently only) head's config; convenience binding
    # since multi-head support is not yet wired through the rest of
    # this script.
    _cca_head_config = run_config.heads[0]

    # Script-local operational params. Not in run_config because they
    # differ between train/eval (BATCH_SIZE: throughput-vs-memory) or
    # don't affect the trained model (shuffle_buffer).
    BATCH_SIZE = 256
    SHUFFLE_BUFFER = 100_000


    # -------------------------------------------------------------------------
    # Load and prepare data
    # -------------------------------------------------------------------------
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

    # Layer-1 schema-aware validation: confirm the dataset contains
    # the columns run_config says it expects, BEFORE the preprocessor
    # trace-time check fires. Tier 3 closeout (addressing I1 from the
    # adversarial review): the design doc claims a 3-layer validation
    # hierarchy — config self-validity, schema-vs-config (this), and
    # call-time data validation; this assertion makes that hierarchy
    # real rather than aspirational.
    _dataset_columns = set(ldc_data["train"]["pos"].element_spec.keys())
    _missing_columns = run_config.expected_columns - _dataset_columns
    if _missing_columns:
        raise ValueError(
            f"Cached dataset at {config.CCA_SET_DIR} does not contain "
            f"every column run_config expects. Missing: {sorted(_missing_columns)}. "
            f"Dataset columns: {sorted(_dataset_columns)}. "
            f"Configured: text_key={run_config.text_key!r}, "
            f"label_keys source columns={sorted(run_config.label_keys.values())}. "
            f"Either rebuild the cache (delete CCA_SET_DIR) with the "
            f"current run config or update the run config to match the "
            f"existing cache."
        )


    # -------------------------------------------------------------------------
    # Preprocessors
    # -------------------------------------------------------------------------
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


    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    # Ratio Batch sampling: every training batch contains a known fraction
    # of labeled positives. The `RatioBatchConfig` defaults match the
    # pre-Tier-3 hardcodes (0.1 train, 0.5 val/test = 1:9 / 1:1) but are
    # now sweepable via run_config — see the Tier-2-pinned "Ratio Batch
    # sensitivity sweep" deferred empirical-check item.
    training_set = src.data_setup.data.dataset_create(
        SHUFFLE_BUFFER,
        BATCH_SIZE,
        train_preprocess,
        data=[ldc_data["train"]["pos"], ldc_data["train"]["unl"]],
        weights=[
            run_config.ratio_batch.train_pos,
            1 - run_config.ratio_batch.train_pos,
        ],
    )
    validation_set = src.data_setup.data.dataset_create(
        SHUFFLE_BUFFER,
        BATCH_SIZE,
        train_preprocess,
        data=[ldc_data["val"]["pos"], ldc_data["val"]["unl"]],
        weights=[
            run_config.ratio_batch.val_pos,
            1 - run_config.ratio_batch.val_pos,
        ],
    )
    # Steps. Train: 18300 positives, 1026418 unlabeled.
    # Val: 1017 positives, 57024 unlabeled.
    # Test set isn't constructed via dataset_create here — Tier 3
    # closeout (I5) replaces the Ratio-Batch + `.repeat()` + `steps=`-
    # approximation pattern with a finite, concat-based test dataset
    # constructed below near the evaluate() call. evaluate iterates
    # the whole test set exactly once without needing `steps=`.
    steps_per_epoch = math.floor(18300 / (BATCH_SIZE / 10))
    validation_steps = math.floor(1017 / (BATCH_SIZE / 2))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)

    # Resolve LR schedule factors against the concrete steps_per_epoch
    # so the sidecar is self-sufficient. See
    # docs/notes/tier4-design.md Piece 2.
    run_config = dataclasses.replace(
        run_config,
        lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch),
    )


    # -------------------------------------------------------------------------
    # Model assembly
    # -------------------------------------------------------------------------
    # Backbone: DAPT-finetuned RoBERTa, weights loaded from the .h5 file
    # at the path declared in run_config.backbone_weights_path (defaults
    # to config.DAPT_BACKBONE_WEIGHTS via DEFAULT_CCA_CONFIG).
    backbone = load_dapt_backbone(run_config.backbone_weights_path)

    # Defense-in-depth: verify the backbone's hidden_dim matches what
    # the head config declares. Catches the bug class "wrong backbone
    # for this run config" *before* weight load (Piece 2's shape-
    # mismatch check is the load-time backstop).
    run_config.validate_against_backbone(backbone)

    # Single-head classifier: FLPU loss handled internally via the head's
    # add_loss path (endpoint mode). Per-head metrics handled internally
    # via the head's metric_objs path (Tier 2 Piece 4c addition; symmetric
    # with loss_fn). The head's name (from run_config) prefixes the
    # metrics for disambiguation when more heads land later.
    #
    # With a ~2% class prior and 50/50-weighted validation batches,
    # BinaryAccuracy alone is misleading (the model can score well by
    # being very cautious about positives). Precision/Recall/PR-AUC
    # capture the actual classification behavior under imbalance.
    # Thresholds are 0.0 because outputs are logits (sigmoid(0)=0.5).
    # F1 isn't included because keras.metrics.F1Score requires threshold
    # in (0, 1] (probability output); compute it post-hoc from precision
    # and recall.
    #
    # Metrics are script-local (not in run_config) — they're monitoring
    # choices, not load-bearing for predict, and serializing keras
    # Metric configs is a layer of ceremony we don't need yet. See
    # `docs/notes/tier3-design.md` Piece 3 "What's in vs. out".
    cca_head = ClassificationHead(
        hidden_dim=_cca_head_config.hidden_dim,
        loss_fn=FLPULoss(
            prior=_cca_head_config.loss.prior,
            kiryo_clawback=_cca_head_config.loss.kiryo_clawback,
        ),
        metrics=make_cca_metrics()
        + make_distribution_metrics(run_config.diagnostics),
        name=_cca_head_config.name,
        expose_loss_components=run_config.diagnostics.enable_loss_components,
    )

    # Pattern A: build train + inference models sharing the head and
    # backbone Layer instances. Fitting the train model trains the
    # inference model's weights by Python identity. The inference model
    # is for predict() only when used this way — see the docstring on
    # build_inference_model for the operational rule.
    cca_classifier = build_endpoint_model(
        backbone=backbone,
        heads={_cca_head_config.name: cca_head},
        seq_length=run_config.seq_length,
        freeze_encoder=True,
        diagnostics=run_config.diagnostics,
    )
    cca_inference = build_inference_model(
        backbone=backbone,
        heads={_cca_head_config.name: cca_head},
        seq_length=run_config.seq_length,
    )


    # -------------------------------------------------------------------------
    # Optimizer and compile
    # -------------------------------------------------------------------------
    # CosineDecay LR schedule with warmup. Parameters come from
    # run_config.lr_schedule, with resolved step counts populated
    # by with_resolved() earlier. Warmup and decay steps are now read
    # directly from the resolved sub-object.
    resolved = run_config.lr_schedule.resolved
    assert resolved is not None, (
        "lr_schedule.resolved should be populated by the with_resolved "
        "call earlier; this is a programmer error if it fires."
    )
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=run_config.lr_schedule.initial_lr,
        decay_steps=resolved.decay_steps,
        alpha=run_config.lr_schedule.decay_alpha,
        warmup_target=run_config.lr_schedule.warmup_target,
        warmup_steps=resolved.warmup_steps,
    )

    # AdamW + LossScaleOptimizer wrapping under mixed_float16 (cluster).
    # Loss scaling protects against fp16 gradient underflow on small
    # gradients, which is the standard practice for CUDA mixed precision.
    # Locally (float32) it's unnecessary and just adds machinery, so
    # we skip the wrap. weight_decay comes from run_config.optimizer.
    base_optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=run_config.optimizer.weight_decay,
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


    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    config.CCA_CLASSIFIER_DIR.mkdir(exist_ok=True)
    config.CCA_LOGS_DIR.mkdir(exist_ok=True)

    _run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSVLogger needs the per-run-stamp directory to exist.
    (config.CCA_LOGS_DIR / _run_stamp).mkdir(parents=True, exist_ok=True)

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.CCA_CLASSIFIER_DIR / f"{_run_stamp}_checkpoint.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.CSVLogger(
            str(config.CCA_LOGS_DIR / _run_stamp / "metrics.csv")
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


    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    cca_classifier.fit(
        training_set,
        validation_data=validation_set,
        epochs=run_config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
    )

    # Save weights only (LayerLRModel isn't fully serializable via the
    # standard `.keras` save without registering custom Keras objects;
    # weights load by name into a freshly-constructed model in the eval
    # script — Pattern 2 for cross-process weight loading).
    cca_classifier.save_weights(str(config.CCA_CLASSIFIER_WEIGHTS))

    # Save the run config sidecar alongside the weights (Tier 3 Piece 3
    # — I4: train/eval coupling). The eval script
    # (`eval_cca_classifier.py`) reads this file at startup and drives
    # its preprocessor + head + assembly construction from the same
    # values used here, eliminating silent drift between the two
    # scripts. Sidecar path is derived from the weights path via
    # `config_path_for_weights` (.weights.h5 -> .config.json).
    sidecar_path = cca_config.config_path_for_weights(config.CCA_CLASSIFIER_WEIGHTS)
    run_config.to_json(sidecar_path)
    print(f"Saved run config sidecar: {sidecar_path}")  # LOG


    # -------------------------------------------------------------------------
    # Evaluate on test set
    # -------------------------------------------------------------------------
    # Build a *finite* test set for evaluate() — Tier 3 closeout
    # (addressing I5 from the adversarial review). The previous version
    # evaluated `test_set` (a Ratio-Batch-sampled, .repeat()-ed dataset
    # from dataset_create) for `steps=test_steps` where
    # `test_steps = validation_steps` (an approximation). That produced
    # evaluate-on-an-arbitrary-prefix behavior, with the test loss being
    # computed on a slightly-too-small or slightly-too-large slice
    # depending on whether test had more or fewer positives than val.
    # Following Piece 2's discipline for predict, build a finite
    # (non-repeated) dataset sized to the actual test data — concat
    # pos + unl, batch, preprocess. evaluate() iterates the whole
    # thing exactly once, no `steps=` approximation needed.
    test_pos_raw = tf.data.Dataset.load(str(config.CCA_SET_DIR / "test_pos.tf"))
    test_unl_raw = tf.data.Dataset.load(str(config.CCA_SET_DIR / "test_unl.tf"))
    test_set_finite = (
        test_pos_raw.concatenate(test_unl_raw)
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_results = cca_classifier.evaluate(test_set_finite, return_dict=True)
    print(f"Test results: {test_results}")  # LOG


    # -------------------------------------------------------------------------
    # Per-subset predictions for qualitative review
    # -------------------------------------------------------------------------
    # Build *finite* (non-repeated) test datasets sized to the actual data
    # rather than reusing dataset_create's repeat()-based pipeline. The
    # old approach passed `steps=validation_steps` (which was wrong — it
    # was sized from val positives, not test) to a repeated dataset,
    # which produced duplicate predictions; downstream code worked around
    # this by slicing to the real dataframe length. Building finite
    # datasets here removes the workaround. Tier 3 closeout extends the
    # same discipline to evaluate (above).
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


if __name__ == "__main__":
    main()
