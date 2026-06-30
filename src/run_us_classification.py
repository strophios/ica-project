# pattern: Imperative Shell
"""
Train the US/not-US filter as a frozen-DAPT linear probe with BCE.

This script integrates the model-construction spine (backbone, heads, assembly)
to train a binary supervised classifier. The US filter differs from CCA:
  - Loss: BinaryCrossentropy (BCE), not FLPU (no prior/nnPU)
  - Data: single natural-balance shuffled stream (PN task), not PU ratio-batch
  - Config: UsRunConfig (parallel to RunConfig, no FLPU coupling)

Importing must not train (`if __name__ == "__main__": main()`).
"""

import dataclasses
import datetime
import math

import keras
import tensorflow as tf
import polars as pl

import src.config as config
import src.us_config as us_config
from src.us_config import config_path_for_weights
import src.data_setup.data
from src.preproc.preprocessor import ClassifierPreprocessor
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_endpoint_model, build_inference_model
from src.us_metrics import make_us_metrics
from src.diagnostics.distribution_metrics import make_distribution_metrics
from src.preproc.dateline_guard import assert_no_dateline_residue
from src.validation.escalation import top_n_group_fn


def main(run_config=None, max_steps=None):
    """
    Train the US filter.

    Args:
        run_config: Optional UsRunConfig instance. If None, uses DEFAULT_US_CONFIG.
        max_steps: Optional cap on steps_per_epoch. If provided, steps_per_epoch
                   is capped at this value.
    """
    # Platform-conditional dtype and seed (mirrors run_cca_classification.py)
    keras.config.set_dtype_policy(config.DTYPE_POLICY)
    keras.utils.set_random_seed(200)

    if run_config is None:
        run_config = us_config.DEFAULT_US_CONFIG

    head_cfg = run_config.head
    BATCH_SIZE = 256
    SHUFFLE_BUFFER = 100_000

    # -------------------------------------------------------------------------
    # Load and prepare data
    # -------------------------------------------------------------------------
    df = src.data_setup.data.data_from_parquet(
        config.PROJECT_ROOT,
        "us_filter",
        addl_columns=["us_label", "label_source"],
        lead_column="stripped_text",
    )
    splits = src.data_setup.data.create_us_filter_data(df)
    for sdf in splits.values():
        assert_no_dateline_residue(sdf["stripped_text"])  # AC2.2 guard

    # -------------------------------------------------------------------------
    # tf.data cache
    # -------------------------------------------------------------------------
    # Cache under US_FILTER_SET_DIR is keyed by split name. If the labeled
    # parquet changes upstream (Phase 1 gazetteers finalized / re-run),
    # DELETE US_FILTER_SET_DIR so it rebuilds — otherwise cached data and
    # recomputed steps silently diverge.
    if not config.US_FILTER_SET_DIR.is_dir():
        config.US_FILTER_SET_DIR.mkdir(parents=True)
        for name, sdf in splits.items():
            ds = tf.data.Dataset.from_tensor_slices({
                "headline_with_lead": sdf["headline_with_lead"].to_list(),
                "us_label": sdf["us_label"].cast(pl.Int8).to_numpy(),
            })
            ds.save(str(config.US_FILTER_SET_DIR / f"{name}.tf"))

    datasets = {
        n: tf.data.Dataset.load(str(config.US_FILTER_SET_DIR / f"{n}.tf"))
        for n in ("train", "val", "test")
    }

    # Freshness check: cached cardinality must match current split sizes.
    # NOTE: this check validates row count but NOT row order; if the split
    # function's ordering semantics change (e.g. shuffling is removed), the cache
    # must be deleted manually. See create_us_filter_data for the within-split
    # shuffle that ensures early tf.data batches are not class-blocked.
    for n in ("train", "val", "test"):
        cached_n = int(datasets[n].cardinality().numpy())
        if cached_n != splits[n].shape[0]:
            raise ValueError(
                f"Stale US set cache for split {n!r}: cache has {cached_n} rows "
                f"but current split has {splits[n].shape[0]}. Delete "
                f"{config.US_FILTER_SET_DIR} and re-run."
            )

    # -------------------------------------------------------------------------
    # Preprocessor (endpoint mode, single head)
    # -------------------------------------------------------------------------
    train_preprocess = ClassifierPreprocessor(
        SEQ_LENGTH=run_config.seq_length,
        text_key=run_config.text_key,
        label_keys=run_config.label_keys,
        endpoint_model=True,
        target_dtype=run_config.target_dtype,
    )

    # -------------------------------------------------------------------------
    # Dataset creation and step sizing
    # -------------------------------------------------------------------------
    train_size = splits["train"].shape[0]
    val_size = splits["val"].shape[0]
    steps_per_epoch = math.floor(train_size / BATCH_SIZE)
    validation_steps = math.floor(val_size / BATCH_SIZE)
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
        validation_steps = min(validation_steps, max(1, max_steps // 5))

    # Resolve LR schedule factors against concrete steps_per_epoch for sidecar
    run_config = dataclasses.replace(
        run_config, lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch)
    )

    # Single natural-balance shuffled stream (not ratio-batch)
    training_set = src.data_setup.data.dataset_create(
        SHUFFLE_BUFFER, BATCH_SIZE, train_preprocess, data=datasets["train"]
    )
    validation_set = src.data_setup.data.dataset_create(
        SHUFFLE_BUFFER, BATCH_SIZE, train_preprocess, data=datasets["val"]
    )

    # -------------------------------------------------------------------------
    # Model assembly
    # -------------------------------------------------------------------------
    backbone = load_dapt_backbone(run_config.backbone_weights_path)
    run_config.validate_against_backbone(backbone)

    # US head with BCE (endpoint mode, add_loss via head's loss_fn)
    us_head = ClassificationHead(
        hidden_dim=head_cfg.hidden_dim,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=make_us_metrics() + make_distribution_metrics(run_config.diagnostics),
        name=head_cfg.name,
        expose_loss_components=False,
    )

    # Escalation knobs: when unfreeze_top_n > 0, use per-layer LR scaling
    # (frozen path unchanged, byte-identical when unfreeze_top_n == 0)
    build_kwargs = {
        "backbone": backbone,
        "heads": {head_cfg.name: us_head},
        "seq_length": run_config.seq_length,
        "diagnostics": run_config.diagnostics,
    }

    if run_config.unfreeze_top_n > 0:
        # Unfreezing path: use top-N layer groups + custom LR multipliers
        build_kwargs.update({
            "freeze_encoder": False,
            "group_fn": top_n_group_fn(run_config.unfreeze_top_n, n_layers=12),
            "layer_multipliers": run_config.layer_multipliers or {
                "head": 1.0,
                "encoder_top": 0.1,
                "encoder_frozen": 0.0,
            },
        })
    else:
        # Frozen probe path (default): freeze encoder, no custom LR scaling
        build_kwargs.update({"freeze_encoder": True})

    # Pattern A: endpoint + inference models share head/backbone instances
    us_model = build_endpoint_model(**build_kwargs)
    us_inference = build_inference_model(
        backbone=backbone,
        heads={head_cfg.name: us_head},
        seq_length=run_config.seq_length,
    )

    # -------------------------------------------------------------------------
    # Optimizer and compile
    # -------------------------------------------------------------------------
    resolved = run_config.lr_schedule.resolved
    if resolved is None:
        raise RuntimeError(
            "lr_schedule.resolved is None after with_resolved() — "
            "programming error in the config resolution flow"
        )
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=run_config.lr_schedule.initial_lr,
        decay_steps=resolved.decay_steps,
        alpha=run_config.lr_schedule.decay_alpha,
        warmup_target=run_config.lr_schedule.warmup_target,
        warmup_steps=resolved.warmup_steps,
    )

    base_opt = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=run_config.optimizer.weight_decay
    )
    optimizer = (
        keras.optimizers.LossScaleOptimizer(base_opt)
        if config.IS_CLUSTER
        else base_opt
    )

    # Compile WITHOUT loss/metrics — head owns both via add_loss
    us_model.compile(optimizer=optimizer, jit_compile="auto")

    # -------------------------------------------------------------------------
    # Callbacks and fit
    # -------------------------------------------------------------------------
    config.US_FILTER_CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    config.US_FILTER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(
                config.US_FILTER_CLASSIFIER_DIR / f"{stamp}_checkpoint.weights.h5"
            ),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.CSVLogger(
            str(config.US_FILTER_LOGS_DIR / f"{stamp}_metrics.csv")
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, start_from_epoch=2, verbose=1
        ),
    ]

    us_model.fit(
        training_set,
        validation_data=validation_set,
        epochs=run_config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
    )

    # -------------------------------------------------------------------------
    # Save weights and config sidecar
    # -------------------------------------------------------------------------
    us_model.save_weights(str(config.US_FILTER_CLASSIFIER_WEIGHTS))
    run_config.to_json(config_path_for_weights(config.US_FILTER_CLASSIFIER_WEIGHTS))

    # -------------------------------------------------------------------------
    # In-distribution test eval: P/R/F1 + majority baseline
    # -------------------------------------------------------------------------
    test_set = (
        datasets["test"]
        .batch(BATCH_SIZE, drop_remainder=False)
        .map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    results = us_model.evaluate(test_set, return_dict=True)

    # Extract P/R from head-prefixed metric keys (head name is "us")
    p = results.get("us_precision")
    r = results.get("us_recall")
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0.0

    # Majority-class baseline
    maj = max(
        splits["test"]["us_label"].mean(), 1 - splits["test"]["us_label"].mean()
    )
    print(f"US test: P={p} R={r} F1={f1} | majority-class acc baseline={maj}")

    return us_model, us_inference, results


if __name__ == "__main__":
    main()
