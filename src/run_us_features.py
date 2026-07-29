# pattern: Imperative Shell
"""Train the US/not-US filter features-mode on cached DAPT-CLS embeddings.

Frozen-probe counterpart of `run_us_classification.py`: trains the BCE US head
directly on cached CLS vectors (`src/embed_corpus.py`), so a full real run takes
minutes instead of a multi-hour token-mode pass over the LDC labeled set. The task
is plain supervised PN (natural-balance), no PU/FLPU. The features-mode wiring
(cache load, `build_feature_endpoint_model`, `_gather`) mirrors `run_cca_doca.py`;
the US specifics (BCE loss, `make_us_metrics`, test-eval + majority baseline) mirror
`run_us_classification.py`. The split reuses `create_us_filter_data` on the cache
meta, so split membership is identical to the token-mode path under the same seed.

Saves to a NEW path (`us_filter/us_classifier_full.weights.h5`) so the prior
smoke-test weights survive for comparison. These are head-only weights (features-mode
has no backbone); apply them to cached CLS via `build_feature_inference_model`.

Run from project root (cache produced by embed_corpus --source-pattern ... --label-column us_label):
    uv run python -m src.run_us_features --suffix us_train_ldc
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import math
from pathlib import Path

import numpy as np
import keras
import polars as pl

import src.config as config
import src.us_config as us_config
from src.artifact_guard import check_no_production_overwrite
from src.us_config import config_path_for_weights
from src.data_setup.data import create_us_filter_data, dataset_from_embeddings
from src.embed_corpus import load_cache
from src.us_metrics import make_us_metrics
from src.diagnostics.distribution_metrics import make_distribution_metrics
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import (
    build_feature_endpoint_model,
    build_feature_inference_model,
)

keras.config.set_dtype_policy(config.DTYPE_POLICY)
keras.utils.set_random_seed(200)

BATCH_SIZE = 256
SHUFFLE_BUFFER = 100_000
# `main()`'s own --suffix default; also the "production cache" identity that
# check_no_production_overwrite compares an explicit --suffix against.
DEFAULT_SUFFIX = "us_train_ldc"


def _gather(cls: np.ndarray, group: pl.DataFrame):
    """Feature rows (by `emb_row`) + float us_label array for a split group."""
    rows = group["emb_row"].to_numpy()
    feats = cls[rows]
    labels = group["us_label"].cast(pl.Float32).to_numpy()
    return feats, labels


def main(suffix=DEFAULT_SUFFIX, epochs=10, max_steps=None, weights_path=None,
         backbone_weights=None):
    weights_path = (
        config.US_FILTER_FULL_WEIGHTS
        if weights_path is None
        else Path(weights_path)
    )
    check_no_production_overwrite(
        cache_suffix=suffix,
        production_cache_suffix=DEFAULT_SUFFIX,
        weights_path=weights_path,
        production_weights_path=config.US_FILTER_FULL_WEIGHTS,
        artifact_label="US filter",
    )
    # Reuse the canonical US config (LR schedule, optimizer, diagnostics) but allow
    # more epochs than the smoke run — the features-mode pass is cheap, so a real
    # train to convergence (early-stopped) is affordable.
    run_config = dataclasses.replace(us_config.DEFAULT_US_CONFIG, epochs=epochs)
    if backbone_weights is not None:
        # Bookkeeping only -- features-mode training never loads a backbone, but
        # the sidecar's backbone_weights_path is read by token-mode consumers
        # (src.validation.slice_eval.apply_us_model) to rebuild the encoder at
        # eval time. Training on a `*_tuned` cache without recording the tuned
        # backbone here would leave those consumers silently re-embedding text
        # through the WRONG (production) backbone -- a features/backbone
        # mismatch that produces incoherent scores without erroring.
        run_config = dataclasses.replace(
            run_config, backbone_weights_path=str(backbone_weights)
        )
    head_cfg = run_config.head

    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    hidden_dim = cls.shape[1]
    if hidden_dim != head_cfg.hidden_dim:
        raise ValueError(
            f"cache feature dim {hidden_dim} != head hidden_dim {head_cfg.hidden_dim}"
        )
    if "us_label" not in meta.columns:
        raise ValueError(
            f"cache {suffix!r} has no us_label column (cols={meta.columns}); "
            f"re-embed with --label-column us_label"
        )

    # Identical split to the token-mode path: pos/neg split 90/5/5 separately, seed 200,
    # within-split shuffle. emb_row rides along for the gather.
    splits = create_us_filter_data(meta)
    tr = _gather(cls, splits["train"])
    va = _gather(cls, splits["val"])
    print(
        f"train={tr[0].shape[0]} (us_frac={tr[1].mean():.3f}) | "
        f"val={va[0].shape[0]} | test={splits['test'].height}"
    )  # LOG

    train_set = dataset_from_embeddings(
        SHUFFLE_BUFFER, BATCH_SIZE, data=tr, head_name=head_cfg.name
    )
    val_set = dataset_from_embeddings(
        SHUFFLE_BUFFER, BATCH_SIZE, data=va, head_name=head_cfg.name
    )

    steps_per_epoch = math.floor(tr[0].shape[0] / BATCH_SIZE)
    validation_steps = max(1, math.floor(va[0].shape[0] / BATCH_SIZE))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    run_config = dataclasses.replace(
        run_config, lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch)
    )

    # US head: BCE (standard PN), endpoint mode (head owns the loss via add_loss).
    us_head = ClassificationHead(
        hidden_dim=head_cfg.hidden_dim,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=make_us_metrics() + make_distribution_metrics(run_config.diagnostics),
        name=head_cfg.name,
    )
    model = build_feature_endpoint_model(
        {head_cfg.name: us_head}, hidden_dim=hidden_dim,
        diagnostics=run_config.diagnostics,
    )
    inference = build_feature_inference_model(
        {head_cfg.name: us_head}, hidden_dim=hidden_dim
    )

    resolved = run_config.lr_schedule.resolved
    assert resolved is not None, "with_resolved should have populated lr_schedule.resolved"
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=run_config.lr_schedule.initial_lr,
        decay_steps=resolved.decay_steps,
        alpha=run_config.lr_schedule.decay_alpha,
        warmup_target=run_config.lr_schedule.warmup_target,
        warmup_steps=resolved.warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=run_config.optimizer.weight_decay
    )
    model.compile(optimizer=optimizer, jit_compile="auto")

    config.US_FILTER_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        keras.callbacks.CSVLogger(str(config.US_FILTER_DIR / f"{stamp}_features_metrics.csv")),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, start_from_epoch=2, verbose=1
        ),
    ]
    model.fit(
        train_set, validation_data=val_set, epochs=run_config.epochs,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        callbacks=callbacks,
    )

    model.save_weights(str(weights_path))
    run_config.to_json(config_path_for_weights(weights_path))
    print(f"Saved weights + sidecar at {weights_path}")  # LOG

    # Test eval on the held-out split: precision/recall/F1 at logit 0, + majority baseline.
    te = _gather(cls, splits["test"])
    logits = inference.predict({"features": te[0]}, verbose=0)[head_cfg.name].reshape(-1)
    preds = (logits > 0).astype(np.float32)
    y = te[1]
    tp = float(((preds == 1) & (y == 1)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    maj = float(max(y.mean(), 1 - y.mean()))
    print(
        f"US test (features-mode): P={p:.4f} R={r:.4f} F1={f1:.4f} | "
        f"majority-class acc baseline={maj:.4f} | n={len(y)}"
    )  # LOG
    return model, inference


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train US filter head (features-mode) on cached embeddings."
    )
    ap.add_argument("--suffix", default=DEFAULT_SUFFIX, help="embedding cache subdir")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="output weights .h5 (default: us_filter/us_classifier_full.weights.h5). "
                         "Must be a non-production path when --suffix is non-default.")
    ap.add_argument("--backbone-weights", default=None,
                    help="backbone weights path to record in the sidecar (bookkeeping "
                         "only -- training itself is features-mode and never loads a "
                         "backbone). Pass the tuned backbone's path (from "
                         "extract_tuned_backbone.py) when --suffix selects a '_tuned' "
                         "cache, so token-mode consumers (apply_us_model) re-embed "
                         "through the matching backbone at eval time.")
    args = ap.parse_args()
    main(suffix=args.suffix, epochs=args.epochs, max_steps=args.max_steps,
         weights_path=args.out, backbone_weights=args.backbone_weights)
