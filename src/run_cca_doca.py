"""
Train the CCA classifier (features-mode) on cached DoCA-labeled embeddings.

The frozen-probe counterpart of `run_cca_classification.py`: instead of
`token_ids/padding_mask -> backbone -> CLS -> head`, it trains the head directly
on cached CLS vectors (see `src/embed_corpus.py`). Positives are DoCA-confirmed
matches; the unlabeled pool is US-restricted; the FLPU prior is the DEDPUL
re-estimate for this population (Phase 2). The full diagnostic instrumentation is
preserved via `build_feature_endpoint_model`. Token-mode training is unchanged.

Run from project root:
    uv run python -m src.run_cca_doca --prior 0.03 --suffix train250k --threshold 0.0
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import math
from pathlib import Path

import numpy as np
import keras

import src.config as config
import src.cca_config as cca_config
from src.cca_metrics import make_cca_metrics
from src.diagnostics.distribution_metrics import make_distribution_metrics
import polars as pl
from src.data_setup.data import create_cca_doca_data, dataset_from_embeddings
from src.embed_corpus import load_cache
from src.build_cca_doca_table import filter_positives_by_form, label_and_restrict
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import (
    build_feature_endpoint_model,
    build_feature_inference_model,
)
from src.loss_functions.loss import FLPULoss

keras.config.set_dtype_policy(config.DTYPE_POLICY)
keras.utils.set_random_seed(200)

BATCH_SIZE = 256
SHUFFLE_BUFFER = 100_000


def _config_with_prior(prior: float, epochs: int) -> cca_config.RunConfig:
    """DEFAULT_CCA_CONFIG with the FLPU prior + epochs replaced (nested frozen)."""
    base = cca_config.DEFAULT_CCA_CONFIG
    head = base.heads[0]
    new_head = dataclasses.replace(
        head, loss=dataclasses.replace(head.loss, prior=prior)
    )
    return dataclasses.replace(base, heads=(new_head,), epochs=epochs)


def _gather(cls: np.ndarray, group: pl.DataFrame, label: float):
    """Feature rows for a split group + a constant label array."""
    rows = group["emb_row"].to_numpy()
    feats = cls[rows]
    labels = np.full(feats.shape[0], label, dtype=np.float32)
    return feats, labels


def _load_holdout_ids(holdout_path: str | None) -> list[str] | None:
    """Read the gold-set coding template's `id` column (leakage-guard holdout).

    The path is the score-stratified coding template (`cca_coding_template.parquet`),
    whose full id set we hold out so any coded prefix stays leakage-clean.
    """
    if holdout_path is None:
        return None
    frame = pl.read_parquet(holdout_path)
    if "id" not in frame.columns:
        raise ValueError(f"holdout file {holdout_path} has no `id` column")
    return frame["id"].to_list()


def main(prior, suffix="train250k", threshold=0.0, epochs=7, max_steps=None,
         holdout_ids=None, weights_path=None, form_filter=None):
    weights_path = config.CCA_DOCA_WEIGHTS if weights_path is None else Path(weights_path)
    run_config = _config_with_prior(prior, epochs)
    head_cfg = run_config.heads[0]

    # Load cache + build the labeled, US-restricted table, then split.
    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    hidden_dim = cls.shape[1]
    if hidden_dim != head_cfg.hidden_dim:
        raise ValueError(
            f"cache feature dim {hidden_dim} != head hidden_dim {head_cfg.hidden_dim}"
        )
    pos_df = pl.read_parquet(config.CCA_DOCA_POSITIVES)
    holdout_ids = list(holdout_ids or [])
    if form_filter:
        positives, nonform = filter_positives_by_form(pos_df, form_filter)
        # Form-restricted CCA definition: only `form_filter` events are positive;
        # non-form DoCA are NOT dropped -- they fall to cca_label=0 and join the
        # unlabeled background as presumed-negatives (known non-form protests are
        # informative hard negatives for the form/not-form boundary).
        print(f"form_filter={form_filter}: {len(positives)} positives; "
              f"{len(nonform)} non-form DoCA -> unlabeled (presumed-negative)")  # LOG
    else:
        positives = pos_df["id"].to_list()
    table = label_and_restrict(meta, positives, threshold)
    if holdout_ids:
        print(f"holdout: dropping {len(holdout_ids)} ids from training pool")  # LOG
    splits = create_cca_doca_data(table, holdout_ids=holdout_ids)

    pos_tr = _gather(cls, splits["train"]["pos"], 1.0)
    unl_tr = _gather(cls, splits["train"]["unl"], 0.0)
    pos_va = _gather(cls, splits["val"]["pos"], 1.0)
    unl_va = _gather(cls, splits["val"]["unl"], 0.0)
    n_pos_tr, n_pos_va = pos_tr[0].shape[0], pos_va[0].shape[0]
    print(f"train pos={n_pos_tr} unl={unl_tr[0].shape[0]} | "
          f"val pos={n_pos_va} unl={unl_va[0].shape[0]} | prior={prior}")  # LOG

    # Ratio-Batch sampled datasets (every batch carries labeled positive signal).
    train_set = dataset_from_embeddings(
        SHUFFLE_BUFFER, BATCH_SIZE, data=[pos_tr, unl_tr],
        weights=[run_config.ratio_batch.train_pos, 1 - run_config.ratio_batch.train_pos],
    )
    val_set = dataset_from_embeddings(
        SHUFFLE_BUFFER, BATCH_SIZE, data=[pos_va, unl_va],
        weights=[run_config.ratio_batch.val_pos, 1 - run_config.ratio_batch.val_pos],
    )
    steps_per_epoch = math.floor(n_pos_tr / (BATCH_SIZE * run_config.ratio_batch.train_pos))
    validation_steps = max(1, math.floor(n_pos_va / (BATCH_SIZE * run_config.ratio_batch.val_pos)))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    run_config = dataclasses.replace(
        run_config, lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch)
    )

    # Head + features-mode models (Pattern A: shared instances).
    cca_head = ClassificationHead(
        hidden_dim=head_cfg.hidden_dim,
        loss_fn=FLPULoss(prior=head_cfg.loss.prior, kiryo_clawback=head_cfg.loss.kiryo_clawback),
        metrics=make_cca_metrics() + make_distribution_metrics(run_config.diagnostics),
        name=head_cfg.name,
        expose_loss_components=run_config.diagnostics.enable_loss_components,
    )
    model = build_feature_endpoint_model(
        {head_cfg.name: cca_head}, hidden_dim=hidden_dim,
        diagnostics=run_config.diagnostics,
    )
    inference = build_feature_inference_model({head_cfg.name: cca_head}, hidden_dim=hidden_dim)

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

    config.CCA_DOCA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        keras.callbacks.CSVLogger(str(config.CCA_DOCA_DIR / f"{stamp}_metrics.csv")),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, verbose=1, start_from_epoch=2
        ),
    ]
    model.fit(
        train_set, validation_data=val_set, epochs=run_config.epochs,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        callbacks=callbacks,
    )

    model.save_weights(str(weights_path))
    run_config.to_json(cca_config.config_path_for_weights(weights_path))
    print(f"Saved weights + sidecar at {weights_path}")  # LOG

    # Spot-check: score held-out test pos/unl and report distributions.
    pos_te = _gather(cls, splits["test"]["pos"], 1.0)
    unl_te = _gather(cls, splits["test"]["unl"], 0.0)
    pos_scores = inference.predict({"features": pos_te[0]}, verbose=0)["cca"].reshape(-1)
    unl_scores = inference.predict({"features": unl_te[0]}, verbose=0)["cca"].reshape(-1)
    print(f"test positives logit[mean/median]="
          f"{pos_scores.mean():.3f}/{np.median(pos_scores):.3f}  "
          f"unlabeled logit[mean/median]={unl_scores.mean():.3f}/{np.median(unl_scores):.3f}")  # LOG
    print(f"  positives scoring > 0: {(pos_scores > 0).mean():.3f}  "
          f"unlabeled scoring > 0: {(unl_scores > 0).mean():.3f}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train CCA head (features-mode) on cached embeddings.")
    ap.add_argument("--prior", type=float, required=True, help="FLPU class prior (DEDPUL re-estimate)")
    ap.add_argument("--suffix", default="train250k", help="embedding cache subdir")
    ap.add_argument("--threshold", type=float, default=0.0, help="US logit threshold")
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--holdout-ids", default=None,
                    help="path to the gold-set coding template parquet; its ids are "
                         "dropped from the training pool (leakage guard)")
    ap.add_argument("--out", default=None,
                    help="output weights .h5 path (default: CCA_DOCA_WEIGHTS); the "
                         "sidecar is derived from it (per-experiment weights)")
    ap.add_argument("--form-filter", default=None,
                    choices=["any_street", "any_boycott", "any_conventional",
                             "any_lawsuit", "no_form"],
                    help="restrict positives to a DoCA form flag; non-form DoCA ids "
                         "are dropped from the table (label-noise hypothesis test)")
    args = ap.parse_args()
    main(prior=args.prior, suffix=args.suffix, threshold=args.threshold,
         epochs=args.epochs, max_steps=args.max_steps,
         holdout_ids=_load_holdout_ids(args.holdout_ids), weights_path=args.out,
         form_filter=args.form_filter)
