# pattern: Imperative Shell
"""Train the immigrant-relevance head (features-mode) on cached CLS embeddings.

The relevance analog of `run_cca_doca`: same frozen-DAPT-backbone + single
`ClassificationHead` spine, same FLPU loss and Ratio-Batch sampling, same
leakage-holdout guard. Differences:

  * POSITIVES are the immigrant-relevance candidates (`relevance/candidates.parquet`
    -- descriptor-selected immigration-content articles UNION the 466 hand-verified
    ICA anchors), not the DoCA/CCA positives.
  * The cache is `relevance_train` (built by `build_relevance_table`, which already
    re-scored `us_logit` with the new US head), so US gating uses the cached
    calibrated probability and no `--us-weights` re-score happens here.
  * The head reuses the CCA head architecture/config (DEFAULT_CCA_CONFIG, head
    name "cca") for pass 1 -- it is a standalone binary FLPU head; the name is
    cosmetic and will be renamed when the multi-head config lands. Weights save to
    a distinct `relevance.weights.h5` artifact.

Run from project root:
    uv run python -m src.run_relevance --prior 0.05 \
        --holdout-ids validation/cca_coding_template.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import math

import keras
import numpy as np
import polars as pl

import src.config as config
import src.cca_config as cca_config
from src.build_cca_doca_table import label_and_restrict
from src.cca_metrics import make_cca_metrics
from src.data_setup.data import (
    assert_holdout_excluded,
    create_relevance_data,
    dataset_from_embeddings,
)
from src.diagnostics.distribution_metrics import make_distribution_metrics
from src.embed_corpus import load_cache
from src.loss_functions.loss import FLPULoss
from src.model_setup.assembly import (
    build_feature_endpoint_model,
    build_feature_inference_model,
)
from src.model_setup.heads import ClassificationHead
from src.preproc.us_location import apply_fused_us_gate, load_location_signals
from src.run_cca_doca import (
    BATCH_SIZE,
    SHUFFLE_BUFFER,
    _config_with_prior,
    _gather,
    _load_holdout_ids,
)

RELEVANCE_DIR = config.PROJECT_ROOT / "relevance"
DEFAULT_WEIGHTS = RELEVANCE_DIR / "relevance.weights.h5"




def main(prior, suffix="relevance_train", threshold=0.5, epochs=7, max_steps=None,
         holdout_ids=None, weights_path=None, nnpnu_eta=0.0, neg_weight=0.15):
    weights_path = weights_path or DEFAULT_WEIGHTS
    run_config = _config_with_prior(prior, epochs)
    # Set the nnPNU mixing weight on the (frozen) loss config so it is recorded in
    # the run sidecar. eta=0 leaves the head as pure nnPU (CCA-identical).
    # Also rename the head from "cca" to "rel" for multi-head assembly.
    head0 = run_config.heads[0]
    run_config = dataclasses.replace(
        run_config,
        heads=(dataclasses.replace(
            head0,
            name="rel",  # Rename from "cca" to "rel" for multi-head assembly
            loss=dataclasses.replace(head0.loss, nnpnu_eta=nnpnu_eta)
        ),),
    )
    head_cfg = run_config.heads[0]

    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    hidden_dim = cls.shape[1]
    if hidden_dim != head_cfg.hidden_dim:
        raise ValueError(
            f"cache feature dim {hidden_dim} != head hidden_dim {head_cfg.hidden_dim}"
        )

    positives = pl.read_parquet(RELEVANCE_DIR / "candidates.parquet")["id"].to_list()
    holdout_ids = list(holdout_ids or [])
    table = label_and_restrict(meta, positives, threshold)

    # Smarter US gate: replace the dateline-only `us` (us_logit>=threshold) with the
    # FUSED gate -- ML passes AND not clearly-foreign (a foreign location signal with
    # no US one). Catches US-datelined foreign news the dateline gate lets through,
    # while keeping diaspora (which carries a US location). See src/preproc/us_location.
    signals = load_location_signals(table["id"].to_list())
    table = table.join(signals, on="id", how="left").with_columns(
        pl.col("any_us").fill_null(False), pl.col("any_not_us").fill_null(False)
    )
    n_us_ml = int(table["us"].sum())
    table = apply_fused_us_gate(table)
    print(f"fused US gate: {int(table['us'].sum())}/{n_us_ml} of ML-gated kept "
          f"(clearly-foreign removed)")  # LOG
    # Relevance differs from CCA here: positives must ALSO be US. CCA keeps
    # positives regardless of `us` (DoCA events are US by construction), but the
    # relevance descriptor-positives include genuinely foreign articles (refugees-
    # abroad, diaspora-homeland) that the US gate correctly rejects. The relevance
    # head deploys behind the US gate, so non-US positives are out-of-distribution
    # and must not be trained as positives.
    n_pos_all = int((table["cca_label"] == 1).sum())
    table = table.with_columns(
        ((pl.col("cca_label") == 1) & pl.col("us")).cast(pl.Int8).alias("cca_label")
    )
    n_pos_us = int((table["cca_label"] == 1).sum())
    print(f"positives US-restricted: {n_pos_us}/{n_pos_all} kept")  # LOG

    # Reliable negatives (label -1): confidently-foreign, no-US-footprint articles
    # carved out of the unlabeled background. eta=0 ignores them (back-compat); the
    # loss only weights them when nnpnu_eta>0.
    reliable_neg_ids = pl.read_parquet(RELEVANCE_DIR / "reliable_negatives.parquet")["id"].to_list()
    table = table.with_columns(pl.col("id").is_in(reliable_neg_ids).alias("reliable_neg"))
    n_neg = int(table["reliable_neg"].sum())
    print(f"reliable negatives: {n_neg} (nnpnu_eta={nnpnu_eta})")  # LOG
    if holdout_ids:
        print(f"holdout: dropping {len(holdout_ids)} ids from training pool")  # LOG
    splits = create_relevance_data(table, holdout_ids=holdout_ids)
    assert_holdout_excluded(splits, holdout_ids)

    pos_tr = _gather(cls, splits["train"]["pos"], 1.0)
    neg_tr = _gather(cls, splits["train"]["neg"], -1.0)
    unl_tr = _gather(cls, splits["train"]["unl"], 0.0)
    pos_va = _gather(cls, splits["val"]["pos"], 1.0)
    neg_va = _gather(cls, splits["val"]["neg"], -1.0)
    unl_va = _gather(cls, splits["val"]["unl"], 0.0)
    n_pos_tr, n_pos_va = pos_tr[0].shape[0], pos_va[0].shape[0]
    print(f"train pos={n_pos_tr} neg={neg_tr[0].shape[0]} unl={unl_tr[0].shape[0]} | "
          f"val pos={n_pos_va} neg={neg_va[0].shape[0]} unl={unl_va[0].shape[0]} | prior={prior}")  # LOG

    # Three-stream Ratio Batch: pos, reliable-neg, unlabeled. Weights sum to 1;
    # the reliable-neg stream is small (neg_weight) but guaranteed-present so the
    # PN term gets signal every batch when eta>0.
    tp, vp = run_config.ratio_batch.train_pos, run_config.ratio_batch.val_pos
    train_set = dataset_from_embeddings(
        SHUFFLE_BUFFER, BATCH_SIZE, data=[pos_tr, neg_tr, unl_tr],
        weights=[tp, neg_weight, 1 - tp - neg_weight],
    )
    val_set = dataset_from_embeddings(
        SHUFFLE_BUFFER, BATCH_SIZE, data=[pos_va, neg_va, unl_va],
        weights=[vp, neg_weight, 1 - vp - neg_weight],
    )
    steps_per_epoch = math.floor(n_pos_tr / (BATCH_SIZE * run_config.ratio_batch.train_pos))
    validation_steps = max(1, math.floor(n_pos_va / (BATCH_SIZE * run_config.ratio_batch.val_pos)))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    run_config = dataclasses.replace(
        run_config, lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch)
    )

    head = ClassificationHead(
        hidden_dim=head_cfg.hidden_dim,
        loss_fn=FLPULoss(
            prior=head_cfg.loss.prior,
            kiryo_clawback=head_cfg.loss.kiryo_clawback,
            nnpnu_eta=head_cfg.loss.nnpnu_eta,
        ),
        metrics=make_cca_metrics() + make_distribution_metrics(run_config.diagnostics),
        name=head_cfg.name,
        expose_loss_components=run_config.diagnostics.enable_loss_components,
    )
    model = build_feature_endpoint_model(
        {head_cfg.name: head}, hidden_dim=hidden_dim, diagnostics=run_config.diagnostics,
    )
    inference = build_feature_inference_model({head_cfg.name: head}, hidden_dim=hidden_dim)

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

    RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks = [
        keras.callbacks.CSVLogger(str(RELEVANCE_DIR / f"{stamp}_metrics.csv")),
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

    pos_te = _gather(cls, splits["test"]["pos"], 1.0)
    unl_te = _gather(cls, splits["test"]["unl"], 0.0)
    pos_scores = inference.predict({"features": pos_te[0]}, verbose=0)[head_cfg.name].reshape(-1)
    unl_scores = inference.predict({"features": unl_te[0]}, verbose=0)[head_cfg.name].reshape(-1)
    print(f"test positives logit[mean/median]="
          f"{pos_scores.mean():.3f}/{np.median(pos_scores):.3f}  "
          f"unlabeled logit[mean/median]={unl_scores.mean():.3f}/{np.median(unl_scores):.3f}")  # LOG
    print(f"  positives scoring > 0: {(pos_scores > 0).mean():.3f}  "
          f"unlabeled scoring > 0: {(unl_scores > 0).mean():.3f}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train relevance head (features-mode).")
    ap.add_argument("--prior", type=float, required=True, help="FLPU class prior")
    ap.add_argument("--suffix", default="relevance_train", help="embedding cache subdir")
    ap.add_argument("--threshold", type=float, default=0.5, help="calibrated US prob gate")
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--holdout-ids", default=None,
                    help="parquet of ids (with `id` col) to drop from the unlabeled pool")
    ap.add_argument("--out", default=None, help="weights output path")
    ap.add_argument("--eta", type=float, default=0.0,
                    help="nnPNU PU<->PN mixing weight (0 = pure nnPU)")
    ap.add_argument("--neg-weight", type=float, default=0.15,
                    help="reliable-negative Ratio-Batch sampling weight")
    args = ap.parse_args()
    main(
        prior=args.prior, suffix=args.suffix, threshold=args.threshold,
        epochs=args.epochs, max_steps=args.max_steps,
        holdout_ids=_load_holdout_ids(args.holdout_ids),
        weights_path=args.out, nnpnu_eta=args.eta, neg_weight=args.neg_weight,
    )
