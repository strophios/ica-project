# pattern: Imperative Shell
"""Train the immigrant-relevance head in TEXT mode, with encoder unfreezing.

The "rel-first sequential" text-mode training path
(docs/notes/encoder-unfreeze-strategy.md): tune the shared frozen-DAPT encoder
using ONLY the relevance head's loss (top-N unfreeze + discriminative LR, the
ULMFiT recipe already proven in `run_us_classification.py:154-179`), leaving
CCA/US to be retrained features-mode on the re-embedded cache afterward.

Differs from BOTH templates it's built from:
  - vs. `run_us_classification.py` (the token-mode / unfreeze template): FLPU
    loss (not BCE), 3-way PNU Ratio-Batch streams (not a single PN stream).
  - vs. `run_relevance.py` (the features-mode rel trainer whose population/
    label derivation and PNU stream weighting this reproduces): token-mode
    (raw text -> tokenizer -> backbone), not cached CLS features, so the
    encoder can actually be fine-tuned (features-mode is definitionally frozen).

Data: `src.build_relevance_text_table` pre-computes the population (candidates,
fused US gate, US-restricted positives, reliable negatives, ICA-eval holdout
already excluded) and attaches `headline_with_lead`; this script only splits,
caches, and trains on it. The FLPU/nnPNU label convention (1.0 pos / -1.0
reliable-neg / 0.0 unlabeled) is NOT a column in that table -- it's a synthetic
`rel_label` column attached per split-group here, at cache-build time.

Run from project root:
    uv run python -m src.run_relevance_text --unfreeze-top-n 1 --epochs 7
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import math

import keras
import numpy as np
import polars as pl
import tensorflow as tf

import src.config as config
import src.cca_config as cca_config
from src.cca_config import config_path_for_weights
from src.cca_metrics import make_cca_metrics
from src.data_setup.data import (
    assert_holdout_excluded,
    create_relevance_data,
    dataset_create,
)
from src.diagnostics.distribution_metrics import make_distribution_metrics
from src.loss_functions.loss import FLPULoss
from src.model_setup.assembly import build_endpoint_model, build_inference_model
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.preproc.preprocessor import ClassifierPreprocessor
from src.validation.escalation import escalation_build_kwargs

RELEVANCE_DIR = config.PROJECT_ROOT / "relevance"

# FLPU/nnPNU label convention (matches _gather in run_cca_doca.py / run_relevance.py).
_LABEL_BY_GROUP = {"pos": 1.0, "neg": -1.0, "unl": 0.0}
_SPLITS = ("train", "val", "test")
_GROUPS = ("pos", "neg", "unl")
# Reliable-negative Ratio-Batch stream weight -- matches run_relevance.py's default.
_NEG_WEIGHT = 0.15


# ---------------------------------------------------------------------------
# Functional core
# ---------------------------------------------------------------------------
def with_rel_label(df: pl.DataFrame, group_name: str) -> pl.DataFrame:
    """Pure: attach the constant `rel_label` column for a PNU split-group.

    `group_name` must be one of "pos"/"neg"/"unl"; raises KeyError otherwise
    (via `_LABEL_BY_GROUP`'s dict lookup) rather than silently mislabeling.
    """
    return df.with_columns(
        pl.lit(_LABEL_BY_GROUP[group_name]).alias("rel_label").cast(pl.Float32)
    )


def resolve_tensorboard_dir(
    tensorboard_dir: str | None, tensorboard: bool, timestamp: str
) -> str | None:
    """Pure: resolve the TensorBoard log dir from the two CLI knobs.

    Precedence: an explicit `tensorboard_dir` always wins. Otherwise,
    `tensorboard=True` defaults to a timestamped path under
    `RELEVANCE_DIR/tb_logs/`. `tensorboard=False` with no explicit dir means
    off (returns None). `timestamp` is caller-supplied (the imperative shell
    computes it via `datetime.datetime.now(datetime.timezone.utc)`) so this
    function has no non-deterministic inputs of its own.
    """
    if tensorboard_dir is not None:
        return tensorboard_dir
    if tensorboard:
        return str(RELEVANCE_DIR / "tb_logs" / timestamp)
    return None


def build_fit_callbacks(metrics_csv_path, tensorboard_log_dir=None):
    """Assemble the `fit()` callback list.

    Deterministic given its inputs: CSVLogger + EarlyStopping are always
    present; a `keras.callbacks.TensorBoard` is appended iff
    `tensorboard_log_dir` is not None. Constructing these callback objects
    has no side effects (no file/dir is touched until `fit()` calls
    `on_train_begin`), which is what makes this testable without invoking
    `fit`.
    """
    callbacks_list = [
        keras.callbacks.CSVLogger(str(metrics_csv_path)),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, start_from_epoch=2, verbose=1
        ),
    ]
    if tensorboard_log_dir is not None:
        callbacks_list.append(
            keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), update_freq="epoch")
        )
    return callbacks_list


def _default_rel_text_config(epochs: int = 7) -> cca_config.RunConfig:
    """DEFAULT_CCA_CONFIG with the head renamed to "rel" (source column
    "rel_label", the synthetic PNU-convention column -- NOT "cca_label", the
    raw indicator column in the text table) and the canonical rel prior/eta
    (prior=0.05, nnpnu_eta=0.0 -- matches the features-mode `run_relevance.py`
    harmonized-retrain invocation). Escalation knobs default to frozen-probe;
    callers derive an unfreezing variant via `dataclasses.replace`.
    """
    base = cca_config.DEFAULT_CCA_CONFIG
    head0 = base.heads[0]
    new_head = dataclasses.replace(
        head0,
        name="rel",
        source_column="rel_label",
        loss=dataclasses.replace(head0.loss, prior=0.05, nnpnu_eta=0.0),
    )
    return dataclasses.replace(base, heads=(new_head,), epochs=epochs)


DEFAULT_REL_TEXT_CONFIG = _default_rel_text_config()


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def main(run_config=None, max_steps=None, batch_size=256, tensorboard_dir=None):
    """
    Train the relevance head in text mode, with optional encoder unfreezing.

    Args:
        run_config: Optional cca_config.RunConfig instance. If None, uses
            DEFAULT_REL_TEXT_CONFIG (frozen-probe, unfreeze_top_n=0). Set
            `unfreeze_top_n > 0` (+ `freeze_encoder=False`) via
            `dataclasses.replace` to escalate.
        max_steps: Optional cap on steps_per_epoch (and a proportional cap on
            validation_steps), for short/smoke runs.
        batch_size: Training/eval batch size (default 256, matching every
            other training script in this project). Does NOT affect the
            tf.data cache under RELEVANCE_SET_DIR (built at row granularity,
            batched downstream) -- safe to shrink for memory-constrained runs
            (e.g. local MPS) without invalidating an existing cache.
        tensorboard_dir: Optional path (str). If not None, appends a
            `keras.callbacks.TensorBoard(log_dir=tensorboard_dir,
            update_freq="epoch")` to the fit callbacks -- everything logged
            (per-step diagnostic trackers, the loss tracker) already rides
            Keras's metrics path (see module docstring / CLAUDE.md), so this
            is the only wiring TensorBoard needs. Default None = off. The
            `__main__` CLI resolves this from `--tensorboard-dir` /
            `--tensorboard` via `resolve_tensorboard_dir`; callers invoking
            `main()` directly pass the final path themselves.
    """
    keras.config.set_dtype_policy(config.DTYPE_POLICY)
    keras.utils.set_random_seed(200)

    if run_config is None:
        run_config = DEFAULT_REL_TEXT_CONFIG

    head_cfg = run_config.heads[0]
    BATCH_SIZE = batch_size
    SHUFFLE_BUFFER = 100_000

    # -------------------------------------------------------------------------
    # Load and split data (text table already has holdout excluded at build
    # time; re-apply + re-verify here -- belt-and-suspenders, matching
    # src/run_us_pnu.py's pattern for the same table-vs-script split of duty).
    # -------------------------------------------------------------------------
    table = pl.read_parquet(config.RELEVANCE_TEXT_TABLE)
    holdout_ids = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].to_list()
    splits = create_relevance_data(table, holdout_ids=holdout_ids)
    assert_holdout_excluded(splits, holdout_ids)

    # -------------------------------------------------------------------------
    # tf.data cache: one file per (split, group) -- 9 total. See
    # run_us_classification.py's caching comment for the staleness contract:
    # if the text table changes upstream, DELETE RELEVANCE_SET_DIR so it
    # rebuilds, or cached data and recomputed steps silently diverge.
    # -------------------------------------------------------------------------
    if not config.RELEVANCE_SET_DIR.is_dir():
        config.RELEVANCE_SET_DIR.mkdir(parents=True)
        for split_name in _SPLITS:
            for group_name in _GROUPS:
                gdf = with_rel_label(splits[split_name][group_name], group_name)
                ds = tf.data.Dataset.from_tensor_slices({
                    "headline_with_lead": gdf["headline_with_lead"].to_list(),
                    "rel_label": gdf["rel_label"].to_numpy(),
                })
                ds.save(str(config.RELEVANCE_SET_DIR / f"{split_name}_{group_name}.tf"))

    datasets = {
        (split_name, group_name): tf.data.Dataset.load(
            str(config.RELEVANCE_SET_DIR / f"{split_name}_{group_name}.tf")
        )
        for split_name in _SPLITS
        for group_name in _GROUPS
    }

    # Freshness check: cached cardinality must match current split-group sizes.
    for split_name in _SPLITS:
        for group_name in _GROUPS:
            cached_n = int(datasets[(split_name, group_name)].cardinality().numpy())
            expected_n = splits[split_name][group_name].height
            if cached_n != expected_n:
                raise ValueError(
                    f"Stale relevance-text set cache for {split_name}/{group_name}: "
                    f"cache has {cached_n} rows but current split has {expected_n}. "
                    f"Delete {config.RELEVANCE_SET_DIR} and re-run."
                )

    # -------------------------------------------------------------------------
    # Preprocessors: one for training (emits rel_targets), one for predict-only
    # spot-checks (empty label_keys -- the ClassifierPreprocessor "predict-only
    # configuration" documented in its own docstring).
    # -------------------------------------------------------------------------
    train_preprocess = ClassifierPreprocessor(
        SEQ_LENGTH=run_config.seq_length,
        text_key=run_config.text_key,
        label_keys=run_config.label_keys,  # {"rel_targets": "rel_label"}
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
    # 3-way weighted Ratio-Batch streams (pos / reliable-neg / unlabeled) --
    # same weighting scheme as run_relevance.py's features-mode
    # dataset_from_embeddings call, here composed over TEXT streams via
    # dataset_create's list-of-datasets + weights path (never previously
    # exercised with 3 TEXT streams -- see tests/test_run_relevance_text.py).
    # -------------------------------------------------------------------------
    tp, vp = run_config.ratio_batch.train_pos, run_config.ratio_batch.val_pos
    training_set = dataset_create(
        SHUFFLE_BUFFER, BATCH_SIZE, train_preprocess,
        data=[datasets[("train", g)] for g in _GROUPS],
        weights=[tp, _NEG_WEIGHT, 1 - tp - _NEG_WEIGHT],
    )
    validation_set = dataset_create(
        SHUFFLE_BUFFER, BATCH_SIZE, train_preprocess,
        data=[datasets[("val", g)] for g in _GROUPS],
        weights=[vp, _NEG_WEIGHT, 1 - vp - _NEG_WEIGHT],
    )

    n_pos_tr = splits["train"]["pos"].height
    n_pos_va = splits["val"]["pos"].height
    steps_per_epoch = math.floor(n_pos_tr / (BATCH_SIZE * tp))
    validation_steps = max(1, math.floor(n_pos_va / (BATCH_SIZE * vp)))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
        validation_steps = min(validation_steps, max(1, max_steps // 5))

    run_config = dataclasses.replace(
        run_config, lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch)
    )

    # -------------------------------------------------------------------------
    # Model assembly: rel head (FLPU) + escalation branch.
    # -------------------------------------------------------------------------
    backbone = load_dapt_backbone(run_config.backbone_weights_path)
    run_config.validate_against_backbone(backbone)

    rel_head = ClassificationHead(
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

    # Escalation knobs: byte-identical frozen-probe path when unfreeze_top_n==0
    # (escalation_build_kwargs returns {"freeze_encoder": True} only).
    build_kwargs = {
        "backbone": backbone,
        "heads": {head_cfg.name: rel_head},
        "seq_length": run_config.seq_length,
        "diagnostics": run_config.diagnostics,
    }
    build_kwargs.update(
        escalation_build_kwargs(run_config.unfreeze_top_n, run_config.layer_multipliers)
    )

    # Pattern A: endpoint + inference models share head/backbone instances.
    rel_model = build_endpoint_model(**build_kwargs)
    rel_inference = build_inference_model(
        backbone=backbone, heads={head_cfg.name: rel_head}, seq_length=run_config.seq_length
    )

    # -------------------------------------------------------------------------
    # Optimizer and compile
    # -------------------------------------------------------------------------
    resolved = run_config.lr_schedule.resolved
    if resolved is None:
        raise RuntimeError(
            "lr_schedule.resolved is None after with_resolved() -- "
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
        keras.optimizers.LossScaleOptimizer(base_opt) if config.IS_CLUSTER else base_opt
    )

    # Compile WITHOUT loss/metrics -- head owns both via add_loss.
    rel_model.compile(optimizer=optimizer, jit_compile="auto")

    # -------------------------------------------------------------------------
    # Callbacks and fit
    # -------------------------------------------------------------------------
    RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tensorboard_dir is not None:
        print(f"TensorBoard log dir: {tensorboard_dir}")  # LOG (operators need this for rsync/tunnel)
    callbacks_list = build_fit_callbacks(
        RELEVANCE_DIR / f"{stamp}_text_metrics.csv", tensorboard_log_dir=tensorboard_dir
    )

    history = rel_model.fit(
        training_set,
        validation_data=validation_set,
        epochs=run_config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
    )

    # -------------------------------------------------------------------------
    # Save weights + config sidecar (records the escalation knobs used)
    # -------------------------------------------------------------------------
    weights_path = config.RELEVANCE_TEXT_WEIGHTS
    rel_model.save_weights(str(weights_path))
    run_config.to_json(config_path_for_weights(weights_path))

    # -------------------------------------------------------------------------
    # Spot-check: score held-out test pos/reliable-neg/unlabeled, report
    # distributions (mirrors run_relevance.py's final block).
    # -------------------------------------------------------------------------
    def _score_group(group_name):
        ds = (
            datasets[("test", group_name)]
            .batch(BATCH_SIZE, drop_remainder=False)
            .map(predict_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        return rel_inference.predict(ds, verbose=0)[head_cfg.name].reshape(-1)

    pos_scores = _score_group("pos")
    neg_scores = _score_group("neg")
    unl_scores = _score_group("unl")
    print(f"test positives logit[mean/median]="
          f"{pos_scores.mean():.3f}/{np.median(pos_scores):.3f}  "
          f"reliable-neg logit[mean/median]={neg_scores.mean():.3f}/{np.median(neg_scores):.3f}  "
          f"unlabeled logit[mean/median]={unl_scores.mean():.3f}/{np.median(unl_scores):.3f}")  # LOG

    return rel_model, rel_inference, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train the relevance head in text mode (encoder-unfreeze rel-first path)."
    )
    ap.add_argument("--unfreeze-top-n", type=int, default=0,
                     help="number of top RoBERTa layers to unfreeze (0 = frozen probe)")
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tensorboard-dir", type=str, default=None,
                     help="explicit TensorBoard log dir (wins over --tensorboard)")
    ap.add_argument("--tensorboard", action="store_true",
                     help="enable TensorBoard logging to a timestamped dir under "
                          "RELEVANCE_DIR/tb_logs/ (ignored if --tensorboard-dir is set)")
    args = ap.parse_args()

    cfg = dataclasses.replace(
        DEFAULT_REL_TEXT_CONFIG,
        epochs=args.epochs,
        unfreeze_top_n=args.unfreeze_top_n,
        freeze_encoder=(args.unfreeze_top_n == 0),
    )
    tb_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tensorboard_dir = resolve_tensorboard_dir(args.tensorboard_dir, args.tensorboard, tb_timestamp)
    main(
        run_config=cfg,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        tensorboard_dir=tensorboard_dir,
    )
