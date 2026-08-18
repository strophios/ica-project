# pattern: Imperative Shell
"""Train BOTH the `cca` and `rel` heads in TEXT mode over a SHARED,
top-N-unfrozen encoder -- the joint CCA+rel fine-tune
(docs/design-plans/2026-08-18-stage4-joint-finetune.md, Components item 3).

The stage-3 rel-first sequential unfreeze (`run_relevance_text.py`) proved a
real rel-head lift that negatively transfers to CCA/US on the SAME backbone
when the two heads are trained sequentially with different loss signals. This
script tests the alternative: fine-tune ONE shared encoder against a SCALARIZED
combination of both heads' FLPU losses (`loss_weight`s summing the design's
lambda mix -- `docs/notes/pinned-questions.md` #1 "Multi-head lookahead"),
so the encoder never sees a rel-only or cca-only gradient signal in isolation.

Modeled on `run_relevance_text.py` (importable `main`, escalation/hard-freeze/
seed knobs, same artifact-guard posture, same tf.data caching pattern), with
these differences:

  - TWO heads (`cca`, `rel`) sharing one `RunConfig`, mixed via
    `HeadConfig.loss_weight` -- `rel` gets `--lam`, `cca` gets `1 - lam`
    (`ClassificationHead.loss_weight`, `docs/design-plans/
    2026-08-18-stage4-joint-finetune.md` Components item 1).
  - Population/split: `src.build_joint_text_table` (the UNION of the CCA and
    rel populations) split via `src.data_setup.data.create_joint_text_data`
    -- a WHOLE-TABLE-FIRST split (unlike this script's per-PU-group-split
    ancestors; see that function's docstring for why the ordering matters
    when a row can be positive for one head and background for the other).
  - THREE Ratio-Batch streams (not run_relevance_text's three PNU streams):
    cca-positives / rel-positives / unlabeled, weighted (0.1, 0.1, 0.8) train
    and (0.25, 0.25, 0.5) val. The two positive streams CAN overlap (a row
    positive for both heads is drawn from both) -- deliberate, same family as
    Ratio-Batch itself. Every batch carries BOTH target keys for every row
    (PU semantics: not-head-X-positive implies head-X-unlabeled for that row).
  - Reliable-negative rows are NOT a fourth stream (unlike run_relevance_text's
    weighted `neg` stream): they fall into `unl` (not cca_label==1, not
    rel_label==1) and rely on `rel_target`'s -1 label (FLPU masks -1 rows out
    of the rel loss at eta=0) to stay inert for rel while still contributing
    ordinary cca-unlabeled background.
  - No `--prior`/`--eta`/`--peak-lr` CLI overrides: both heads' priors (0.02)
    and eta (0.0) are fixed by the design doc, not sweep knobs here.
  - `--weights-out` is REQUIRED (no positional default): there is no single
    "production" artifact for this two-head family yet. `_guard_weights_out`
    still refuses to write over any of the FOUR existing single-head
    production artifacts (repurposing `check_no_production_overwrite`'s
    (cache_suffix, weights_path) shape the same way `fit_fusion.py`'s
    `resolve_fusion_inputs` does for its multi-input guard -- see that
    function's docstring).

Data: `src.build_joint_text_table` pre-computes the union population (cca_label,
rel_label, us, reliable_neg, headline_with_lead, holdout already excluded).
This script derives the synthetic `rel_target` FLPU/nnPNU-convention column
(`derive_rel_target`), splits + streams it (`create_joint_text_data`), caches,
and trains.

Run from project root:
    uv run python -m src.run_joint_text --lam 0.5 --unfreeze-top-n 1 \
        --weights-out ../relevance/joint_sweep/joint_N1_lam050_s200.weights.h5
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import math
from pathlib import Path

import keras
import numpy as np
import polars as pl
import tensorflow as tf

import src.config as config
import src.cca_config as cca_config
from src.artifact_guard import check_no_production_overwrite
from src.cca_config import config_path_for_weights
from src.cca_metrics import make_cca_metrics
from src.data_setup.data import (
    assert_holdout_excluded,
    create_joint_text_data,
    dataset_create,
)
from src.diagnostics.distribution_metrics import make_distribution_metrics
from src.loss_functions.loss import FLPULoss
from src.model_setup.assembly import build_endpoint_model, build_inference_model
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.preproc.preprocessor import ClassifierPreprocessor
from src.validation.escalation import escalation_build_kwargs, graded_multipliers

RELEVANCE_DIR = config.PROJECT_ROOT / "relevance"
# Text-cache dir for THIS trainer -- distinct from run_relevance_text.py's
# config.RELEVANCE_SET_DIR (different columns: cca_label + rel_target, not
# rel_label). Not promoted to src/config.py -- this trainer's own concern.
JOINT_SET_DIR = RELEVANCE_DIR / "joint_set"

_SPLITS = ("train", "val", "test")
_GROUPS = ("cca_pos", "rel_pos", "unl")
# Ratio-Batch stream weights, ordered to match _GROUPS. Design doc: train
# (0.1, 0.1, 0.8), val (0.25, 0.25, 0.5) -- mirrors run_relevance_text.py's
# val-boosted positive share, doubled here (one boost per head).
_TRAIN_WEIGHTS = (0.1, 0.1, 0.8)
_VAL_WEIGHTS = (0.25, 0.25, 0.5)

# The FOUR existing single-head production artifacts _guard_weights_out
# refuses to write over. (label, production_path) pairs.
_PRODUCTION_WEIGHTS_PATHS = (
    ("relevance (features-mode)", config.RELEVANCE_DOCA_WEIGHTS),
    ("relevance (text-mode, rel-first)", config.RELEVANCE_TEXT_WEIGHTS),
    ("CCA/DoCA (features-mode)", config.CCA_DOCA_WEIGHTS),
    ("US filter (features-mode)", config.US_FILTER_FULL_WEIGHTS),
)
# Sentinel "cache_suffix" value that can never equal a real production
# cache-suffix default -- see _guard_weights_out's docstring for why this
# repurposes check_no_production_overwrite's (cache_suffix, weights_path)
# shape to guard a single path against multiple targets.
_NO_CACHE_SUFFIX_SENTINEL = "__joint_text_has_no_cache_suffix_knob__"


# ---------------------------------------------------------------------------
# Functional core
# ---------------------------------------------------------------------------
def derive_rel_target(table: pl.DataFrame) -> pl.DataFrame:
    """Pure: attach the synthetic `rel_target` FLPU/nnPNU-convention column.

    1.0 where `rel_label == 1` (a rel positive, already US-restricted at
    table-build time); -1.0 where `reliable_neg` (confidently-foreign,
    no-US-footprint background -- FLPU masks -1 rows out of the rel loss
    entirely at eta=0, per `docs/design-plans/2026-08-18-stage4-joint-
    finetune.md` Components item 2: "reliable negatives are excluded from
    rel's unlabeled background"); 0.0 otherwise (ordinary unlabeled
    background). `rel_label == 1` is checked FIRST: by construction
    (`scripts/build_reliable_negatives.py` asserts zero overlap between
    reliable negatives and rel candidates/anchors) the two conditions never
    co-occur in practice, but the precedence is still the correct read if
    that upstream invariant were ever violated.
    """
    return table.with_columns(
        pl.when(pl.col("rel_label") == 1)
        .then(pl.lit(1.0))
        .when(pl.col("reliable_neg"))
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
        .cast(pl.Float32)
        .alias("rel_target")
    )


def validate_lam(lam) -> float:
    """Pure: validate `--lam` (the rel-side loss_weight) is a finite number
    strictly in (0, 1). Returns the validated float unchanged.

    Strict-open interval: lam=0 or lam=1 would zero out one head's loss
    entirely, which is a single-head run wearing a joint-trainer costume --
    reject it here rather than let `HeadConfig.loss_weight`'s own
    finite-and->0 check catch only the lam=0 case (1-0=1 is a valid
    loss_weight; the corresponding cca_head.loss_weight=1.0 would slip
    through that check silently).
    """
    if (
        not isinstance(lam, (int, float))
        or isinstance(lam, bool)
        or not math.isfinite(float(lam))
    ):
        raise ValueError(
            f"--lam must be a finite number; got {lam!r} (type {type(lam).__name__})."
        )
    lam = float(lam)
    if not (0.0 < lam < 1.0):
        raise ValueError(f"--lam must be in (0, 1); got {lam}.")
    return lam


def resolve_tensorboard_dir(
    tensorboard_dir: str | None, tensorboard: bool, timestamp: str
) -> str | None:
    """Pure: resolve the TensorBoard log dir from the two CLI knobs.

    Byte-identical port of `run_relevance_text.resolve_tensorboard_dir`
    (see that function's docstring) -- this trainer is self-contained
    rather than importing from a sibling script, matching this project's
    per-script convention.
    """
    if tensorboard_dir is not None:
        return tensorboard_dir
    if tensorboard:
        return str(RELEVANCE_DIR / "tb_logs" / timestamp)
    return None


def build_fit_callbacks(metrics_csv_path, tensorboard_log_dir=None):
    """Assemble the `fit()` callback list. Port of
    `run_relevance_text.build_fit_callbacks` -- see that function's
    docstring for the determinism/testability rationale."""
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


def _default_joint_text_config(lam, epochs: int = 7) -> cca_config.RunConfig:
    """Two-head `RunConfig` for the joint trainer: `cca` (source_column
    "cca_label", used as-is) and `rel` (source_column "rel_target", the
    synthetic -1/0/1 column `derive_rel_target` produces -- NOT the raw
    0/1 `rel_label` indicator column, which the split/grouping logic
    consumes instead; see `create_joint_text_data`'s docstring for that
    split of duty).

    Both heads share prior=0.02 (the settled operating value, Phase 2) and
    eta=0.0 (pure nnPU; the reliable-negative PN term stays off for both,
    matching the harmonized-retrain convention), and hidden_dim=768
    (DEFAULT_CCA_CONFIG's roberta_base_en dim). `loss_weight` is the lambda
    mix: rel=lam, cca=1-lam (`validate_lam` enforces lam strictly in (0,1),
    so both weights are always positive per `HeadConfig`'s own validation).

    `ratio_batch` is recorded as `RatioBatchConfig(train_pos=0.1, val_pos=0.25)`
    for sidecar provenance -- documenting the PER-STREAM weight shared by
    BOTH positive streams (there are two, at that weight each; the schema
    has room for only one "positive" fraction). The actual stream weights
    used at training time are the module constants `_TRAIN_WEIGHTS` /
    `_VAL_WEIGHTS`, not read back from this field.

    Escalation knobs default to frozen-probe; callers derive an unfreezing
    variant via `dataclasses.replace` (mirrors `_default_rel_text_config`).
    """
    lam = validate_lam(lam)
    base = cca_config.DEFAULT_CCA_CONFIG
    head0 = base.heads[0]
    shared_loss = dataclasses.replace(head0.loss, prior=0.02, nnpnu_eta=0.0)
    cca_head = dataclasses.replace(
        head0,
        name="cca",
        source_column="cca_label",
        loss=shared_loss,
        loss_weight=1.0 - lam,
    )
    rel_head = dataclasses.replace(
        head0,
        name="rel",
        source_column="rel_target",
        loss=shared_loss,
        loss_weight=lam,
    )
    return dataclasses.replace(
        base,
        heads=(cca_head, rel_head),
        epochs=epochs,
        ratio_batch=cca_config.RatioBatchConfig(train_pos=0.1, val_pos=0.25),
    )


DEFAULT_JOINT_TEXT_CONFIG = _default_joint_text_config(0.5)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def _guard_weights_out(weights_out) -> None:
    """Refuse to write over any of the FOUR existing single-head production
    artifacts.

    This trainer has no cache-suffix CLI knob of its own (`config.
    JOINT_TEXT_TABLE` is a fixed path, and `--weights-out` is always
    REQUIRED with no production default for this two-head artifact family --
    `docs/design-plans/2026-08-18-stage4-joint-finetune.md` Components item
    5), so there is no natural "suffix moved but weights path didn't" pairing
    for `check_no_production_overwrite` to check. Reused anyway, the same way
    `fit_fusion.py`'s `resolve_fusion_inputs` guards ITS multi-input shape:
    `cache_suffix` stands in for "the input value" (here, always the literal
    resolved `weights_out` path, stringified), compared against a SENTINEL
    `production_cache_suffix` that can never legitimately equal it -- which
    collapses the guard's `cache_suffix != production_cache_suffix` branch to
    always-true, leaving a pure `weights_path == production_weights_path`
    check. Looped once per production artifact (four calls, one per
    known single-head weights path).
    """
    weights_out = Path(weights_out)
    for artifact_label, production_path in _PRODUCTION_WEIGHTS_PATHS:
        check_no_production_overwrite(
            cache_suffix=str(weights_out),
            production_cache_suffix=_NO_CACHE_SUFFIX_SENTINEL,
            weights_path=weights_out,
            production_weights_path=production_path,
            artifact_label=f"joint CCA+rel trainer writing over the {artifact_label} artifact",
        )


def main(run_config=None, weights_out=None, max_steps=None, batch_size=256,
         tensorboard_dir=None):
    """
    Train the joint cca+rel heads in text mode over a shared encoder.

    Args:
        run_config: Optional cca_config.RunConfig instance. If None, uses
            DEFAULT_JOINT_TEXT_CONFIG (lam=0.5, frozen-probe). Callers derive
            a variant via `_default_joint_text_config(lam)` +
            `dataclasses.replace` (unfreeze/hard_freeze/seed).
        weights_out: REQUIRED path (str or Path) to write the trained
            weights to. Guarded by `_guard_weights_out` against the four
            existing single-head production artifacts.
        max_steps: Optional cap on steps_per_epoch (and a proportional cap
            on validation_steps), for short/smoke runs.
        batch_size: Training/eval batch size (default 256).
        tensorboard_dir: Optional TensorBoard log dir; see
            `run_relevance_text.main`'s docstring for the same knob.
    """
    if weights_out is None:
        raise ValueError(
            "weights_out is required -- there is no production default for "
            "the joint CCA+rel artifact family (docs/design-plans/"
            "2026-08-18-stage4-joint-finetune.md Components item 5)."
        )
    weights_out = Path(weights_out)
    _guard_weights_out(weights_out)

    keras.config.set_dtype_policy(config.DTYPE_POLICY)

    if run_config is None:
        run_config = DEFAULT_JOINT_TEXT_CONFIG
    keras.utils.set_random_seed(run_config.seed)

    cca_head_cfg, rel_head_cfg = run_config.heads
    BATCH_SIZE = batch_size
    SHUFFLE_BUFFER = 100_000

    # -------------------------------------------------------------------------
    # Load, derive rel_target, split (whole-table-first), re-verify holdout.
    # The joint table already drops the ICA-eval holdout at build time
    # (build_joint_text_table.py); this re-drop + assert_holdout_excluded is
    # belt-and-suspenders, matching run_relevance_text.py's pattern for the
    # same table-vs-script split of duty.
    # -------------------------------------------------------------------------
    table = pl.read_parquet(config.JOINT_TEXT_TABLE)
    table = derive_rel_target(table)
    holdout_ids = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].to_list()
    splits = create_joint_text_data(table, holdout_ids=holdout_ids)

    # assert_holdout_excluded expects the {"pos", "unl"} PU-group shape;
    # create_joint_text_data's groups are {"cca_pos", "rel_pos", "unl"}, which
    # carry no PU semantics assert_holdout_excluded cares about -- it only
    # checks id membership across whatever's under "pos"/"unl". Passing the
    # SAME union of all three groups under both keys is a correct (if
    # redundant) way to reuse that check without inventing a parallel one.
    def _holdout_check_shape(split_name):
        union = pl.concat([splits[split_name][g] for g in _GROUPS])
        return {"pos": union, "unl": union}

    assert_holdout_excluded(
        {sn: _holdout_check_shape(sn) for sn in ("train", "val")}, holdout_ids
    )

    # -------------------------------------------------------------------------
    # tf.data cache: one file per (split, group) -- 9 total, under
    # JOINT_SET_DIR (distinct from run_relevance_text's RELEVANCE_SET_DIR --
    # different columns: cca_label + rel_target, not rel_label). Same
    # staleness contract as run_relevance_text.py: if JOINT_TEXT_TABLE
    # changes upstream, DELETE JOINT_SET_DIR so it rebuilds.
    # -------------------------------------------------------------------------
    if not JOINT_SET_DIR.is_dir():
        JOINT_SET_DIR.mkdir(parents=True)
        for split_name in _SPLITS:
            for group_name in _GROUPS:
                gdf = splits[split_name][group_name]
                ds = tf.data.Dataset.from_tensor_slices({
                    "headline_with_lead": gdf["headline_with_lead"].to_list(),
                    "cca_label": gdf["cca_label"].cast(pl.Float32).to_numpy(),
                    "rel_target": gdf["rel_target"].to_numpy(),
                })
                ds.save(str(JOINT_SET_DIR / f"{split_name}_{group_name}.tf"))

    datasets = {
        (split_name, group_name): tf.data.Dataset.load(
            str(JOINT_SET_DIR / f"{split_name}_{group_name}.tf")
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
                    f"Stale joint-text set cache for {split_name}/{group_name}: "
                    f"cache has {cached_n} rows but current split has {expected_n}. "
                    f"Delete {JOINT_SET_DIR} and re-run."
                )

    # -------------------------------------------------------------------------
    # Preprocessors: one for training (emits both cca_targets and
    # rel_targets), one for predict-only spot-checks.
    # -------------------------------------------------------------------------
    train_preprocess = ClassifierPreprocessor(
        SEQ_LENGTH=run_config.seq_length,
        text_key=run_config.text_key,
        label_keys=run_config.label_keys,  # {"cca_targets": "cca_label", "rel_targets": "rel_target"}
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
    # 3-way weighted Ratio-Batch streams (cca_pos / rel_pos / unl). Every
    # batch carries both cca_targets and rel_targets regardless of which
    # stream a row was drawn from (PU semantics: a cca_pos-drawn row's
    # rel_target is whatever it naturally is -- 0.0 or, rarely, 1.0 for the
    # deliberate cca/rel overlap -- and vice versa).
    # -------------------------------------------------------------------------
    training_set = dataset_create(
        SHUFFLE_BUFFER, BATCH_SIZE, train_preprocess,
        data=[datasets[("train", g)] for g in _GROUPS],
        weights=list(_TRAIN_WEIGHTS),
    )
    validation_set = dataset_create(
        SHUFFLE_BUFFER, BATCH_SIZE, train_preprocess,
        data=[datasets[("val", g)] for g in _GROUPS],
        weights=list(_VAL_WEIGHTS),
    )

    n_cca_pos_tr = splits["train"]["cca_pos"].height
    n_rel_pos_tr = splits["train"]["rel_pos"].height
    n_cca_pos_va = splits["val"]["cca_pos"].height
    n_rel_pos_va = splits["val"]["rel_pos"].height

    # steps_per_epoch from the SMALLER positive stream (design doc item 3):
    # the larger stream just gets seen less often per epoch, not truncated.
    steps_per_epoch = math.floor(
        min(n_cca_pos_tr, n_rel_pos_tr) / (BATCH_SIZE * _TRAIN_WEIGHTS[0])
    )
    validation_steps = max(
        1,
        math.floor(min(n_cca_pos_va, n_rel_pos_va) / (BATCH_SIZE * _VAL_WEIGHTS[0])),
    )
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
        validation_steps = min(validation_steps, max(1, max_steps // 5))

    run_config = dataclasses.replace(
        run_config, lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch)
    )

    # -------------------------------------------------------------------------
    # Model assembly: cca + rel heads (FLPU, loss_weight-mixed) + escalation.
    # -------------------------------------------------------------------------
    backbone = load_dapt_backbone(run_config.backbone_weights_path)
    run_config.validate_against_backbone(backbone)

    def _build_head(head_cfg):
        return ClassificationHead(
            hidden_dim=head_cfg.hidden_dim,
            loss_fn=FLPULoss(
                prior=head_cfg.loss.prior,
                kiryo_clawback=head_cfg.loss.kiryo_clawback,
                nnpnu_eta=head_cfg.loss.nnpnu_eta,
            ),
            metrics=make_cca_metrics() + make_distribution_metrics(run_config.diagnostics),
            name=head_cfg.name,
            expose_loss_components=run_config.diagnostics.enable_loss_components,
            loss_weight=head_cfg.loss_weight,
        )

    cca_head = _build_head(cca_head_cfg)
    rel_head = _build_head(rel_head_cfg)
    heads = {cca_head_cfg.name: cca_head, rel_head_cfg.name: rel_head}

    # Escalation knobs: byte-identical frozen-probe path when unfreeze_top_n==0.
    build_kwargs = {
        "backbone": backbone,
        "heads": heads,
        "seq_length": run_config.seq_length,
        "diagnostics": run_config.diagnostics,
    }
    build_kwargs.update(
        escalation_build_kwargs(
            run_config.unfreeze_top_n, run_config.layer_multipliers,
            hard_freeze=run_config.hard_freeze,
        )
    )

    # Pattern A: endpoint + inference models share head/backbone instances.
    joint_model = build_endpoint_model(**build_kwargs)
    joint_inference = build_inference_model(
        backbone=backbone, heads=heads, seq_length=run_config.seq_length
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

    # Compile WITHOUT loss/metrics -- heads own both via add_loss.
    joint_model.compile(optimizer=optimizer, jit_compile="auto")

    # -------------------------------------------------------------------------
    # Callbacks and fit
    # -------------------------------------------------------------------------
    RELEVANCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tensorboard_dir is not None:
        print(f"TensorBoard log dir: {tensorboard_dir}")  # LOG
    callbacks_list = build_fit_callbacks(
        RELEVANCE_DIR / f"{stamp}_joint_text_metrics.csv", tensorboard_log_dir=tensorboard_dir
    )

    history = joint_model.fit(
        training_set,
        validation_data=validation_set,
        epochs=run_config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
    )

    # -------------------------------------------------------------------------
    # Save weights + config sidecar (records lambda via each head's
    # loss_weight, plus the escalation knobs used).
    # -------------------------------------------------------------------------
    joint_model.save_weights(str(weights_out))
    run_config.to_json(config_path_for_weights(weights_out))

    # -------------------------------------------------------------------------
    # Spot-check: score held-out test cca_pos/rel_pos/unl per head, report
    # distributions (mirrors run_relevance_text.py's final block).
    # -------------------------------------------------------------------------
    def _score_group(group_name, head_name):
        ds = (
            datasets[("test", group_name)]
            .batch(BATCH_SIZE, drop_remainder=False)
            .map(predict_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        return joint_inference.predict(ds, verbose=0)[head_name].reshape(-1)

    for head_name in (cca_head_cfg.name, rel_head_cfg.name):
        cca_pos_scores = _score_group("cca_pos", head_name)
        rel_pos_scores = _score_group("rel_pos", head_name)
        unl_scores = _score_group("unl", head_name)
        print(
            f"[{head_name}] test cca_pos logit[mean/median]="
            f"{cca_pos_scores.mean():.3f}/{np.median(cca_pos_scores):.3f}  "
            f"rel_pos logit[mean/median]={rel_pos_scores.mean():.3f}/{np.median(rel_pos_scores):.3f}  "
            f"unl logit[mean/median]={unl_scores.mean():.3f}/{np.median(unl_scores):.3f}"
        )  # LOG

    return joint_model, joint_inference, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train the joint cca+rel text-mode heads (shared-encoder fine-tune)."
    )
    ap.add_argument("--lam", type=float, required=True,
                     help="rel-side loss_weight (lambda) in (0, 1); cca gets 1-lambda.")
    ap.add_argument("--unfreeze-top-n", type=int, default=0,
                     help="number of top RoBERTa layers to unfreeze (0 = frozen probe)")
    ap.add_argument("--graded-decay", type=float, default=None,
                     help="ULMFiT-style graded per-layer LRs (see run_relevance_text.py's "
                          "same knob). Requires --unfreeze-top-n > 0.")
    ap.add_argument("--weights-out", type=str, required=True,
                     help="weights output path (REQUIRED -- no production default exists "
                          "for the joint CCA+rel artifact family)")
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=200,
                     help="training-time random seed (keras.utils.set_random_seed); "
                          "does NOT affect the seed=200 data-split in "
                          "src/data_setup/data.py")
    ap.add_argument("--hard-freeze", action=argparse.BooleanOptionalAction, default=True,
                     help="hard-freeze (trainable=False) backbone sub-layers below the "
                          "unfrozen top-N block (see run_relevance_text.py's same knob). "
                          "Default ON; --no-hard-freeze to opt out (the sweep script "
                          "passes --no-hard-freeze -- the stage-3 deploy rule is "
                          "multiplier-freeze-train + graft-at-deploy).")
    ap.add_argument("--tensorboard-dir", type=str, default=None,
                     help="explicit TensorBoard log dir (wins over --tensorboard)")
    ap.add_argument("--tensorboard", action="store_true",
                     help="enable TensorBoard logging to a timestamped dir under "
                          "RELEVANCE_DIR/tb_logs/ (ignored if --tensorboard-dir is set)")
    args = ap.parse_args()

    if args.graded_decay is not None and args.unfreeze_top_n < 1:
        ap.error("--graded-decay requires --unfreeze-top-n > 0")

    cfg = _default_joint_text_config(args.lam, epochs=args.epochs)
    cfg = dataclasses.replace(
        cfg,
        unfreeze_top_n=args.unfreeze_top_n,
        freeze_encoder=(args.unfreeze_top_n == 0),
        layer_multipliers=(
            graded_multipliers(args.unfreeze_top_n, decay=args.graded_decay)
            if args.graded_decay is not None
            else cfg.layer_multipliers
        ),
        hard_freeze=args.hard_freeze,
        seed=args.seed,
    )
    tb_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tensorboard_dir = resolve_tensorboard_dir(args.tensorboard_dir, args.tensorboard, tb_timestamp)
    main(
        run_config=cfg,
        weights_out=args.weights_out,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        tensorboard_dir=tensorboard_dir,
    )
