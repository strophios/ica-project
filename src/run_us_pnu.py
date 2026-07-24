# pattern: Imperative Shell (attach_emb_rows, gather_group_features,
#           estimate_prior_from_calibrated_logits, pn_val_metrics, pu_val_risk
#           are the pure core)
"""Train the v1 US-head retrain candidate (stripped channel, nnPNU).

Design: docs/notes/us-head-retrain-plan.md. Corpus: `src/build_us_pnu_table.py`'s
`us_filter/us_pnu_table.parquet` (columns id/cache/pnu_label/source/year; 367,876
rows: 15,053 pos [14,708 DoCA-API + 345 DoCA-LDC] / 102,823 neg [dateline-resolved
foreign] / 250,000 unl [random API sample]). Rows draw their CLS feature vector
from FOUR different embed caches depending on the `cache` column (train250k |
us_pos_ldc345 | us_train_ldc | full) -- see build_us_pnu_table.py's module
docstring for the source mapping. `attach_emb_rows` joins each cache-scoped
subset of the table to that cache's own (id, emb_row) so `gather_group_features`
can pull the right CLS rows for an arbitrary split group (a group may span more
than one cache -- the "pos" group does: train250k for DoCA-API, us_pos_ldc345 for
DoCA-LDC).

CONFIG CHOICE: `UsRunConfig` (the production US head's BCE-only config) cannot
express a PU prior or an nnPNU eta -- it has no `loss` field at all (see
`src/us_config.py`'s module docstring: "plain supervised PN with BCE"). This
retrain needs both, so -- per the task's explicit instruction to mirror
`run_relevance.py` rather than invent a third config style -- it reuses
`cca_config.RunConfig` / `HeadConfig` / `FLPULossConfig` (the same machinery the
CCA and relevance heads use). The head is named "us_pnu" (NOT "us") so its
sidecar is unambiguously distinct from the production head's `UsRunConfig`
sidecar at `us_classifier_full.config.json` -- different schema, different
weights file, same directory. A future swap into the assembled `IcaModel` would
rename it to "us" at that point (see the design doc's "Sequencing" section).

MODEL SELECTION RULE (stated here, before any metric is computed, exactly as
required by the task): for each eta in the grid, after training, compute on the
us_pnu_table's OWN VAL SPLIT ONLY -- `pn_val_metrics` (PR-AUC + F1 of val pos vs
val neg; the ONLY groups with ground-truth labels -- the unlabeled group has
none) and `pu_val_risk` (the held-out nnPNU loss over all three val streams).
Select the eta with the highest val PR-AUC. THE HAND-CODED ICA EVAL SET IS NEVER
TOUCHED IN THIS SCRIPT -- it is reserved entirely for scripts/eval_us_retrain.py's
post-hoc validate-before-swap comparison.

PRIOR ESTIMATION: pi_hat = mean CALIBRATED us-probability the CURRENT production
head (`us_classifier_full`) assigns to the table's 250k-row unlabeled sample.
The `full` embed cache carries `us_logit` (the current head's raw logit, co-
computed at embed time by `embed_corpus.py`); `estimate_prior_from_calibrated_logits`
Platt-transforms it via the current head's calibration sidecar and averages. This
substitutes for a fresh DEDPUL run (the CCA/relevance precedent) with a cheaper
"trust the existing head's aggregate belief" proxy -- reasonable here because
recovering exactly the current head's failure mode (diaspora ICA) is the whole
point of the retrain, so its prior estimate is a defensible starting point even
though it is somewhat circular. See docs/notes/us-head-retrain-plan.md.

Run from project root:
    uv run python -m src.run_us_pnu
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
from collections.abc import Collection

import keras
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, f1_score

import src.config as config
import src.cca_config as cca_config
from src.calibration.sidecar import calibration_path_for_weights, load_calibration
from src.cca_metrics import make_cca_metrics
from src.data_setup.data import (
    assert_holdout_excluded,
    create_us_pnu_data,
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

keras.config.set_dtype_policy(config.DTYPE_POLICY)
keras.utils.set_random_seed(200)

CACHE_NAMES = ("train250k", "us_pos_ldc345", "us_train_ldc", "full")
HEAD_NAME = "us_pnu"
ETA_GRID = (0.0, 0.25, 0.5, 1.0)
NEG_WEIGHT = 0.15  # reliable-negative Ratio-Batch stream weight (matches run_relevance.py)
BATCH_SIZE = 256
SHUFFLE_BUFFER = 100_000


# ---------------------------------------------------------------------------
# Functional core: table/cache joins, feature gather, prior + val metrics
# ---------------------------------------------------------------------------
def attach_emb_rows(table: pl.DataFrame, cache_metas: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Join each row to its OWN cache's `emb_row` (dispatched by the `cache` column).

    `cache_metas` maps cache name -> that cache's meta DataFrame (must carry `id`
    + `emb_row`, e.g. from `embed_corpus.load_cache`/`load_cache_meta`). Every
    cache value present in `table["cache"]` must have an entry in `cache_metas`
    (raises `ValueError` naming the missing cache), and every row must resolve to
    exactly one `emb_row` (raises `ValueError` enumerating unresolved ids) -- a
    silent drop here would silently shrink a training population.
    """
    parts = []
    for cache_name in table["cache"].unique().to_list():
        if cache_name not in cache_metas:
            raise ValueError(
                f"no cache_metas entry for cache={cache_name!r} "
                f"(have {sorted(cache_metas.keys())})"
            )
        sub = table.filter(pl.col("cache") == cache_name)
        meta = cache_metas[cache_name].select(
            pl.col("id").cast(pl.Utf8), pl.col("emb_row")
        )
        joined = sub.join(meta, on="id", how="left")
        missing = joined.filter(pl.col("emb_row").is_null())
        if missing.height > 0:
            raise ValueError(
                f"{missing.height} ids in cache={cache_name!r} not found in its "
                f"cache meta (e.g. {missing['id'].to_list()[:5]})"
            )
        parts.append(joined)
    return pl.concat(parts, how="vertical")


def gather_group_features(
    group: pl.DataFrame, cls_by_cache: dict[str, np.ndarray], label: float
) -> tuple[np.ndarray, np.ndarray]:
    """CLS feature rows + a constant label array for a split group.

    `group` carries `cache` + `emb_row` (see `attach_emb_rows`) -- a group may
    span more than one cache (the "pos" group draws from both `train250k` and
    `us_pos_ldc345`), so this gathers per-cache and concatenates. Row order need
    not match `group`'s own order: the label is uniform across the whole group,
    so alignment only needs to hold WITHIN each per-cache slice, which it does by
    construction (`emb_row` indexes directly into `cls_by_cache[cache_name]`).

    `label` follows FLPULoss's convention: 1.0 positive / -1.0 reliable-negative /
    0.0 unlabeled.
    """
    feats_parts = []
    for cache_name in group["cache"].unique().to_list():
        sub = group.filter(pl.col("cache") == cache_name)
        rows = sub["emb_row"].to_numpy().astype(int)
        feats_parts.append(cls_by_cache[cache_name][rows])
    if feats_parts:
        feats = np.concatenate(feats_parts, axis=0)
    else:
        feats = np.empty((0, 0), dtype=np.float32)
    labels = np.full(feats.shape[0], label, dtype=np.float32)
    return feats, labels


def estimate_prior_from_calibrated_logits(logits: np.ndarray, calibrator) -> float:
    """pi_hat = mean Platt-calibrated probability over an array of raw logits.

    Pure given already-loaded logits + a `PlattCalibrator` instance (the caller
    loads the calibration sidecar; see module docstring "PRIOR ESTIMATION").
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        raise ValueError("estimate_prior_from_calibrated_logits: logits array is empty")
    probs = calibrator.transform(logits)
    return float(np.mean(probs))


def pn_val_metrics(pos_logits: np.ndarray, neg_logits: np.ndarray) -> dict:
    """PR-AUC + F1(logit>0) of val positives vs val reliable-negatives.

    The ONLY val groups with ground-truth labels -- deliberately excludes the
    unlabeled group (see module docstring "MODEL SELECTION RULE").
    """
    pos_logits = np.asarray(pos_logits, dtype=np.float64).reshape(-1)
    neg_logits = np.asarray(neg_logits, dtype=np.float64).reshape(-1)
    scores = np.concatenate([pos_logits, neg_logits])
    labels = np.concatenate([np.ones_like(pos_logits), np.zeros_like(neg_logits)])
    preds = scores > 0
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "n_pos": int(pos_logits.size),
        "n_neg": int(neg_logits.size),
    }


def pu_val_risk(
    pos_logits: np.ndarray, neg_logits: np.ndarray, unl_logits: np.ndarray,
    prior: float, eta: float,
) -> float:
    """The held-out nnPNU training objective, evaluated on the val split.

    Combines the three val streams into FLPULoss's three-label convention
    (1.0/-1.0/0.0) and returns the scalar loss -- the "PU risk" figure in the
    model-selection rule, alongside `pn_val_metrics`.

    SHAPE NOTE (load-bearing, found while writing tests for this function):
    `y_pred` MUST be 2-D `(n, 1)`, matching the real endpoint-mode call shape
    (`ClassificationHead.call` produces `(batch, 1)` logits; the target Input is
    `(batch,)` 1-D -- see `src/model_setup/heads.py`'s `call` docstring). Calling
    `FLPULoss` with BOTH `y_true` and `y_pred` flattened to 1-D silently produces
    an eta-INVARIANT loss (verified empirically: `tests/test_run_us_pnu.py`
    `TestPuValRisk` would fail against a 1-D y_pred). This does not affect actual
    training (Keras always calls the loss with the head's native (batch,1) output
    via `add_loss`) -- it is a footgun specific to reconstructing the loss
    call OUTSIDE the model graph, as this diagnostic function does.
    """
    pos_logits = np.asarray(pos_logits, dtype=np.float32).reshape(-1)
    neg_logits = np.asarray(neg_logits, dtype=np.float32).reshape(-1)
    unl_logits = np.asarray(unl_logits, dtype=np.float32).reshape(-1)
    y_true = np.concatenate([
        np.ones_like(pos_logits), -np.ones_like(neg_logits), np.zeros_like(unl_logits),
    ])
    y_pred = np.concatenate([pos_logits, neg_logits, unl_logits]).reshape(-1, 1)
    loss_fn = FLPULoss(prior=prior, nnpnu_eta=eta)
    loss = loss_fn(
        keras.ops.convert_to_tensor(y_true), keras.ops.convert_to_tensor(y_pred)
    )
    return float(keras.ops.convert_to_numpy(loss))


def _base_run_config(prior: float, eta: float, epochs: int) -> cca_config.RunConfig:
    """DEFAULT_CCA_CONFIG with the head renamed HEAD_NAME and prior/eta set.

    See module docstring "CONFIG CHOICE". `source_column`/`text_key` are
    inherited unchanged from DEFAULT_CCA_CONFIG -- as in run_relevance.py, these
    fields are vestigial for the features-mode path (dataset_from_embeddings
    builds targets directly; ClassifierPreprocessor/label_keys never run).
    """
    base = cca_config.DEFAULT_CCA_CONFIG
    head0 = base.heads[0]
    new_head = dataclasses.replace(
        head0, name=HEAD_NAME,
        loss=dataclasses.replace(head0.loss, prior=prior, nnpnu_eta=eta),
    )
    return dataclasses.replace(base, heads=(new_head,), epochs=epochs)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------
def _load_holdout_ids() -> list[str]:
    api = pl.read_parquet(config.ICA_HOLDOUT_IDS)["id"].cast(pl.Utf8).to_list()
    ldc = pl.read_parquet(config.ICA_HOLDOUT_IDS_LDC)["id"].cast(pl.Utf8).to_list()
    return sorted(set(api) | set(ldc))


def _load_caches() -> dict[str, tuple[pl.DataFrame, np.ndarray]]:
    return {name: load_cache(config.CCA_EMBED_CACHE_DIR / name) for name in CACHE_NAMES}


def _gather_splits(
    splits: dict, cls_by_cache: dict[str, np.ndarray]
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Gather (features, labels) for every {split}/{group} cell in one pass."""
    label_by_group = {"pos": 1.0, "neg": -1.0, "unl": 0.0}
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for split_name, groups in splits.items():
        out[split_name] = {
            group_name: gather_group_features(df, cls_by_cache, label_by_group[group_name])
            for group_name, df in groups.items()
        }
    return out


def _estimate_prior(table: pl.DataFrame, full_meta: pl.DataFrame) -> float:
    unl_ids = table.filter(pl.col("pnu_label") == "unl")["id"].to_list()
    full_meta_utf8 = full_meta.with_columns(pl.col("id").cast(pl.Utf8))
    unl_logits = (
        full_meta_utf8.filter(pl.col("id").is_in(unl_ids))["us_logit"].to_numpy()
    )
    if unl_logits.shape[0] != len(unl_ids):
        raise ValueError(
            f"prior estimation: resolved {unl_logits.shape[0]} of "
            f"{len(unl_ids)} unlabeled ids in the `full` cache"
        )
    cal = load_calibration(calibration_path_for_weights(config.US_FILTER_FULL_WEIGHTS))
    pi_hat = estimate_prior_from_calibrated_logits(unl_logits, cal)
    # FLPULossConfig/FLPULoss require prior strictly in (0, 1); clip narrowly
    # against a pathological calibrator output at the boundary.
    clipped = min(max(pi_hat, 1e-3), 0.999)
    if clipped != pi_hat:
        print(f"WARNING: pi_hat={pi_hat:.6f} clipped to {clipped:.6f}")  # LOG
    return clipped


def _train_variant(
    run_config: cca_config.RunConfig,
    streams: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    max_steps: int | None,
    log_dir,
    stamp: str,
):
    """Build, compile, and fit ONE (prior, eta) variant. Returns (model, inference)."""
    head_cfg = run_config.heads[0]
    pos_tr, neg_tr, unl_tr = streams["train"]["pos"], streams["train"]["neg"], streams["train"]["unl"]
    pos_va, neg_va, unl_va = streams["val"]["pos"], streams["val"]["neg"], streams["val"]["unl"]
    n_pos_tr, n_pos_va = pos_tr[0].shape[0], pos_va[0].shape[0]

    tp, vp = run_config.ratio_batch.train_pos, run_config.ratio_batch.val_pos
    train_set = dataset_from_embeddings(
        SHUFFLE_BUFFER, BATCH_SIZE, data=[pos_tr, neg_tr, unl_tr],
        weights=[tp, NEG_WEIGHT, 1 - tp - NEG_WEIGHT], head_name=head_cfg.name,
    )
    val_set = dataset_from_embeddings(
        SHUFFLE_BUFFER, BATCH_SIZE, data=[pos_va, neg_va, unl_va],
        weights=[vp, NEG_WEIGHT, 1 - vp - NEG_WEIGHT], head_name=head_cfg.name,
    )
    steps_per_epoch = math.floor(n_pos_tr / (BATCH_SIZE * tp))
    validation_steps = max(1, math.floor(n_pos_va / (BATCH_SIZE * vp)))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    run_config = dataclasses.replace(
        run_config, lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch)
    )

    head = ClassificationHead(
        hidden_dim=head_cfg.hidden_dim,
        loss_fn=FLPULoss(prior=head_cfg.loss.prior, nnpnu_eta=head_cfg.loss.nnpnu_eta),
        metrics=make_cca_metrics() + make_distribution_metrics(run_config.diagnostics),
        name=head_cfg.name,
        expose_loss_components=run_config.diagnostics.enable_loss_components,
    )
    model = build_feature_endpoint_model(
        {head_cfg.name: head}, hidden_dim=head_cfg.hidden_dim, diagnostics=run_config.diagnostics,
    )
    inference = build_feature_inference_model({head_cfg.name: head}, hidden_dim=head_cfg.hidden_dim)

    resolved = run_config.lr_schedule.resolved
    assert resolved is not None, "with_resolved should have populated lr_schedule.resolved"
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=run_config.lr_schedule.initial_lr,
        decay_steps=resolved.decay_steps, alpha=run_config.lr_schedule.decay_alpha,
        warmup_target=run_config.lr_schedule.warmup_target, warmup_steps=resolved.warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=run_config.optimizer.weight_decay
    )
    model.compile(optimizer=optimizer, jit_compile="auto")

    callbacks = [
        keras.callbacks.CSVLogger(str(log_dir / f"{stamp}_eta{head_cfg.loss.nnpnu_eta}_metrics.csv")),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=1, start_from_epoch=2),
    ]
    model.fit(
        train_set, validation_data=val_set, epochs=run_config.epochs,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        callbacks=callbacks, verbose=2,
    )
    return model, inference, run_config


def main(
    run_config: cca_config.RunConfig | None = None,
    max_steps: int | None = None,
    weights_path=None,
    etas: Collection[float] = ETA_GRID,
    epochs: int = 7,
    run_sensitivity: bool = True,
    holdout_ids: list[str] | None = None,
):
    """Full workflow: prior estimate -> eta grid -> val-split selection -> save.

    `run_config` (default None -> DEFAULT_CCA_CONFIG-derived base, epochs=`epochs`)
    supplies the fixed architectural parameters (backbone, ratio_batch, lr_schedule,
    optimizer); prior and nnpnu_eta are swept per-variant regardless of what
    `run_config.heads[0].loss` carries (the eta grid + estimated pi_hat always
    override it -- `run_config` is a base to derive FROM, not a single variant to
    run once, since this script's whole job is the grid + selection).
    """
    weights_path = weights_path or config.US_PNU_WEIGHTS
    base_config = run_config or _base_run_config(prior=0.5, eta=0.0, epochs=epochs)
    # prior=0.5 above is a throwaway placeholder immediately overridden per-variant
    # below by the estimated pi_hat; __post_init__ just needs SOME value in (0,1).

    config.US_FILTER_DIR.mkdir(parents=True, exist_ok=True)
    (config.CCA_DOCA_DIR / "experiments").mkdir(parents=True, exist_ok=True)

    table = pl.read_parquet(config.US_PNU_TABLE)
    holdout_ids = holdout_ids if holdout_ids is not None else _load_holdout_ids()
    print(f"holdout: {len(holdout_ids)} ids (belt-and-suspenders on top of "
          f"table-build-time exclusion)")  # LOG

    caches = _load_caches()
    cache_metas = {name: meta for name, (meta, _cls) in caches.items()}
    cls_by_cache = {name: cls for name, (_meta, cls) in caches.items()}

    resolved_table = attach_emb_rows(table, cache_metas)
    splits = create_us_pnu_data(resolved_table, holdout_ids=holdout_ids)
    assert_holdout_excluded(splits, holdout_ids)
    streams = _gather_splits(splits, cls_by_cache)
    for split_name in ("train", "val", "test"):
        s = streams[split_name]
        print(f"{split_name}: pos={s['pos'][0].shape[0]} neg={s['neg'][0].shape[0]} "
              f"unl={s['unl'][0].shape[0]}")  # LOG

    pi_hat = _estimate_prior(resolved_table, cache_metas["full"])
    print(f"pi_hat (estimated prior) = {pi_hat:.4f}")  # LOG

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_results = []
    trained = {}  # eta -> (model, inference, run_config)
    for eta in etas:
        variant_config = dataclasses.replace(
            base_config,
            heads=(dataclasses.replace(
                base_config.heads[0],
                loss=dataclasses.replace(base_config.heads[0].loss, prior=pi_hat, nnpnu_eta=eta),
            ),),
        )
        print(f"--- training eta={eta} prior={pi_hat:.4f} ---")  # LOG
        model, inference, variant_config = _train_variant(
            variant_config, streams, max_steps, config.US_FILTER_DIR, stamp
        )
        trained[eta] = (model, inference, variant_config)

        pos_va, neg_va, unl_va = streams["val"]["pos"], streams["val"]["neg"], streams["val"]["unl"]
        pos_logits = inference.predict({"features": pos_va[0]}, verbose=0)[HEAD_NAME].reshape(-1)
        neg_logits = inference.predict({"features": neg_va[0]}, verbose=0)[HEAD_NAME].reshape(-1)
        unl_logits = inference.predict({"features": unl_va[0]}, verbose=0)[HEAD_NAME].reshape(-1)
        pn = pn_val_metrics(pos_logits, neg_logits)
        pu_risk = pu_val_risk(pos_logits, neg_logits, unl_logits, pi_hat, eta)
        result = {"eta": eta, "prior": pi_hat, **pn, "pu_val_risk": pu_risk}
        grid_results.append(result)
        print(f"eta={eta}: val_pr_auc={pn['pr_auc']:.4f} val_f1={pn['f1']:.4f} "
              f"val_pu_risk={pu_risk:.4f}")  # LOG

    # --- Model selection: highest val PR-AUC (stated rule above). Gold set untouched. ---
    best = max(grid_results, key=lambda r: r["pr_auc"])
    best_eta = best["eta"]
    best_model, best_inference, best_config = trained[best_eta]
    print(f"SELECTED eta={best_eta} (val_pr_auc={best['pr_auc']:.4f})")  # LOG

    sensitivity_results = []
    if run_sensitivity:
        for delta in (-0.1, 0.1):
            sens_prior = min(max(pi_hat + delta, 1e-3), 0.999)
            sens_config = dataclasses.replace(
                base_config,
                heads=(dataclasses.replace(
                    base_config.heads[0],
                    loss=dataclasses.replace(
                        base_config.heads[0].loss, prior=sens_prior, nnpnu_eta=best_eta
                    ),
                ),),
            )
            print(f"--- sensitivity eta={best_eta} prior={sens_prior:.4f} (delta={delta:+.1f}) ---")  # LOG
            _model, s_inference, _cfg = _train_variant(
                sens_config, streams, max_steps, config.US_FILTER_DIR, stamp
            )
            pos_va, neg_va = streams["val"]["pos"], streams["val"]["neg"]
            pos_logits = s_inference.predict({"features": pos_va[0]}, verbose=0)[HEAD_NAME].reshape(-1)
            neg_logits = s_inference.predict({"features": neg_va[0]}, verbose=0)[HEAD_NAME].reshape(-1)
            pn = pn_val_metrics(pos_logits, neg_logits)
            sensitivity_results.append({
                "prior": sens_prior, "delta": delta, "pr_auc": pn["pr_auc"], "f1": pn["f1"],
                "pr_auc_delta_vs_selected": pn["pr_auc"] - best["pr_auc"],
            })
            print(f"  prior={sens_prior:.4f}: val_pr_auc={pn['pr_auc']:.4f} "
                  f"(delta vs selected: {pn['pr_auc'] - best['pr_auc']:+.4f})")  # LOG

    best_model.save_weights(str(weights_path))
    best_config.to_json(cca_config.config_path_for_weights(weights_path))
    print(f"Saved weights + sidecar at {weights_path}")  # LOG

    experiment_record = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "pi_hat": pi_hat,
        "eta_grid": list(etas),
        "grid_results": grid_results,
        "selected_eta": best_eta,
        "sensitivity_results": sensitivity_results,
        "weights_path": str(weights_path),
        "config_path": str(cca_config.config_path_for_weights(weights_path)),
        "n_train": {k: v[0].shape[0] for k, v in streams["train"].items()},
        "n_val": {k: v[0].shape[0] for k, v in streams["val"].items()},
        "n_test": {k: v[0].shape[0] for k, v in streams["test"].items()},
    }
    exp_path = config.CCA_DOCA_DIR / "experiments" / "us_pnu_eta_grid.json"
    exp_path.write_text(json.dumps(experiment_record, indent=2))
    print(f"Wrote {exp_path}")  # LOG

    # Spot-check the SELECTED model on its own held-out test split.
    pos_te, neg_te, unl_te = streams["test"]["pos"], streams["test"]["neg"], streams["test"]["unl"]
    pos_scores = best_inference.predict({"features": pos_te[0]}, verbose=0)[HEAD_NAME].reshape(-1)
    neg_scores = best_inference.predict({"features": neg_te[0]}, verbose=0)[HEAD_NAME].reshape(-1)
    unl_scores = best_inference.predict({"features": unl_te[0]}, verbose=0)[HEAD_NAME].reshape(-1)
    print(f"test positives logit[mean/median]="
          f"{pos_scores.mean():.3f}/{np.median(pos_scores):.3f}  "
          f"reliable-neg logit[mean/median]={neg_scores.mean():.3f}/{np.median(neg_scores):.3f}  "
          f"unlabeled logit[mean/median]={unl_scores.mean():.3f}/{np.median(unl_scores):.3f}")  # LOG

    return experiment_record


if __name__ == "__main__":
    main()
