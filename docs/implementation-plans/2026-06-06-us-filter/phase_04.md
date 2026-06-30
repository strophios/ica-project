# US/not-US Pre-Filter — Phase 4 Implementation Plan

**Goal:** A frozen-DAPT linear probe trained with binary cross-entropy, producing in-distribution test metrics, with the shared diagnostic instrumentation firing.

**Architecture:** Maximal reuse of the existing model-construction spine. The US head is a `ClassificationHead` wired through `build_endpoint_model` (so `LayerLRModel` + diagnostics + prediction-distribution metrics all ride along), but its `loss_fn` is `BinaryCrossentropy(from_logits=True)` instead of FLPU — BCE rides the head's `add_loss` path and `compile` takes no loss, exactly as the CCA path does. A separate `UsRunConfig` parallels `RunConfig` (no FLPU/prior coupling). US is PN with mild imbalance, so training uses a single natural-balance shuffled stream, not the PU ratio-batch. The frozen encoder is trained end-to-end (feature precompute deferred).

**Tech Stack:** Python, Keras 3 + TensorFlow, keras_hub RoBERTa, polars, pytest.

**Scope:** Phase 4 of 8.

**Codebase verified:** 2026-06-09 (codebase-investigator full contract map of `cca_config.py`, `heads.py`, `assembly.py`, `backbone.py`, `run_cca_classification.py`, `cca_metrics.py`, `preprocessor.py`).

---

## Acceptance Criteria Coverage

This phase implements and tests **us-filter.AC3** (in-distribution portion):

### us-filter.AC3: Model trains as supervised PN and its transfer is measured
- **us-filter.AC3.1 Success:** trains to completion on the confidently-labeled set; in-distribution test F1 exceeds a majority-class baseline.
- **us-filter.AC3.2 Success:** prediction-distribution diagnostics/metrics populate for train and val phases.
- **us-filter.AC3.4 Guard:** no FLPU / prior / nnPU in the path — the config has no prior field and the loss is BCE.
- **us-filter.AC3.5 Edge:** the same seed yields the same split and the same metrics within tolerance.

(AC3.3 — transfer to the pre-1986 slice — is covered in Phase 6.)

**Scope note on AC3.1:** the short capped run (Task 5) verifies the training *path* (trains end-to-end, metrics produced, diagnostics fire). AC3.1's quality bar — *in-distribution test F1 exceeds a majority-class baseline* — requires a **full training run**, which is operator-invoked (`main()` uncapped, likely on cluster), consistent with how this project runs its other full trainings. The script computes and reports F1 and the majority-class baseline so the bar can be checked when the full run completes.

---

## Key contract facts (verbatim from investigation)

- `ClassificationHead.__init__(self, hidden_dim, dropout=0.1, loss_fn=None, metrics=None, *, name, expose_loss_components=False)`. Endpoint mode (`loss_fn` not None): `call(features, targets)` does `add_loss(loss_fn(targets, logits))` and per-head `metric.update_state(targets, logits)`. `expose_loss_components=True` requires a `loss_fn` with a `return_intermediates` param → for BCE keep it **False**.
- `build_endpoint_model(backbone, heads, seq_length, target_dtype="float32", freeze_encoder=False, layer_multipliers=None, group_fn=None, diagnostics=None)` → `LayerLRModel`. Creates `<head>_targets` Inputs; outputs keyed by head name. **Compile with no loss/metrics** (head owns both). `freeze_encoder=True` sets `backbone.trainable=False` before gathering diagnostic trackables (frozen encoder → no spurious backbone grad tracker).
- `build_inference_model(backbone, heads, seq_length)` → plain `keras.Model`; Pattern-A weight sharing when given the same head instances in-process.
- `load_dapt_backbone(weights_path)` → backbone with `.trainable` left True; assembly freezes it.
- `ClassifierPreprocessor(SEQ_LENGTH, text_key, label_keys, tokenizer=None, endpoint_model=False, target_dtype="float32")`. Endpoint mode emits `{token_ids, padding_mask, "us_targets": <target>}`; `label_keys={"us_targets": "us_label"}`.
- `make_cca_metrics()` → `[BinaryAccuracy(threshold=0.0), Precision(thresholds=0.0), Recall(thresholds=0.0), AUC(curve="PR", from_logits=True, name="pr_auc")]`. F1 omitted (Keras F1 needs a probability threshold) — compute post-hoc from P/R.
- `make_distribution_metrics(diagnostics)` rides the head `metrics=` path (train + val, no extra forward pass).
- `DiagnosticsConfig(enable_gradient_norms, enable_overflow_proxy, enable_loss_components, enable_batch_balance, enable_prediction_distribution, gradient_norm_aggregations, prediction_summary_stats)` — all default True/tuples. For US set `enable_loss_components=False`.
- CCA training compiles `cca_classifier.compile(optimizer=optimizer, jit_compile="auto")` (no loss); `LossScaleOptimizer` wraps AdamW only on cluster; `keras.utils.set_random_seed(200)`; `keras.config.set_dtype_policy(config.DTYPE_POLICY)`.
- `create_classifier_data` splits by `cca_label` into pos/unl — **CCA-specific**; US needs its own split.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: `src/us_config.py` — UsRunConfig (parallel to RunConfig, no FLPU coupling)

**Verifies:** us-filter.AC3.4 (no prior field), partial AC3.5 (config reproducibility surface).

**Files:**
- Create: `src/us_config.py`
- Create: `tests/test_us_config.py`

**Implementation:** Frozen dataclasses reusing `cca_config` sub-configs. Module docstring states the **convergence path**: when the multi-head config lands, CCA and US configs unify via a loss-type discriminated union on `HeadConfig.loss`; until then `UsRunConfig` is a deliberate parallel that reuses every shared sub-config and mirrors `RunConfig`'s property surface so the merge is mechanical.

```python
# pattern: Functional Core
"""US/not-US run configuration.

Parallel to cca_config.RunConfig but with NO FLPU/prior/nnPU coupling: the US
filter is plain supervised PN with BCE. This is a deliberate "separate for now"
config that REUSES every shared sub-config (LRScheduleConfig, OptimizerConfig,
DiagnosticsConfig, ResolvedSteps) and MIRRORS RunConfig's property surface
(label_keys, expected_columns, validate_against_backbone, to_json/from_json).
Convergence path: when the multi-head config is built, CCA + US unify via a
loss-type discriminated union on the head config; this module merges in mechanically.
"""
from __future__ import annotations
import dataclasses
import json
from pathlib import Path

from src import config
from src.cca_config import (
    LRScheduleConfig, OptimizerConfig, DiagnosticsConfig, config_path_for_weights,
)


@dataclasses.dataclass(frozen=True)
class UsHeadConfig:
    name: str
    source_column: str
    hidden_dim: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("UsHeadConfig.name must be a non-empty string")
        if "/" in self.name:
            raise ValueError("UsHeadConfig.name must not contain '/'")
        if not isinstance(self.source_column, str) or not self.source_column:
            raise ValueError("UsHeadConfig.source_column must be a non-empty string")
        if not isinstance(self.hidden_dim, int) or self.hidden_dim <= 0:
            raise ValueError("UsHeadConfig.hidden_dim must be a positive int")


@dataclasses.dataclass(frozen=True)
class UsRunConfig:
    seq_length: int
    text_key: str
    target_dtype: str
    head: UsHeadConfig
    epochs: int
    backbone_weights_path: str
    lr_schedule: LRScheduleConfig
    optimizer: OptimizerConfig
    diagnostics: DiagnosticsConfig = dataclasses.field(
        default_factory=lambda: DiagnosticsConfig(enable_loss_components=False)
    )

    def __post_init__(self):
        if self.seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if not isinstance(self.text_key, str) or not self.text_key:
            raise ValueError("text_key must be a non-empty string")
        # target_dtype validated by reusing Keras dtype check (mirror RunConfig)
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if not isinstance(self.backbone_weights_path, str) or not self.backbone_weights_path:
            raise ValueError("backbone_weights_path must be a non-empty string")
        if not isinstance(self.head, UsHeadConfig):
            raise ValueError("head must be a UsHeadConfig")
        # BCE has no loss components; loss-component diagnostics must be disabled.
        if self.diagnostics.enable_loss_components:
            raise ValueError(
                "US filter uses BCE (no loss components); set "
                "DiagnosticsConfig.enable_loss_components=False"
            )

    @property
    def label_keys(self) -> dict[str, str]:
        return {f"{self.head.name}_targets": self.head.source_column}

    @property
    def expected_columns(self) -> set[str]:
        return {self.text_key, self.head.source_column}

    def validate_against_backbone(self, backbone) -> None:
        if self.head.hidden_dim != backbone.hidden_dim:
            raise ValueError(
                f"head hidden_dim {self.head.hidden_dim} != backbone.hidden_dim "
                f"{backbone.hidden_dim}"
            )

    def to_json(self, path: Path | str) -> None:
        payload = dataclasses.asdict(self)
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> "UsRunConfig":
        payload = json.loads(Path(path).read_text())
        # Delegate sub-config reconstruction to cca_config's existing
        # _from_dict classmethods: LRScheduleConfig._from_dict rebuilds the
        # nested ResolvedSteps, DiagnosticsConfig._from_dict coerces the
        # JSON-array tuple fields back to tuples, and all three filter
        # unknown fields. This is robust to the populated `resolved` field
        # (present in a sidecar written after with_resolved) and to future
        # sub-config field additions.
        return cls(
            seq_length=payload["seq_length"],
            text_key=payload["text_key"],
            target_dtype=payload["target_dtype"],
            head=UsHeadConfig(**payload["head"]),
            epochs=payload["epochs"],
            backbone_weights_path=payload["backbone_weights_path"],
            lr_schedule=LRScheduleConfig._from_dict(payload["lr_schedule"]),
            optimizer=OptimizerConfig._from_dict(payload["optimizer"]),
            diagnostics=DiagnosticsConfig._from_dict(payload["diagnostics"]),
        )


DEFAULT_US_CONFIG = UsRunConfig(
    seq_length=128,
    text_key="headline_with_lead",
    target_dtype="float32",
    head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=768),
    epochs=7,
    backbone_weights_path=str(config.DAPT_BACKBONE_WEIGHTS),
    lr_schedule=LRScheduleConfig(),
    optimizer=OptimizerConfig(),
)
```
Note: `from_json` delegates to the verified `cca_config` sub-config deserializers (`LRScheduleConfig._from_dict` reconstructs the nested `ResolvedSteps`; `OptimizerConfig._from_dict`; `DiagnosticsConfig._from_dict` coerces JSON arrays back to tuples — all confirmed present at `src/cca_config.py:400,455,543`). This round-trips a sidecar written *after* `with_resolved` (populated `resolved`) without loss.

**Testing** (`tests/test_us_config.py`), mirroring `tests/test_cca_config.py`:
- Construction + each `__post_init__` validation (bad name/dim/epochs raise).
- **AC3.4**: assert `not hasattr(DEFAULT_US_CONFIG.head, "prior")` and there is no `prior`/FLPU field anywhere in the dataclass tree (walk `dataclasses.fields`).
- `enable_loss_components=True` raises in `UsRunConfig.__post_init__`.
- `label_keys == {"us_targets": "us_label"}`; `expected_columns == {"headline_with_lead", "us_label"}`.
- Sidecar round-trip: `to_json` → `from_json` reproduces an equal config — **including a populated `resolved`**: call `dataclasses.replace(cfg, lr_schedule=cfg.lr_schedule.with_resolved(100))` before `to_json`, then assert the reloaded config's `lr_schedule.resolved` equals the original `ResolvedSteps` (exercises the nested-deserialization path, AC3.5).

**Verification:** `uv run pytest tests/test_us_config.py` → all pass.
**Commit:** `feat(us-filter): UsRunConfig (BCE, no-prior parallel of RunConfig)`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `src/us_metrics.py` — make_us_metrics

**Verifies:** supports AC3.1/AC3.2 reporting.

**Files:**
- Create: `src/us_metrics.py`
- Create: `tests/test_us_metrics.py`

**Implementation:**

```python
# pattern: Functional Core
import keras


def make_us_metrics() -> list[keras.metrics.Metric]:
    """Canonical binary-classification metrics for the US head (logits space).
    Same set as the CCA head: thresholds at 0.0 because outputs are logits.
    F1 is computed post-hoc from precision/recall (Keras F1 needs prob thresholds).
    """
    return [
        keras.metrics.BinaryAccuracy(threshold=0.0),
        keras.metrics.Precision(thresholds=0.0, name="precision"),
        keras.metrics.Recall(thresholds=0.0, name="recall"),
        keras.metrics.AUC(curve="PR", from_logits=True, name="pr_auc"),
    ]
```

**Testing:** returns 4 fresh, independent metric instances (two calls yield distinct objects); names as expected.
**Verification:** `uv run pytest tests/test_us_metrics.py` → pass.
**Commit:** `feat(us-filter): us head metric set`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `create_us_filter_data` — stratified PN split

**Verifies:** us-filter.AC3.5 (deterministic split), class stratification, null-drop.

**Files:**
- Modify: `src/data_setup/data.py` (add `create_us_filter_data`)
- Create: `tests/test_us_data_splits.py`

**Implementation:** add to `src/data_setup/data.py`:

```python
def create_us_filter_data(dataset):
    """Stratified 90/5/5 train/val/test split for the US filter (PN task).

    Drops rows with null `us_label` (unresolved/conflict), then splits the
    `us_label=True` and `us_label=False` groups separately (seed=200) and
    concatenates, so class proportions are stable across splits. Returns
    {"train":..., "val":..., "test":...} polars DataFrames.
    """
    data = dataset.filter(pl.col("us_label").is_not_null())
    assert data["id"].n_unique() == data.shape[0], (
        f"`id` not unique: {data.shape[0]} rows, {data['id'].n_unique()} ids"
    )

    def _split(group):
        train = group.sample(fraction=0.9, seed=200)
        rest = group.filter(pl.col("id").is_in(train["id"].implode()).not_())
        test = rest.sample(fraction=0.5, seed=200)
        val = rest.filter(pl.col("id").is_in(test["id"].implode()).not_())
        return train, val, test

    pos = data.filter(pl.col("us_label"))
    neg = data.filter(pl.col("us_label").not_())
    p_tr, p_va, p_te = _split(pos)
    n_tr, n_va, n_te = _split(neg)
    return {
        "train": pl.concat([p_tr, n_tr]),
        "val": pl.concat([p_va, n_va]),
        "test": pl.concat([p_te, n_te]),
    }
```
Note: `us_label` arrives from parquet as boolean; ensure the model target column is the integer/float cast — the preprocessor casts `label_keys` source columns to `target_dtype`, so pass `us_label` as the source column and let the preprocessor cast. (If polars boolean → tf needs int, cast `us_label` to `pl.Int8` when building the tf.data dict in Task 4.)

**Testing** (`tests/test_us_data_splits.py`), mirroring `tests/test_data_splits.py`:
- **AC3.5**: two calls on the same input yield identical split membership (by `id` sets).
- Null `us_label` rows are dropped from all splits.
- 90/5/5 proportions (within rounding) hold within each class; both classes present in each split.
- `id` uniqueness assertion fires on duplicated ids.

**Verification:** `uv run pytest tests/test_us_data_splits.py` → pass.
**Commit:** `feat(us-filter): stratified PN train/val/test split`
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->

<!-- START_TASK_4 -->
### Task 4: `src/run_us_classification.py` — training shell

**Verifies:** us-filter.AC3.4 (BCE in path, no FLPU), us-filter.AC3.2 (distribution metrics present). Operational AC3.1 path via Task 5.

**Files:**
- Create: `src/run_us_classification.py`
- Create: `tests/test_run_us_classification.py` (targeted unit tests; no full train)
- Modify: `src/config.py` (add `US_FILTER_SET_DIR`, `US_FILTER_CLASSIFIER_DIR`, `US_FILTER_CLASSIFIER_WEIGHTS`, `US_FILTER_LOGS_DIR`)

**Implementation** (`# pattern: Imperative Shell`). Parallels `run_cca_classification.main` with the US-specific differences. Importing must not train (`if __name__ == "__main__": main()`):

```python
import dataclasses, math, datetime
import keras
import tensorflow as tf
import polars as pl

from src import config
from src import us_config
from src.us_config import config_path_for_weights  # re-exported from cca_config
import src.data_setup.data
from src.preproc.preprocessor import ClassifierPreprocessor
from src.model_setup.backbone import load_dapt_backbone
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_endpoint_model, build_inference_model
from src.us_metrics import make_us_metrics
from src.diagnostics.distribution_metrics import make_distribution_metrics
from src.preproc.dateline_guard import assert_no_dateline_residue


def main(run_config=None, max_steps=None):
    keras.config.set_dtype_policy(config.DTYPE_POLICY)
    keras.utils.set_random_seed(200)
    if run_config is None:
        run_config = us_config.DEFAULT_US_CONFIG
    head_cfg = run_config.head
    BATCH_SIZE = 256
    SHUFFLE_BUFFER = 100_000

    # ----- data load + split + guard -----
    df = src.data_setup.data.data_from_parquet(
        config.PROJECT_ROOT, "us_filter",
        addl_columns=["us_label", "label_source"],
        lead_column="stripped_text",
    )
    splits = src.data_setup.data.create_us_filter_data(df)
    for name, sdf in splits.items():
        assert_no_dateline_residue(sdf["stripped_text"])  # runtime AC2.2 guard

    # ----- tf.data (cache; natural-balance single stream) -----
    # NOTE: the cache under US_FILTER_SET_DIR is keyed only by split name, not by
    # label content. If the labeled parquet changes upstream (e.g. Phase 1
    # gazetteers are finalized / re-run), DELETE US_FILTER_SET_DIR so it rebuilds —
    # otherwise the loaded data and the recomputed steps below silently diverge.
    if not config.US_FILTER_SET_DIR.is_dir():
        config.US_FILTER_SET_DIR.mkdir(parents=True)
        for name, sdf in splits.items():
            ds = tf.data.Dataset.from_tensor_slices({
                "headline_with_lead": sdf["headline_with_lead"].to_list(),
                "us_label": sdf["us_label"].cast(pl.Int8).to_numpy(),
            })
            ds.save(str(config.US_FILTER_SET_DIR / f"{name}.tf"))
    datasets = {n: tf.data.Dataset.load(str(config.US_FILTER_SET_DIR / f"{n}.tf"))
                for n in ("train", "val", "test")}
    # Cache/freshness sanity check: cached cardinality must match the freshly
    # computed split sizes; a mismatch means a stale cache vs. current labels.
    for n in ("train", "val", "test"):
        cached_n = int(datasets[n].cardinality().numpy())
        if cached_n != splits[n].shape[0]:
            raise ValueError(
                f"Stale US set cache for split {n!r}: cache has {cached_n} rows "
                f"but current split has {splits[n].shape[0]}. Delete "
                f"{config.US_FILTER_SET_DIR} and re-run."
            )

    # ----- preprocessor (endpoint mode, single head) -----
    train_preprocess = ClassifierPreprocessor(
        SEQ_LENGTH=run_config.seq_length, text_key=run_config.text_key,
        label_keys=run_config.label_keys, endpoint_model=True,
        target_dtype=run_config.target_dtype,
    )

    train_size = splits["train"].shape[0]
    val_size = splits["val"].shape[0]
    steps_per_epoch = math.floor(train_size / BATCH_SIZE)
    validation_steps = math.floor(val_size / BATCH_SIZE)
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
        validation_steps = min(validation_steps, max(1, max_steps // 5))
    run_config = dataclasses.replace(
        run_config, lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch)
    )

    training_set = src.data_setup.data.dataset_create(
        SHUFFLE_BUFFER, BATCH_SIZE, train_preprocess, data=datasets["train"]
    )
    validation_set = src.data_setup.data.dataset_create(
        SHUFFLE_BUFFER, BATCH_SIZE, train_preprocess, data=datasets["val"]
    )

    # ----- model (BCE via endpoint add_loss path) -----
    backbone = load_dapt_backbone(run_config.backbone_weights_path)
    run_config.validate_against_backbone(backbone)
    us_head = ClassificationHead(
        hidden_dim=head_cfg.hidden_dim,
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=make_us_metrics() + make_distribution_metrics(run_config.diagnostics),
        name=head_cfg.name,
        expose_loss_components=False,
    )
    us_model = build_endpoint_model(
        backbone=backbone, heads={head_cfg.name: us_head},
        seq_length=run_config.seq_length, freeze_encoder=True,
        diagnostics=run_config.diagnostics,
    )
    us_inference = build_inference_model(
        backbone=backbone, heads={head_cfg.name: us_head},
        seq_length=run_config.seq_length,
    )

    resolved = run_config.lr_schedule.resolved
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=run_config.lr_schedule.initial_lr,
        decay_steps=resolved.decay_steps, alpha=run_config.lr_schedule.decay_alpha,
        warmup_target=run_config.lr_schedule.warmup_target,
        warmup_steps=resolved.warmup_steps,
    )
    base_opt = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=run_config.optimizer.weight_decay)
    optimizer = keras.optimizers.LossScaleOptimizer(base_opt) if config.IS_CLUSTER else base_opt
    us_model.compile(optimizer=optimizer, jit_compile="auto")  # NO loss — head add_loss

    # ----- callbacks, fit, save sidecar -----
    config.US_FILTER_CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    config.US_FILTER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.US_FILTER_CLASSIFIER_DIR / f"{stamp}_checkpoint.weights.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True),
        keras.callbacks.CSVLogger(str(config.US_FILTER_LOGS_DIR / f"{stamp}_metrics.csv")),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, start_from_epoch=2, verbose=1),
    ]
    us_model.fit(training_set, validation_data=validation_set, epochs=run_config.epochs,
                 steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                 callbacks=callbacks_list)

    us_model.save_weights(str(config.US_FILTER_CLASSIFIER_WEIGHTS))
    run_config.to_json(config_path_for_weights(config.US_FILTER_CLASSIFIER_WEIGHTS))

    # ----- in-distribution test eval: P/R/PR-AUC + F1 + majority baseline -----
    test_set = (datasets["test"].batch(BATCH_SIZE, drop_remainder=False)
                .map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    results = us_model.evaluate(test_set, return_dict=True)
    # Metric keys are head-prefixed (head name "us"); confirm exact keys from
    # results.keys() at first run. Use explicit None checks so a missing key
    # (absent metric) is distinguishable from a legitimately-zero precision/recall.
    p = results.get("us_precision")
    r = results.get("us_recall")
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0.0
    maj = max(splits["test"]["us_label"].mean(), 1 - splits["test"]["us_label"].mean())
    print(f"US test: P={p} R={r} F1={f1} | majority-class acc baseline={maj}")
    return us_model, us_inference, results


if __name__ == "__main__":
    main()
```
Notes for the implementer: confirm the head-prefixed metric key names from `evaluate(return_dict=True)` at first run (the head name is `"us"`, so keys are `us_precision`/`us_recall`/`us_pr_auc`/`us_binary_accuracy`); adjust the F1 key lookups if they differ. Mirror `run_cca_classification.py`'s exact callback/TensorBoard setup if desired.

**Config additions** (`src/config.py`):
```python
US_FILTER_SET_DIR: Path = US_FILTER_DIR / "us_set"
US_FILTER_CLASSIFIER_DIR: Path = US_FILTER_DIR / "classifier"
US_FILTER_CLASSIFIER_WEIGHTS: Path = US_FILTER_DIR / "us_classifier.weights.h5"
US_FILTER_LOGS_DIR: Path = US_FILTER_DIR / "logs"
```

**Testing** (`tests/test_run_us_classification.py`) — targeted, no full train (use the fake-backbone pattern from `tests/test_assembly.py` where a real backbone is too heavy):
- **AC3.4**: build the US head + endpoint model as `main` does (with a fake/tiny backbone) and assert the head's `loss_fn` is a `keras.losses.BinaryCrossentropy` with `from_logits=True`, and that no FLPU loss / prior is present in the model or config.
- **AC3.2**: assert the assembled US head's metric set includes the prediction-distribution metrics by their **actual head-prefixed names** — `us_pred_dist/mean`, `us_pred_dist/std`, `us_pred_dist/frac_above_0.5` (the base names are `pred_dist/mean` etc. from `distribution_metrics.py`, head-prefixed by `ClassificationHead`). Assert against these exact names (not substrings) so the test genuinely confirms they populate for train and val.
- Importing `src.run_us_classification` does not trigger training (no side effects at import).

**Verification:** `uv run pytest tests/test_run_us_classification.py` → pass.
**Commit:** `feat(us-filter): BCE-endpoint training script + config paths`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: `scripts/us_short_run.py` — capped operational shakedown

**Files:**
- Create: `scripts/us_short_run.py`

**Implementation:** mirror `scripts/tier5_short_run.py` — call `src.run_us_classification.main(run_config=dataclasses.replace(DEFAULT_US_CONFIG, epochs=1), max_steps=200)` for a reproducible short run.

**Verification (operational):**
Run: `uv run python scripts/us_short_run.py`
Expected: trains end-to-end without error; prints per-epoch metrics including the prediction-distribution metrics for **train and val** (AC3.2); the diagnostic trackers fire (grad-overflow / batch-balance appear in logs); a test-eval line prints P/R/F1 and the majority baseline. (Under local float32 the overflow proxy is ≈0; the cluster `mixed_float16` run is the AC7.3 cluster shakedown — operator-run.)

**Commit:** `feat(us-filter): short-run shakedown wrapper`
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->

---

## Phase 4 Done When

- `UsRunConfig` constructs, validates (no prior field), and round-trips through its JSON sidecar.
- `create_us_filter_data` produces a deterministic, stratified, null-dropped split.
- The training script builds the US head with **BCE on the endpoint path** (no FLPU/prior), assembles via `build_endpoint_model` with `freeze_encoder=True`, and the short capped run completes end-to-end with metrics + diagnostics firing for train and val.
- Config exposes the US artifact paths.

Covers **us-filter.AC3** (in-distribution: AC3.2, AC3.4, AC3.5; AC3.1 path). AC3.1's F1-beats-baseline bar and the cluster `mixed_float16` shakedown (AC7.3) are operator-run full executions, flagged but not automated here. AC3.3 (transfer) is Phase 6.
