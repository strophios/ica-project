# Tier 5 Implementation Plan — Phase 6: Assembly Wiring and Smoke Test

**Goal:** Wire diagnostics end-to-end: per-step trackers through `build_endpoint_model` → `LayerLRModel`, per-head distribution metrics through the head's `metrics=` list, a `CSVLogger` into the training script, and a diagnostics-enabled smoke test that asserts the diagnostic columns appear and the save/load round-trip reinitializes tracker state cleanly.

**Architecture:** `build_endpoint_model` gains an optional `diagnostics` param. When set, it gathers the **constituent layers'** trainable variables (`backbone` + each `head`) — realized after the head calls, *after* the `freeze_encoder` block — calls `build_trackers` for group enumeration, and passes `diagnostic_trackers` + `diagnostic_head_refs` into `LayerLRModel.__init__` (no Phase 4 change; runtime dispatch still uses the model's own authoritative `self.trainable_variables`). Distribution metrics ride the head's `metrics=` list (Phase 5), constructed at the caller's head-construction site. `build_inference_model` is unchanged (diagnostic-free; inference heads get no targets so distribution metrics stay inert).

**Tech Stack:** Python ≥3.12, Keras 3.12, TensorFlow, pytest. No new src files → modified files only (`assembly.py`, `run_cca_classification.py`, `scripts/smoke_test_integrated_stack.py`) get **no** FCIS comment (existing files).

**Scope:** Phase 6 of 8. Depends on Phases 1–5 (trackers, factory, head contract, train-step dispatch, distribution metrics).

**Codebase verified:** 2026-05-17 via codebase-investigator (verbatim, line-numbered).

**Codebase verification findings:**
- `build_endpoint_model(backbone, heads, seq_length, target_dtype="float32", freeze_encoder=False, layer_multipliers=None, group_fn=None)` (`assembly.py:68-76`). Internal order: `if freeze_encoder: backbone.trainable = False` (**lines 148-149**); `backbone_out = backbone({...})` (163); `cls_features = backbone_out[:, 0, :]` (164); head-call loop `outputs[head_name] = head(cls_features, targets=target_inputs[f"{head_name}_targets"])` (167-170); `return LayerLRModel(inputs=all_inputs, outputs=outputs, group_fn=group_fn, multipliers=layer_multipliers or {})` (174-179).
- **Trainable-vars timing (empirically confirmed):** after the head-call loop and the freeze block, `backbone.trainable_variables` is `[]` when `freeze_encoder=True` (else backbone vars) and each `head.trainable_variables` is populated. This is the correct gather point — *after* line 170 and the freeze block, *before* line 174.
- Heads are constructed by the **caller**:
  - `run_cca_classification.py:260-268`: `ClassificationHead(hidden_dim=_cca_head_config.hidden_dim, loss_fn=FLPULoss(prior=_cca_head_config.loss.prior, kiryo_clawback=_cca_head_config.loss.kiryo_clawback), metrics=make_cca_metrics(), name=_cca_head_config.name)`; `build_endpoint_model(backbone=backbone, heads={_cca_head_config.name: cca_head}, seq_length=run_config.seq_length, freeze_encoder=True)` (275-285); `cca_classifier.compile(optimizer=optimizer, jit_compile="auto")` (326); callbacks list `[ModelCheckpoint, TensorBoard(update_freq="epoch"), EarlyStopping]` (336-356) — **no CSVLogger**.
  - `scripts/smoke_test_integrated_stack.py`: synthetic `RunConfig` (98-115; not `DEFAULT_CCA_CONFIG`); head construction (230-241) with a hand-built `metrics=[BinaryAccuracy, Precision]`; `train_model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3)); train_model.fit(training_set, epochs=1, steps_per_epoch=4, verbose=1)` (259-266); assertions (366-401): pred shape, Pattern-A vs Pattern-2 max-diff < 1e-4, finite preds, backbone-weight-norm match. No callbacks currently.
- `build_inference_model(backbone, heads, seq_length)` calls `head(cls_features)` with no targets (no metric/loss update). Unchanged in Phase 6.
- `_default_group_fn` (`assembly.py:54-65`): `return variable.path.split("/")[0]`.
- `tests/test_assembly.py`: `TestBuildEndpointModel`@120-199 (pattern: `model = build_endpoint_model(backbone=fresh_backbone, heads={"cca": fresh_head}, seq_length=SEQ_LEN)`), `TestFreezeEncoder`@324-345, `fresh_backbone`/`fresh_head` fixtures. Grep: no `build_trackers`/`make_distribution_metrics`/`CSVLogger` anywhere — greenfield.
- `RunConfig` (post-Phase-2) has `diagnostics: DiagnosticsConfig` via `default_factory`, so `run_config.diagnostics` always exists (all-enabled by default), including in the smoke test's hand-built `RunConfig`.

**Critical executor invariant (user-flagged):** the constituent-variable gather **must** be placed after the `if freeze_encoder: backbone.trainable = False` block *and* after the head-call loop. If gathered earlier, `backbone.trainable_variables` would still contain encoder vars and `build_trackers` would enumerate a spurious encoder group → an unwanted `grad_norm/<backbone>/...` tracker. Task 1 includes a test that fails if this ordering is violated.

**Task structure (6 tasks):**
- A — assembly wiring (Task 1)
- B — training-script wiring (Tasks 2–3)
- C — smoke test (Tasks 4–5)
- Phase integration verification (Task 6)

---

<!-- START_SUBCOMPONENT_A (task 1) -->
<!-- START_TASK_1 -->
### Task 1: build_endpoint_model — diagnostics param + constituent-var gather + bundle passthrough

**Files:**
- Modify: `src/model_setup/assembly.py`
- Modify: `tests/test_assembly.py` (new class `TestEndpointDiagnosticsWiring`)

**Step 1: Write the failing tests**

Append to `tests/test_assembly.py` (reuse `fresh_backbone`, `fresh_head` fixtures, `SEQ_LEN`; import `from src.cca_config import DiagnosticsConfig`):

```python
class TestEndpointDiagnosticsWiring:
    def test_diagnostics_none_is_backcompat(self, fresh_backbone, fresh_head):
        model = build_endpoint_model(
            backbone=fresh_backbone, heads={"cca": fresh_head}, seq_length=SEQ_LEN
        )
        assert model._diagnostic_trackers is None
        assert model._head_refs_by_name == {}

    def test_diagnostics_wires_bundle_and_head_refs(self, fresh_backbone, fresh_head):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
            freeze_encoder=True,
            diagnostics=DiagnosticsConfig(),
        )
        assert model._diagnostic_trackers is not None
        assert set(model._diagnostic_trackers["per_step"].keys()) == {
            "gradient", "loss_component", "batch_target"
        }
        assert model._diagnostic_trackers["periodic"] == []
        assert "cca" in model._head_refs_by_name

    def test_frozen_encoder_no_backbone_grad_tracker(self, fresh_backbone, fresh_head):
        # USER-FLAGGED INVARIANT: the constituent-var gather must run AFTER
        # the freeze_encoder block. With freeze_encoder=True the only
        # trainable group is the head ("cca"); no backbone/encoder
        # grad-norm tracker may be built. If this fails, the gather was
        # placed before backbone.trainable=False.
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
            freeze_encoder=True,
            diagnostics=DiagnosticsConfig(),
        )
        grad_names = [
            t.name for t in model._diagnostic_trackers["per_step"]["gradient"]
        ]
        assert any(n.startswith("grad_norm/cca/") for n in grad_names)
        assert not any(
            n.startswith("grad_norm/") and "/cca/" not in n for n in grad_names
        ), f"unexpected non-cca grad-norm tracker(s): {grad_names}"

    def test_unfrozen_encoder_includes_backbone_group(self, fresh_backbone, fresh_head):
        model = build_endpoint_model(
            backbone=fresh_backbone,
            heads={"cca": fresh_head},
            seq_length=SEQ_LEN,
            freeze_encoder=False,
            diagnostics=DiagnosticsConfig(),
        )
        grad_names = [
            t.name for t in model._diagnostic_trackers["per_step"]["gradient"]
        ]
        # At least one non-cca (backbone) grad-norm group present.
        assert any(
            n.startswith("grad_norm/") and "/cca/" not in n for n in grad_names
        )

    def test_inference_model_unaffected(self, fresh_backbone, fresh_head):
        inf = build_inference_model(
            backbone=fresh_backbone, heads={"cca": fresh_head}, seq_length=SEQ_LEN
        )
        assert not hasattr(inf, "_diagnostic_trackers") or \
            getattr(inf, "_diagnostic_trackers", None) is None
```

> Executor note: `fresh_head` is a `ClassificationHead` without
> `expose_loss_components`; `build_trackers` with `enable_loss_components=True`
> will introspect `fresh_head.loss_fn`. If `fresh_head` has an `FLPULoss`
> (Phase 3 adds `return_intermediates`), loss-component trackers build. If
> the fixture's loss lacks it, the Phase-2 guard raises — in that case pass
> `diagnostics=DiagnosticsConfig(enable_loss_components=False)` in the
> bundle-wiring tests, or ensure the fixture uses an FLPULoss. Decide based
> on the actual `fresh_head` fixture loss; document the choice in the test.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_assembly.py::TestEndpointDiagnosticsWiring -v`
Expected: fail — `build_endpoint_model` has no `diagnostics` param (`TypeError: unexpected keyword argument 'diagnostics'`).

**Step 3: Write minimal implementation**

Read `src/model_setup/assembly.py:build_endpoint_model` (lines 68-179) first. Make three changes:

(a) Add the param to the signature (after `group_fn=None`):

```python
def build_endpoint_model(
    backbone,
    heads,
    seq_length,
    target_dtype="float32",
    freeze_encoder=False,
    layer_multipliers=None,
    group_fn=None,
    diagnostics=None,
):
```

(b) Add imports at the top of `assembly.py` (with the other `src.*` imports):

```python
from src.diagnostics.factory import build_trackers
```

(c) Immediately **before** the `return LayerLRModel(...)` (currently lines 174-179) — i.e., after the head-call loop AND after the `if freeze_encoder: backbone.trainable = False` block — insert the gather + bundle build, and thread the new kwargs into the `LayerLRModel(...)` call:

```python
    # Tier 5 diagnostics. The constituent-variable gather MUST happen here:
    # after the head-call loop (so head/backbone variables are realized) AND
    # after the freeze_encoder block above (so backbone.trainable is already
    # False and frozen-encoder builds enumerate only head groups). build_trackers
    # uses this list ONLY for group enumeration; train_step's runtime dispatch
    # uses the model's own self.trainable_variables.
    diagnostic_trackers = None
    diagnostic_head_refs = None
    if diagnostics is not None:
        constituent_trainable = list(backbone.trainable_variables)
        for _h in heads.values():
            constituent_trainable.extend(_h.trainable_variables)
        diagnostic_trackers = build_trackers(
            diagnostics,
            group_fn=group_fn or _default_group_fn,
            heads=heads,
            trainable_variables=constituent_trainable,
        )
        diagnostic_head_refs = list(heads.values())

    return LayerLRModel(
        inputs=all_inputs,
        outputs=outputs,
        group_fn=group_fn or _default_group_fn,
        multipliers=layer_multipliers or {},
        diagnostic_trackers=diagnostic_trackers,
        diagnostic_head_refs=diagnostic_head_refs,
    )
```

> Executor note: match the exact existing `LayerLRModel(...)` kwargs
> (`group_fn`, `multipliers`) as currently written in lines 174-179; only
> ADD the two diagnostic kwargs. Do not change `inputs`/`outputs`/existing
> `group_fn`/`multipliers` expressions. If the current call passes
> `group_fn=group_fn or _default_group_fn` already, keep it; ensure the
> gather uses the same resolved group_fn the model will use.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_assembly.py::TestEndpointDiagnosticsWiring -v`
Expected: 5 passed (esp. `test_frozen_encoder_no_backbone_grad_tracker`).

Run: `pytest tests/test_assembly.py -q`
Expected: all existing assembly tests still pass (diagnostics defaults to None → behavior unchanged) + 5 new.

Run: `pytest -q`
Expected: Phase-5 total + 5, zero failures.

**Step 5: Commit**

```bash
git add src/model_setup/assembly.py tests/test_assembly.py
git commit -m "tier5 phase 6: build_endpoint_model diagnostics wiring (constituent-var gather post-freeze)"
```
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2-3) -->
<!-- START_TASK_2 -->
### Task 2: run_cca_classification.py — head metrics + expose_loss_components + diagnostics passthrough

**Files:**
- Modify: `src/run_cca_classification.py`

**Type:** Infrastructure (operational wiring; verified by Task 6's run + the smoke test in Task 4). No unit test — this is script glue exercised end-to-end.

**Step 1: Make the changes**

Read `src/run_cca_classification.py` around lines 260-285 first.

(a) Add imports (with existing `from src.cca_metrics import make_cca_metrics` / `from src.model_setup...` imports):

```python
from src.diagnostics.distribution_metrics import make_distribution_metrics
```

(b) The head construction (currently ~260-268) — add `expose_loss_components` and combine metrics. `run_config` is already in scope (it builds from `DEFAULT_CCA_CONFIG`):

```python
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
```

(c) The `build_endpoint_model(...)` call (currently ~275-285) — pass `diagnostics`:

```python
    cca_classifier = build_endpoint_model(
        backbone=backbone,
        heads={_cca_head_config.name: cca_head},
        seq_length=run_config.seq_length,
        freeze_encoder=True,
        diagnostics=run_config.diagnostics,
    )
```

> Executor note: preserve every other existing argument exactly as written
> (do not drop `freeze_encoder=True` or alter the heads dict). `_cca_head_config`
> and `run_config` are existing locals — confirm their names by reading the
> surrounding code; adapt if the script uses different identifiers.

**Step 2: Verify operationally**

Run: `python -c "import ast; ast.parse(open('src/run_cca_classification.py').read()); print('parse-ok')"`
Expected: `parse-ok` (syntactic check; full run is Task 6 / Phase 7).

Run: `pytest -q`
Expected: unchanged total (no tests target the script directly; nothing broke on import paths). Zero failures.

**Step 3: Commit**

```bash
git add src/run_cca_classification.py
git commit -m "tier5 phase 6: wire distribution metrics + expose_loss_components + diagnostics into training script"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: run_cca_classification.py — add CSVLogger to callbacks

**Files:**
- Modify: `src/run_cca_classification.py`

**Type:** Infrastructure.

**Step 1: Make the change**

In the callbacks list (currently ~336-356), add a `CSVLogger` **before** the `TensorBoard` entry (ordering: a custom/log-producing callback must precede loggers; the per-step trackers reach `logs` via Keras's metrics pipeline before any callback runs, but keeping `CSVLogger` early and consistent with TensorBoard avoids surprises). Use a per-run path mirroring the TensorBoard `log_dir` convention:

```python
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
```

> Executor note: `config.CCA_LOGS_DIR / _run_stamp` is the existing
> TensorBoard dir; `CSVLogger` needs its parent dir to exist. Keras's
> `CSVLogger` does not create parent dirs — add `(config.CCA_LOGS_DIR /
> _run_stamp).mkdir(parents=True, exist_ok=True)` immediately before the
> callbacks list if the script does not already create that directory
> (check the surrounding code; the TensorBoard callback may create it
> lazily but CSVLogger will not). Match `_run_stamp`'s actual identifier.

**Step 2: Verify operationally**

Run: `python -c "import ast; ast.parse(open('src/run_cca_classification.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

Run: `pytest -q` — unchanged total, zero failures.

**Step 3: Commit**

```bash
git add src/run_cca_classification.py
git commit -m "tier5 phase 6: add CSVLogger to training callbacks (flat diagnostic CSV artifact)"
```
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 4-5) -->
<!-- START_TASK_4 -->
### Task 4: Smoke test — diagnostics-enabled run + CSVLogger + column assertions

**Files:**
- Modify: `scripts/smoke_test_integrated_stack.py`

**Type:** Infrastructure (operational verification script).

**Step 1: Make the changes**

Read `scripts/smoke_test_integrated_stack.py` (RunConfig 98-115, head construction 230-241, fit 259-266, assertions 366-401) first.

(a) Imports: add `from src.diagnostics.distribution_metrics import make_distribution_metrics`. `run_config.diagnostics` already exists (Phase 2 `default_factory` → all-enabled `DiagnosticsConfig()`), so the synthetic `RunConfig` needs no change for diagnostics to be on.

(b) Head construction (~230-241): replace the hand-built `metrics=[...]` with the combined list and add `expose_loss_components`:

```python
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
```
(Add `from src.cca_metrics import make_cca_metrics` if not already imported.)

(c) The `build_endpoint_model(...)` call in the smoke test: add `diagnostics=run_config.diagnostics`.

(d) The fit (~259-266): add a `CSVLogger` to a temp path and pass `callbacks=`:

```python
    csv_path = tmp_dir / "smoke_metrics.csv"
    train_model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3))
    train_model.fit(
        training_set,
        epochs=1,
        steps_per_epoch=4,
        verbose=1,
        callbacks=[keras.callbacks.CSVLogger(str(csv_path))],
    )
```
(`tmp_dir` is the smoke test's existing temp directory — confirm its identifier.)

(e) After fit, before/with the existing assertions (~366), add diagnostic-column checks:

```python
    import csv as _csv
    with open(csv_path, newline="") as f:
        header = next(_csv.reader(f))
    # Per-step trackers (via LayerLRModel.metrics property, Phase 4):
    assert any(h.startswith("grad_norm/cca/") for h in header), header
    assert "grad_overflow_rate" in header, header
    assert any(h.startswith("cca/") and h.endswith("/mean") for h in header), header
    assert "cca/positive_fraction" in header, header
    # Per-head distribution metrics (via head metric_objs, Phase 5):
    assert "cca_pred_dist/mean" in header, header
    assert "cca_pred_dist/std" in header, header
    assert "cca_pred_dist/frac_above_0.5" in header, header
    print("[OK] diagnostic CSV columns present:", sorted(header))
```

> Executor note: the loss-component column name is `cca/<key>/mean` (e.g.
> `cca/positive_risk/mean`) per Phase 1's `LossComponentTracker` name
> pattern; the substring assertion `h.startswith("cca/") and
> h.endswith("/mean")` tolerates the exact key. The smoke test fits without
> `validation_data`, so only train-side columns appear (no `val_*`). Keep
> ALL existing assertions (shape, Pattern-A/2 max-diff, finite, backbone
> norm) — append, don't replace.

**Step 2: Verify operationally**

Run: `python scripts/smoke_test_integrated_stack.py`
Expected: completes; prints `[OK] diagnostic CSV columns present: [...]`; all original assertions still pass; exit 0.

**Step 3: Commit**

```bash
git add scripts/smoke_test_integrated_stack.py
git commit -m "tier5 phase 6: smoke test exercises diagnostics + asserts CSV columns"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Smoke test — save/load round-trip leaves trackers reinitialized

**Files:**
- Modify: `scripts/smoke_test_integrated_stack.py`

**Type:** Infrastructure. Validates the design's Phase 6 "Done when": save (weights+sidecar) → load → predict leaves diagnostic tracker state cleanly reinitialized (trackers are `Metric` state, not `Layer` weights, so they are NOT in `.weights.h5`; a freshly loaded model has zeroed/empty trackers until training resets them).

**Step 1: Make the change**

The smoke test already exercises a Pattern-2 reload (it asserts Pattern-A vs Pattern-2 max-diff and backbone-norm match). Extend the post-reload section: build the reloaded endpoint model with `diagnostics=run_config.diagnostics` and assert its tracker state is at initialization (result == 0.0 before any training), and that a single `train_on_batch`/one fit step then populates them — proving load doesn't carry stale tracker state and trackers re-engage cleanly.

```python
    # Reloaded endpoint model: trackers exist but are at init (Metric state
    # is not persisted in .weights.h5).
    reloaded_endpoint = build_endpoint_model(
        backbone=reloaded_backbone,
        heads={_cca_head_config.name: reloaded_head},
        seq_length=run_config.seq_length,
        freeze_encoder=True,
        diagnostics=run_config.diagnostics,
    )
    for _t in reloaded_endpoint._diagnostic_trackers["per_step"]["gradient"]:
        assert float(_t.result()) == 0.0, f"{_t.name} not reinitialized after load"
    print("[OK] diagnostic trackers reinitialized cleanly after load")
```

> Executor note: reuse whatever reloaded backbone/head identifiers the
> existing Pattern-2 section already constructs (the smoke test already does
> a cross-process-style reload — extend that block; do not build a second
> independent reload pipeline). If the existing reload only builds an
> inference model, add a minimal endpoint rebuild solely for this assertion.

**Step 2: Verify operationally**

Run: `python scripts/smoke_test_integrated_stack.py`
Expected: completes; prints both `[OK]` lines; exit 0.

**Step 3: Commit**

```bash
git add scripts/smoke_test_integrated_stack.py
git commit -m "tier5 phase 6: smoke test asserts trackers reinitialize cleanly after load"
```
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_C -->

<!-- START_TASK_6 -->
### Task 6: Phase 6 integration verification

**Type:** Verification (no new code; no commit).

**Step 1: Full suite, no regressions**

Run: `pytest -q`
Expected: Phase-5 total + Task-1's 5 assembly tests, zero failures. Confirm all pre-existing `tests/test_assembly.py` (incl. `TestFreezeEncoder`, `TestPatternAWeightSharing`, `TestPatternTwoSerialization`) green — `diagnostics=None` default keeps assembly behavior unchanged. Record the number (Phase 7 baseline).

**Step 2: Smoke test end-to-end**

Run: `python scripts/smoke_test_integrated_stack.py`
Expected: exit 0; prints the diagnostic-CSV-columns `[OK]` and the trackers-reinitialized `[OK]`; all original round-trip assertions still pass. This is the operational proof that the full stack (per-step trackers + distribution metrics + CSVLogger + save/load) works on synthetic data — the gate before Phase 7's real-data run.

**Step 3: Training-script static check**

Run: `python -c "import ast; ast.parse(open('src/run_cca_classification.py').read()); print('parse-ok')"`
Expected: `parse-ok`. (Full real-data execution is Phase 7.)

**Step 4: Working tree clean**

Run: `git status --short` → clean.
Run: `git log --oneline 9136195..HEAD` → Phase 1–5 commits + 5 Phase-6 task commits (Tasks 1-5; Task 6 no commit).

**Phase 6 Done-when criteria (from design plan, adjusted for the Phase-5 supersession):**
- Smoke test passes including diagnostic column verification. ✓ (Step 2, Task 4)
- Save/load round-trip leaves trackers reinitialized cleanly. ✓ (Step 2, Task 5)
- Assembly tests pass; `build_inference_model` does not wire diagnostics. ✓ (Step 1, Task 1)
- `run_cca_classification.py` parses and is wired (real-data run is Phase 7). ✓ (Step 3)
- No `DiagnosticsCallback` wired (superseded — distribution metrics ride the head metric path). ✓

Phase 6 is complete. The full diagnostic stack is assembled and synthetic-validated; Phase 7 runs it on real cached CCA data.
<!-- END_TASK_6 -->
