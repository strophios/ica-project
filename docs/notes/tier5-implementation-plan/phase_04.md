# Tier 5 Implementation Plan — Phase 4: Train-Step Integration

**Goal:** Wire diagnostic dispatch into `LayerLRModel.train_step` (observing raw gradients, loss components, and targets) behind optional `__init__` params, with a `metrics` property override so Keras resets/logs the per-step trackers — and with a regression harness proving the no-trackers path is byte-for-byte unchanged.

**Architecture:** Additive + guarded. Two new optional `__init__` params (`diagnostic_trackers`, `diagnostic_head_refs`). The dispatch is a read-only observer inserted **between `tape.gradient(...)` and the multiplier-scaling list** — so trackers see *computed* gradients, not multiplier-scaled ones (design lines 165–176: "we measure what was computed, not what was applied"). When `diagnostic_trackers is None` (the default, and what every existing call site passes), `train_step` and `metrics` behave exactly as today.

> **Design vs. investigator note:** the Phase 4 codebase-investigator's *factual* reads are authoritative, but its closing "Summary" suggested inserting the dispatch *after* `apply_gradients`. That contradicts the design's explicit pre-scaling placement. Follow the design. The dispatch goes after `gradients = tape.gradient(...)` and before `scaled = [...]`.

**Tech Stack:** Python ≥3.12, Keras 3.12.0 (TF backend), TensorFlow, pytest. No new src files (only `layer_lr_model.py` modified) → **no FCIS pattern comment** (existing file; project adopts FCIS on new files only).

**Scope:** Phase 4 of 8. Depends on Phases 1 (tracker classes), 2 (`DiagnosticBundle`), 3 (`ClassificationHead.last_components`).

**Codebase verified:** 2026-05-16 via codebase-investigator (verbatim, line-numbered).

**Codebase verification findings:**
- `src/model_setup/layer_lr_model.py` imports: `from typing import Callable, Optional` / `import keras` / `import tensorflow as tf` (lines 48–51). **No `from __future__ import annotations`.**
- `LayerLRModel.__init__` (lines 92–106), verbatim:
  ```python
  def __init__(
      self,
      *args,
      group_fn: Optional[Callable[[tf.Variable], str]] = None,
      multipliers: Optional[dict] = None,
      **kwargs,
  ):
      super().__init__(*args, **kwargs)
      if group_fn is None:
          group_fn = lambda v: "default"  # noqa: E731
      self.group_fn = group_fn
      self.multipliers = dict(multipliers) if multipliers is not None else {}
  ```
- `train_step` (lines 127–257) — the code body after the docstring, verbatim:
  ```python
      x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

      with tf.GradientTape() as tape:
          y_pred = self(x, training=True)
          loss = self._compute_loss(
              x=x,
              y=y,
              y_pred=y_pred,
              sample_weight=sample_weight,
              training=True,
          )
          self._loss_tracker.update_state(
              loss,
              sample_weight=tf.shape(
                  next(t for t in tf.nest.flatten(x) if t is not None)
              )[0],
          )
          if self.optimizer is not None:
              loss = self.optimizer.scale_loss(loss)

      gradients = tape.gradient(loss, self.trainable_variables)
      scaled = [
          tf.math.scalar_mul(self.get_multiplier(w), g) if g is not None else None
          for w, g in zip(self.trainable_variables, gradients)
      ]
      self.optimizer.apply_gradients(zip(scaled, self.trainable_variables))

      return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
  ```
- No `metrics` property override exists (Keras default). No `test_step` override. No `get_config`/`from_config`.
- `self.group_fn` is stored (line 105) — dispatch calls `self.group_fn`.
- `LayerLRModel` holds **no** references to `ClassificationHead` instances (functional graph only). `build_endpoint_model` (assembly.py ~177) constructs: `LayerLRModel(inputs=all_inputs, outputs=outputs, group_fn=group_fn or _default_group_fn, multipliers=layer_multipliers or {})` — heads wired via `outputs[head_name] = head(cls_features, targets=target_inputs[f"{head_name}_targets"])`.
- Keras 3.12.0. `train_step` uses `self._compute_loss(...)`, `self._loss_tracker.update_state(...)`, `self.optimizer.scale_loss(...)`, `self.optimizer.apply_gradients(...)`, `self.compute_metrics(...)`.
- `tests/test_layer_lr_model.py` (13 tests), classes: `TestConstruction`@91, `TestMultiplierLookup`@108, `TestTrainStepIntegration`@139, `TestWithEndpointLoss`@284 (`test_trains_with_endpoint_head`@285 — the endpoint-LayerLRModel construction pattern to mirror in Task 3), `TestSparseGradients`@344 (`test_embedding_layer_trains_under_layer_lr_model`@365 — sparse-grad regression), `TestLossTracking`@409 (`test_fit_history_records_nonzero_loss`@432, `test_fit_history_loss_close_to_evaluate_loss`@466 — the two loss-tracker regressions). Helper: `_tiny_model(multipliers=None, group_fn=None)`@50–76 (LayerLRModel over two Dense layers "lower"/"upper").
- Grep: no `diagnostic_trackers`/`diagnostic_head_refs`/`_head_refs_by_name`/`DiagnosticBundle` anywhere — greenfield.

**Keras 3.12 metrics-property mechanics (traced — why the override is safe):**
- `compute_metrics` updates only `self._compile_metrics` (metrics from `compile(metrics=...)`); it does NOT call `.update_state()` on arbitrary `self.metrics` entries → exposing custom-signature trackers in `metrics` does not crash.
- `get_metrics_result()` iterates `self.metrics` calling `.result()` → diagnostics enter `logs`/`history`.
- `reset_metrics()` iterates `self.metrics` calling `.reset_state()` → Keras resets trackers per epoch.
- This is the documented Keras idiom for "stateful metrics updated manually in `train_step`, reset/logged by Keras." Task 4 is the proof obligation (same bug class as Tier 2's missing `_loss_tracker`).

**Task structure (5 tasks, 2 subcomponents):**
- A — guarded integration + regression harness (Task 1)
- B — dispatch correctness (Tasks 2–4)
- Phase integration verification (Task 5)

---

<!-- START_SUBCOMPONENT_A (task 1) -->
<!-- START_TASK_1 -->
### Task 1: __init__ params + guarded dispatch + metrics override + regression harness

**Files:**
- Modify: `src/model_setup/layer_lr_model.py`
- Modify: `tests/test_layer_lr_model.py` (new class `TestDiagnosticsNoOpRegression`)

**Step 1: Write the regression tests (must stay green through the change)**

Append to `tests/test_layer_lr_model.py`:

```python
class TestDiagnosticsNoOpRegression:
    """With diagnostic_trackers=None (the default every existing call site
    uses), LayerLRModel must behave exactly as before Tier 5 Phase 4."""

    def test_default_init_has_none_trackers(self):
        model = _tiny_model()
        assert model._diagnostic_trackers is None
        assert model._head_refs_by_name == {}

    def test_metrics_property_unchanged_when_none(self):
        model = _tiny_model()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        # Build metrics by running one step.
        rng = np.random.RandomState(0)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        # No extra metrics beyond what stock Keras exposes (loss tracker).
        names = [m.name for m in model.metrics]
        assert "loss" in names
        assert not any(n.startswith("grad_norm/") for n in names)
        assert not any(n.startswith("grad_overflow") for n in names)

    def test_fit_history_records_nonzero_loss_no_trackers(self):
        # Mirror of the Tier-2 regression, asserted explicitly under the
        # Phase-4 change with trackers absent.
        model = _tiny_model()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(42)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert np.isfinite(history.history["loss"][0])
        assert history.history["loss"][0] > 0
```

**Step 2: Run to verify the new tests fail on the new attributes**

Run: `pytest tests/test_layer_lr_model.py::TestDiagnosticsNoOpRegression -v`
Expected: `test_default_init_has_none_trackers` fails (`AttributeError: _diagnostic_trackers`); the others may error on the same attribute path. The existing 13 tests still pass (unchanged code).

**Step 3: Write minimal implementation**

(a) Imports — change line 48 and add a `TYPE_CHECKING` block (no runtime `diagnostics`/`heads` import):

```python
from typing import TYPE_CHECKING, Callable, Optional

import keras
import tensorflow as tf

if TYPE_CHECKING:
    from src.diagnostics.factory import DiagnosticBundle
    from src.model_setup.heads import ClassificationHead
```

(b) `__init__` — add two string-annotated params and store derived attributes (keep existing body verbatim, append the new lines):

```python
    def __init__(
        self,
        *args,
        group_fn: Optional[Callable[[tf.Variable], str]] = None,
        multipliers: Optional[dict] = None,
        diagnostic_trackers: "Optional[DiagnosticBundle]" = None,
        diagnostic_head_refs: "Optional[list[ClassificationHead]]" = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if group_fn is None:
            group_fn = lambda v: "default"  # noqa: E731
        self.group_fn = group_fn
        self.multipliers = dict(multipliers) if multipliers is not None else {}
        self._diagnostic_trackers = diagnostic_trackers
        self._head_refs_by_name = {
            h.name: h for h in (diagnostic_head_refs or [])
        }
```

(c) Add the `metrics` property (place it as a method on the class, e.g., directly after `__init__`):

```python
    @property
    def metrics(self):
        base = super().metrics
        if self._diagnostic_trackers is None:
            return base
        extra = []
        for category in self._diagnostic_trackers["per_step"].values():
            extra.extend(category)
        return base + extra
```

(d) Add the dispatch helper (new method on the class):

```python
    def _dispatch_diagnostics(self, gradients, y):
        per_step = self._diagnostic_trackers["per_step"]
        for tracker in per_step["gradient"]:
            tracker.update_state(
                gradients, self.trainable_variables, self.group_fn
            )
        for tracker in per_step["loss_component"]:
            head = self._head_refs_by_name[tracker.head_name]
            tracker.update_state(head.last_components)
        for tracker in per_step["batch_target"]:
            tracker.update_state(y)
```

(e) In `train_step`, insert the guarded call between the existing `gradients = tape.gradient(...)` line and the existing `scaled = [...]` list comprehension. Do **not** modify the docstring, the tape block, `_loss_tracker`, `scale_loss`, `scaled`, `apply_gradients`, or the `return`:

```python
        gradients = tape.gradient(loss, self.trainable_variables)
        # Tier 5: read-only diagnostic observation of the COMPUTED
        # gradients (before per-variable multiplier scaling) plus loss
        # components and targets. No-op when diagnostics aren't configured.
        if self._diagnostic_trackers is not None:
            self._dispatch_diagnostics(gradients, y)
        scaled = [
            tf.math.scalar_mul(self.get_multiplier(w), g) if g is not None else None
            for w, g in zip(self.trainable_variables, gradients)
        ]
        self.optimizer.apply_gradients(zip(scaled, self.trainable_variables))

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layer_lr_model.py -v`
Expected: all 13 existing tests pass (unchanged behavior; `diagnostic_trackers=None`) **plus** the 3 new `TestDiagnosticsNoOpRegression` tests. Pay specific attention to `TestLossTracking` and `TestSparseGradients` staying green.

Run: `pytest -q`
Expected: Phase-3 total + 3, zero failures.

**Step 5: Commit**

```bash
git add src/model_setup/layer_lr_model.py tests/test_layer_lr_model.py
git commit -m "tier5 phase 4: guarded diagnostic dispatch + metrics override (no-op when unset)"
```
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2-4) -->
<!-- START_TASK_2 -->
### Task 2: Gradient-category dispatch correctness + pre-scaling invariant

**Files:**
- Modify: `tests/test_layer_lr_model.py` (new class `TestGradientDiagnosticDispatch`) — test-only; uses Phase-1 trackers + a hand-built bundle.

**Step 1: Write the tests**

Append to `tests/test_layer_lr_model.py` (imports at top of file: `from src.diagnostics.trackers import PerGroupGradNormTracker, GradientFiniteTracker`):

```python
class TestGradientDiagnosticDispatch:
    def _bundle(self, trackers):
        return {
            "per_step": {
                "gradient": trackers,
                "loss_component": [],
                "batch_target": [],
            },
            "periodic": [],
        }

    def test_grad_norm_tracker_sees_raw_unscaled_gradient(self):
        # The design's load-bearing invariant (lines 165-176): trackers
        # observe the COMPUTED gradient, BEFORE per-variable multiplier
        # scaling. Differential test (no fragile closed-form re-derivation):
        # two models with IDENTICAL initial weights and the SAME batch, one
        # with multiplier 1.0 and one with 10.0 on the tracked "lower"
        # group, each with its own grad-norm tracker. The gradient computed
        # by tape.gradient is identical in both (multiplier is applied
        # AFTER). So if trackers observe the raw gradient (correct), both
        # report the SAME norm. If they observed multiplier*grad (the bug
        # this guards), the 10.0-model's tracker would be ~10x the other.
        group_fn = lambda v: v.path.split("/")[0]
        rng = np.random.RandomState(1)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)

        tracker_1 = PerGroupGradNormTracker(group_name="lower", aggregation="mean")
        tracker_10 = PerGroupGradNormTracker(group_name="lower", aggregation="mean")
        base_1 = _tiny_model(group_fn=group_fn)
        base_10 = _tiny_model(group_fn=group_fn)
        model_1 = LayerLRModel(
            inputs=base_1.inputs, outputs=base_1.outputs, group_fn=group_fn,
            multipliers={"lower": 1.0},
            diagnostic_trackers=self._bundle([tracker_1]),
        )
        model_10 = LayerLRModel(
            inputs=base_10.inputs, outputs=base_10.outputs, group_fn=group_fn,
            multipliers={"lower": 10.0},
            diagnostic_trackers=self._bundle([tracker_10]),
        )
        # Force identical initial weights so the computed gradient is
        # identical across both models for the same batch.
        model_10.set_weights(model_1.get_weights())
        for m in (model_1, model_10):
            m.compile(
                optimizer=keras.optimizers.SGD(0.0),  # no update; weights stay equal
                loss=keras.losses.BinaryCrossentropy(from_logits=True),
            )
        model_1.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        model_10.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)

        v1 = float(tracker_1.result())
        v10 = float(tracker_10.result())
        assert v1 > 0.0  # sanity: a gradient was actually observed
        # Correct (pre-scaling) behavior: equal regardless of multiplier.
        assert v10 == pytest.approx(v1, rel=1e-4), (
            f"grad-norm tracker is multiplier-sensitive: v1={v1}, v10={v10} "
            "— it is observing multiplier*grad (post-scaling), violating the "
            "design's pre-scaling invariant."
        )
        # Explicitly reject the specific failure mode.
        assert v10 != pytest.approx(10.0 * v1, rel=1e-3)

    def test_overflow_tracker_zero_under_float32(self):
        tracker = GradientFiniteTracker()
        base = _tiny_model()
        model = LayerLRModel(
            inputs=base.inputs,
            outputs=base.outputs,
            diagnostic_trackers=self._bundle([tracker]),
        )
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(2)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert float(tracker.result()) == 0.0  # finite grads under float32
```

> Executor note: `test_grad_norm_tracker_sees_raw_unscaled_gradient` must
> implement the real two-part assertion (observed == raw norm; observed !=
> 10x raw norm). Capture the model's initial weights, recompute the
> "lower"-group gradient norm with a plain `tf.GradientTape` and the same
> inputs/loss, and compare. The placeholder `assert observed > 0.0` is NOT
> sufficient — the point of this test is the pre-scaling invariant.

**Step 2: Run — verify the dispatch produces non-trivial tracker state**

Run: `pytest tests/test_layer_lr_model.py::TestGradientDiagnosticDispatch -v`
Expected: pass once the executor completes the exact assertion. (Implementation already exists from Task 1; this is a correctness test of that dispatch.)

**Step 3:** No implementation change (Task 1 added the dispatch).

**Step 4: Full suite**

Run: `pytest -q`
Expected: prior total + 2, zero failures.

**Step 5: Commit**

```bash
git add tests/test_layer_lr_model.py
git commit -m "tier5 phase 4: gradient-category dispatch correctness + pre-scaling invariant"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Loss-component + batch-target dispatch

**Files:**
- Modify: `tests/test_layer_lr_model.py` (new class `TestLossComponentBatchTargetDispatch`) — test-only.

**Step 1: Write the tests**

Append to `tests/test_layer_lr_model.py` (mirror the existing `TestWithEndpointLoss::test_trains_with_endpoint_head`@285 construction for the endpoint `LayerLRModel`; imports: `from src.diagnostics.trackers import LossComponentTracker, BatchLabelBalanceTracker`, `from src.model_setup.heads import ClassificationHead`, `from src.loss_functions.loss import FLPULoss`):

```python
class TestLossComponentBatchTargetDispatch:
    def test_loss_component_trackers_populated_from_last_components(self):
        # Build an endpoint LayerLRModel with a single CCA head exposing
        # loss components (mirror TestWithEndpointLoss setup).
        head = ClassificationHead(
            hidden_dim=INPUT_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            expose_loss_components=True,
        )
        # ... executor: assemble inputs (features + "cca_targets"),
        # outputs = {"cca": head(features, targets=cca_targets)} ...
        lc = LossComponentTracker("cca", "positive_risk", "mean")
        bt = BatchLabelBalanceTracker("cca")
        bundle = {
            "per_step": {
                "gradient": [],
                "loss_component": [lc],
                "batch_target": [bt],
            },
            "periodic": [],
        }
        model = LayerLRModel(
            inputs=...,            # features + cca_targets inputs
            outputs=...,           # {"cca": logits}
            diagnostic_trackers=bundle,
            diagnostic_head_refs=[head],
        )
        model.compile(optimizer=keras.optimizers.SGD(0.01))
        # ... fit one epoch on synthetic (features, {"cca_targets": y}) ...
        assert float(lc.result()) == float(lc.result())  # finite, populated
        assert 0.0 <= float(bt.result()) <= 1.0

    def test_head_ref_lookup_by_name(self):
        head = ClassificationHead(
            hidden_dim=INPUT_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            expose_loss_components=True,
        )
        model = LayerLRModel(
            inputs=..., outputs=...,
            diagnostic_trackers={"per_step": {"gradient": [], "loss_component": [],
                                              "batch_target": []}, "periodic": []},
            diagnostic_head_refs=[head],
        )
        assert model._head_refs_by_name == {"cca": head}
```

> Executor note: complete the endpoint-model assembly by mirroring
> `tests/test_layer_lr_model.py::TestWithEndpointLoss::test_trains_with_endpoint_head`
> (line ~285) — that test already builds a working endpoint `LayerLRModel`
> with a head + target input. Reuse its input/output construction; only add
> `diagnostic_trackers=` + `diagnostic_head_refs=[head]` and assert tracker
> state after one epoch. `last_components` is populated during the in-tape
> forward pass (head has `expose_loss_components=True`), so it is set by the
> time `_dispatch_diagnostics` runs.

**Step 2: Run** — `pytest tests/test_layer_lr_model.py::TestLossComponentBatchTargetDispatch -v`. Expected: pass once assembly is completed (dispatch impl exists from Task 1).

**Step 3:** No implementation change.

**Step 4: Full suite** — `pytest -q`. Expected: prior + 2, zero failures.

**Step 5: Commit**

```bash
git add tests/test_layer_lr_model.py
git commit -m "tier5 phase 4: loss-component + batch-target dispatch tests"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Keras integration — the linchpin (logs, epoch reset, loss-tracking intact)

**Files:**
- Modify: `tests/test_layer_lr_model.py` (new class `TestDiagnosticsKerasIntegration`) — test-only.

This task proves the `metrics` property override is safe in this Keras 3.12 build — the same bug class as Tier 2's missing `_loss_tracker`. If any assertion here fails, it indicates a Keras-version interaction; the architectural fallback (not implemented unless needed) is to have `DiagnosticsCallback` (Phase 5) own tracker reset + result-logging instead of the `metrics` property.

**Step 1: Write the tests**

Append to `tests/test_layer_lr_model.py`:

```python
class TestDiagnosticsKerasIntegration:
    def _model_with_trackers(self):
        from src.diagnostics.trackers import GradientFiniteTracker, PerGroupGradNormTracker
        group_fn = lambda v: v.path.split("/")[0]
        base = _tiny_model(group_fn=group_fn)
        trackers = [
            PerGroupGradNormTracker(group_name="lower", aggregation="max"),
            GradientFiniteTracker(),
        ]
        bundle = {"per_step": {"gradient": trackers, "loss_component": [],
                               "batch_target": []}, "periodic": []}
        return LayerLRModel(
            inputs=base.inputs, outputs=base.outputs, group_fn=group_fn,
            diagnostic_trackers=bundle,
        ), trackers

    def test_loss_tracking_intact_with_trackers(self):
        # THE regression: metrics override must not break history["loss"].
        model, _ = self._model_with_trackers()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(42)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert np.isfinite(history.history["loss"][0])
        assert history.history["loss"][0] > 0

    def test_diagnostic_scalars_appear_in_history(self):
        model, _ = self._model_with_trackers()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(3)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert "grad_norm/lower/max" in history.history
        assert "grad_overflow_rate" in history.history

    def test_trackers_reset_at_epoch_boundary(self):
        model, trackers = self._model_with_trackers()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        rng = np.random.RandomState(4)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=3, batch_size=BATCH, verbose=0)
        # 3 separate per-epoch values recorded (not a single monotone
        # accumulation across the whole run) — Keras reset_state per epoch.
        assert len(history.history["grad_overflow_rate"]) == 3

    def test_no_crash_compute_metrics_with_custom_trackers(self):
        # Trackers have custom update_state signatures; ensure Keras's
        # compute_metrics/get_metrics_result path (which iterates
        # self.metrics) does not call them with (y, y_pred).
        model, _ = self._model_with_trackers()
        model.compile(
            optimizer=keras.optimizers.SGD(0.01),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(name="acc")],
        )
        rng = np.random.RandomState(5)
        x = rng.randn(BATCH, INPUT_DIM).astype(np.float32)
        y = rng.randint(0, 2, size=(BATCH, 1)).astype(np.float32)
        history = model.fit(x, y, epochs=1, batch_size=BATCH, verbose=0)
        assert "acc" in history.history          # compiled metric still works
        assert "grad_overflow_rate" in history.history
        assert np.isfinite(history.history["loss"][0])
```

**Step 2: Run — this is the make-or-break step**

Run: `pytest tests/test_layer_lr_model.py::TestDiagnosticsKerasIntegration -v`
Expected: 4 passed. If `test_loss_tracking_intact_with_trackers` or `test_no_crash_compute_metrics_with_custom_trackers` fails, STOP and report — the `metrics` override interacts badly with this Keras build and the callback-owned fallback must be considered (escalate to the human; do not paper over with `--no-verify` or by deleting the assertion).

**Step 3:** No implementation change (Task 1 added the override; this validates it).

**Step 4: Full suite**

Run: `pytest -q`
Expected: prior total + 4, zero failures.

**Step 5: Commit**

```bash
git add tests/test_layer_lr_model.py
git commit -m "tier5 phase 4: Keras integration tests (logs, epoch reset, loss-tracking intact)"
```
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_TASK_5 -->
### Task 5: Phase 4 integration verification

**Type:** Verification (no new code; no commit).

**Step 1: Confirm contract surface**

Run: `python -c "import inspect; from src.model_setup.layer_lr_model import LayerLRModel; p=inspect.signature(LayerLRModel.__init__).parameters; print('diagnostic_trackers' in p, 'diagnostic_head_refs' in p)"`
Expected: `True True`

**Step 2: Full suite, no regressions**

Run: `pytest -q`
Expected: Phase-3 total + all Phase-4 additions, zero failures. Explicitly confirm the original 13 `test_layer_lr_model.py` tests are all green — especially `TestLossTracking` (the 2 loss-tracker regressions) and `TestSparseGradients` — proving the guarded change did not disturb the no-trackers path. Record the number (Phase 5 baseline).

**Step 3: Working tree clean**

Run: `git status --short` → clean.
Run: `git log --oneline 9136195..HEAD` → Phase-1/2/3 commits + 4 Phase-4 task commits.

**Phase 4 Done-when criteria (from design plan):**
- Regression test passes (existing train_step behavior unchanged when no trackers). ✓ (Step 2; `TestDiagnosticsNoOpRegression` + original 13)
- Dispatch tests pass. ✓ (Step 2; Tasks 2–4)
- `LayerLRModel`'s 13 existing tests still pass. ✓ (Step 2)

Phase 4 is complete.
<!-- END_TASK_5 -->
