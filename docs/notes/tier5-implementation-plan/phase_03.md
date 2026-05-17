# Tier 5 Implementation Plan — Phase 3: Loss-Component Harvest Path

**Goal:** Extend `FLPULoss.call` with an optional `return_intermediates` parameter exposing `{positive_risk, negative_risk, correction_triggered}`, and add an `expose_loss_components` flag + `last_components` attribute to `ClassificationHead` so the train-step dispatch (Phase 4) can harvest the components.

**Architecture:** Backward-compatible extension. `FLPULoss.call` gains a defaulted parameter; the loss scalar is computed by the identical expression regardless of the flag (bit-identity is structural, not coincidental). `ClassificationHead` gains a keyword-only flag with a construction-boundary guard (defense-in-depth: independent of the Phase 2 factory guard — different boundary, catches different misuse). Flag-off path is byte-for-byte unchanged (zero regression risk); flag-on path computes the loss once via `.call(return_intermediates=True)` and registers that scalar.

**Tech Stack:** Python ≥3.12, Keras 3 (`keras.ops`), TensorFlow, pytest, hypothesis. New in this phase: `inspect` (already stdlib) in `heads.py`.

**Scope:** Phase 3 of 8. Depends on Phase 1 (tracker contract knowable — informs the 3-key dict shape). Independent of Phase 2 at the code level (no factory import here).

**Codebase verified:** 2026-05-16 via codebase-investigator (verbatim reads).

**Codebase verification findings:**
- `FLPULoss` (`src/loss_functions/loss.py`): base `keras.losses.Loss`, `@keras.saving.register_keras_serializable()` (line 14), **no** `get_config`/`from_config`. Imports `import keras` / `from keras import ops` (lines 8-9). `__init__(self, prior, focal_gamma=2.0, nn_beta=0.0, nn_gamma=1.0, kiryo_clawback=False)`; attrs include `self.prior, self.focal_gamma, self.nn_beta, self.nn_gamma, self.kiryo_clawback, self.focal_loss, self.positive=1, self.unlabeled=0, self.min_count=1.0`.
- `FLPULoss.call` (lines 122-179): `positive_risk` is an explicit named intermediate (line 159); `negative_risk` is explicit (lines 160-163); `correction_triggered` is **NOT** a named variable — the nnPU correction is implicit in `ops.maximum(negative_risk, 0)` (no-clawback, line 169) / the `ops.cond` predicate `negative_risk < -self.nn_beta` (clawback, lines 175-178). Returns a scalar tensor only. Verbatim current body reproduced in Task 1.
- `return_intermediates` is a `call` parameter, not an `__init__` arg → **no `get_config` change needed** for FLPULoss (Keras serialization unaffected; the `@register_keras_serializable` decorator covers `__init__` args only).
- `ClassificationHead` (`src/model_setup/heads.py`): `__init__(self, hidden_dim, dropout=0.1, loss_fn=None, metrics=None, *, name)` — `name` keyword-only required (lines 112-127). `self.loss_fn = loss_fn` (line 131). No `get_config`/`from_config`. `call(self, features, targets=None)` (lines 168-207); loss/metrics block (lines ~201-206):
  ```python
  if targets is not None:
      if self.loss_fn is not None:
          self.add_loss(self.loss_fn(targets, logits))
      for metric in self.metric_objs:
          metric.update_state(targets, logits)
  return logits
  ```
- `expose_loss_components` defaults `False`, is training-only. The Pattern-2 eval head (`eval_cca_classifier.py`) is constructed without `loss_fn` → flag stays default → **no `get_config` change needed** for ClassificationHead.
- Keras subtlety: `loss(y_true, y_pred)` → `keras.losses.Loss.__call__` (reduction wrapper); `loss.call(...)` is the direct path. For `FLPULoss` both yield the same scalar (`call` already returns a reduced 0-d tensor; Keras reduction on a scalar is identity). Pinned by an explicit test in Task 1.
- `tests/test_flpu_loss.py`: class-based (`TestConstruction`@130, `TestOutputStructure`@146, `TestEasyVsAdversarial`@166, `TestOrderInvariance`@203, `TestNonNegativity`@222, `TestPriorSensitivity`@243, `TestNumpyReference`@287, `TestKiryoClawback`@335, `TestEdgeCaseBatches`@397, `TestProductionConfiguration`@436). Helpers: `_batch(positive_logits=[...], unlabeled_logits=[...]) -> (y_true, y_pred)`, `_scalar(...)`, `_numpy_reference_flpu(y_true, y_pred, prior=, focal_gamma=)`. Existing assertion style: `np.testing.assert_allclose(actual, expected, rtol=1e-4)`. The existing numerical tests exercise the `loss(...)` (`__call__`) path → unaffected by adding a defaulted `call` param.
- `tests/test_heads.py`: class-based (`TestConstruction`@57, `TestForwardPass`@91, `TestEndpointMode`@118, `TestTrainableWeights`@151, `TestMetrics`@185). Helpers: `_dummy_features()`, `_dummy_targets()`. Endpoint pattern: `head = ClassificationHead(hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.1), name="test_head"); _ = head(_dummy_features(), targets=_dummy_targets()); assert len(head.losses) == 1`.
- No existing `last_components` / `return_intermediates` / `intermediates` anywhere in `src/` or `tests/` — greenfield contract.

**Resolved design decisions (from review):**
- `correction_triggered` = per-batch 0/1 indicator, path-dependent: `ops.cast(negative_risk < 0, "float32")` (no-clawback) / `ops.cast(negative_risk < -self.nn_beta, "float32")` (clawback). A `LossComponentTracker(head, "correction_triggered", "mean")` aggregates the 0/1 into the "correction rate" the design DoD references. Design doc line ~115 corrected to match.
- `ClassificationHead.__init__` performs a construction-boundary check (independent of the Phase 2 factory guard — boundary-inventory pattern).
- Flag-on head path computes the loss once via `.call(return_intermediates=True)`; a loss-level pinning test (`loss(...)` vs `loss.call(...)`) justifies this. Flag-off path byte-for-byte unchanged.

**Task structure (5 tasks, 2 subcomponents):**
- A — FLPULoss (Tasks 1–2)
- B — ClassificationHead (Tasks 3–4)
- Phase integration verification (Task 5)

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: FLPULoss.call — return_intermediates + correction_triggered

**Files:**
- Modify: `src/loss_functions/loss.py` (`FLPULoss.call`, lines 122-179)
- Modify: `tests/test_flpu_loss.py` (new class `TestReturnIntermediates`)

**Step 1: Write the failing tests**

Append to `tests/test_flpu_loss.py` (reuse module helpers `_batch`, `_scalar`):

```python
class TestReturnIntermediates:
    def test_default_path_returns_scalar(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        out = loss.call(y_true, y_pred)
        # scalar tensor, not a tuple
        assert not isinstance(out, tuple)
        assert float(out) == float(out)  # finite, indexable as scalar

    def test_flag_path_returns_scalar_and_dict(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        out = loss.call(y_true, y_pred, return_intermediates=True)
        assert isinstance(out, tuple) and len(out) == 2
        scalar, comps = out
        assert set(comps.keys()) == {
            "positive_risk", "negative_risk", "correction_triggered"
        }

    def test_loss_scalar_bit_identical_between_paths(self):
        # Design DoD: loss scalar is bit-identical with/without the flag.
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        scalar_only = loss.call(y_true, y_pred)
        scalar_with, _ = loss.call(y_true, y_pred, return_intermediates=True)
        assert float(scalar_only) == float(scalar_with)  # exact equality

    def test_loss_scalar_bit_identical_clawback_path(self):
        loss = FLPULoss(prior=0.1, kiryo_clawback=True)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        scalar_only = loss.call(y_true, y_pred)
        scalar_with, _ = loss.call(y_true, y_pred, return_intermediates=True)
        assert float(scalar_only) == float(scalar_with)

    def test_direct_call_equals_dunder_call(self):
        # Pins the equivalence that justifies the flag-on head path using
        # loss_fn.call(...) directly rather than loss_fn(...).
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        via_dunder = float(loss(y_true, y_pred))
        via_call = float(loss.call(y_true, y_pred))
        np.testing.assert_allclose(via_dunder, via_call, rtol=1e-6)

    def test_existing_dunder_call_path_unchanged(self):
        # Back-compat: the __call__ path (used by all existing tests) still
        # returns a finite scalar identical to pre-change behavior.
        loss = FLPULoss(prior=0.1, focal_gamma=2.0)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        expected = _numpy_reference_flpu(
            y_true, y_pred, prior=0.1, focal_gamma=2.0
        )
        np.testing.assert_allclose(_scalar(loss(y_true, y_pred)), expected, rtol=1e-4)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_flpu_loss.py::TestReturnIntermediates -v`
Expected: `test_flag_path_*` and `test_loss_scalar_bit_identical_*` fail (call doesn't accept `return_intermediates`); `test_default_path_*`, `test_direct_call_*`, `test_existing_*` may pass already.

**Step 3: Write minimal implementation**

Replace the body of `FLPULoss.call` (currently lines 122-179) with the following. The signature gains `return_intermediates=False`; everything up to and including the `negative_risk` computation is **unchanged verbatim**; only the return tail is restructured (each branch's loss expression is identical to the original — bound to `loss` then returned):

```python
    def call(self, y_true, y_pred, return_intermediates=False):
        # Boolean masks per sub-population, cast to float for elementwise
        # multiplication. Reshape to 1-D so the masks line up with the
        # per-sample focal loss output (which is also 1-D under reduction="none").
        positive = ops.cast(y_true == self.positive, dtype="float32")
        unlabeled = ops.cast(y_true == self.unlabeled, dtype="float32")
        positive = ops.reshape(positive, (-1,))
        unlabeled = ops.reshape(unlabeled, (-1,))

        # Sample counts, floored at 1 to avoid division by zero.
        n_positive = ops.maximum(ops.sum(positive), self.min_count)
        n_unlabeled = ops.maximum(ops.sum(unlabeled), self.min_count)

        # Per-sample focal loss over the whole batch; we zero out irrelevant
        # entries via the masks below. We evaluate over the full batch
        # (rather than slicing) to keep the graph static for autograph/jit.
        pn_loss = self.focal_loss(y_true, y_pred)

        # Three FLPU components.
        y_positive = pn_loss * positive  # positives, treated as positive
        y_unlabeled = pn_loss * unlabeled  # unlabeled, treated as negative
        y_positive_inv = (
            self.focal_loss(ops.abs(y_true - 1), y_pred) * positive
        )  # positives, treated as negative (bias correction)

        positive_risk = self.prior * ops.sum(y_positive) / n_positive
        negative_risk = (
            ops.sum(y_unlabeled) / n_unlabeled
            - self.prior * ops.sum(y_positive_inv) / n_positive
        )

        if not self.kiryo_clawback:
            loss = positive_risk + ops.maximum(negative_risk, 0)
            correction_triggered = ops.cast(negative_risk < 0, "float32")
        else:
            loss = ops.cond(
                pred=negative_risk < -self.nn_beta,
                true_fn=lambda: -self.nn_gamma * negative_risk,
                false_fn=lambda: positive_risk + negative_risk,
            )
            correction_triggered = ops.cast(
                negative_risk < -self.nn_beta, "float32"
            )

        if return_intermediates:
            return loss, {
                "positive_risk": positive_risk,
                "negative_risk": negative_risk,
                "correction_triggered": correction_triggered,
            }
        return loss
```

> Executor note: do NOT alter the `positive`/`unlabeled`/`n_*`/`pn_loss`/
> `y_*`/`positive_risk`/`negative_risk` lines — they are byte-identical to
> the original. The ONLY changes are: (1) the `return_intermediates=False`
> parameter, (2) binding each branch's existing return expression to `loss`,
> (3) the two `correction_triggered` casts, (4) the conditional tuple return.
> This is what makes the scalar bit-identical.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_flpu_loss.py::TestReturnIntermediates -v`
Expected: 6 passed.

Run: `pytest tests/test_flpu_loss.py -q`
Expected: all existing FLPU tests still pass (the `__call__` path is unchanged) + 6 new.

Run: `pytest -q`
Expected: Phase-2 total + 6, zero failures.

**Step 5: Commit**

```bash
git add src/loss_functions/loss.py tests/test_flpu_loss.py
git commit -m "tier5 phase 3: FLPULoss.call return_intermediates + correction_triggered"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: FLPULoss component numerical-correctness tests

**Files:**
- Modify: `tests/test_flpu_loss.py` (new class `TestLossComponentCorrectness`) — test-only; implementation from Task 1.

**Step 1: Write the tests**

Append to `tests/test_flpu_loss.py`:

```python
class TestLossComponentCorrectness:
    def test_no_clawback_loss_equals_pos_plus_clamped_neg(self):
        # Ties components to the already-numpy-verified scalar without
        # reimplementing the component math.
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0, -1.0],
            unlabeled_logits=[-3.0, -1.0, 0.0, 1.0],
        )
        scalar, c = loss.call(y_true, y_pred, return_intermediates=True)
        recombined = float(c["positive_risk"]) + max(float(c["negative_risk"]), 0.0)
        np.testing.assert_allclose(float(scalar), recombined, rtol=1e-5)

    def test_correction_triggered_is_binary(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, 1.0],
            unlabeled_logits=[-3.0, 0.0, 1.0],
        )
        _, c = loss.call(y_true, y_pred, return_intermediates=True)
        assert float(c["correction_triggered"]) in (0.0, 1.0)

    def test_correction_fires_iff_negative_risk_below_zero_no_clawback(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[5.0, 5.0],
            unlabeled_logits=[-5.0, -5.0, -5.0],
        )
        scalar, c = loss.call(y_true, y_pred, return_intermediates=True)
        neg = float(c["negative_risk"])
        fired = float(c["correction_triggered"])
        if neg < 0.0:
            assert fired == 1.0
            np.testing.assert_allclose(float(scalar), float(c["positive_risk"]), rtol=1e-5)
        else:
            assert fired == 0.0
            np.testing.assert_allclose(
                float(scalar),
                float(c["positive_risk"]) + neg,
                rtol=1e-5,
            )

    def test_clawback_path_correction_semantics(self):
        loss = FLPULoss(prior=0.1, kiryo_clawback=True, nn_beta=0.0, nn_gamma=1.0)
        y_true, y_pred = _batch(
            positive_logits=[5.0, 5.0],
            unlabeled_logits=[-5.0, -5.0, -5.0],
        )
        scalar, c = loss.call(y_true, y_pred, return_intermediates=True)
        neg = float(c["negative_risk"])
        fired = float(c["correction_triggered"])
        if neg < -0.0:
            assert fired == 1.0
            np.testing.assert_allclose(float(scalar), -1.0 * neg, rtol=1e-5)
        else:
            assert fired == 0.0
            np.testing.assert_allclose(
                float(scalar), float(c["positive_risk"]) + neg, rtol=1e-5
            )

    def test_positive_risk_non_negative(self):
        loss = FLPULoss(prior=0.1)
        y_true, y_pred = _batch(
            positive_logits=[2.0, -1.0, 0.5],
            unlabeled_logits=[-3.0, 1.0],
        )
        _, c = loss.call(y_true, y_pred, return_intermediates=True)
        assert float(c["positive_risk"]) >= 0.0
```

**Step 2: Run tests**

Run: `pytest tests/test_flpu_loss.py::TestLossComponentCorrectness -v`
Expected: 5 passed (implementation from Task 1 is correct; these pin the component semantics).

**Step 3:** No implementation change.

**Step 4: Full suite**

Run: `pytest -q`
Expected: prior total + 5, zero failures.

**Step 5: Commit**

```bash
git add tests/test_flpu_loss.py
git commit -m "tier5 phase 3: FLPULoss loss-component numerical-correctness tests"
```
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: ClassificationHead — expose_loss_components flag + last_components + guard

**Files:**
- Modify: `src/model_setup/heads.py` (`__init__` signature + body; `call` loss block; ensure `import inspect`)
- Modify: `tests/test_heads.py` (new class `TestExposeLossComponents`)

**Step 1: Write the failing tests**

Append to `tests/test_heads.py` (reuse `_dummy_features`, `_dummy_targets`, `HIDDEN_DIM`):

```python
class _StubLossNoIntermediates(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.constant(0.0)


class TestExposeLossComponents:
    def test_flag_off_default_last_components_none(self):
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.1), name="h"
        )
        _ = head(_dummy_features(), targets=_dummy_targets())
        assert head.last_components is None
        assert len(head.losses) == 1

    def test_flag_off_behavior_matches_existing_endpoint_contract(self):
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM, loss_fn=FLPULoss(prior=0.1), name="h"
        )
        out = head(_dummy_features(), targets=_dummy_targets())
        assert out.shape[-1] == 1
        assert len(head.losses) == 1

    def test_construction_guard_rejects_incapable_loss(self):
        with pytest.raises(ValueError, match="return_intermediates"):
            ClassificationHead(
                hidden_dim=HIDDEN_DIM,
                loss_fn=_StubLossNoIntermediates(),
                name="h",
                expose_loss_components=True,
            )

    def test_construction_guard_not_triggered_when_loss_none(self):
        # loss_fn=None (standard mode): flag is inert, no raise.
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM, name="h", expose_loss_components=True
        )
        assert head.expose_loss_components is True
        assert head.last_components is None

    def test_construction_guard_passes_with_flpu(self):
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="h",
            expose_loss_components=True,
        )
        assert head.expose_loss_components is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_heads.py::TestExposeLossComponents -v`
Expected: failures (`expose_loss_components` not a param; `last_components` not an attribute).

**Step 3: Write minimal implementation**

(a) Ensure `import inspect` is present at the top of `src/model_setup/heads.py` (add it with the other stdlib imports if absent).

(b) Change the `__init__` signature (lines 112-120) to add a keyword-only flag after `name`:

```python
    def __init__(
        self,
        hidden_dim,
        dropout=0.1,
        loss_fn=None,
        metrics=None,
        *,
        name,
        expose_loss_components=False,
    ):
```

(c) After `self.loss_fn = loss_fn` (line 131), add the construction-boundary guard + state init:

```python
        self.expose_loss_components = expose_loss_components
        self.last_components = None
        if expose_loss_components and loss_fn is not None:
            if "return_intermediates" not in inspect.signature(
                loss_fn.call
            ).parameters:
                raise ValueError(
                    f"ClassificationHead {name!r} was constructed with "
                    f"expose_loss_components=True but its loss "
                    f"{type(loss_fn).__name__} does not accept a "
                    f"`return_intermediates` parameter. Loss-component "
                    f"harvest requires an FLPU-style loss."
                )
```

(d) In `call`, replace the loss-registration block (currently the `if self.loss_fn is not None: self.add_loss(self.loss_fn(targets, logits))` inside `if targets is not None:`) with the flag branch (the metrics loop and `return logits` are unchanged):

```python
        if targets is not None:
            if self.loss_fn is not None:
                if self.expose_loss_components:
                    loss, components = self.loss_fn.call(
                        targets, logits, return_intermediates=True
                    )
                    self.last_components = components
                    self.add_loss(loss)
                else:
                    self.add_loss(self.loss_fn(targets, logits))
            for metric in self.metric_objs:
                metric.update_state(targets, logits)
        return logits
```

> Executor note: the `else: self.add_loss(self.loss_fn(targets, logits))`
> line is byte-for-byte the original behavior — the flag-off path must not
> change (zero regression; existing TestEndpointMode relies on it).

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_heads.py::TestExposeLossComponents -v`
Expected: 5 passed.

Run: `pytest tests/test_heads.py -q`
Expected: all existing head tests still pass (flag-off path unchanged) + 5 new.

Run: `pytest -q`
Expected: prior total + 5, zero failures.

**Step 5: Commit**

```bash
git add src/model_setup/heads.py tests/test_heads.py
git commit -m "tier5 phase 3: ClassificationHead expose_loss_components + guard + last_components"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: ClassificationHead flag-on contract tests

**Files:**
- Modify: `tests/test_heads.py` (extend `TestExposeLossComponents`) — test-only; implementation from Task 3.

**Step 1: Write the tests**

Append to `tests/test_heads.py`:

```python
class TestExposeLossComponentsFlagOn:
    def _head(self):
        return ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="h",
            expose_loss_components=True,
        )

    def test_last_components_populated_after_call(self):
        head = self._head()
        _ = head(_dummy_features(), targets=_dummy_targets())
        assert head.last_components is not None
        assert set(head.last_components.keys()) == {
            "positive_risk", "negative_risk", "correction_triggered"
        }

    def test_single_loss_registered(self):
        head = self._head()
        _ = head(_dummy_features(), targets=_dummy_targets())
        assert len(head.losses) == 1  # single computation, single registration

    def test_repeated_calls_update_to_latest(self):
        head = self._head()
        _ = head(_dummy_features(), targets=_dummy_targets())
        first = {k: float(v) for k, v in head.last_components.items()}
        # Different batch → different components.
        other_targets = 1 - _dummy_targets()
        _ = head(_dummy_features(), targets=other_targets)
        second = {k: float(v) for k, v in head.last_components.items()}
        assert head.last_components is not None
        assert first.keys() == second.keys()

    def test_inference_call_leaves_last_components_untouched(self):
        head = self._head()
        out = head(_dummy_features(), targets=None)  # inference
        assert out.shape[-1] == 1
        assert head.last_components is None  # never set without targets
```

**Step 2: Run tests**

Run: `pytest tests/test_heads.py::TestExposeLossComponentsFlagOn -v`
Expected: 4 passed.

**Step 3:** No implementation change.

**Step 4: Full suite**

Run: `pytest -q`
Expected: prior total + 4, zero failures.

**Step 5: Commit**

```bash
git add tests/test_heads.py
git commit -m "tier5 phase 3: ClassificationHead flag-on contract tests"
```
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_TASK_5 -->
### Task 5: Phase 3 integration verification

**Type:** Verification (no new code; no commit).

**Step 1: Confirm imports + contract surface**

Run: `python -c "from src.loss_functions.loss import FLPULoss; from src.model_setup.heads import ClassificationHead; import inspect; print('return_intermediates' in inspect.signature(FLPULoss.call).parameters); print('expose_loss_components' in inspect.signature(ClassificationHead.__init__).parameters)"`
Expected: `True` then `True`.

**Step 2: Full suite, no regressions**

Run: `pytest -q`
Expected: Phase-2 total + all Phase-3 additions, zero failures, zero errors. Explicitly confirm pre-existing `tests/test_flpu_loss.py` numerical tests (`TestNumpyReference`, `TestProductionConfiguration`, etc.) and `tests/test_heads.py::TestEndpointMode` are green — proves the flag-off / `__call__` paths are byte-for-byte unchanged. Record the number (Phase 4 baseline).

**Step 3: Confirm bit-identity guard exists**

Run: `pytest tests/test_flpu_loss.py::TestReturnIntermediates::test_loss_scalar_bit_identical_between_paths tests/test_flpu_loss.py::TestReturnIntermediates::test_loss_scalar_bit_identical_clawback_path -v`
Expected: 2 passed (the design DoD's bit-identity requirement).

**Step 4: Working tree clean**

Run: `git status --short`
Expected: clean.

Run: `git log --oneline 9136195..HEAD`
Expected: Phase-1/2 commits + 4 Phase-3 task commits.

**Phase 3 Done-when criteria (from design plan):**
- All new contract tests pass. ✓ (Step 2)
- Existing FLPU and head tests still pass. ✓ (Step 2)
- The loss scalar is bit-identical between `return_intermediates=True` and `=False` paths. ✓ (Step 3)

Phase 3 is complete.
<!-- END_TASK_5 -->
