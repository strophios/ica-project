# Tier 5 Implementation Plan — Phase 1: Tracker Module Foundation

**Goal:** Establish `src/diagnostics/` with four concrete per-step `keras.metrics.Metric` subclasses (no abstract base) and their unit + property-based tests.

**Architecture:** Flat hierarchy — each tracker is a direct `keras.metrics.Metric` subclass managing its own aggregation state internally, so `aggregation="max"` and `aggregation="mean"` are configurable per instance rather than separate classes. Categories (`gradient`, `loss_component`, `batch_target`) are NOT enforced by inheritance; they are enforced by registration in Phase 2's factory output dict.

**Tech Stack:** Python ≥3.12, Keras 3 (standalone `import keras`), TensorFlow (platform-conditional), pytest ≥9.0.3, hypothesis (new dev dependency, added in Task 1). New code files carry FCIS `# pattern:` classification comments (adopted on new files only for Tier 5).

**Scope:** Phase 1 of 8 (tracker module foundation). Phases 2–8 are separate plan files.

**Codebase verified:** 2026-05-15 via codebase-investigator.

**Codebase verification findings:**
- `src/diagnostics/` does not exist; `tests/test_diagnostics_trackers.py` does not exist.
- No pre-existing `keras.metrics.Metric` subclasses anywhere in `src/`. Only stock Keras metrics, constructed in `src/cca_metrics.py:make_cca_metrics()` (line 39).
- Test conventions: class-based (`class TestXxx:`), no `conftest.py`, no pytest fixtures (module-level helper functions instead). Assertions use bare `assert` + `pytest.raises`. Imports follow `import keras` / `import tensorflow as tf` / `import numpy as np` / `from src.x import y`.
- `pyproject.toml`: `pytest>=9.0.3` in `[dependency-groups].dev`; `[tool.pytest.ini_options]` has `pythonpath = ["."]` and `testpaths = ["tests"]`; `requires-python = ">=3.12"`; Keras 3 standalone.
- `hypothesis` not present — added in Task 1.
- No FCIS `# pattern:` comments anywhere in `src/` — introduced on new files only.
- Baseline test count: 220 passing.

**Tracker contracts (all four):**
- `PerGroupGradNormTracker(group_name: str, aggregation: str)` → name `grad_norm/{group}/{agg}`; `update_state(gradients, variables, group_fn)`.
- `GradientFiniteTracker()` → name `grad_overflow_rate`; `update_state(gradients, variables=None, group_fn=None)` (uniform gradient-category signature; ignores extras).
- `LossComponentTracker(head_name: str, component_key: str, aggregation: str)` → name `{head}/{key}/{agg}`; `update_state(components_dict)`.
- `BatchLabelBalanceTracker(head_name: str)` → name `{head}/positive_fraction`; `update_state(y)` where `y` is a `dict[str, Tensor]`-shaped target.

All four return a scalar `tf.Tensor` from `result()` and support `reset_state()` for epoch boundaries.

**Task structure (7 tasks, 5 subcomponents):**
- Subcomponent A — Setup (Task 1)
- Subcomponent B — PerGroupGradNormTracker (Tasks 2–3)
- Subcomponent C — GradientFiniteTracker (Task 4)
- Subcomponent D — LossComponentTracker (Task 5)
- Subcomponent E — BatchLabelBalanceTracker (Task 6)
- Phase integration verification (Task 7)

---

<!-- START_SUBCOMPONENT_A (task 1) -->
<!-- START_TASK_1 -->
### Task 1: Setup — add hypothesis dev dep, scaffold trackers.py

**Type:** Infrastructure (3-step template).

**Files:**
- Modify: `pyproject.toml` (add `hypothesis` to `[dependency-groups].dev`)
- Create: `src/diagnostics/trackers.py` (scaffold only)
- Implicitly created: `src/diagnostics/` directory (no `__init__.py` — project uses implicit namespace packages)

**Step 1: Create/modify the files**

Edit `pyproject.toml` `[dependency-groups].dev` to add hypothesis (keep existing entries):

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.3",
    "hypothesis>=6.100",
]
```

Create `src/diagnostics/trackers.py`:

```python
# pattern: Mixed (unavoidable)
# Reason: Keras Metric subclasses must hold tf.Variable state for cross-step
# aggregation; they cannot be pure functions. The aggregation arithmetic is
# pure, but the surrounding Metric protocol (update_state + result +
# reset_state over persistent state vars) is inherently stateful.

"""Per-step diagnostic trackers for Tier 5.

Concrete keras.metrics.Metric subclasses observed inside
LayerLRModel.train_step. Categories ('gradient', 'loss_component',
'batch_target') are enforced by registration in src/diagnostics/factory.py,
not by inheritance.
"""

from __future__ import annotations

import keras
import tensorflow as tf

__all__ = [
    "PerGroupGradNormTracker",
    "GradientFiniteTracker",
    "LossComponentTracker",
    "BatchLabelBalanceTracker",
]
```

**Step 2: Verify operationally**

Run: `uv sync --group dev`
Expected: hypothesis installed without errors; `uv.lock` updated.

Run: `python -c "import src.diagnostics.trackers; print(src.diagnostics.trackers.__all__)"`
Expected: `['PerGroupGradNormTracker', 'GradientFiniteTracker', 'LossComponentTracker', 'BatchLabelBalanceTracker']`

Run: `pytest -q`
Expected: 220 passed (no regressions; new module not yet exercised).

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock src/diagnostics/trackers.py
git commit -m "tier5 phase 1: scaffold src/diagnostics/trackers.py + add hypothesis dev dep"
```
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2-3) -->
<!-- START_TASK_2 -->
### Task 2: PerGroupGradNormTracker — construction, dense + sparse gradients, both aggregations

**Files:**
- Modify: `src/diagnostics/trackers.py`
- Create: `tests/test_diagnostics_trackers.py`

**Step 1: Write the failing test**

Create `tests/test_diagnostics_trackers.py`:

```python
# pattern: Functional Core

"""Unit + property-based tests for src/diagnostics/trackers.py."""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from src.diagnostics.trackers import (
    BatchLabelBalanceTracker,
    GradientFiniteTracker,
    LossComponentTracker,
    PerGroupGradNormTracker,
)


def _group_fn_by_first_path_segment(var):
    """Mirror of assembly._default_group_fn: split var.path on / and take the
    first segment as the group name."""
    return var.path.split("/", 1)[0]


def _make_two_var_setup(group_a_norm, group_b_norm):
    """Two trainable vars + matching gradients with known norms.

    variables[0] is in group 'a', variables[1] is in group 'b'.
    """
    var_a = tf.Variable(tf.zeros([4]), name="a/w")
    var_b = tf.Variable(tf.zeros([4]), name="b/w")
    grad_a = tf.constant([group_a_norm, 0.0, 0.0, 0.0])
    grad_b = tf.constant([group_b_norm, 0.0, 0.0, 0.0])
    return [var_a, var_b], [grad_a, grad_b]


class TestPerGroupGradNormConstruction:
    def test_name_pattern_mean(self):
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        assert t.name == "grad_norm/head/mean"

    def test_name_pattern_max(self):
        t = PerGroupGradNormTracker(group_name="encoder", aggregation="max")
        assert t.name == "grad_norm/encoder/max"

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="aggregation"):
            PerGroupGradNormTracker(group_name="head", aggregation="median")

    def test_empty_group_name_raises(self):
        with pytest.raises(ValueError, match="group_name"):
            PerGroupGradNormTracker(group_name="", aggregation="mean")

    def test_result_default_zero(self):
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        assert float(t.result()) == 0.0


class TestPerGroupGradNormUpdateMean:
    def test_single_group_mean_of_norms(self):
        var_h1 = tf.Variable(tf.zeros([4]), name="head/w1")
        var_h2 = tf.Variable(tf.zeros([4]), name="head/w2")
        grad_h1 = tf.constant([3.0, 0.0, 0.0, 0.0])  # norm 3.0
        grad_h2 = tf.constant([0.0, 4.0, 0.0, 0.0])  # norm 4.0
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state(
            [grad_h1, grad_h2], [var_h1, var_h2], _group_fn_by_first_path_segment
        )
        assert float(t.result()) == pytest.approx(3.5, rel=1e-5)

    def test_filters_other_groups(self):
        variables, gradients = _make_two_var_setup(group_a_norm=5.0, group_b_norm=100.0)
        t = PerGroupGradNormTracker(group_name="a", aggregation="mean")
        t.update_state(gradients, variables, _group_fn_by_first_path_segment)
        assert float(t.result()) == pytest.approx(5.0, rel=1e-5)

    def test_empty_group_reports_zero(self):
        variables, gradients = _make_two_var_setup(5.0, 7.0)
        t = PerGroupGradNormTracker(group_name="nonexistent", aggregation="mean")
        t.update_state(gradients, variables, _group_fn_by_first_path_segment)
        assert float(t.result()) == 0.0

    def test_none_gradient_skipped(self):
        var_a = tf.Variable(tf.zeros([4]), name="head/a")
        var_b = tf.Variable(tf.zeros([4]), name="head/b")
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state(
            [None, tf.constant([3.0, 0.0, 0.0, 0.0])],
            [var_a, var_b],
            _group_fn_by_first_path_segment,
        )
        assert float(t.result()) == pytest.approx(3.0, rel=1e-5)


class TestPerGroupGradNormUpdateMax:
    def test_max_of_norms_within_step(self):
        var_h1 = tf.Variable(tf.zeros([4]), name="head/w1")
        var_h2 = tf.Variable(tf.zeros([4]), name="head/w2")
        grad_h1 = tf.constant([3.0, 0.0, 0.0, 0.0])
        grad_h2 = tf.constant([0.0, 4.0, 0.0, 0.0])
        t = PerGroupGradNormTracker(group_name="head", aggregation="max")
        t.update_state(
            [grad_h1, grad_h2], [var_h1, var_h2], _group_fn_by_first_path_segment
        )
        assert float(t.result()) == pytest.approx(4.0, rel=1e-5)

    def test_max_running_across_steps(self):
        var = tf.Variable(tf.zeros([2]), name="head/w")
        t = PerGroupGradNormTracker(group_name="head", aggregation="max")
        t.update_state([tf.constant([3.0, 0.0])], [var], _group_fn_by_first_path_segment)
        t.update_state([tf.constant([1.0, 0.0])], [var], _group_fn_by_first_path_segment)
        t.update_state([tf.constant([5.0, 0.0])], [var], _group_fn_by_first_path_segment)
        assert float(t.result()) == pytest.approx(5.0, rel=1e-5)

    def test_reset_state_clears_max(self):
        var = tf.Variable(tf.zeros([2]), name="head/w")
        t = PerGroupGradNormTracker(group_name="head", aggregation="max")
        t.update_state([tf.constant([5.0, 0.0])], [var], _group_fn_by_first_path_segment)
        t.reset_state()
        assert float(t.result()) == 0.0


class TestPerGroupGradNormSparse:
    def test_indexed_slices_norm(self):
        var = tf.Variable(tf.zeros([10, 4]), name="head/embedding")
        values = tf.constant([[3.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
        sparse_grad = tf.IndexedSlices(
            values=values, indices=tf.constant([1, 7]), dense_shape=[10, 4]
        )
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state([sparse_grad], [var], _group_fn_by_first_path_segment)
        # Norm of the values tensor = sqrt(3^2 + 4^2) = 5.0
        assert float(t.result()) == pytest.approx(5.0, rel=1e-5)

    def test_mixed_dense_and_sparse_in_same_group(self):
        var_dense = tf.Variable(tf.zeros([4]), name="head/dense")
        var_emb = tf.Variable(tf.zeros([10, 4]), name="head/embedding")
        grad_dense = tf.constant([3.0, 0.0, 0.0, 0.0])  # norm 3
        grad_sparse = tf.IndexedSlices(
            values=tf.constant([[0.0, 4.0, 0.0, 0.0]]),
            indices=tf.constant([0]),
            dense_shape=[10, 4],
        )  # norm 4
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state(
            [grad_dense, grad_sparse],
            [var_dense, var_emb],
            _group_fn_by_first_path_segment,
        )
        assert float(t.result()) == pytest.approx(3.5, rel=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostics_trackers.py -v`
Expected: all tests fail with `ImportError`/`AttributeError` (class not yet implemented).

**Step 3: Write minimal implementation**

Append to `src/diagnostics/trackers.py`:

```python
_VALID_AGGREGATIONS = ("max", "mean")


def _norm_of_gradient(grad: tf.Tensor | tf.IndexedSlices) -> tf.Tensor:
    """L2 norm of a gradient, handling sparse IndexedSlices.

    For IndexedSlices the unselected rows are implicitly zero, so the norm of
    the dense-equivalent tensor equals the norm of the .values block.
    """
    if isinstance(grad, tf.IndexedSlices):
        return tf.norm(grad.values)
    return tf.norm(grad)


class PerGroupGradNormTracker(keras.metrics.Metric):
    """Aggregated L2 norm of gradients whose variable belongs to a named group.

    aggregation='mean' is the running mean of per-variable norms across steps;
    aggregation='max' is the running max.

    Frozen-encoder note: if the group has no variables in the trainable set
    (e.g., 'encoder' under freeze_encoder=True), update_state is a no-op and
    result() returns 0.0. The tracker reports zero because nothing was
    computed, NOT because computed gradients were zero.
    """

    def __init__(self, group_name: str, aggregation: str, **kwargs):
        if not group_name:
            raise ValueError("group_name must be a non-empty string")
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATIONS}, got {aggregation!r}"
            )
        super().__init__(name=f"grad_norm/{group_name}/{aggregation}", **kwargs)
        self.group_name = group_name
        self.aggregation = aggregation
        self._total = self.add_variable(shape=(), initializer="zeros", name="total")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")
        self._running_max = self.add_variable(
            shape=(), initializer="zeros", name="running_max"
        )

    def update_state(self, gradients, variables, group_fn):
        in_group_norms = []
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            if group_fn(var) != self.group_name:
                continue
            in_group_norms.append(_norm_of_gradient(grad))

        if not in_group_norms:
            return  # empty group is a no-op (e.g., frozen encoder)

        norms = tf.stack(in_group_norms)
        if self.aggregation == "mean":
            self._total.assign_add(tf.reduce_sum(norms))
            self._count.assign_add(tf.cast(tf.size(norms), self._count.dtype))
        else:  # "max"
            self._running_max.assign(
                tf.maximum(self._running_max, tf.reduce_max(norms))
            )

    def result(self):
        if self.aggregation == "mean":
            return tf.math.divide_no_nan(self._total, self._count)
        return self._running_max

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)
        self._running_max.assign(0.0)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_diagnostics_trackers.py -v`
Expected: 15 passed (5 construction + 4 mean + 3 max + 2 sparse + 1 none-skip).

Run: `pytest -q`
Expected: 235 passed (220 existing + 15 new).

**Step 5: Commit**

```bash
git add src/diagnostics/trackers.py tests/test_diagnostics_trackers.py
git commit -m "tier5 phase 1: PerGroupGradNormTracker (dense+sparse, mean+max)"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: PerGroupGradNormTracker — property-based tests

**Files:**
- Modify: `tests/test_diagnostics_trackers.py` (no implementation change expected)

**Step 1: Write the tests**

Append to `tests/test_diagnostics_trackers.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

# Realistic positive gradient norms.
norm_lists = st.lists(
    st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=10,
)


def _build_grads_with_norms(norms):
    """Trainable vars + 1-D gradients all in group 'head' with the given norms."""
    variables = [
        tf.Variable(tf.zeros([1]), name=f"head/w{i}") for i in range(len(norms))
    ]
    gradients = [tf.constant([n]) for n in norms]
    return variables, gradients


class TestPerGroupGradNormProperties:
    @given(norm_lists)
    @settings(max_examples=50, deadline=None)
    def test_mean_equals_sum_div_count(self, norms):
        variables, gradients = _build_grads_with_norms(norms)
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state(gradients, variables, _group_fn_by_first_path_segment)
        expected = sum(norms) / len(norms)
        assert float(t.result()) == pytest.approx(expected, rel=1e-4, abs=1e-5)

    @given(norm_lists)
    @settings(max_examples=50, deadline=None)
    def test_permutation_invariance_within_group(self, norms):
        va, ga = _build_grads_with_norms(norms)
        vb, gb = _build_grads_with_norms(list(reversed(norms)))
        ta = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        tb = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        ta.update_state(ga, va, _group_fn_by_first_path_segment)
        tb.update_state(gb, vb, _group_fn_by_first_path_segment)
        assert float(ta.result()) == pytest.approx(float(tb.result()), rel=1e-5)

    @given(st.lists(norm_lists, min_size=1, max_size=5))
    @settings(max_examples=30, deadline=None)
    def test_max_monotone_non_decreasing(self, norm_sequences):
        var = tf.Variable(tf.zeros([1]), name="head/w0")
        t = PerGroupGradNormTracker(group_name="head", aggregation="max")
        prev = 0.0
        for norms in norm_sequences:
            t.update_state(
                [tf.constant([norms[0]])], [var], _group_fn_by_first_path_segment
            )
            current = float(t.result())
            assert current >= prev
            prev = current
```

**Step 2: Run tests**

Run: `pytest tests/test_diagnostics_trackers.py::TestPerGroupGradNormProperties -v`
Expected: 3 passed (implementation correct from Task 2; properties confirm contracts over the input space). If a property fails, fix the Task 2 implementation, not the property.

**Step 3:** No implementation change.

**Step 4: Run full suite**

Run: `pytest -q`
Expected: 238 passed (235 + 3).

**Step 5: Commit**

```bash
git add tests/test_diagnostics_trackers.py
git commit -m "tier5 phase 1: property-based tests for PerGroupGradNormTracker"
```
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (task 4) -->
<!-- START_TASK_4 -->
### Task 4: GradientFiniteTracker — implementation + unit + property tests

**Files:**
- Modify: `src/diagnostics/trackers.py`
- Modify: `tests/test_diagnostics_trackers.py`

**Step 1: Write the failing tests**

Append to `tests/test_diagnostics_trackers.py`:

```python
class TestGradientFiniteTracker:
    def test_name(self):
        assert GradientFiniteTracker().name == "grad_overflow_rate"

    def test_all_finite_rate_zero(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([1.0, 2.0]), tf.constant([3.0])], None, None)
        assert float(t.result()) == 0.0

    def test_nan_increments_rate(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([float("nan"), 1.0])], None, None)
        assert float(t.result()) == pytest.approx(1.0)

    def test_inf_increments_rate(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([float("inf"), 1.0])], None, None)
        assert float(t.result()) == pytest.approx(1.0)

    def test_mixed_steps_average_correctly(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([1.0])], None, None)             # finite
        t.update_state([tf.constant([float("nan")])], None, None)    # overflow
        t.update_state([tf.constant([1.0])], None, None)             # finite
        t.update_state([tf.constant([1.0])], None, None)             # finite
        assert float(t.result()) == pytest.approx(0.25, rel=1e-5)

    def test_ignores_none_gradients(self):
        t = GradientFiniteTracker()
        t.update_state([None, tf.constant([1.0])], None, None)
        assert float(t.result()) == 0.0

    def test_handles_indexed_slices(self):
        t = GradientFiniteTracker()
        slices = tf.IndexedSlices(
            values=tf.constant([[float("nan")]]),
            indices=tf.constant([0]),
            dense_shape=[3, 1],
        )
        t.update_state([slices], None, None)
        assert float(t.result()) == pytest.approx(1.0)

    def test_reset_state(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([float("nan")])], None, None)
        t.reset_state()
        assert float(t.result()) == 0.0


class TestGradientFiniteProperties:
    @given(st.lists(st.booleans(), min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_rate_in_zero_one(self, overflow_per_step):
        t = GradientFiniteTracker()
        for is_overflow in overflow_per_step:
            grad = tf.constant([float("nan")] if is_overflow else [1.0])
            t.update_state([grad], None, None)
        assert 0.0 <= float(t.result()) <= 1.0

    @given(st.lists(st.booleans(), min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_rate_matches_fraction(self, overflow_per_step):
        t = GradientFiniteTracker()
        for is_overflow in overflow_per_step:
            grad = tf.constant([float("nan")] if is_overflow else [1.0])
            t.update_state([grad], None, None)
        expected = sum(overflow_per_step) / len(overflow_per_step)
        assert float(t.result()) == pytest.approx(expected, rel=1e-5, abs=1e-6)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnostics_trackers.py::TestGradientFiniteTracker -v`
Expected: all fail with `AttributeError` (only the class name is exported, no working body).

**Step 3: Write minimal implementation**

Append to `src/diagnostics/trackers.py`:

```python
def _gradient_is_finite(grad: tf.Tensor | tf.IndexedSlices) -> tf.Tensor:
    """True iff all elements of the gradient are finite."""
    values = grad.values if isinstance(grad, tf.IndexedSlices) else grad
    return tf.reduce_all(tf.math.is_finite(values))


class GradientFiniteTracker(keras.metrics.Metric):
    """Rate at which a training step contains any non-finite gradient.

    A step counts as 'overflow' if at least one non-None gradient contains
    NaN or Inf. Under local float32 this is effectively a constant 0.0; under
    mixed_float16 it is the active diagnostic that observes LossScaleOptimizer
    dynamic-loss-scaling floor behavior (Tier 5 level-2 acceptance criterion).
    """

    def __init__(self, **kwargs):
        super().__init__(name="grad_overflow_rate", **kwargs)
        self._overflow_steps = self.add_variable(
            shape=(), initializer="zeros", name="overflow_steps"
        )
        self._total_steps = self.add_variable(
            shape=(), initializer="zeros", name="total_steps"
        )

    def update_state(self, gradients, variables=None, group_fn=None):
        # variables/group_fn accepted for uniform gradient-category signature.
        del variables, group_fn
        any_nonfinite = tf.constant(False)
        for grad in gradients:
            if grad is None:
                continue
            any_nonfinite = tf.logical_or(
                any_nonfinite, tf.logical_not(_gradient_is_finite(grad))
            )
        self._overflow_steps.assign_add(
            tf.cast(any_nonfinite, self._overflow_steps.dtype)
        )
        self._total_steps.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self._overflow_steps, self._total_steps)

    def reset_state(self):
        self._overflow_steps.assign(0.0)
        self._total_steps.assign(0.0)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnostics_trackers.py::TestGradientFiniteTracker tests/test_diagnostics_trackers.py::TestGradientFiniteProperties -v`
Expected: 10 passed (8 unit + 2 property).

Run: `pytest -q`
Expected: 248 passed (238 + 10).

**Step 5: Commit**

```bash
git add src/diagnostics/trackers.py tests/test_diagnostics_trackers.py
git commit -m "tier5 phase 1: GradientFiniteTracker + unit/property tests"
```
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_C -->

<!-- START_SUBCOMPONENT_D (task 5) -->
<!-- START_TASK_5 -->
### Task 5: LossComponentTracker — implementation + key extraction + cross-head + property tests

**Files:**
- Modify: `src/diagnostics/trackers.py`
- Modify: `tests/test_diagnostics_trackers.py`

**Step 1: Write the failing tests**

Append to `tests/test_diagnostics_trackers.py`:

```python
class TestLossComponentTrackerBasics:
    def test_name_pattern(self):
        t = LossComponentTracker(
            head_name="cca", component_key="positive_risk", aggregation="mean"
        )
        assert t.name == "cca/positive_risk/mean"

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="aggregation"):
            LossComponentTracker(
                head_name="cca", component_key="positive_risk", aggregation="median"
            )

    def test_empty_head_name_raises(self):
        with pytest.raises(ValueError, match="head_name"):
            LossComponentTracker(
                head_name="", component_key="positive_risk", aggregation="mean"
            )

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="component_key"):
            LossComponentTracker(head_name="cca", component_key="", aggregation="mean")

    def test_update_mean_aggregation(self):
        t = LossComponentTracker("cca", "positive_risk", "mean")
        t.update_state({"positive_risk": tf.constant(0.4), "negative_risk": tf.constant(0.1)})
        t.update_state({"positive_risk": tf.constant(0.6), "negative_risk": tf.constant(0.2)})
        assert float(t.result()) == pytest.approx(0.5, rel=1e-5)

    def test_update_max_aggregation(self):
        t = LossComponentTracker("cca", "correction_triggered", "max")
        t.update_state({"correction_triggered": tf.constant(0.0)})
        t.update_state({"correction_triggered": tf.constant(1.0)})
        t.update_state({"correction_triggered": tf.constant(0.5)})
        assert float(t.result()) == pytest.approx(1.0)

    def test_missing_key_raises(self):
        # Defense-in-depth Layer 2 (business): an absent key signals a
        # tracker/loss mismatch the factory should have prevented.
        t = LossComponentTracker("cca", "ghost", "mean")
        with pytest.raises(KeyError, match="ghost"):
            t.update_state({"positive_risk": tf.constant(0.4)})

    def test_reset_state(self):
        t = LossComponentTracker("cca", "positive_risk", "mean")
        t.update_state({"positive_risk": tf.constant(0.4)})
        t.reset_state()
        assert float(t.result()) == 0.0


class TestLossComponentTrackerCrossHead:
    def test_same_key_different_heads_independent(self):
        t_cca = LossComponentTracker("cca", "positive_risk", "mean")
        t_immig = LossComponentTracker("immig", "positive_risk", "mean")
        t_cca.update_state({"positive_risk": tf.constant(0.1)})
        t_immig.update_state({"positive_risk": tf.constant(0.9)})
        assert float(t_cca.result()) == pytest.approx(0.1)
        assert float(t_immig.result()) == pytest.approx(0.9)

    def test_different_keys_independent(self):
        t_pos = LossComponentTracker("cca", "positive_risk", "mean")
        t_neg = LossComponentTracker("cca", "negative_risk", "mean")
        components = {
            "positive_risk": tf.constant(0.3),
            "negative_risk": tf.constant(0.7),
        }
        t_pos.update_state(components)
        t_neg.update_state(components)
        assert float(t_pos.result()) == pytest.approx(0.3)
        assert float(t_neg.result()) == pytest.approx(0.7)


class TestLossComponentTrackerProperties:
    _scalar_lists = st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=20,
    )

    @given(_scalar_lists)
    @settings(max_examples=50, deadline=None)
    def test_mean_equals_sample_mean(self, values):
        t = LossComponentTracker("cca", "k", "mean")
        for v in values:
            t.update_state({"k": tf.constant(v)})
        expected = sum(values) / len(values)
        assert float(t.result()) == pytest.approx(expected, rel=1e-4, abs=1e-4)

    @given(_scalar_lists)
    @settings(max_examples=50, deadline=None)
    def test_max_equals_sample_max(self, values):
        t = LossComponentTracker("cca", "k", "max")
        for v in values:
            t.update_state({"k": tf.constant(v)})
        # running_max starts at 0.0, so the tracked max is max(0.0, max(values))
        assert float(t.result()) == pytest.approx(max(0.0, max(values)), rel=1e-4, abs=1e-4)
```

> Implementation note for the executor: the `test_max_equals_sample_max`
> property encodes that `running_max` is initialized to `0.0` (consistent with
> `PerGroupGradNormTracker`). This is intentional and matches the design's
> "holds its own tf.Variables for max-style accumulation" — do not change the
> initializer to `-inf` to make a different property pass.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnostics_trackers.py::TestLossComponentTrackerBasics -v`
Expected: all fail with `AttributeError`.

**Step 3: Write minimal implementation**

Append to `src/diagnostics/trackers.py`:

```python
class LossComponentTracker(keras.metrics.Metric):
    """Aggregated scalar loss-component value across steps.

    Reads components_dict[component_key] per update. Raises KeyError if absent
    (Layer-2 mismatch signal — the factory should have ensured the tracker
    subscribes only to keys the loss emits). aggregation='mean' is the running
    mean of the scalar; 'max' is the running max (running_max starts at 0.0).
    """

    def __init__(
        self, head_name: str, component_key: str, aggregation: str, **kwargs
    ):
        if not head_name:
            raise ValueError("head_name must be a non-empty string")
        if not component_key:
            raise ValueError("component_key must be a non-empty string")
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATIONS}, got {aggregation!r}"
            )
        super().__init__(name=f"{head_name}/{component_key}/{aggregation}", **kwargs)
        self.head_name = head_name
        self.component_key = component_key
        self.aggregation = aggregation
        self._total = self.add_variable(shape=(), initializer="zeros", name="total")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")
        self._running_max = self.add_variable(
            shape=(), initializer="zeros", name="running_max"
        )

    def update_state(self, components_dict):
        if self.component_key not in components_dict:
            raise KeyError(
                f"LossComponentTracker {self.name!r} expects key "
                f"{self.component_key!r}; got keys {list(components_dict.keys())}"
            )
        value = tf.cast(components_dict[self.component_key], self._total.dtype)
        if self.aggregation == "mean":
            self._total.assign_add(value)
            self._count.assign_add(1.0)
        else:
            self._running_max.assign(tf.maximum(self._running_max, value))

    def result(self):
        if self.aggregation == "mean":
            return tf.math.divide_no_nan(self._total, self._count)
        return self._running_max

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)
        self._running_max.assign(0.0)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnostics_trackers.py::TestLossComponentTrackerBasics tests/test_diagnostics_trackers.py::TestLossComponentTrackerCrossHead tests/test_diagnostics_trackers.py::TestLossComponentTrackerProperties -v`
Expected: 12 passed (8 basics + 2 cross-head + 2 property).

Run: `pytest -q`
Expected: 260 passed (248 + 12).

**Step 5: Commit**

```bash
git add src/diagnostics/trackers.py tests/test_diagnostics_trackers.py
git commit -m "tier5 phase 1: LossComponentTracker + key/cross-head/property tests"
```
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_D -->

<!-- START_SUBCOMPONENT_E (task 6) -->
<!-- START_TASK_6 -->
### Task 6: BatchLabelBalanceTracker — implementation + edge cases + property tests

**Files:**
- Modify: `src/diagnostics/trackers.py`
- Modify: `tests/test_diagnostics_trackers.py`

**Step 1: Write the failing tests**

Append to `tests/test_diagnostics_trackers.py`:

```python
class TestBatchLabelBalanceTracker:
    def test_name(self):
        assert BatchLabelBalanceTracker(head_name="cca").name == "cca/positive_fraction"

    def test_empty_head_name_raises(self):
        with pytest.raises(ValueError, match="head_name"):
            BatchLabelBalanceTracker(head_name="")

    def test_balanced_batch(self):
        t = BatchLabelBalanceTracker(head_name="cca")
        t.update_state({"cca": tf.constant([1.0, 0.0, 1.0, 0.0])})
        assert float(t.result()) == pytest.approx(0.5)

    def test_all_positive_batch(self):
        t = BatchLabelBalanceTracker(head_name="cca")
        t.update_state({"cca": tf.constant([1.0, 1.0, 1.0])})
        assert float(t.result()) == pytest.approx(1.0)

    def test_all_negative_batch(self):
        t = BatchLabelBalanceTracker(head_name="cca")
        t.update_state({"cca": tf.constant([0.0, 0.0])})
        assert float(t.result()) == pytest.approx(0.0)

    def test_running_mean_across_batches(self):
        t = BatchLabelBalanceTracker(head_name="cca")
        t.update_state({"cca": tf.constant([1.0, 1.0, 0.0, 0.0])})  # 0.5
        t.update_state({"cca": tf.constant([1.0, 1.0, 1.0, 1.0])})  # 1.0
        assert float(t.result()) == pytest.approx(0.75, rel=1e-5)

    def test_missing_head_key_raises(self):
        t = BatchLabelBalanceTracker(head_name="cca")
        with pytest.raises(KeyError, match="cca"):
            t.update_state({"immig": tf.constant([1.0, 0.0])})

    def test_accepts_int_targets(self):
        t = BatchLabelBalanceTracker(head_name="cca")
        t.update_state({"cca": tf.constant([1, 0, 1, 1], dtype=tf.int32)})
        assert float(t.result()) == pytest.approx(0.75)

    def test_reset_state(self):
        t = BatchLabelBalanceTracker(head_name="cca")
        t.update_state({"cca": tf.constant([1.0, 1.0])})
        t.reset_state()
        assert float(t.result()) == 0.0


class TestBatchLabelBalanceProperties:
    _label_lists = st.lists(
        st.integers(min_value=0, max_value=1), min_size=1, max_size=64
    )

    @given(_label_lists)
    @settings(max_examples=50, deadline=None)
    def test_positive_fraction_in_unit_interval(self, labels):
        t = BatchLabelBalanceTracker(head_name="cca")
        t.update_state({"cca": tf.constant(labels, dtype=tf.float32)})
        assert 0.0 <= float(t.result()) <= 1.0

    @given(_label_lists)
    @settings(max_examples=50, deadline=None)
    def test_matches_numpy_mean(self, labels):
        t = BatchLabelBalanceTracker(head_name="cca")
        t.update_state({"cca": tf.constant(labels, dtype=tf.float32)})
        assert float(t.result()) == pytest.approx(
            float(np.mean(labels)), rel=1e-5, abs=1e-6
        )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnostics_trackers.py::TestBatchLabelBalanceTracker -v`
Expected: all fail with `AttributeError`.

**Step 3: Write minimal implementation**

Append to `src/diagnostics/trackers.py`:

```python
class BatchLabelBalanceTracker(keras.metrics.Metric):
    """Running mean of mean(y[head_name]) per batch — the positive-class
    fraction. Raises KeyError if the head's targets are absent from the y dict
    (Layer-2 mismatch signal).
    """

    def __init__(self, head_name: str, **kwargs):
        if not head_name:
            raise ValueError("head_name must be a non-empty string")
        super().__init__(name=f"{head_name}/positive_fraction", **kwargs)
        self.head_name = head_name
        self._total = self.add_variable(shape=(), initializer="zeros", name="total")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y):
        if self.head_name not in y:
            raise KeyError(
                f"BatchLabelBalanceTracker {self.name!r} expects key "
                f"{self.head_name!r}; got keys {list(y.keys())}"
            )
        targets = tf.cast(y[self.head_name], self._total.dtype)
        self._total.assign_add(tf.reduce_mean(targets))
        self._count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)
```

**Step 4: Run tests + full suite**

Run: `pytest tests/test_diagnostics_trackers.py -v`
Expected: ~50 passed total across all four trackers.

Run: `pytest -q`
Expected: prior total + the new `BatchLabelBalanceTracker` tests, **zero failures, zero errors**. The exact integer is illustrative only — the gate is "all green," and Task 7 records the authoritative final number as the Phase 2 baseline.

**Step 5: Commit**

```bash
git add src/diagnostics/trackers.py tests/test_diagnostics_trackers.py
git commit -m "tier5 phase 1: BatchLabelBalanceTracker + edge/property tests"
```
<!-- END_TASK_6 -->
<!-- END_SUBCOMPONENT_E -->

<!-- START_TASK_7 -->
### Task 7: Phase 1 integration verification

**Type:** Verification (no new code; confirms phase-end state). No commit.

**Step 1: Confirm module imports cleanly**

Run: `python -c "from src.diagnostics.trackers import PerGroupGradNormTracker, GradientFiniteTracker, LossComponentTracker, BatchLabelBalanceTracker; print('OK')"`
Expected: `OK`

**Step 2: Confirm full suite passes with no regressions**

Run: `pytest -q`
Expected: All tests pass — 220 baseline + all new Phase 1 tracker tests. Zero failures, zero errors. Record the exact number; it becomes the new baseline for Phase 2.

**Step 3: Confirm working tree is clean**

Run: `git status --short`
Expected: clean (all Phase 1 work committed).

Run: `git log --oneline 9136195..HEAD`
Expected: exactly 6 new Phase-1 commits — one per Task 1–6 (Task 3 commits test-only changes; Task 7 is verification-only and produces no commit).

**Phase 1 Done-when criteria (from design plan):**
- All new tracker unit and property tests pass. ✓ (Step 2)
- Existing 220-test suite still passes. ✓ (Step 2)
- New module imports cleanly. ✓ (Step 1)

Phase 1 is complete.
<!-- END_TASK_7 -->
