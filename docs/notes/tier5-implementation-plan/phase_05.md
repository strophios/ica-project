# Tier 5 Implementation Plan — Phase 5: Prediction-Distribution Metrics

> **This phase supersedes the original design's "Periodic diagnostic + callback."**
> See the supersession note in `docs/notes/tier5-design.md` ("Diagnostic
> instrumentation: module layout"). Decision rationale: the periodic /
> reference-batch / `DiagnosticsCallback` subsystem was found largely
> redundant with the validation step and prone to a metric-pollution problem
> (the endpoint model's forward pass mutates head metric state because
> `ClassificationHead.call` is targets-gated, not training-gated). The
> valuable signal — output-distribution shape — is obtained more cheaply and
> correctly as **per-head `keras.metrics.Metric`s** riding the existing
> validated `ClassificationHead.metric_objs` path. No `PeriodicDiagnostic`,
> no `DiagnosticsCallback`, no reference batch, no `every_n_batches`.

**Goal:** Implement per-head prediction-distribution metrics (`sigmoid(logits)` mean / std / frac-above-0.5) as ordinary `keras.metrics.Metric`s plus a `make_distribution_metrics(config)` factory, so Phase 6 can add them to the head's `metrics=` list alongside `make_cca_metrics()`. They then compute for both the train and val phases per epoch, with zero extra forward passes and no metric pollution.

**Architecture:** Three small `keras.metrics.Metric` subclasses with the standard `update_state(y_true, y_pred, sample_weight=None)` signature (they use only `y_pred` = logits; `y_true` ignored). They ride the head's existing metric machinery: `ClassificationHead.__init__` already accepts `metrics=`, clones each via `m.__class__.from_config(m.get_config())` with a `{head}_` name prefix, stores them in `self.metric_objs`, and updates them inside `call()` whenever targets are present (train + val). `DiagnosticBundle["periodic"]` stays the permanently-empty forward-compat slot from Phase 2 — these metrics do **not** go through the bundle or `LayerLRModel.train_step` dispatch.

**Tech Stack:** Python ≥3.12, Keras 3.12 (`keras.ops`), TensorFlow, pytest, hypothesis. New file `src/diagnostics/distribution_metrics.py` → FCIS `# pattern: Mixed (unavoidable)` (same justification as `trackers.py`: Keras `Metric` requires `tf.Variable` state).

**Scope:** Phase 5 of 8. Depends on Phase 2 (`DiagnosticsConfig` with `enable_prediction_distribution` + `prediction_summary_stats`). Independent of Phases 1/3/4 at the code level. Phase 6 wires the factory output into the head.

**Codebase verified:** 2026-05-16 via codebase-investigator + Phase 3 verbatim read of `heads.py`.

**Codebase verification findings (relevant subset):**
- `ClassificationHead.__init__(self, hidden_dim, dropout=0.1, loss_fn=None, metrics=None, *, name, expose_loss_components=False)` (the `expose_loss_components` param is added by Phase 3). The `metrics=` list is cloned per head: `config = m.get_config(); if not config["name"].startswith(f"{self.name}_"): config["name"] = f"{self.name}_{config['name']}"; self.metric_objs.append(m.__class__.from_config(config))` (heads.py ~133-152). **Contract consequence:** each distribution metric's `__init__` must accept `name` and `dtype` with defaults (so `cls.from_config({"name": ..., "dtype": ...})` reconstructs it) and add no required-without-default args. Keras's base `Metric.get_config` returns `{"name", "dtype"}` and `from_config` is `cls(**config)` — sufficient if we add no new constructor args.
- Metrics update inside `head.call` when `targets is not None` — fires in both `train_step` and `test_step` (val), so each metric yields a train value and a `val_` value per epoch automatically; Keras resets metric state between train and val phases and across epochs.
- `make_cca_metrics()` (`src/cca_metrics.py:39`) is the pattern to mirror: `() -> list[keras.metrics.Metric]`, fresh instances each call, names like `precision`/`recall`/`pr_auc` (head prefixes `{head}_`). `make_distribution_metrics` mirrors this but takes the `DiagnosticsConfig` (for the enable flag + stat list).
- `DiagnosticsConfig` (Phase 2): `enable_prediction_distribution: bool = True`, `prediction_summary_stats: tuple[str, ...] = ("mean", "std", "frac_above_0.5")`, validated against `_VALID_SUMMARY_STATS = ("mean", "std", "frac_above_0.5")` — so the factory may trust the values.
- `src/diagnostics/` already hosts `trackers.py`, `factory.py` (Phases 1-2). New sibling `distribution_metrics.py`.
- No existing `pred_dist`/distribution metric anywhere — greenfield.

**Key contracts:**
- `PredictionMeanMetric(name="pred_dist/mean", dtype=None)` — running mean of `sigmoid(logits)`.
- `PredictionStdMetric(name="pred_dist/std", dtype=None)` — population std of `sigmoid(logits)` over all samples seen since last reset: `sqrt(max(E[s²] − E[s]², 0))`.
- `PredictionFracAboveMetric(name="pred_dist/frac_above_0.5", dtype=None)` — fraction with `sigmoid(logits) > 0.5`.
- All: `update_state(y_true, y_pred, sample_weight=None)` (standard Keras head-metric signature; `y_true`/`sample_weight` ignored), `result()`, `reset_state()`, default-arg `__init__` so head-cloning via `from_config` works.
- `make_distribution_metrics(config: DiagnosticsConfig) -> list[keras.metrics.Metric]` — `[]` if `not config.enable_prediction_distribution`; otherwise one metric per entry of `config.prediction_summary_stats` (order preserved), fresh instances per call.

**Task structure (4 tasks, 1 subcomponent):**
- A — distribution metrics + factory (Tasks 1–2)
- head-integration test (Task 3)
- Phase integration verification (Task 4)

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: distribution metric classes + unit/property tests

**Files:**
- Create: `src/diagnostics/distribution_metrics.py`
- Create: `tests/test_diagnostics_distribution_metrics.py`

**Step 1: Write the failing tests**

Create `tests/test_diagnostics_distribution_metrics.py`:

```python
# pattern: Functional Core

"""Unit + property-based tests for src/diagnostics/distribution_metrics.py."""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf
from hypothesis import given, settings
from hypothesis import strategies as st

from src.diagnostics.distribution_metrics import (
    PredictionFracAboveMetric,
    PredictionMeanMetric,
    PredictionStdMetric,
)


def _logits(*vals):
    return tf.constant([[v] for v in vals], dtype=tf.float32)


def _sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.asarray(a, dtype=np.float64)))


class TestPredictionMeanMetric:
    def test_name_default(self):
        assert PredictionMeanMetric().name == "pred_dist/mean"

    def test_mean_of_sigmoid(self):
        m = PredictionMeanMetric()
        m.update_state(None, _logits(0.0, 0.0))  # sigmoid(0)=0.5
        assert float(m.result()) == pytest.approx(0.5, rel=1e-5)

    def test_accumulates_across_batches(self):
        m = PredictionMeanMetric()
        m.update_state(None, _logits(0.0))            # 0.5
        m.update_state(None, _logits(1e9, 1e9))       # ~1.0, ~1.0
        # mean of [0.5, 1.0, 1.0]
        assert float(m.result()) == pytest.approx(2.5 / 3.0, rel=1e-4)

    def test_reset_state(self):
        m = PredictionMeanMetric()
        m.update_state(None, _logits(1e9))
        m.reset_state()
        assert float(m.result()) == 0.0

    def test_from_config_roundtrip(self):
        # Head cloning relies on cls.from_config(m.get_config()).
        m = PredictionMeanMetric()
        clone = PredictionMeanMetric.from_config(m.get_config())
        assert clone.name == m.name
        clone.update_state(None, _logits(0.0))
        assert float(clone.result()) == pytest.approx(0.5, rel=1e-5)

    def test_head_prefix_clone_pattern(self):
        # Exactly what ClassificationHead.__init__ does.
        m = PredictionMeanMetric()
        cfg = m.get_config()
        cfg["name"] = f"cca_{cfg['name']}"
        clone = PredictionMeanMetric.from_config(cfg)
        assert clone.name == "cca_pred_dist/mean"


class TestPredictionFracAboveMetric:
    def test_name_default(self):
        assert PredictionFracAboveMetric().name == "pred_dist/frac_above_0.5"

    def test_fraction_above_half(self):
        m = PredictionFracAboveMetric()
        # sigmoid>0.5 iff logit>0. Two of four above.
        m.update_state(None, _logits(2.0, 3.0, -1.0, -2.0))
        assert float(m.result()) == pytest.approx(0.5, rel=1e-5)

    def test_all_below(self):
        m = PredictionFracAboveMetric()
        m.update_state(None, _logits(-5.0, -5.0))
        assert float(m.result()) == 0.0

    def test_reset_state(self):
        m = PredictionFracAboveMetric()
        m.update_state(None, _logits(5.0))
        m.reset_state()
        assert float(m.result()) == 0.0


class TestPredictionStdMetric:
    def test_name_default(self):
        assert PredictionStdMetric().name == "pred_dist/std"

    def test_zero_std_constant_input(self):
        m = PredictionStdMetric()
        m.update_state(None, _logits(0.0, 0.0, 0.0))  # all 0.5
        assert float(m.result()) == pytest.approx(0.0, abs=1e-5)

    def test_matches_numpy_population_std(self):
        m = PredictionStdMetric()
        logits = [2.0, -1.0, 0.5, -3.0, 1.0]
        m.update_state(None, _logits(*logits))
        expected = float(np.std(_sigmoid(logits)))  # population std (ddof=0)
        assert float(m.result()) == pytest.approx(expected, rel=1e-4, abs=1e-5)

    def test_accumulates_across_batches(self):
        m = PredictionStdMetric()
        m.update_state(None, _logits(2.0, -1.0))
        m.update_state(None, _logits(0.5, -3.0, 1.0))
        expected = float(np.std(_sigmoid([2.0, -1.0, 0.5, -3.0, 1.0])))
        assert float(m.result()) == pytest.approx(expected, rel=1e-4, abs=1e-5)

    def test_reset_state(self):
        m = PredictionStdMetric()
        m.update_state(None, _logits(2.0, -3.0))
        m.reset_state()
        assert float(m.result()) == 0.0


_logit_lists = st.lists(
    st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=64,
)


class TestDistributionMetricProperties:
    @given(_logit_lists)
    @settings(max_examples=50, deadline=None)
    def test_mean_in_unit_interval(self, logits):
        m = PredictionMeanMetric()
        m.update_state(None, _logits(*logits))
        assert 0.0 <= float(m.result()) <= 1.0

    @given(_logit_lists)
    @settings(max_examples=50, deadline=None)
    def test_frac_in_unit_interval(self, logits):
        m = PredictionFracAboveMetric()
        m.update_state(None, _logits(*logits))
        assert 0.0 <= float(m.result()) <= 1.0

    @given(_logit_lists)
    @settings(max_examples=50, deadline=None)
    def test_std_non_negative_and_matches_numpy(self, logits):
        m = PredictionStdMetric()
        m.update_state(None, _logits(*logits))
        r = float(m.result())
        assert r >= 0.0
        assert r == pytest.approx(float(np.std(_sigmoid(logits))), rel=1e-3, abs=1e-4)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnostics_distribution_metrics.py -v`
Expected: all fail with `ModuleNotFoundError: No module named 'src.diagnostics.distribution_metrics'`.

**Step 3: Write minimal implementation**

Create `src/diagnostics/distribution_metrics.py`:

```python
# pattern: Mixed (unavoidable)
# Reason: Keras Metric subclasses must hold tf.Variable state for cross-batch
# aggregation; they cannot be pure functions. The statistic arithmetic is
# pure, but the surrounding Metric protocol (update_state + result +
# reset_state over persistent state vars) is inherently stateful.

"""Per-head prediction-distribution metrics for Tier 5.

These are ordinary keras.metrics.Metric subclasses (standard
update_state(y_true, y_pred) signature; y_true ignored). They are passed
into ClassificationHead(metrics=...) alongside make_cca_metrics() and ride
the head's metric_objs path — computed for both the train and val phases
per epoch, no extra forward pass, no metric pollution. They do NOT go
through DiagnosticBundle or LayerLRModel.train_step dispatch.

Supersedes the original design's PeriodicDiagnostic/DiagnosticsCallback
subsystem (see tier5-design.md supersession note).
"""

from __future__ import annotations

import keras
from keras import ops

from src.cca_config import DiagnosticsConfig

__all__ = [
    "PredictionMeanMetric",
    "PredictionStdMetric",
    "PredictionFracAboveMetric",
    "make_distribution_metrics",
]


class PredictionMeanMetric(keras.metrics.Metric):
    """Running mean of sigmoid(logits) since last reset."""

    def __init__(self, name="pred_dist/mean", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self._total = self.add_variable(shape=(), initializer="zeros", name="total")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        s = ops.sigmoid(ops.cast(y_pred, "float32"))
        self._total.assign_add(ops.sum(s))
        self._count.assign_add(ops.cast(ops.size(s), self._count.dtype))

    def result(self):
        return ops.divide_no_nan(self._total, self._count)

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)


class PredictionFracAboveMetric(keras.metrics.Metric):
    """Fraction of sigmoid(logits) strictly greater than 0.5
    (equivalently, fraction of logits > 0)."""

    def __init__(self, name="pred_dist/frac_above_0.5", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self._above = self.add_variable(shape=(), initializer="zeros", name="above")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        s = ops.sigmoid(ops.cast(y_pred, "float32"))
        above = ops.cast(s > 0.5, "float32")
        self._above.assign_add(ops.sum(above))
        self._count.assign_add(ops.cast(ops.size(s), self._count.dtype))

    def result(self):
        return ops.divide_no_nan(self._above, self._count)

    def reset_state(self):
        self._above.assign(0.0)
        self._count.assign(0.0)


class PredictionStdMetric(keras.metrics.Metric):
    """Population std (ddof=0) of sigmoid(logits) over all samples seen
    since last reset: sqrt(max(E[s^2] - E[s]^2, 0))."""

    def __init__(self, name="pred_dist/std", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self._sum = self.add_variable(shape=(), initializer="zeros", name="sum")
        self._sum_sq = self.add_variable(shape=(), initializer="zeros", name="sum_sq")
        self._count = self.add_variable(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):
        s = ops.sigmoid(ops.cast(y_pred, "float32"))
        self._sum.assign_add(ops.sum(s))
        self._sum_sq.assign_add(ops.sum(s * s))
        self._count.assign_add(ops.cast(ops.size(s), self._count.dtype))

    def result(self):
        mean = ops.divide_no_nan(self._sum, self._count)
        mean_sq = ops.divide_no_nan(self._sum_sq, self._count)
        var = ops.maximum(mean_sq - mean * mean, 0.0)
        return ops.sqrt(var)

    def reset_state(self):
        self._sum.assign(0.0)
        self._sum_sq.assign(0.0)
        self._count.assign(0.0)


_STAT_TO_METRIC = {
    "mean": PredictionMeanMetric,
    "std": PredictionStdMetric,
    "frac_above_0.5": PredictionFracAboveMetric,
}


def make_distribution_metrics(
    config: DiagnosticsConfig,
) -> list[keras.metrics.Metric]:
    """Construct the configured per-head prediction-distribution metrics.

    Returns [] when config.enable_prediction_distribution is False.
    Otherwise one fresh metric per entry of config.prediction_summary_stats
    (order preserved). Fresh instances each call — ClassificationHead clones
    them per head anyway; callers should not share the list across heads
    (mirrors make_cca_metrics).
    """
    if not config.enable_prediction_distribution:
        return []
    return [_STAT_TO_METRIC[stat]() for stat in config.prediction_summary_stats]
```

> Executor note: `ops.divide_no_nan` and `ops.size` are Keras 3 ops (same
> namespace `loss.py` uses). If `ops.size` is unavailable in the pinned
> Keras build, substitute `ops.cast(ops.shape(s)[0], "float32")` for the
> 1-D logit column (shape `(batch, 1)` → use the flattened element count;
> prefer `ops.size` if present). Verify during GREEN.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnostics_distribution_metrics.py -v`
Expected: all unit + property tests pass (~20 tests).

Run: `pytest -q`
Expected: Phase-4 baseline + new, zero failures.

**Step 5: Commit**

```bash
git add src/diagnostics/distribution_metrics.py tests/test_diagnostics_distribution_metrics.py
git commit -m "tier5 phase 5: per-head prediction-distribution metrics (mean/std/frac_above)"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: make_distribution_metrics factory + tests

**Files:**
- Modify: `tests/test_diagnostics_distribution_metrics.py` (new class `TestMakeDistributionMetrics`) — implementation already added in Task 1 Step 3.

**Step 1: Write the tests**

Append to `tests/test_diagnostics_distribution_metrics.py`:

```python
from src.cca_config import DiagnosticsConfig
from src.diagnostics.distribution_metrics import make_distribution_metrics


class TestMakeDistributionMetrics:
    def test_default_config_builds_all_three_in_order(self):
        metrics = make_distribution_metrics(DiagnosticsConfig())
        assert [m.name for m in metrics] == [
            "pred_dist/mean", "pred_dist/std", "pred_dist/frac_above_0.5"
        ]

    def test_disabled_returns_empty(self):
        metrics = make_distribution_metrics(
            DiagnosticsConfig(enable_prediction_distribution=False)
        )
        assert metrics == []

    def test_subset_and_order_preserved(self):
        cfg = DiagnosticsConfig(prediction_summary_stats=("frac_above_0.5", "mean"))
        metrics = make_distribution_metrics(cfg)
        assert [m.name for m in metrics] == [
            "pred_dist/frac_above_0.5", "pred_dist/mean"
        ]

    def test_fresh_instances_each_call(self):
        cfg = DiagnosticsConfig()
        a = make_distribution_metrics(cfg)
        b = make_distribution_metrics(cfg)
        assert all(x is not y for x, y in zip(a, b))

    def test_instances_are_metric_subclasses(self):
        import keras
        for m in make_distribution_metrics(DiagnosticsConfig()):
            assert isinstance(m, keras.metrics.Metric)
```

**Step 2: Run tests** — `pytest tests/test_diagnostics_distribution_metrics.py::TestMakeDistributionMetrics -v`
Expected: 5 passed (factory implemented in Task 1).

**Step 3:** No implementation change.

**Step 4: Full suite** — `pytest -q`. Expected: prior + 5, zero failures.

**Step 5: Commit**

```bash
git add tests/test_diagnostics_distribution_metrics.py
git commit -m "tier5 phase 5: make_distribution_metrics factory tests"
```
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_3 -->
### Task 3: Head-integration test — distribution metrics ride the head metric path

**Files:**
- Modify: `tests/test_heads.py` (new class `TestDistributionMetricsIntegration`) — test-only; verifies the head clones our metrics correctly and they update in `call()`.

This is the proof obligation: `ClassificationHead.__init__` clones each `metrics=` entry via `m.__class__.from_config(m.get_config())` with a `{head}_` name prefix. If our metrics don't survive that round-trip, head construction or per-head naming breaks.

**Step 1: Write the tests**

Append to `tests/test_heads.py` (reuse `_dummy_features`, `_dummy_targets`, `HIDDEN_DIM`):

```python
class TestDistributionMetricsIntegration:
    def _metrics(self):
        from src.cca_config import DiagnosticsConfig
        from src.diagnostics.distribution_metrics import make_distribution_metrics
        return make_distribution_metrics(DiagnosticsConfig())

    def test_head_clones_distribution_metrics_with_prefix(self):
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            metrics=self._metrics(),
        )
        names = {m.name for m in head.metric_objs}
        assert "cca_pred_dist/mean" in names
        assert "cca_pred_dist/std" in names
        assert "cca_pred_dist/frac_above_0.5" in names

    def test_distribution_metrics_update_during_call(self):
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            metrics=self._metrics(),
        )
        _ = head(_dummy_features(), targets=_dummy_targets())
        by_name = {m.name: m for m in head.metric_objs}
        mean_v = float(by_name["cca_pred_dist/mean"].result())
        frac_v = float(by_name["cca_pred_dist/frac_above_0.5"].result())
        std_v = float(by_name["cca_pred_dist/std"].result())
        assert 0.0 <= mean_v <= 1.0
        assert 0.0 <= frac_v <= 1.0
        assert std_v >= 0.0

    def test_distribution_metrics_coexist_with_cca_metrics(self):
        from src.cca_metrics import make_cca_metrics
        combined = make_cca_metrics() + self._metrics()
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            metrics=combined,
        )
        names = {m.name for m in head.metric_objs}
        # quality metrics + distribution metrics both present, prefixed
        assert "cca_precision" in names
        assert "cca_pred_dist/mean" in names

    def test_no_distribution_metrics_when_disabled(self):
        from src.cca_config import DiagnosticsConfig
        from src.diagnostics.distribution_metrics import make_distribution_metrics
        head = ClassificationHead(
            hidden_dim=HIDDEN_DIM,
            loss_fn=FLPULoss(prior=0.1),
            name="cca",
            metrics=make_distribution_metrics(
                DiagnosticsConfig(enable_prediction_distribution=False)
            ),
        )
        assert not any("pred_dist" in m.name for m in head.metric_objs)
```

**Step 2: Run tests** — `pytest tests/test_heads.py::TestDistributionMetricsIntegration -v`
Expected: 4 passed. If `test_head_clones_distribution_metrics_with_prefix` fails with a `from_config` error, the metric `__init__` signature is incompatible with the head's clone path — fix `__init__` to accept only `name`/`dtype` (no required args), do not work around by changing the head.

**Step 3:** No implementation change.

**Step 4: Full suite** — `pytest -q`. Expected: prior + 4, zero failures (existing `test_heads.py` unchanged — `metrics=` path already validated pre-Tier-5).

**Step 5: Commit**

```bash
git add tests/test_heads.py
git commit -m "tier5 phase 5: head-integration tests for distribution metrics (clone-safe)"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Phase 5 integration verification

**Type:** Verification (no new code; no commit).

**Step 1: Confirm imports + contract**

Run: `python -c "from src.diagnostics.distribution_metrics import PredictionMeanMetric, PredictionStdMetric, PredictionFracAboveMetric, make_distribution_metrics; from src.cca_config import DiagnosticsConfig; print([m.name for m in make_distribution_metrics(DiagnosticsConfig())])"`
Expected: `['pred_dist/mean', 'pred_dist/std', 'pred_dist/frac_above_0.5']`

**Step 2: Full suite, no regressions**

Run: `pytest -q`
Expected: Phase-4 total + Phase-5 additions, zero failures. Confirm `tests/test_heads.py` pre-existing tests still green (the `metrics=` clone path is unchanged; we only added new metric instances that flow through it). Record the number (Phase 6 baseline).

**Step 3: Working tree clean**

Run: `git status --short` → clean.
Run: `git log --oneline 9136195..HEAD` → Phase-1..4 commits + 3 Phase-5 task commits.

**Phase 5 Done-when criteria (revised per supersession):**
- The three distribution `Metric` classes pass unit + property tests (sigmoid applied; mean/frac ∈ [0,1]; std ≥ 0 and matches numpy population std; reset works). ✓ (Task 1)
- `make_distribution_metrics` honors `enable_prediction_distribution` and `prediction_summary_stats` (order preserved, `[]` when disabled). ✓ (Task 2)
- Metrics survive the `ClassificationHead` `get_config`/`from_config` clone with `{head}_` prefix and update during `call()`. ✓ (Task 3)
- No `PeriodicDiagnostic`/`DiagnosticsCallback`/reference-batch artifacts created; `DiagnosticBundle["periodic"]` untouched (still permanently `[]`). ✓
- Existing suite green. ✓ (Step 2)

Phase 5 is complete. **Phase 6 dependency:** wire `make_cca_metrics() + make_distribution_metrics(config.diagnostics)` into the CCA head's `metrics=` list (in `build_endpoint_model`/`run_cca_classification.py`); the smoke test will then see `pred_dist/*` (and `val_*pred_dist/*`) as ordinary history/CSV columns — no callback wiring.
<!-- END_TASK_4 -->
