# Tier 5 Implementation Plan — Phase 2: Config and Factory

**Goal:** Integrate `DiagnosticsConfig` into `RunConfig` (mirroring the existing sub-config pattern, with `default_factory` back-compat) and implement `src/diagnostics/factory.py:build_trackers()` constructing the tracker set from config + model state.

**Architecture:** `DiagnosticsConfig` is a frozen dataclass embedded as the **last** `RunConfig` field via `dataclasses.field(default_factory=DiagnosticsConfig)`. The factory walks `trainable_variables` once to enumerate groups, honors enable flags, and gates loss-component trackers behind a loss-introspection guard. Validation hierarchy follows the project's defense-in-depth split: `DiagnosticsConfig.__post_init__` (own fields) → `RunConfig.__post_init__` type-defense (correct sub-config type) → factory call-site (derived group/head/loss consistency with the model). See the "Diagnostics vs. head metrics" subsection in `docs/notes/tier5-design.md` for why diagnostics and head metrics are separate, non-shared surfaces.

**Tech Stack:** Python ≥3.12, Keras 3, TensorFlow, pytest, hypothesis. New in this phase: `typing.TypedDict`, `inspect.signature`, `TYPE_CHECKING`-guarded import.

**Scope:** Phase 2 of 8. Depends on Phase 1 (tracker classes exist).

**Codebase verified:** 2026-05-16 via codebase-investigator + direct reads.

**Codebase verification findings:**
- Sub-config pattern (verbatim from `FLPULossConfig`, `src/cca_config.py:104–133`): `@dataclasses.dataclass(frozen=True)`; `__post_init__` raises `ValueError` (no `object.__setattr__` coercion); `_from_dict(cls, payload, _source="<dict>")` classmethod calling `_filter_known_fields(cls, payload, _source=_source)` then `cls(**kwargs)`.
- `_filter_known_fields` (`src/cca_config.py:762–795`): warns via `warnings.warn(msg, stacklevel=3)` on unknown keys; raises `ValueError` only on missing fields with neither `default` nor `default_factory`. `DiagnosticsConfig` is all-defaulted → no required-field raises.
- JSON: `RunConfig.to_json` uses `dataclasses.asdict(self)` then `json.dump`; `from_json` → `_from_dict` reconstructs sub-configs bottom-up. `dataclasses.asdict` + `json` serializes `tuple` fields as JSON arrays → they deserialize as **lists**; `_from_dict` must coerce list→tuple.
- Back-compat precedent: `LRScheduleConfig.resolved` (`src/cca_config.py:321`) uses `"ResolvedSteps | None" = None` + `payload.get("resolved")`. `DiagnosticsConfig` deliberately differs: a missing key should yield a *fully-enabled* config, so `default_factory=DiagnosticsConfig`; `RunConfig._from_dict` omits the key from kwargs when absent (lets the default_factory fire) rather than passing `None`.
- `RunConfig` fields (`src/cca_config.py:489–497`): `seq_length, text_key, target_dtype, heads, epochs, backbone_weights_path, ratio_batch, lr_schedule, optimizer` — **all 9 required, no defaults**. A defaulted field cannot precede a non-defaulted one, so `diagnostics` (defaulted) must be field #10, immediately after `optimizer: OptimizerConfig` (line 497). This also yields in-code back-compat: existing `RunConfig(...)` calls without `diagnostics` get the default.
- `_default_group_fn` (`src/model_setup/assembly.py:54–65`): `return variable.path.split("/")[0]` → `Callable[[Variable], str]`. The factory receives `group_fn`; Phase 2 tests pass an explicit one (real `_default_group_fn` wired in Phase 6).
- `ClassificationHead.loss_fn` (`src/model_setup/heads.py:131`): a `keras.losses.Loss` or `None` (standard mode). Guard introspects `inspect.signature(head.loss_fn.call)`.
- `tests/test_cca_config.py`: class-based; back-compat precedent `test_loads_pre_resolved_sidecar_without_key` (deletes a key from an `asdict` payload, reloads, asserts default fired).
- No `TypedDict`/`DiagnosticBundle` exists yet. `typing` usage in `src/` is minimal (`Any`, `Callable`, `Optional`). `from __future__ import annotations` is in use in `cca_config.py`.

**Resolved design decisions (folded in):**
- `src/diagnostics/factory.py` is FCIS `# pattern: Functional Core` (config + model state in, a `DiagnosticBundle` of objects out, no I/O; constructed trackers are stateful but construction is pure).
- `cca_config.py` defines its **own** `_VALID_GRADIENT_AGGREGATIONS = ("max", "mean")` literal (duplicate of `trackers._VALID_AGGREGATIONS`) with a linking comment — preserves the invariant that `cca_config` describes runs without importing the machinery it configures (`FLPULossConfig` doesn't import `FLPULoss`, etc.). Drift is prevented by a dedicated **sync test** (Task 1) that imports both and asserts equality.
- Loss-component tracker aggregation is hardcoded `"mean"` via module constant `_FLPU_LOSS_COMPONENT_AGG` (design DoD only needs mean-tracked components; YAGNI — no config knob).
- Factory tests use a synthetic `_StubHead` (object exposing only `.loss_fn`), not real `ClassificationHead` — decouples Phase 2 from `heads.py` and from Phase 3's not-yet-existing `FLPULoss.return_intermediates`. Real wiring is exercised in Phase 6.

**Key contracts:**

> **Revised 2026-05-16 (Phase 5 simplification ripple).** The Phase 5
> periodic/reference-batch subsystem was superseded by per-head distribution
> metrics (see the supersession note in `tier5-design.md` and
> `phase_05.md`). Consequently the `prediction_reference_batch_path`,
> `prediction_reference_batch_n`, `periodic_update_freq`, and
> `periodic_update_n` fields are **removed** from `DiagnosticsConfig` —
> carrying them would be validated-but-dead serialized config.
> `enable_prediction_distribution` and `prediction_summary_stats` are
> **kept** (they now gate/parameterize the per-head distribution metrics
> built in Phase 5/6). `DiagnosticBundle["periodic"]` is retained as a
> documented permanently-empty forward-compat slot (Phase 4 never reads it).

`DiagnosticsConfig` (frozen dataclass, all fields defaulted):
```
enable_gradient_norms: bool = True
enable_overflow_proxy: bool = True
enable_loss_components: bool = True
enable_batch_balance: bool = True
enable_prediction_distribution: bool = True
gradient_norm_aggregations: tuple[str, ...] = ("max", "mean")
prediction_summary_stats: tuple[str, ...] = ("mean", "std", "frac_above_0.5")
```

`build_trackers(config: DiagnosticsConfig, *, group_fn, heads: dict[str, "ClassificationHead"], trainable_variables) -> DiagnosticBundle`:
- Groups: `sorted({group_fn(v) for v in trainable_variables})` (deterministic). **Frozen-encoder reconciliation:** under the default frozen encoder `trainable_variables` holds only head variables, so `"encoder"` is not a group and no encoder grad-norm tracker is built. This matches design line 231 ("walks `trainable_variables`"); design line 184's "PerGroupGradNormTracker('encoder',…) reports 0.0" describes the Phase 1 tracker's *defensive* empty-group behavior for configs where the group IS enumerated but a step lacks those grads — not a contradiction.
- `per_step = {"gradient": [...], "loss_component": [...], "batch_target": [...]}`; `periodic = []` — **permanently empty forward-compat slot.** Per-head prediction-distribution metrics (the Phase 5 supersession) ride the head's `metric_objs` path, NOT the `DiagnosticBundle`; the factory does not build them. `enable_prediction_distribution` / `prediction_summary_stats` are consumed by `make_distribution_metrics` (Phase 5) wired into the head's `metrics=` list (Phase 6), not by `build_trackers`. Task 5 asserts `periodic == []` regardless of config.
- Loss-component guard: `enable_loss_components` and **zero** heads' losses expose `return_intermediates` → `raise ValueError`; per individual head whose loss lacks it → `warnings.warn(..., stacklevel=2)` and skip that head's loss-component trackers.
- `_FLPU_COMPONENT_KEYS = ("positive_risk", "negative_risk", "correction_triggered")`; `_FLPU_LOSS_COMPONENT_AGG = "mean"`.
- **Phase-ordering note:** `FLPULoss.return_intermediates` lands in Phase 3. Phase 2 tests the guard only with synthetic stand-in losses. The factory meets the real `FLPULoss` at Phase 6 (after Phase 3). Executor MUST NOT introspect the real `FLPULoss` in Phase 2.

`DiagnosticBundle` (TypedDict, in `factory.py`):
```python
class DiagnosticBundle(TypedDict):
    per_step: dict[str, list[keras.metrics.Metric]]   # keys: gradient | loss_component | batch_target
    periodic: list                                     # permanently [] — forward-compat slot (no current consumer)
```

**Task structure (7 tasks, 3 subcomponents):**
- A — DiagnosticsConfig (Tasks 1–2)
- B — RunConfig integration (Task 3)
- C — factory (Tasks 4–6)
- Phase integration verification (Task 7)

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: DiagnosticsConfig dataclass + `__post_init__` validation + sync test

**Files:**
- Modify: `src/cca_config.py` (add module constants + `DiagnosticsConfig` class, placed after `OptimizerConfig` block ending ~line 467 and before `@dataclasses.dataclass(frozen=True)` for `RunConfig` at line 468)
- Modify: `tests/test_cca_config.py` (new class `TestDiagnosticsConfigValidation` + sync test)

**Step 1: Write the failing tests**

Append to `tests/test_cca_config.py`:

```python
class TestDiagnosticsConfigValidation:
    def test_default_constructs(self):
        from src.cca_config import DiagnosticsConfig

        c = DiagnosticsConfig()
        assert c.enable_gradient_norms is True
        assert c.enable_prediction_distribution is True
        assert c.gradient_norm_aggregations == ("max", "mean")
        assert c.prediction_summary_stats == ("mean", "std", "frac_above_0.5")

    def test_non_bool_enable_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="enable_gradient_norms"):
            DiagnosticsConfig(enable_gradient_norms=1)

    def test_empty_gradient_aggregations_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="gradient_norm_aggregations"):
            DiagnosticsConfig(gradient_norm_aggregations=())

    def test_invalid_gradient_aggregation_value_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="gradient_norm_aggregations"):
            DiagnosticsConfig(gradient_norm_aggregations=("median",))

    def test_non_tuple_gradient_aggregations_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="gradient_norm_aggregations"):
            DiagnosticsConfig(gradient_norm_aggregations=["max"])

    def test_invalid_summary_stat_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="prediction_summary_stats"):
            DiagnosticsConfig(prediction_summary_stats=("variance",))

    def test_non_tuple_summary_stats_raises(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.raises(ValueError, match="prediction_summary_stats"):
            DiagnosticsConfig(prediction_summary_stats=["mean"])

    def test_enable_prediction_distribution_false_ok(self):
        from src.cca_config import DiagnosticsConfig

        assert DiagnosticsConfig(enable_prediction_distribution=False).enable_prediction_distribution is False


class TestDiagnosticsAggregationConstantSync:
    def test_valid_aggregations_in_sync_with_trackers(self):
        # Drift guard: cca_config deliberately duplicates the
        # ("max","mean") literal rather than importing from
        # diagnostics.trackers (preserves the "config does not import
        # the machinery it configures" invariant). This test recovers
        # drift-prevention at the test boundary.
        from src.cca_config import _VALID_GRADIENT_AGGREGATIONS
        from src.diagnostics.trackers import _VALID_AGGREGATIONS

        assert _VALID_GRADIENT_AGGREGATIONS == _VALID_AGGREGATIONS
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cca_config.py::TestDiagnosticsConfigValidation tests/test_cca_config.py::TestDiagnosticsAggregationConstantSync -v`
Expected: all fail with `ImportError` (`DiagnosticsConfig` / `_VALID_GRADIENT_AGGREGATIONS` not defined).

**Step 3: Write minimal implementation**

In `src/cca_config.py`, after the `OptimizerConfig` class block and before `@dataclasses.dataclass(frozen=True)` / `class RunConfig` (line 468), add:

```python
# Valid per-step gradient-norm aggregations a run may request. Deliberately
# duplicates src.diagnostics.trackers._VALID_AGGREGATIONS rather than
# importing it: cca_config describes runs declaratively and must not import
# the machinery it configures (FLPULossConfig does not import FLPULoss,
# OptimizerConfig does not import the optimizer, etc.). The two literals are
# pinned equal by TestDiagnosticsAggregationConstantSync in
# tests/test_cca_config.py — change both together.
_VALID_GRADIENT_AGGREGATIONS = ("max", "mean")
_VALID_SUMMARY_STATS = ("mean", "std", "frac_above_0.5")


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    """Tier 5 diagnostic instrumentation configuration.

    Embedded as the (defaulted) last field of RunConfig. A pre-Tier-5
    sidecar lacking the 'diagnostics' key reconstructs as DiagnosticsConfig()
    (all enabled) via RunConfig's default_factory — see RunConfig._from_dict.

    Group names are NOT stored here; they are derived from the model at
    factory build time by walking trainable_variables with group_fn.
    """

    enable_gradient_norms: bool = True
    enable_overflow_proxy: bool = True
    enable_loss_components: bool = True
    enable_batch_balance: bool = True
    enable_prediction_distribution: bool = True
    gradient_norm_aggregations: tuple[str, ...] = ("max", "mean")
    prediction_summary_stats: tuple[str, ...] = ("mean", "std", "frac_above_0.5")

    def __post_init__(self):
        for field_name in (
            "enable_gradient_norms",
            "enable_overflow_proxy",
            "enable_loss_components",
            "enable_batch_balance",
            "enable_prediction_distribution",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(
                    f"DiagnosticsConfig.{field_name} must be bool; "
                    f"got {type(value).__name__}."
                )

        if (
            not isinstance(self.gradient_norm_aggregations, tuple)
            or len(self.gradient_norm_aggregations) == 0
        ):
            raise ValueError(
                "DiagnosticsConfig.gradient_norm_aggregations must be a "
                f"non-empty tuple; got {self.gradient_norm_aggregations!r}."
            )
        for agg in self.gradient_norm_aggregations:
            if agg not in _VALID_GRADIENT_AGGREGATIONS:
                raise ValueError(
                    "DiagnosticsConfig.gradient_norm_aggregations entries "
                    f"must be in {_VALID_GRADIENT_AGGREGATIONS}; got {agg!r}."
                )

        if (
            not isinstance(self.prediction_summary_stats, tuple)
            or len(self.prediction_summary_stats) == 0
        ):
            raise ValueError(
                "DiagnosticsConfig.prediction_summary_stats must be a "
                f"non-empty tuple; got {self.prediction_summary_stats!r}."
            )
        for stat in self.prediction_summary_stats:
            if stat not in _VALID_SUMMARY_STATS:
                raise ValueError(
                    "DiagnosticsConfig.prediction_summary_stats entries must "
                    f"be in {_VALID_SUMMARY_STATS}; got {stat!r}."
                )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cca_config.py::TestDiagnosticsConfigValidation tests/test_cca_config.py::TestDiagnosticsAggregationConstantSync -v`
Expected: 15 passed (14 validation + 1 sync). The sync test requires `src/diagnostics/trackers.py:_VALID_AGGREGATIONS` from Phase 1 — confirm Phase 1 is merged.

Run: `pytest -q`
Expected: Phase-1 baseline + 15 new, zero failures.

**Step 5: Commit**

```bash
git add src/cca_config.py tests/test_cca_config.py
git commit -m "tier5 phase 2: DiagnosticsConfig dataclass + validation + agg-sync test"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: DiagnosticsConfig._from_dict + list→tuple coercion + round-trip tests

**Files:**
- Modify: `src/cca_config.py` (add `_from_dict` classmethod to `DiagnosticsConfig`)
- Modify: `tests/test_cca_config.py` (new class `TestDiagnosticsConfigFromDict`)

**Step 1: Write the failing tests**

Append to `tests/test_cca_config.py`:

```python
class TestDiagnosticsConfigFromDict:
    def test_roundtrip_default(self):
        from src.cca_config import DiagnosticsConfig

        original = DiagnosticsConfig()
        payload = json.loads(json.dumps(dataclasses.asdict(original)))
        reconstructed = DiagnosticsConfig._from_dict(payload)
        assert reconstructed == original

    def test_roundtrip_non_default(self):
        from src.cca_config import DiagnosticsConfig

        original = DiagnosticsConfig(
            enable_overflow_proxy=False,
            enable_prediction_distribution=False,
            gradient_norm_aggregations=("mean",),
            prediction_summary_stats=("mean", "std"),
        )
        payload = json.loads(json.dumps(dataclasses.asdict(original)))
        reconstructed = DiagnosticsConfig._from_dict(payload)
        assert reconstructed == original

    def test_list_payload_coerced_to_tuple(self):
        from src.cca_config import DiagnosticsConfig

        # JSON arrays deserialize as lists; _from_dict must coerce.
        payload = {
            "gradient_norm_aggregations": ["max", "mean"],
            "prediction_summary_stats": ["mean", "std", "frac_above_0.5"],
        }
        c = DiagnosticsConfig._from_dict(payload)
        assert isinstance(c.gradient_norm_aggregations, tuple)
        assert isinstance(c.prediction_summary_stats, tuple)
        assert c.gradient_norm_aggregations == ("max", "mean")

    def test_unknown_key_warns_and_ignored(self):
        from src.cca_config import DiagnosticsConfig

        with pytest.warns(UserWarning, match="Unknown field"):
            c = DiagnosticsConfig._from_dict({"enable_gradient_norms": False, "bogus": 1})
        assert c.enable_gradient_norms is False

    def test_missing_keys_use_defaults(self):
        from src.cca_config import DiagnosticsConfig

        # All fields defaulted → empty payload reconstructs the default.
        assert DiagnosticsConfig._from_dict({}) == DiagnosticsConfig()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cca_config.py::TestDiagnosticsConfigFromDict -v`
Expected: fail with `AttributeError: type object 'DiagnosticsConfig' has no attribute '_from_dict'`.

**Step 3: Write minimal implementation**

Add to the `DiagnosticsConfig` class body (after `__post_init__`):

```python
    @classmethod
    def _from_dict(
        cls, payload: dict, _source: str = "<dict>"
    ) -> "DiagnosticsConfig":
        kwargs = _filter_known_fields(cls, payload, _source=_source)
        # dataclasses.asdict + json serializes tuples as arrays; they
        # deserialize as lists. Coerce back so __post_init__'s tuple
        # checks (and equality with default tuples) hold.
        for tuple_field in ("gradient_norm_aggregations", "prediction_summary_stats"):
            if tuple_field in kwargs and isinstance(kwargs[tuple_field], list):
                kwargs[tuple_field] = tuple(kwargs[tuple_field])
        return cls(**kwargs)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cca_config.py::TestDiagnosticsConfigFromDict -v`
Expected: 5 passed.

Run: `pytest -q`
Expected: prior total + 5, zero failures.

**Step 5: Commit**

```bash
git add src/cca_config.py tests/test_cca_config.py
git commit -m "tier5 phase 2: DiagnosticsConfig._from_dict + list->tuple coercion"
```
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (task 3) -->
<!-- START_TASK_3 -->
### Task 3: Embed `diagnostics` on RunConfig — field, type-defense, back-compat `_from_dict`

**Files:**
- Modify: `src/cca_config.py` (RunConfig field #10; `RunConfig.__post_init__` type-defense; `RunConfig._from_dict` reconstruction)
- Modify: `tests/test_cca_config.py` (new class `TestRunConfigDiagnosticsIntegration`)

**Step 1: Write the failing tests**

Append to `tests/test_cca_config.py`:

```python
class TestRunConfigDiagnosticsIntegration:
    def test_default_factory_fires_when_constructed_without_diagnostics(self):
        from src.cca_config import DEFAULT_CCA_CONFIG, DiagnosticsConfig

        # DEFAULT_CCA_CONFIG does not pass diagnostics → default fires.
        assert DEFAULT_CCA_CONFIG.diagnostics == DiagnosticsConfig()

    def test_back_compat_sidecar_missing_diagnostics_key(self, tmp_path):
        from src.cca_config import DEFAULT_CCA_CONFIG, DiagnosticsConfig, RunConfig

        payload = dataclasses.asdict(DEFAULT_CCA_CONFIG)
        payload.pop("diagnostics", None)  # simulate pre-Tier-5 sidecar
        sidecar = tmp_path / "old.config.json"
        with open(sidecar, "w") as f:
            json.dump(payload, f)
        reconstructed = RunConfig.from_json(sidecar)
        assert reconstructed.diagnostics == DiagnosticsConfig()

    def test_present_diagnostics_key_roundtrips(self, tmp_path):
        from src.cca_config import DEFAULT_CCA_CONFIG, DiagnosticsConfig, RunConfig

        custom = dataclasses.replace(
            DEFAULT_CCA_CONFIG,
            diagnostics=DiagnosticsConfig(enable_overflow_proxy=False,
                                          gradient_norm_aggregations=("mean",)),
        )
        sidecar = tmp_path / "new.config.json"
        custom.to_json(sidecar)
        reconstructed = RunConfig.from_json(sidecar)
        assert reconstructed.diagnostics == custom.diagnostics

    def test_type_defense_rejects_non_diagnosticsconfig(self):
        from src.cca_config import DEFAULT_CCA_CONFIG

        with pytest.raises(ValueError, match="diagnostics"):
            dataclasses.replace(DEFAULT_CCA_CONFIG, diagnostics={"enable_gradient_norms": True})

    def test_null_diagnostics_in_payload_uses_default(self, tmp_path):
        from src.cca_config import DEFAULT_CCA_CONFIG, DiagnosticsConfig, RunConfig

        payload = dataclasses.asdict(DEFAULT_CCA_CONFIG)
        payload["diagnostics"] = None  # explicit null
        sidecar = tmp_path / "null.config.json"
        with open(sidecar, "w") as f:
            json.dump(payload, f)
        reconstructed = RunConfig.from_json(sidecar)
        assert reconstructed.diagnostics == DiagnosticsConfig()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cca_config.py::TestRunConfigDiagnosticsIntegration -v`
Expected: fail (`AttributeError: 'RunConfig' object has no attribute 'diagnostics'` and/or type-defense not present).

**Step 3: Write minimal implementation**

(a) Add field #10 to `RunConfig`, immediately after `optimizer: OptimizerConfig` (line 497):

```python
    diagnostics: DiagnosticsConfig = dataclasses.field(
        default_factory=DiagnosticsConfig
    )
```

(b) In `RunConfig.__post_init__`, in the sub-config type-defense block (alongside the existing `ratio_batch`/`lr_schedule`/`optimizer` `isinstance` checks), add:

```python
        if not isinstance(self.diagnostics, DiagnosticsConfig):
            raise ValueError(
                "RunConfig.diagnostics must be a DiagnosticsConfig instance; "
                f"got {type(self.diagnostics).__name__}."
            )
```

(c) In `RunConfig._from_dict`, do NOT add `"diagnostics"` to the strict required-sub-config loop (that loop raises on missing keys). After that loop and after `kwargs = _filter_known_fields(cls, payload, ...)`, add separate optional handling, then ensure it is applied to kwargs:

```python
        # diagnostics is optional (back-compat): absent or null → let the
        # RunConfig default_factory produce DiagnosticsConfig(); present
        # dict → reconstruct. Distinct from the strict sub-config loop
        # above, which raises on missing keys.
        diag_payload = payload.get("diagnostics")
        if diag_payload is not None:
            kwargs["diagnostics"] = DiagnosticsConfig._from_dict(
                diag_payload, _source=f"{_source}.diagnostics"
            )
        else:
            kwargs.pop("diagnostics", None)  # ensure default_factory fires
```

> Executor note: place the `diag_payload` block so it runs after
> `_filter_known_fields` populated `kwargs` and after the existing
> `sub_configs`/`heads` assignments, mirroring how those override filtered
> kwargs. The `kwargs.pop` guards the case where a `None` payload value
> survived `_filter_known_fields`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cca_config.py::TestRunConfigDiagnosticsIntegration -v`
Expected: 5 passed.

Run: `pytest -q`
Expected: prior total + 5, zero failures (existing `TestRunConfigJSONRoundTrip` etc. still green — `dataclasses.asdict` now includes `diagnostics`, reconstructed identically).

**Step 5: Commit**

```bash
git add src/cca_config.py tests/test_cca_config.py
git commit -m "tier5 phase 2: embed DiagnosticsConfig on RunConfig (default_factory + back-compat)"
```
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 4-6) -->
<!-- START_TASK_4 -->
### Task 4: factory.py + DiagnosticBundle + gradient-category trackers

**Files:**
- Create: `src/diagnostics/factory.py`
- Create: `tests/test_diagnostics_factory.py`

**Step 1: Write the failing tests**

Create `tests/test_diagnostics_factory.py`:

```python
# pattern: Functional Core

"""Tests for src/diagnostics/factory.py:build_trackers."""

from __future__ import annotations

import pytest
import tensorflow as tf

from src.cca_config import DiagnosticsConfig
from src.diagnostics.factory import DiagnosticBundle, build_trackers


def _group_fn(var):
    return var.name.split("/", 1)[0]


def _vars(*names):
    return [tf.Variable(tf.zeros([1]), name=n) for n in names]


class _StubHead:
    """Synthetic stand-in: build_trackers only touches `.loss_fn`."""

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn


class TestBuildTrackersGradientCategory:
    def test_returns_diagnostic_bundle_shape(self):
        bundle = build_trackers(
            DiagnosticsConfig(),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("cca/w"),
        )
        assert set(bundle.keys()) == {"per_step", "periodic"}
        assert set(bundle["per_step"].keys()) == {
            "gradient", "loss_component", "batch_target"
        }
        assert bundle["periodic"] == []

    def test_grad_norm_trackers_per_group_times_agg(self):
        bundle = build_trackers(
            DiagnosticsConfig(gradient_norm_aggregations=("max", "mean")),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("cca/w1", "cca/w2", "extra/w"),
        )
        names = sorted(t.name for t in bundle["per_step"]["gradient"])
        # 2 groups {cca, extra} x 2 aggs + 1 overflow tracker
        assert "grad_norm/cca/max" in names
        assert "grad_norm/cca/mean" in names
        assert "grad_norm/extra/max" in names
        assert "grad_norm/extra/mean" in names
        assert "grad_overflow_rate" in names
        assert len(bundle["per_step"]["gradient"]) == 5

    def test_groups_are_sorted_and_deduped(self):
        bundle = build_trackers(
            DiagnosticsConfig(gradient_norm_aggregations=("mean",)),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("b/w", "a/w", "b/w2", "a/w2"),
        )
        grad_norm_names = [
            t.name for t in bundle["per_step"]["gradient"]
            if t.name.startswith("grad_norm/")
        ]
        assert grad_norm_names == ["grad_norm/a/mean", "grad_norm/b/mean"]

    def test_disable_gradient_norms(self):
        bundle = build_trackers(
            DiagnosticsConfig(enable_gradient_norms=False),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("cca/w"),
        )
        names = [t.name for t in bundle["per_step"]["gradient"]]
        assert all(not n.startswith("grad_norm/") for n in names)
        assert "grad_overflow_rate" in names  # overflow still on

    def test_disable_overflow_proxy(self):
        bundle = build_trackers(
            DiagnosticsConfig(enable_overflow_proxy=False),
            group_fn=_group_fn,
            heads={},
            trainable_variables=_vars("cca/w"),
        )
        names = [t.name for t in bundle["per_step"]["gradient"]]
        assert "grad_overflow_rate" not in names
        assert any(n.startswith("grad_norm/") for n in names)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnostics_factory.py::TestBuildTrackersGradientCategory -v`
Expected: fail with `ModuleNotFoundError: No module named 'src.diagnostics.factory'`.

**Step 3: Write minimal implementation**

Create `src/diagnostics/factory.py`:

```python
# pattern: Functional Core

"""Constructs the Tier 5 DiagnosticBundle from config + model state.

Pure construction: parameters in, a bundle of (stateful) tracker objects
out. No I/O. One source of truth for what diagnostics a run instantiates,
mirroring src/cca_metrics.py:make_cca_metrics.
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, TypedDict

import keras

from src.cca_config import DiagnosticsConfig
from src.diagnostics.trackers import (
    BatchLabelBalanceTracker,
    GradientFiniteTracker,
    LossComponentTracker,
    PerGroupGradNormTracker,
)

if TYPE_CHECKING:
    from src.model_setup.heads import ClassificationHead

__all__ = ["DiagnosticBundle", "build_trackers"]

# FLPU loss-component intermediates (Phase 3 exposes these via
# FLPULoss.call(return_intermediates=True)). Aggregation is fixed to "mean"
# (design DoD only needs mean-tracked components; no config knob — YAGNI).
_FLPU_COMPONENT_KEYS = ("positive_risk", "negative_risk", "correction_triggered")
_FLPU_LOSS_COMPONENT_AGG = "mean"


class DiagnosticBundle(TypedDict):
    per_step: dict[str, list[keras.metrics.Metric]]
    periodic: list  # permanently []; forward-compat slot, no current consumer
                     # (Phase 5 prediction-distribution metrics ride the head
                     # metric path, not this bundle — see phase_05.md)


def build_trackers(
    config: DiagnosticsConfig,
    *,
    group_fn,
    heads: dict[str, "ClassificationHead"],
    trainable_variables,
) -> DiagnosticBundle:
    """Construct the per-step + periodic diagnostic bundle.

    Groups are derived by walking `trainable_variables` with `group_fn`
    (deterministic, sorted). Under the frozen-encoder default, only head
    variables are trainable, so no encoder grad-norm tracker is built — this
    is correct (see tier5-design.md).
    """
    per_step: dict[str, list[keras.metrics.Metric]] = {
        "gradient": [],
        "loss_component": [],
        "batch_target": [],
    }
    periodic: list = []  # permanently empty forward-compat slot (no consumer)

    groups = sorted({group_fn(v) for v in trainable_variables})

    if config.enable_gradient_norms:
        for group in groups:
            for agg in config.gradient_norm_aggregations:
                per_step["gradient"].append(PerGroupGradNormTracker(group, agg))

    if config.enable_overflow_proxy:
        per_step["gradient"].append(GradientFiniteTracker())

    return DiagnosticBundle(per_step=per_step, periodic=periodic)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnostics_factory.py::TestBuildTrackersGradientCategory -v`
Expected: 5 passed.

Run: `pytest -q`
Expected: prior total + 5, zero failures.

**Step 5: Commit**

```bash
git add src/diagnostics/factory.py tests/test_diagnostics_factory.py
git commit -m "tier5 phase 2: factory.py + DiagnosticBundle + gradient-category trackers"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: build_trackers batch-target category + periodic phase boundary

**Files:**
- Modify: `src/diagnostics/factory.py`
- Modify: `tests/test_diagnostics_factory.py`

**Step 1: Write the failing tests**

Append to `tests/test_diagnostics_factory.py`:

```python
class TestBuildTrackersBatchTarget:
    def test_one_balance_tracker_per_head(self):
        heads = {"cca": _StubHead(None), "immig": _StubHead(None)}
        bundle = build_trackers(
            DiagnosticsConfig(enable_loss_components=False),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        names = sorted(t.name for t in bundle["per_step"]["batch_target"])
        assert names == ["cca/positive_fraction", "immig/positive_fraction"]

    def test_disable_batch_balance(self):
        heads = {"cca": _StubHead(None)}
        bundle = build_trackers(
            DiagnosticsConfig(enable_batch_balance=False, enable_loss_components=False),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        assert bundle["per_step"]["batch_target"] == []

    def test_periodic_empty_regardless_of_enable_flag(self):
        # periodic is a permanently-empty forward-compat slot. The factory
        # never populates it; enable_prediction_distribution gates the
        # per-head distribution metrics (Phase 5), which ride the head
        # metric path, NOT this bundle.
        heads = {"cca": _StubHead(None)}
        bundle = build_trackers(
            DiagnosticsConfig(enable_prediction_distribution=True,
                              enable_loss_components=False),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        assert bundle["periodic"] == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnostics_factory.py::TestBuildTrackersBatchTarget -v`
Expected: `test_one_balance_tracker_per_head` fails (empty list); the other two pass already. Confirm the failure is the missing batch-target construction.

**Step 3: Write minimal implementation**

In `build_trackers`, before the final `return`, add:

```python
    if config.enable_batch_balance:
        for head_name in heads:
            per_step["batch_target"].append(BatchLabelBalanceTracker(head_name))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnostics_factory.py::TestBuildTrackersBatchTarget -v`
Expected: 3 passed.

Run: `pytest -q`
Expected: prior total + 3, zero failures.

**Step 5: Commit**

```bash
git add src/diagnostics/factory.py tests/test_diagnostics_factory.py
git commit -m "tier5 phase 2: batch-target trackers + periodic phase-boundary assertion"
```
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: build_trackers loss-component category + loss-introspection guard

**Files:**
- Modify: `src/diagnostics/factory.py`
- Modify: `tests/test_diagnostics_factory.py`

**Step 1: Write the failing tests**

Append to `tests/test_diagnostics_factory.py`:

```python
import keras as _keras


class _LossWithIntermediates(_keras.losses.Loss):
    def call(self, y_true, y_pred, return_intermediates=False):
        return tf.constant(0.0)


class _LossWithoutIntermediates(_keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.constant(0.0)


class TestBuildTrackersLossComponentGuard:
    def test_all_heads_supporting_yield_three_trackers_each(self):
        heads = {
            "cca": _StubHead(_LossWithIntermediates()),
            "immig": _StubHead(_LossWithIntermediates()),
        }
        bundle = build_trackers(
            DiagnosticsConfig(),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        names = sorted(t.name for t in bundle["per_step"]["loss_component"])
        assert names == [
            "cca/correction_triggered/mean",
            "cca/negative_risk/mean",
            "cca/positive_risk/mean",
            "immig/correction_triggered/mean",
            "immig/negative_risk/mean",
            "immig/positive_risk/mean",
        ]

    def test_zero_supporting_raises(self):
        heads = {"cca": _StubHead(_LossWithoutIntermediates())}
        with pytest.raises(ValueError, match="return_intermediates"):
            build_trackers(
                DiagnosticsConfig(),
                group_fn=_group_fn,
                heads=heads,
                trainable_variables=_vars("cca/w"),
            )

    def test_loss_fn_none_treated_as_unsupported(self):
        heads = {"cca": _StubHead(None)}
        with pytest.raises(ValueError, match="return_intermediates"):
            build_trackers(
                DiagnosticsConfig(),
                group_fn=_group_fn,
                heads=heads,
                trainable_variables=_vars("cca/w"),
            )

    def test_partial_support_warns_and_skips_unsupported(self):
        heads = {
            "cca": _StubHead(_LossWithIntermediates()),
            "bce_head": _StubHead(_LossWithoutIntermediates()),
        }
        with pytest.warns(UserWarning, match="bce_head"):
            bundle = build_trackers(
                DiagnosticsConfig(),
                group_fn=_group_fn,
                heads=heads,
                trainable_variables=_vars("cca/w"),
            )
        names = sorted(t.name for t in bundle["per_step"]["loss_component"])
        assert names == [
            "cca/correction_triggered/mean",
            "cca/negative_risk/mean",
            "cca/positive_risk/mean",
        ]

    def test_disable_loss_components_skips_guard_entirely(self):
        # enable_loss_components=False → no guard, no raise even with
        # zero supporting losses.
        heads = {"cca": _StubHead(_LossWithoutIntermediates())}
        bundle = build_trackers(
            DiagnosticsConfig(enable_loss_components=False),
            group_fn=_group_fn,
            heads=heads,
            trainable_variables=_vars("cca/w"),
        )
        assert bundle["per_step"]["loss_component"] == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_diagnostics_factory.py::TestBuildTrackersLossComponentGuard -v`
Expected: failures (no loss-component construction / guard yet).

**Step 3: Write minimal implementation**

In `src/diagnostics/factory.py`, add the helper above `build_trackers`:

```python
def _loss_exposes_intermediates(loss_fn) -> bool:
    """True iff loss_fn.call accepts a `return_intermediates` parameter.

    Phase-ordering: the real FLPULoss gains this parameter in Phase 3.
    Phase 2 only ever sees synthetic stand-in losses; the factory meets
    the real FLPULoss at Phase 6.
    """
    if loss_fn is None:
        return False
    call = getattr(loss_fn, "call", None)
    if call is None:
        return False
    try:
        sig = inspect.signature(call)
    except (ValueError, TypeError):
        return False
    return "return_intermediates" in sig.parameters
```

In `build_trackers`, before the final `return`, add:

```python
    if config.enable_loss_components:
        supporting = [
            name for name, head in heads.items()
            if _loss_exposes_intermediates(head.loss_fn)
        ]
        if not supporting:
            raise ValueError(
                "DiagnosticsConfig.enable_loss_components is True but no "
                "head's loss exposes `return_intermediates`; loss-component "
                f"tracking would produce nothing. Heads: {sorted(heads)}."
            )
        for name in heads:
            if name not in supporting:
                warnings.warn(
                    f"Head {name!r} loss does not expose "
                    f"`return_intermediates`; skipping its loss-component "
                    f"trackers.",
                    stacklevel=2,
                )
                continue
            for key in _FLPU_COMPONENT_KEYS:
                per_step["loss_component"].append(
                    LossComponentTracker(name, key, _FLPU_LOSS_COMPONENT_AGG)
                )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_diagnostics_factory.py::TestBuildTrackersLossComponentGuard -v`
Expected: 5 passed.

Run: `pytest -q`
Expected: prior total + 5, zero failures.

**Step 5: Commit**

```bash
git add src/diagnostics/factory.py tests/test_diagnostics_factory.py
git commit -m "tier5 phase 2: loss-component trackers + introspection guard (raise/warn)"
```
<!-- END_TASK_6 -->
<!-- END_SUBCOMPONENT_C -->

<!-- START_TASK_7 -->
### Task 7: Phase 2 integration verification

**Type:** Verification (no new code; no commit).

**Step 1: Confirm imports**

Run: `python -c "from src.diagnostics.factory import build_trackers, DiagnosticBundle; from src.cca_config import DiagnosticsConfig, DEFAULT_CCA_CONFIG; print('OK')"`
Expected: `OK`

**Step 2: Confirm config round-trip end-to-end**

Run: `python -c "import dataclasses, json, tempfile, os; from src.cca_config import DEFAULT_CCA_CONFIG, RunConfig; p=os.path.join(tempfile.mkdtemp(),'c.json'); DEFAULT_CCA_CONFIG.to_json(p); r=RunConfig.from_json(p); print('roundtrip-ok', r.diagnostics == DEFAULT_CCA_CONFIG.diagnostics)"`
Expected: `roundtrip-ok True`

**Step 3: Full suite, no regressions**

Run: `pytest -q`
Expected: Phase-1 total + all Phase-2 additions, zero failures, zero errors. Record the number (Phase 3 baseline).

**Step 4: Working tree clean**

Run: `git status --short`
Expected: clean.

Run: `git log --oneline 9136195..HEAD`
Expected: Phase-1 commits + 6 Phase-2 task commits.

**Phase 2 Done-when criteria (from design plan):**
- All factory and config tests pass. ✓ (Step 3)
- JSON round-trip of a `RunConfig` with `DiagnosticsConfig` produces an identical config. ✓ (Step 2, `TestRunConfigDiagnosticsIntegration`)

Phase 2 is complete.
<!-- END_TASK_7 -->
