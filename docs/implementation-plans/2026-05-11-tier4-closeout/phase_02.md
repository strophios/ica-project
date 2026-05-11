# Tier 4 Phase 2: I4 — LR Schedule Resolution Implementation Plan

**Goal:** Make the RunConfig sidecar self-sufficient for LR schedule reconstruction by extending `LRScheduleConfig` with a nested `ResolvedSteps` sub-object populated from `steps_per_epoch` at training time.

**Architecture:** New frozen dataclass `ResolvedSteps(warmup_steps, decay_steps, steps_per_epoch)` lives in `src/cca_config.py`. `LRScheduleConfig` gains an optional `resolved: ResolvedSteps | None = None` field and a `with_resolved(steps_per_epoch)` method that returns a new instance with `resolved` populated via `math.floor(factor * steps_per_epoch)`. The training script computes `steps_per_epoch` once (existing line 214), calls `with_resolved` to update `run_config.lr_schedule`, then reads resolved counts from the config when constructing the Keras schedule — eliminating the existing in-script multiplication at lines 286-291 and making the config the single source of truth. `_from_dict` on `LRScheduleConfig` is updated to reconstruct the nested `ResolvedSteps` instance; backward compat with older sidecars (missing `resolved` key) flows through the default `None`.

**Tech Stack:** Python 3.12, frozen dataclasses, pytest. No new dependencies.

**Scope:** 2 of 3 phases. Design reference: `docs/notes/tier4-design.md` "Piece 2: I4 — LR schedule resolution".

**Codebase verified:** 2026-05-11.

**Behavior change note.** The existing factor resolution at `run_cca_classification.py:286-291` does NOT floor — it passes `steps_per_epoch * factor` (a float) to Keras's `CosineDecay`, which coerces internally. This phase introduces explicit `math.floor` in `with_resolved`, matching the `math.floor` pattern at `run_cca_classification.py:214` used for `steps_per_epoch` itself. The numerical effect is tiny (e.g., a `571.75` becoming `571`). Existing trained models are not byte-identical reproducible against this change, but the project doesn't currently rely on byte-exact reproduction and the next training run will use the corrected prior (≈0.02) anyway.

---

<!-- START_SUBCOMPONENT_A (tasks 1-4) -->
<!-- START_TASK_1 -->
### Task 1: ResolvedSteps dataclass

**Files:**
- Modify: `src/cca_config.py` (add new `ResolvedSteps` frozen dataclass before `LRScheduleConfig` at line 250)
- Modify: `tests/test_cca_config.py` (add new test class `TestResolvedSteps`)

**Context.** `ResolvedSteps` is a small frozen dataclass holding the resolved step counts that the existing `LRScheduleConfig` factors compute against `steps_per_epoch`. It serializes via `dataclasses.asdict` recursion in `RunConfig.to_json` and needs a `_from_dict` classmethod for round-tripping (matching the project convention used by `FLPULossConfig`, `RatioBatchConfig`, etc.).

**Step 1: Write failing tests**

Add a new test class to `tests/test_cca_config.py`:

```python
class TestResolvedSteps:
    """ResolvedSteps captures the resolved LR schedule step counts
    that LRScheduleConfig factors are multiplied against at
    training time. See docs/notes/tier4-design.md Piece 2."""

    def test_construction_with_valid_positive_ints(self):
        rs = ResolvedSteps(
            warmup_steps=1250,
            decay_steps=15000,
            steps_per_epoch=5000,
        )
        assert rs.warmup_steps == 1250
        assert rs.decay_steps == 15000
        assert rs.steps_per_epoch == 5000

    def test_is_frozen(self):
        rs = ResolvedSteps(warmup_steps=1, decay_steps=1, steps_per_epoch=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rs.warmup_steps = 2  # type: ignore[misc]

    @pytest.mark.parametrize("field,value", [
        ("warmup_steps", 0),
        ("warmup_steps", -1),
        ("decay_steps", 0),
        ("decay_steps", -1),
        ("steps_per_epoch", 0),
        ("steps_per_epoch", -1),
    ])
    def test_rejects_non_positive(self, field, value):
        kwargs = {"warmup_steps": 1, "decay_steps": 1, "steps_per_epoch": 1}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            ResolvedSteps(**kwargs)

    @pytest.mark.parametrize("field,value", [
        ("warmup_steps", 1.5),
        ("warmup_steps", "1"),
        ("decay_steps", 1.5),
        ("steps_per_epoch", 1.5),
    ])
    def test_rejects_non_int(self, field, value):
        kwargs = {"warmup_steps": 1, "decay_steps": 1, "steps_per_epoch": 1}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            ResolvedSteps(**kwargs)

    def test_from_dict_round_trips(self):
        original = ResolvedSteps(
            warmup_steps=1250, decay_steps=15000, steps_per_epoch=5000
        )
        payload = dataclasses.asdict(original)
        reconstructed = ResolvedSteps._from_dict(payload)
        assert reconstructed == original
```

Ensure `dataclasses`, `pytest`, and `ResolvedSteps` are importable at the top of the test module.

**Step 2: Run tests to verify failure**

Run:
```bash
pytest tests/test_cca_config.py::TestResolvedSteps -v
```

Expected: All fail with `ImportError` or `NameError` for `ResolvedSteps`.

**Step 3: Implement ResolvedSteps**

In `src/cca_config.py`, immediately before the `@dataclasses.dataclass(frozen=True)` line for `LRScheduleConfig` (line 250), add:

```python
@dataclasses.dataclass(frozen=True)
class ResolvedSteps:
    """LR schedule step counts resolved from LRScheduleConfig
    factors against a concrete steps_per_epoch.

    Populated by `LRScheduleConfig.with_resolved` at training
    time; `steps_per_epoch` is recorded for provenance. See
    docs/notes/tier4-design.md Piece 2 for the rationale.
    """

    warmup_steps: int
    decay_steps: int
    steps_per_epoch: int

    def __post_init__(self):
        for name, val in (
            ("warmup_steps", self.warmup_steps),
            ("decay_steps", self.decay_steps),
            ("steps_per_epoch", self.steps_per_epoch),
        ):
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(
                    f"ResolvedSteps.{name} must be an int; got "
                    f"{val!r} (type {type(val).__name__})."
                )
            if val <= 0:
                raise ValueError(
                    f"ResolvedSteps.{name} must be > 0; got {val}."
                )

    @classmethod
    def _from_dict(cls, payload: dict, _source: str = "<dict>") -> "ResolvedSteps":
        kwargs = _filter_known_fields(cls, payload, _source=_source)
        return cls(**kwargs)
```

(The `isinstance(val, bool)` guard is necessary because `bool` is a subclass of `int` in Python; without it, `ResolvedSteps(warmup_steps=True, ...)` would pass the `isinstance(val, int)` check.)

**Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_cca_config.py::TestResolvedSteps -v
```

Expected: All pass.

No commit yet.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: LRScheduleConfig.resolved field + __post_init__ validation

**Files:**
- Modify: `src/cca_config.py` (`LRScheduleConfig` at lines 250-299: add `resolved` field, extend `__post_init__`)
- Modify: `tests/test_cca_config.py` (add tests to a new or existing `TestLRScheduleConfig` class)

**Context.** Adding the new optional field with sensible default keeps backward compatibility — older sidecars without `resolved` parse cleanly through the default `None`.

**Step 1: Write failing tests**

Add to `tests/test_cca_config.py` (locate or create a class for `LRScheduleConfig` field-level tests; the existing `TestJSONRoundTrip` covers serialization only):

```python
class TestLRScheduleConfigResolvedField:
    """LRScheduleConfig.resolved holds optional resolved step counts.
    See docs/notes/tier4-design.md Piece 2 for the rationale."""

    def test_default_is_none(self):
        cfg = LRScheduleConfig()
        assert cfg.resolved is None

    def test_accepts_resolved_steps_instance(self):
        rs = ResolvedSteps(
            warmup_steps=1, decay_steps=1, steps_per_epoch=1
        )
        cfg = LRScheduleConfig(resolved=rs)
        assert cfg.resolved is rs

    def test_rejects_non_resolved_steps_non_none(self):
        with pytest.raises(ValueError, match="resolved"):
            LRScheduleConfig(resolved={"warmup_steps": 1})  # type: ignore[arg-type]
```

**Step 2: Run tests to verify failure**

Run:
```bash
pytest tests/test_cca_config.py::TestLRScheduleConfigResolvedField -v
```

Expected: First two fail (no `resolved` field); third fails (no validation).

**Step 3: Add the `resolved` field and validation**

In `src/cca_config.py`, modify `LRScheduleConfig` (lines 250-299):

(a) Add the new field after the existing factor fields (after line 270, `decay_steps_factor`):

```python
    resolved: "ResolvedSteps | None" = None
```

(b) Extend `__post_init__` (currently lines 272-292) with a validation block at the end:

```python
        if self.resolved is not None and not isinstance(
            self.resolved, ResolvedSteps
        ):
            raise ValueError(
                f"LRScheduleConfig.resolved must be a ResolvedSteps "
                f"instance or None; got {self.resolved!r} (type "
                f"{type(self.resolved).__name__})."
            )
```

**Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_cca_config.py::TestLRScheduleConfigResolvedField -v
pytest tests/test_cca_config.py -v  # full module regression check
```

Expected: New tests pass; all existing tests still pass. Note specifically: existing `TestJSONRoundTrip` tests use default `LRScheduleConfig()` (per investigator findings — see `_valid_run_config` helper described in Task 4 Step 1), which means `resolved=None` everywhere. With `resolved=None`, `dataclasses.asdict` writes `"resolved": null`, the round-trip restores `resolved=None`, and the new validation accepts it cleanly. The first round-trip with a *populated* `resolved` happens in Task 4's new tests, which is also where the `_from_dict` update lands — both pieces land together in the same task.

**If any existing JSON round-trip tests fail anyway:** Inspect what `_filter_known_fields` does with a missing `resolved` key. The dataclass default should handle it, but if the helper requires every dataclass field to be present in the payload, the helper may need updating. Diagnose before proceeding to Task 3.

No commit yet.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: LRScheduleConfig.with_resolved method

**Files:**
- Modify: `src/cca_config.py` (add `with_resolved` method to `LRScheduleConfig`)
- Modify: `tests/test_cca_config.py` (add `TestLRScheduleConfigWithResolved` test class)

**Context.** `with_resolved(steps_per_epoch)` returns a new `LRScheduleConfig` instance with the `resolved` field populated. Uses `math.floor(factor * steps_per_epoch)` for integer step counts. Implementation note: `dataclasses.replace` is the idiomatic way to "update" a frozen dataclass field.

**Why `math.floor` specifically?** Keras's `CosineDecay` accepts the step counts as ints or floats and does its math in floats internally (the decay computation is `tf.minimum(step / decay_steps, 1.0)`). So passing 15000.0 vs 15000 produces the same training trajectory. The existing in-script code at `run_cca_classification.py:286-291` passes floats. Switching to `math.floor` here gives us *deterministic int values in the sidecar* (the design goal) at the cost of a sub-step shift in warmup duration (e.g., 571.75 → 571 means warmup ends 0.75 of one optimizer step earlier). At an `initial_lr` of 1e-4 and warmup spanning hundreds of steps, this is well below the noise floor of training. `math.floor` (rather than `int()` truncation or `round()`) also matches the existing `math.floor` pattern at `run_cca_classification.py:214` for `steps_per_epoch` itself — internal consistency.

**Step 1: Write failing tests**

Add to `tests/test_cca_config.py`:

```python
class TestLRScheduleConfigWithResolved:
    """LRScheduleConfig.with_resolved populates the resolved field
    via math.floor(factor * steps_per_epoch). See
    docs/notes/tier4-design.md Piece 2."""

    def test_populates_resolved_from_factors(self):
        cfg = LRScheduleConfig(
            warmup_steps_factor=0.25, decay_steps_factor=3.0
        )
        resolved_cfg = cfg.with_resolved(steps_per_epoch=5000)

        assert resolved_cfg.resolved is not None
        assert resolved_cfg.resolved.warmup_steps == 1250  # floor(0.25 * 5000)
        assert resolved_cfg.resolved.decay_steps == 15000  # floor(3.0 * 5000)
        assert resolved_cfg.resolved.steps_per_epoch == 5000

    def test_uses_floor_for_non_integer_results(self):
        cfg = LRScheduleConfig(
            warmup_steps_factor=0.25, decay_steps_factor=3.0
        )
        # 2287 * 0.25 = 571.75 → 571
        resolved_cfg = cfg.with_resolved(steps_per_epoch=2287)
        assert resolved_cfg.resolved.warmup_steps == 571

    def test_returns_new_instance_not_mutation(self):
        cfg = LRScheduleConfig()
        assert cfg.resolved is None
        resolved_cfg = cfg.with_resolved(steps_per_epoch=100)

        # Original unchanged
        assert cfg.resolved is None
        # Returned is a different instance
        assert resolved_cfg is not cfg
        assert resolved_cfg.resolved is not None

    def test_preserves_other_fields(self):
        cfg = LRScheduleConfig(
            initial_lr=2e-4,
            warmup_target=2e-3,
            decay_alpha=5e-2,
            warmup_steps_factor=0.5,
            decay_steps_factor=4.0,
        )
        resolved_cfg = cfg.with_resolved(steps_per_epoch=1000)

        assert resolved_cfg.initial_lr == 2e-4
        assert resolved_cfg.warmup_target == 2e-3
        assert resolved_cfg.decay_alpha == 5e-2
        assert resolved_cfg.warmup_steps_factor == 0.5
        assert resolved_cfg.decay_steps_factor == 4.0

    def test_rejects_non_positive_steps_per_epoch(self):
        cfg = LRScheduleConfig()
        with pytest.raises(ValueError, match="steps_per_epoch"):
            cfg.with_resolved(steps_per_epoch=0)
        with pytest.raises(ValueError, match="steps_per_epoch"):
            cfg.with_resolved(steps_per_epoch=-1)
```

**Step 2: Run tests to verify failure**

Run:
```bash
pytest tests/test_cca_config.py::TestLRScheduleConfigWithResolved -v
```

Expected: All fail with `AttributeError` for `with_resolved`.

**Step 3: Implement with_resolved**

Add to `LRScheduleConfig` class body (after `__post_init__`, before `_from_dict`):

```python
    def with_resolved(self, steps_per_epoch: int) -> "LRScheduleConfig":
        """Return a new LRScheduleConfig instance with the resolved
        field populated.

        Computes warmup_steps = floor(warmup_steps_factor *
        steps_per_epoch) and decay_steps = floor(decay_steps_factor
        * steps_per_epoch), matching the math.floor pattern used at
        src/run_cca_classification.py:214 for steps_per_epoch itself.

        Existing behavior at lines 286-291 of run_cca_classification.py
        multiplied factors by steps_per_epoch without flooring,
        passing floats to Keras's CosineDecay (which coerces).
        Switching to explicit floor here makes the resolved values
        deterministic integers — the numerical effect is tiny (e.g.,
        571.75 → 571) and the project doesn't rely on byte-exact
        reproduction.
        """
        if not isinstance(steps_per_epoch, int) or isinstance(
            steps_per_epoch, bool
        ):
            raise ValueError(
                f"steps_per_epoch must be a positive int; got "
                f"{steps_per_epoch!r} (type "
                f"{type(steps_per_epoch).__name__})."
            )
        if steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be > 0; got {steps_per_epoch}."
            )

        resolved = ResolvedSteps(
            warmup_steps=math.floor(
                self.warmup_steps_factor * steps_per_epoch
            ),
            decay_steps=math.floor(
                self.decay_steps_factor * steps_per_epoch
            ),
            steps_per_epoch=steps_per_epoch,
        )
        return dataclasses.replace(self, resolved=resolved)
```

If `math` isn't already imported at the top of `src/cca_config.py`, add `import math`.

**Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_cca_config.py::TestLRScheduleConfigWithResolved -v
```

Expected: All pass.

No commit yet.
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: LRScheduleConfig._from_dict update + JSON round-trip tests

**Files:**
- Modify: `src/cca_config.py` (`LRScheduleConfig._from_dict` at lines 294-299)
- Modify: `tests/test_cca_config.py` (add tests for JSON round-trip with `resolved` populated)

**Context.** Current `LRScheduleConfig._from_dict` (lines 294-299) uses `_filter_known_fields` to handle forward-compat and then calls `cls(**kwargs)`. The `resolved` key in the dict will be a plain dict (not a `ResolvedSteps` instance) post-`asdict` round-trip, and `LRScheduleConfig.__post_init__` validation (added Task 2) rejects non-`ResolvedSteps` non-`None` values. So `_from_dict` must reconstruct `ResolvedSteps` from the dict explicitly. This mirrors how `HeadConfig._from_dict` reconstructs `FLPULossConfig` and `RunConfig._from_dict` reconstructs all sub-configs.

**Step 1: Locate the `_valid_run_config` test helper**

The tests below use `_valid_run_config()` as a test-fixture helper that constructs a known-good `RunConfig`. Per the investigator's report (item 10), this helper exists at `tests/test_cca_config.py` around line 58 and is used by `TestJSONRoundTrip` (lines 366-383). Verify:

```bash
grep -n "_valid_run_config\|^def _valid" tests/test_cca_config.py
```

Expected: one `def _valid_run_config(...)` definition near the top of the file, and several usages in `TestJSONRoundTrip`. If the helper doesn't exist (investigator was wrong), construct one inline in this test class or in a fixture — minimum it needs to return a `RunConfig` with a default `LRScheduleConfig`.

**Step 2: Write failing tests**

Add to `tests/test_cca_config.py`:

```python
class TestLRScheduleConfigJSONRoundTripWithResolved:
    """JSON round-trip preserves the resolved field when populated.
    See docs/notes/tier4-design.md Piece 2."""

    def test_round_trips_with_resolved_populated(self, tmp_path):
        # Construct a RunConfig with a resolved LRScheduleConfig
        original_run_config = _valid_run_config()  # existing helper
        original_run_config = dataclasses.replace(
            original_run_config,
            lr_schedule=original_run_config.lr_schedule.with_resolved(
                steps_per_epoch=5000
            ),
        )

        # Round-trip via JSON sidecar
        sidecar = tmp_path / "config.json"
        original_run_config.to_json(sidecar)
        reconstructed = RunConfig.from_json(sidecar)

        assert reconstructed.lr_schedule.resolved is not None
        assert reconstructed.lr_schedule.resolved.warmup_steps == 1250
        assert reconstructed.lr_schedule.resolved.decay_steps == 15000
        assert reconstructed.lr_schedule.resolved.steps_per_epoch == 5000
        # Equality check across the whole config
        assert reconstructed == original_run_config

    def test_round_trips_with_resolved_none(self, tmp_path):
        # Default (resolved=None) round-trip
        original_run_config = _valid_run_config()
        assert original_run_config.lr_schedule.resolved is None

        sidecar = tmp_path / "config.json"
        original_run_config.to_json(sidecar)
        reconstructed = RunConfig.from_json(sidecar)

        assert reconstructed.lr_schedule.resolved is None
        assert reconstructed == original_run_config

    def test_loads_pre_resolved_sidecar_without_key(self, tmp_path):
        """Backward compat: older sidecars written before Piece 2
        have no 'resolved' key in the lr_schedule dict."""
        # Construct an old-shape sidecar manually
        run_config = _valid_run_config()
        payload = dataclasses.asdict(run_config)
        del payload["lr_schedule"]["resolved"]

        sidecar = tmp_path / "old_config.json"
        with open(sidecar, "w") as f:
            json.dump(payload, f)

        reconstructed = RunConfig.from_json(sidecar)
        assert reconstructed.lr_schedule.resolved is None
```

Ensure `json` is imported in the test module.

**Step 3: Run tests to verify failure**

Run:
```bash
pytest tests/test_cca_config.py::TestLRScheduleConfigJSONRoundTripWithResolved -v
```

Expected: `test_round_trips_with_resolved_populated` FAILS with a `ValueError` from `LRScheduleConfig.__post_init__` complaining that `resolved` is a dict, not a `ResolvedSteps` instance. The other two may pass (resolved-as-None and missing-key both flow through the default).

**Step 4: Update LRScheduleConfig._from_dict**

In `src/cca_config.py`, replace the existing `_from_dict` body (lines 294-299):

```python
    @classmethod
    def _from_dict(
        cls, payload: dict, _source: str = "<dict>"
    ) -> "LRScheduleConfig":
        # Reconstruct the nested ResolvedSteps if present.
        resolved_payload = payload.get("resolved")
        if resolved_payload is None:
            resolved = None
        elif isinstance(resolved_payload, dict):
            resolved = ResolvedSteps._from_dict(
                resolved_payload, _source=f"{_source}.resolved"
            )
        else:
            raise ValueError(
                f"Expected 'resolved' in {_source} to be a dict or "
                f"null; got {type(resolved_payload).__name__}."
            )

        kwargs = _filter_known_fields(cls, payload, _source=_source)
        kwargs["resolved"] = resolved
        return cls(**kwargs)
```

**Step 5: Run tests to verify they pass**

Run:
```bash
pytest tests/test_cca_config.py::TestLRScheduleConfigJSONRoundTripWithResolved -v
pytest tests/test_cca_config.py -v  # full module regression check
```

Expected: New tests pass; all existing JSON round-trip and other tests still pass.

No commit yet.
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 5-6) -->
<!-- START_TASK_5 -->
### Task 5: Wire training script to use config-level resolution

**Files:**
- Modify: `src/run_cca_classification.py` (lines 214 and 286-291)

**Context.** After Tasks 1-4, the new infrastructure exists but the training script still computes resolution inline. This task switches the script to call `with_resolved` on the config and read resolved counts from the config when constructing the Keras schedule. The config becomes the single source of truth; the existing in-script multiplication is removed.

**Step 1: Read the current training-script section**

Use the Read tool: `src/run_cca_classification.py` with `offset=210, limit=90` to inspect lines 210-300.

Capture **exactly**:
(a) The full line containing `steps_per_epoch = math.floor(...)` around line 214 — needed verbatim for Step 2's positioning.
(b) The full block lines 286-291 (or wherever `keras.optimizers.schedules.CosineDecay` is constructed with `decay_steps=...` and `warmup_steps=...`) — needed verbatim for Step 3's `Edit` `old_string`.
(c) The exact variable name holding the RunConfig (likely `run_config`, but confirm by reading).

Do not proceed to Steps 2-3 until you have the exact text — the `Edit` tool requires precise old_string matching.

**Step 2: Insert resolution call after steps_per_epoch computation**

Immediately after line 214 (the `steps_per_epoch = math.floor(...)` line), add:

```python
# Resolve LR schedule factors against the concrete steps_per_epoch
# so the sidecar is self-sufficient. See
# docs/notes/tier4-design.md Piece 2.
run_config = dataclasses.replace(
    run_config,
    lr_schedule=run_config.lr_schedule.with_resolved(steps_per_epoch),
)
```

If `dataclasses` isn't already imported at the top of `run_cca_classification.py`, add `import dataclasses`.

**Step 3: Replace the in-script factor resolution**

Using the exact text captured in Step 1(b), `Edit` the `CosineDecay` construction at lines 286-291. The structural transformation is:

- Anywhere the existing code multiplies `steps_per_epoch * run_config.lr_schedule.decay_steps_factor`, replace with `resolved.decay_steps`.
- Anywhere it multiplies `steps_per_epoch * run_config.lr_schedule.warmup_steps_factor`, replace with `resolved.warmup_steps`.
- Add the `resolved = run_config.lr_schedule.resolved` binding plus a defensive assertion immediately before the `CosineDecay` construction:

```python
resolved = run_config.lr_schedule.resolved
assert resolved is not None, (
    "lr_schedule.resolved should be populated by the with_resolved "
    "call earlier; this is a programmer error if it fires."
)
```

The non-resolved-dependent kwargs (`initial_learning_rate`, `alpha`, `warmup_target`) stay as reads from `run_config.lr_schedule.*`. Preserve the existing argument order in `CosineDecay` — just swap the values for the two resolved-dependent ones.

**Step 4: Verify training script syntax is valid**

Run:
```bash
python -c "import ast; ast.parse(open('src/run_cca_classification.py').read())"
```

Expected: No output.

**Step 5: Verify training script imports succeed**

Run:
```bash
python -c "import src.run_cca_classification" 2>&1 | head -10
```

Expected: Imports cleanly (or, if there are environmental errors like missing GPU drivers, they should be unrelated to the changes). Watch for `NameError` / `AttributeError` indicating mistakes in the new code.

No commit yet.
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Update smoke test to exercise with_resolved

**Files:**
- Modify: `scripts/smoke_test_integrated_stack.py` (around the existing RunConfig save/load section near lines 270-290)

**Context.** The smoke test exercises the RunConfig → fit → save → load → predict round-trip. It needs to (a) call `with_resolved` before save so the sidecar contains resolved values, (b) assert that the loaded config's `resolved` field is populated and matches the saved values.

**Step 1: Read the relevant smoke-test sections**

Run:
Use the Read tool: `scripts/smoke_test_integrated_stack.py` with `offset=200, limit=100` to inspect lines 200-300.

Capture **exactly**:
(a) Where `run_config` is constructed.
(b) The `validate_against_backbone` call (around line 226).
(c) The `to_json` save (around line 279), including the surrounding variable bindings.
(d) The `from_json` load (around line 289), including the variable name receiving the loaded config.
(e) Whether the smoke test already computes any `steps_per_epoch`-like value (it runs `fit`, so it almost certainly does). If yes, capture the variable name and value/computation.

**Step 2: Insert with_resolved call before save**

Find the location where `run_config` is finalized but before `to_json`. Use the actual `steps_per_epoch` value from Step 1(e) if one exists. If the smoke test doesn't compute one explicitly, use the same expression as the smoke-test's training run (typically `math.floor(synthetic_row_count / batch_size)` or similar).

Insert immediately before the `to_json` call:

```python
# Resolve LR schedule for sidecar self-sufficiency
# (mirrors what run_cca_classification.py does after computing
# steps_per_epoch). See docs/notes/tier4-design.md Piece 2.
run_config = dataclasses.replace(
    run_config,
    lr_schedule=run_config.lr_schedule.with_resolved(
        steps_per_epoch=<actual smoke-test steps_per_epoch from Step 1>,
    ),
)
```

If `dataclasses` isn't imported in the smoke-test module, add `import dataclasses`.

**Step 3: Add assertion after the load**

After the `RunConfig.from_json(sidecar_path)` call (around line 289), add (substituting `loaded_config` with the actual variable name from Step 1(d)):

```python
# Verify the resolved sub-object round-tripped correctly.
assert <loaded_var>.lr_schedule.resolved is not None, (
    "Loaded sidecar should have resolved populated."
)
assert (
    <loaded_var>.lr_schedule.resolved
    == run_config.lr_schedule.resolved
), "Resolved values should round-trip exactly."
```

**Step 4: Run the smoke test end-to-end**

Run:
```bash
python scripts/smoke_test_integrated_stack.py 2>&1 | tail -30
```

Expected: Smoke test completes successfully, including the new assertions. If it fails, the most likely cause is a variable-name mismatch — fix and re-run.

No commit yet.
<!-- END_TASK_6 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_TASK_7 -->
### Task 7: Full suite + commit

**Step 1: Run the complete test suite**

Run:
```bash
pytest
```

Expected: All tests pass. Phase 1 ended with the new baseline (192 + 4 new = 196 typical). Phase 2 adds approximately +17 tests: `TestResolvedSteps` (~6), `TestLRScheduleConfigResolvedField` (~3), `TestLRScheduleConfigWithResolved` (~5), `TestLRScheduleConfigJSONRoundTripWithResolved` (3). Target post-Phase-2 count: ~213. Record the actual count in the commit message; small deviation (±2) is acceptable.

**Step 2: Re-run smoke test to confirm**

Run:
```bash
python scripts/smoke_test_integrated_stack.py 2>&1 | tail -5
```

Expected: Smoke test passes (already confirmed in Task 6 but worth re-verifying after any test fixes).

**Step 3: Review the diff**

Run:
```bash
git status
git diff --stat
```

Expected: Changes to `src/cca_config.py`, `src/run_cca_classification.py`, `scripts/smoke_test_integrated_stack.py`, `tests/test_cca_config.py`. Nothing else.

**Step 4: Stage and commit**

Run:
```bash
git add src/cca_config.py src/run_cca_classification.py scripts/smoke_test_integrated_stack.py tests/test_cca_config.py
git commit -m "$(cat <<'EOF'
Tier 4 Piece 2: I4 LR schedule resolution via nested ResolvedSteps

Make the RunConfig sidecar self-sufficient for LR schedule
reconstruction. Closes I4 (Important, inherited from Tier 3
closeout deferred list). See docs/notes/tier4-design.md
"Piece 2".

Changes:
- New frozen dataclass ResolvedSteps in src/cca_config.py with
  positive-int validation on warmup_steps, decay_steps,
  steps_per_epoch.
- LRScheduleConfig extended with resolved: ResolvedSteps | None
  = None field, with_resolved(steps_per_epoch) method that
  returns a new instance with resolved populated via
  math.floor(factor * steps_per_epoch), and _from_dict update
  to reconstruct the nested ResolvedSteps from JSON payload.
- src/run_cca_classification.py: calls with_resolved after
  computing steps_per_epoch (around line 214) and reads
  resolved counts when constructing keras.optimizers.schedules.
  CosineDecay. The existing in-script factor multiplication
  (lines 286-291) is removed — the config is now the single
  source of truth.
- scripts/smoke_test_integrated_stack.py exercises the
  with_resolved path before save and asserts the resolved
  sub-object round-trips through the sidecar.

Numerical note: existing factor resolution passed floats to
Keras (e.g., 571.75 as warmup_steps); switching to explicit
math.floor produces deterministic ints (e.g., 571). The
project doesn't rely on byte-exact reproduction of prior
trained models, and the next training run will use the
corrected prior estimate (≈0.02) anyway.

Backward compat: older sidecars without the 'resolved' key
load via the None default; eval (unchanged this tier) doesn't
reconstruct the schedule, so missing resolved is non-fatal.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Step 5: Verify commit landed**

Run:
```bash
git log --oneline -2
git status
```

Expected: New HEAD commit; working tree clean for in-scope files.

**Step 6: Request code review (per project Tier 2/3 convention)**

Dispatch the code-reviewer subagent referencing `docs/notes/tier4-design.md` "Piece 2" as the spec.
<!-- END_TASK_7 -->
