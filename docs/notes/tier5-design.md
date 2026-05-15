# Tier 5 Design: Empirical Stress Test + Diagnostic Instrumentation

## Summary

Tier 5 adds two things to the post-Tier-4 stack: a structured empirical stress test on real CCA data, and permanent diagnostic instrumentation that makes training dynamics directly observable. The stress test has three levels — mechanical (the assembled pipeline runs on real cached data without crashes), numerical (loss decreases, gradients behave as nnPU structure predicts, no silent floating-point failures), and research-relevant (retrain-vs-baseline quality comparison) — where only the first two are in scope here.

The instrumentation is a new `src/diagnostics/` module introducing four concrete per-step `keras.metrics.Metric` subclasses (gradient norms by variable group, overflow detection, FLPU loss-component breakdown, per-head label balance) and one periodic diagnostic (`PredictionDistributionDiagnostic`). These are wired into `LayerLRModel.train_step` via an optional `DiagnosticBundle`, configured through a new `DiagnosticsConfig` frozen-dataclass sub-config on `RunConfig`, and surfaced to `CSVLogger`/`TensorBoard` via a `DiagnosticsCallback`. The design follows the sub-config pattern, boundary-inventory enforcement, and wrapped-vs-flat forward-compat already established in Tiers 3–4. Tier 5 also retires two inherited deferrals: the I2 synthetic-backbone gap (real-data run replaces the synthetic stand-in) and partial progress on I8 (shared train/eval config). The empirical run at the end of Phase 8 produces the corrected-prior (π=0.02) candidate model that the subsequent level-3 research workstream will evaluate.

## Definition of Done

Tier 5 closes when the empirical stress test — a two-level mechanical + numerical validation of the post-Tier-4 stack — passes on real CCA data, supported by permanent diagnostic instrumentation that makes the level-2 numerical claims directly observable.

**Level 1 (mechanical) — pass criteria:**

- End-to-end training pipeline runs on real cached CCA data (`cca_set/`) — not just synthetic — with the diagnostic instrumentation enabled.
- A short run (a few hundred steps) completes without crashes, shape errors, or NaN in the final loss.
- Save (weights + sidecar JSON) → load (sidecar + weights) → predict round-trip succeeds on a held-out batch.
- All existing tests still pass (220 → ~280–320 with diagnostics test additions).

**Level 2 (numerical) — pass criteria:**

- Loss decreases monotonically (final-epoch mean < initial-epoch mean by at least an order of magnitude on the labeled-positive risk component) over a full-length local run.
- Per-group gradient norms: encoder group reports zero (frozen-encoder default); head group reports non-zero, bounded magnitudes (no exploding gradients — max-aggregated norm within a reasonable bound across all steps).
- No NaN/Inf in any tracked scalar across the full local run.
- Cluster acceptance bar: one short cluster run (any cluster-specific issue triage if needed) followed by one full-length cluster run completes with `grad_overflow_rate` < threshold (low single-digit percent; refine threshold during the short cluster run).
- FLPU loss-component breakdown is observable across the run: `positive_risk`, `negative_risk`, `correction_triggered` rate all logged and within expected ranges (positive risk decreasing, correction rate not pinned at 1.0 — which would indicate the prior estimate is materially off).

**Level 3 (research-relevant comparison) is explicitly scoped *outside* this design.** Comparing CCA classification quality between the prior π=0.03 model and the corrected π=0.02 model — including hand-review of confident-yes/confident-no examples — is a separate research workstream that begins after Tier 5 closes. The diagnostic-instrumented retrain run will produce the level-3 candidate model, but the comparison itself is downstream work.

## Glossary

- **PU learning**: Positive-Unlabeled learning — a binary classification setting where only some positives are labeled; unlabeled examples may contain unobserved positives. This project uses it because there are no confirmed-negative NYT articles.
- **nnPU (non-negative PU)**: A PU loss formulation (Kiryo 2017) that clamps the estimated negative-class risk to zero when it goes negative, preventing the model from exploiting the unlabeled-positive contamination. Basis of `FLPULoss`.
- **FLPULoss**: This project's composite loss — focal cross-entropy (focal loss, Lin 2020, γ=2) wrapped inside the nnPU risk estimator. Lives in `src/loss_functions/loss.py`. The "FL" prefix is focal loss, not false-label.
- **Prior (π)**: The fraction of all articles that are true positives. Estimated at π≈0.02 via DEDPUL; the prior parameterizes the nnPU risk estimator. The corrected value (0.02 rather than the earlier 0.03) is what Tier 5 uses for the retrain.
- **DAPT (Domain-Adaptive Pre-Training)**: Fine-tuning a pre-trained language model (RoBERTa) on in-domain unlabeled text (NYT headline/lede pairs) before task fine-tuning. Phase 1 of the pipeline; produces the backbone weights used downstream.
- **CCA**: The protest-event detection task that forms the first classification head — binary classification of whether an article reports a collective action event. Label set derived from NYT indexer descriptors ("demonstrations and riots", "strikes", etc.).
- **ICA (Immigrant Collective Action)**: The project's research target — collective action events involving immigrants. Requires CCA + immigration + US heads combined; only the CCA head exists yet.
- **RoBERTa / backbone**: The pre-trained transformer encoder shared across classification heads. Loaded via `src/model_setup/backbone.py:load_dapt_backbone`; its weights can be frozen or fine-tuned via discriminative learning rates.
- **ClassificationHead**: A `keras.layers.Layer` subclass (`src/model_setup/heads.py`) that wraps the task-specific dense layer, loss function, and per-head metrics. Supports both standard Keras compile mode and endpoint-layer mode (loss registered via `add_loss`).
- **LayerLRModel**: A `keras.Model` subclass (`src/model_setup/layer_lr_model.py`) that overrides `train_step` to apply per-variable gradient multipliers before `optimizer.apply`, enabling discriminative fine-tuning (different learning rates for encoder vs. head groups).
- **RunConfig**: A frozen dataclass (`src/cca_config.py`) that captures all architectural and research-dimension parameters for a training run — backbone, head config, loss config, optimizer, LR schedule, batch config, and (in Tier 5) diagnostics. Serialized to a JSON sidecar alongside the `.weights.h5` checkpoint.
- **Endpoint-layer pattern**: The training pattern where `ClassificationHead` registers its own loss via Keras's `add_loss` mechanism, allowing the head to own the loss computation. Contrasted with "standard mode" where the loss is passed to `model.compile()`.
- **Pattern A vs. Pattern 2**: Two weight-sharing strategies across train/inference models. Pattern A: same Layer instances shared in-process between models. Pattern 2: fresh model loaded from saved weights by layer structure, used when train and eval are in separate processes.
- **Boundary-inventory pattern**: This project's testing discipline of enforcing each invariant at every boundary it crosses (construction, call site, runtime dispatch), so each layer catches the bugs the others miss. Codified in `docs/notes/engineering-patterns.md`.
- **`mixed_float16` / `LossScaleOptimizer`**: Keras's mixed-precision training policy — weights stored in `float32`, compute in `float16` for GPU throughput. `LossScaleOptimizer` wraps the optimizer to apply dynamic loss scaling, preventing underflow in `float16` gradients. Active on the cluster; `float32` used locally.
- **DiagnosticBundle**: A `TypedDict` grouping per-step tracker dicts by category (`gradient`, `loss_component`, `batch_target`) and a list of periodic diagnostics. Passed to `LayerLRModel.__init__`; constructed by `build_trackers` in `src/diagnostics/factory.py`.
- **correction_triggered**: An nnPU diagnostic — the rate at which the negative-risk clamping (non-negative correction) fires. If pinned near 1.0, the prior estimate is likely too high relative to actual positive prevalence in the batch.

## Architecture

Tier 5 has two coupled but distinct architectural layers: the **stress-test framing** (workflow for validating the post-Tier-4 stack before the retrain) and the **diagnostic instrumentation** (the code surface that makes level-2 claims observable).

### Three-level stress-test structure

- **Level 1 (mechanical):** end-to-end on real data, no crashes, no NaN. Validates that the assembled stack — preprocessor → model → loss → save/load → predict — works when fed real cached samples rather than synthetic stand-ins. Naturally retires the I2 deferred item from Tier 3 closeout (smoke-test-with-synthetic-backbone gap, which is the boundary case for the synthetic-stand-ins engineering pattern).
- **Level 2 (numerical):** training dynamics are correct. Loss decreases. Gradients flow through unfrozen layers and zero through frozen ones. Mixed-precision doesn't silently overflow. Loss components behave the way nnPU's structure predicts.
- **Level 3 (research-relevant):** retrained model with π=0.02 matches or improves on the previous π=0.03 model. Explicitly out of scope here; level-3 lives in a separate workstream.

Levels 1 and 2 happen mostly local (macOS, `float32`, `tensorflow-metal`). One short cluster run validates `mixed_float16` behavior and any cluster-specific issues; one full-length cluster run serves as the level-2 acceptance bar.

### Diagnostic instrumentation: module layout

New module `src/diagnostics/`:

```
src/diagnostics/
├── trackers.py       # concrete per-step Metric subclasses
├── periodic.py       # PeriodicDiagnostic base + concrete subclass(es)
├── callback.py       # DiagnosticsCallback (consumes periodic diagnostics)
└── factory.py        # build_trackers(...) -> DiagnosticBundle
```

**Class hierarchy is deliberately flat.** Concrete per-step trackers inherit directly from `keras.metrics.Metric`. No abstract `DiagnosticTracker` layer; no aggregation sub-base layer. Aggregation strategy lives inside each concrete class (either by inheriting from `keras.metrics.Mean` when that fits, or by holding its own `tf.Variable`s for max-style or last-value accumulation). Categories are enforced by **registration** in the factory's output dict, not by inheritance — `train_step` dispatches based on dict keys, not `isinstance` checks.

`PeriodicDiagnostic` is a separate base class, not a `Metric` subclass. It defines `run(model, reference_batch) -> dict[str, float]` and is invoked by `DiagnosticsCallback` at epoch boundaries (or every N batches per config). Its API surface differs from per-step trackers because what it does is different — separate forward pass on a fixed reference batch, not a byproduct of a training step.

### Concrete tracker catalog

Five concrete classes; most instantiated multiple times per run.

**Per-step trackers (direct `keras.metrics.Metric` subclasses):**

| Class | Category | Per-run instances | Name pattern |
|---|---|---|---|
| `PerGroupGradNormTracker(group_name, aggregation)` | gradient | N_groups × N_aggregations | `grad_norm/{group}/{agg}` |
| `GradientFiniteTracker()` | gradient | 1 | `grad_overflow_rate` |
| `LossComponentTracker(head_name, component_key, aggregation)` | loss_component | N_heads × N_components | `{head}/{key}/{agg}` |
| `BatchLabelBalanceTracker(head_name)` | batch_target | N_heads | `{head}/positive_fraction` |

For the current frozen-encoder single-head CCA setup: ~9 per-step trackers (4 grad-norm + 1 overflow + 3 loss-component + 1 batch-balance).

**Periodic diagnostic (`PeriodicDiagnostic` subclass):**

| Class | Per-run instances | Emits |
|---|---|---|
| `PredictionDistributionDiagnostic(head_name, reference_batch_source, summary_stats)` | N_heads | `{head}/pred_dist/{stat}` per summary stat |

Default summary stats: `mean`, `std`, `frac_above_0.5`. Sigmoid is applied to logits before computing stats.

### Contracts

**`FLPULoss.call`** gains an optional `return_intermediates` parameter:

```python
def call(
    self,
    y_true,
    y_pred,
    return_intermediates: bool = False,
) -> tf.Tensor | tuple[tf.Tensor, dict[str, tf.Tensor]]:
    ...
```

With `return_intermediates=True`, returns `(loss, {"positive_risk": ..., "negative_risk": ..., "correction_triggered": ...})`. The three components are already explicit `ops.sum()` calls in the existing implementation (per Tier 3 codebase investigation); exposing them is plumbing, not refactoring.

**`ClassificationHead`** gains a `last_components: dict | None` attribute populated during `call()` when diagnostics are enabled. Plain Python attribute, not a `tf.Variable`. Read by `train_step` immediately after the forward pass via head references registered on `LayerLRModel`.

**`LayerLRModel.__init__`** gains two optional parameters:

```python
diagnostic_trackers: DiagnosticBundle | None = None,
diagnostic_head_refs: list[ClassificationHead] | None = None,
```

Where `DiagnosticBundle` is:

```python
DiagnosticBundle = TypedDict('DiagnosticBundle', {
    'per_step': dict[str, list[keras.metrics.Metric]],  # keys: 'gradient', 'loss_component', 'batch_target'
    'periodic': list[PeriodicDiagnostic],
})
```

**`DiagnosticsConfig`** is a frozen dataclass field on `RunConfig`, default factory so it's back-compat with sidecars written before Tier 5:

```python
@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    enable_gradient_norms: bool = True
    enable_overflow_proxy: bool = True
    enable_loss_components: bool = True
    enable_batch_balance: bool = True
    enable_prediction_distribution: bool = True
    gradient_norm_aggregations: tuple[str, ...] = ("max", "mean")
    prediction_reference_batch_path: str | None = None
    prediction_summary_stats: tuple[str, ...] = ("mean", "std", "frac_above_0.5")
    prediction_reference_batch_n: int = 128
    periodic_update_freq: str = "epoch"  # "epoch" or "every_n_batches"
    periodic_update_n: int = 0  # used if freq is "every_n_batches"
```

Group names are *not* stored on the config — they're derived from the model at factory build time by walking `trainable_variables` with `group_fn`. The empirical record of which groups were tracked lives in the CSV/TensorBoard column names produced by the run.

### Data flow

**Per training step (new bits in `LayerLRModel.train_step`):**

```
existing: x, y unpacked → forward pass (y_pred = self(x, training=True))
                       → ClassificationHead.call invokes FLPULoss(return_intermediates=True)
                       → head.last_components populated
existing: loss aggregated via Keras → scaled by optimizer.scale_loss
existing: gradients = tape.gradient(loss, trainable_variables)
NEW dispatch (between gradient computation and multiplier application):
  for tracker in per_step["gradient"]:
      tracker.update_state(gradients, self.trainable_variables, self.group_fn)
  for tracker in per_step["loss_component"]:
      head = self._head_refs_by_name[tracker.head_name]
      tracker.update_state(head.last_components)
  for tracker in per_step["batch_target"]:
      tracker.update_state(y)
existing: scaled = [mult * g], optimizer.apply
```

Gradients are observed **after** `tape.gradient` and **before** multiplier scaling — we measure what was computed, not what was applied. Under `mixed_float16`, the gradients are scale-multiplied (Keras's `LossScaleOptimizer.scale_loss` applies before `tape.gradient`). We accept this: grad-norm trends within an environment remain valid; cross-environment absolutes do not. This is consistent with the proxy-signal stance for `GradientFiniteTracker` (which observes finite-ness, not magnitudes).

**Periodic flow:** `DiagnosticsCallback` loads the reference batch once in `on_train_begin`. At configured intervals (`on_epoch_end` by default, or `on_train_batch_end` with stride for `every_n_batches`), it calls `diag.run(model, reference_batch)` for each periodic diagnostic and merges returned dicts into `logs`. Since callbacks share `logs` with `CSVLogger` / `TensorBoard`, periodic scalars appear alongside per-step scalars — no separate logging path. Logger `update_freq` must align with `periodic_update_freq` for clean output (documented in `DiagnosticsConfig` field comments).

### Environment behavior

- **Local (`float32`):** `GradientFiniteTracker` is effectively a no-op (always emits 0.0). Per-group grad norms are unscaled, comparable across local runs. Loss components computed in `float32` throughout.
- **Cluster (`mixed_float16`):** `GradientFiniteTracker` becomes the active diagnostic for the level-2 mixed-precision acceptance criterion. Per-group grad norms are scale-multiplied; trends are valid, absolutes are not cross-comparable to local. Loss components computed in `float16` for the forward pass but accumulated in `float32` (Keras `Metric` state vars default to `float32`).
- **Frozen-encoder default:** encoder variables are not in `trainable_variables`; `tape.gradient` doesn't compute them. `PerGroupGradNormTracker("encoder", ...)` filters to empty list and reports 0.0. This is the expected behavior — the diagnostic reports zero because no encoder variables are in the trainable set, not because computed encoder gradients were zero. Documented in tracker docstring.

### Mode boundaries

- Dispatch is in `train_step` only. `test_step` (evaluation) does not run diagnostics — no gradients, no need.
- Inference models built by `build_inference_model` (Pattern 2) do not wire trackers. Diagnostics are training-only.
- Tracker state vars are `Metric` state, not `Layer` state, so they're not written to `.weights.h5` by `save_weights`. Loading weights into a fresh model leaves trackers initialized at zero — fine, because tracker state is per-epoch and gets reset at the start of training.

## Existing Patterns

Tier 5 builds on patterns established across Tiers 2–4 and codified in `docs/notes/engineering-patterns.md` / `docs/notes/process-patterns.md`.

**Patterns followed:**

- **Sub-config pattern (`cca_config.py`).** `DiagnosticsConfig` mirrors the structure of `FLPULossConfig`, `RatioBatchConfig`, `LRScheduleConfig`, `OptimizerConfig`: frozen dataclass with `__post_init__` field validation, `_from_dict` classmethod using `_filter_known_fields`, embedded as a typed field on `RunConfig` with type-defense in `RunConfig.__post_init__`. Default factory ensures back-compat with pre-Tier-5 sidecars.
- **Boundary-inventory pattern (engineering-patterns.md, Validated).** Diagnostic enablement is validated at multiple boundaries: `DiagnosticsConfig.__post_init__` for internal config validity; factory call site for derived group/head consistency with the model; runtime dispatch for tracker state shape. Each catches what the others miss.
- **Wrapped-vs-flat forward-compat (engineering-patterns.md, Validated).** `DiagnosticsConfig` wraps the diagnostic fields rather than putting them flat on `RunConfig`. Anticipates future expansion: ALUM adds new loss-component keys, multi-head adds per-head trackers, future categories (e.g., `ForwardPassTracker` when ALUM lands) — all extend without flattening `RunConfig`.
- **Pedagogical pattern (process-patterns.md).** Loss-component dispatch through `ClassificationHead.last_components` (rather than recomputation in `train_step`) preserves the architectural reason: components are computed where the loss is invoked; the head-attribute side channel mirrors how Keras's own `Layer.losses` works.
- **`make_cca_metrics` factoring (`src/cca_metrics.py`).** Diagnostic factory mirrors this pattern: one source of truth for what gets constructed, parameterized by run config and model state.

**No divergence from existing patterns.** The class-hierarchy debate in brainstorming (Mean vs. Metric base, aggregation as subclass vs. parameter, one-class-per-loss-component vs. one-class-multiple-instances) was resolved toward the *simpler* design at each step, matching the project's existing preference for direct-Keras-idiomatic structure over abstract hierarchies.

**One genuinely new contract addition:** `FLPULoss.return_intermediates` parameter and `ClassificationHead.last_components` attribute. These extend existing classes without changing default behavior — back-compat is preserved.

## Implementation Phases

Eight phases. Phases 1–6 deliver code with in-phase tests. Phases 7–8 are empirical validation runs (operational verification, not test code).

### Phase 1: Tracker module foundation

**Goal:** Establish `src/diagnostics/` module with concrete per-step trackers and their unit tests.

**Components:**
- `src/diagnostics/trackers.py` — `PerGroupGradNormTracker`, `GradientFiniteTracker`, `LossComponentTracker`, `BatchLabelBalanceTracker` as direct `keras.metrics.Metric` subclasses, each managing its own aggregation state internally.
- `tests/test_diagnostics_trackers.py` — unit tests per class covering: correct value computation against tape-computed ground truth on tiny models; aggregation semantics (max running max within epoch, mean = sum/count); sparse-gradient (`tf.IndexedSlices`) handling for `PerGroupGradNormTracker`; cross-head independence for `LossComponentTracker`; edge cases (empty group, all-zero batch, all-one batch).
- Property-based tests (per `house-style:property-based-testing`): per-group grad norm is invariant under variable permutation within a group; `mean` aggregation equals `sum/count` over a session; `max` aggregation is monotone non-decreasing within an epoch; batch label balance ∈ [0, 1].

**Dependencies:** None (first phase).

**Done when:** All new tracker unit and property tests pass; existing 220-test suite still passes; new module imports cleanly.

### Phase 2: Config and factory

**Goal:** Integrate `DiagnosticsConfig` into `RunConfig` and implement the factory that constructs trackers from config + model state.

**Components:**
- `src/cca_config.py` — add `DiagnosticsConfig` frozen dataclass; add `diagnostics` field to `RunConfig` with default factory; extend `_from_dict` to reconstruct the nested config; add type-defense in `RunConfig.__post_init__`.
- `src/diagnostics/factory.py` — `build_trackers(config, *, group_fn, head_names, trainable_variables) -> DiagnosticBundle`. Walks `trainable_variables` once to enumerate groups; constructs the configured tracker set; handles enable/disable flags.
- `tests/test_cca_config.py` — extend with `DiagnosticsConfig` validation tests, JSON round-trip, back-compat (sidecar missing the field → default factory fires), `RunConfig` type-defense.
- `tests/test_diagnostics_factory.py` — factory tests: all enables on → expected tracker set; each disable individually → that category empty; multi-head case → per-head trackers multiply; group enumeration matches model structure; loss-type guard (FLPU detected via `return_intermediates` parameter introspection).

**Dependencies:** Phase 1.

**Done when:** All factory and config tests pass; JSON round-trip of a `RunConfig` with `DiagnosticsConfig` produces an identical config.

### Phase 3: Loss-component harvest path

**Goal:** Extend `FLPULoss` and `ClassificationHead` to expose loss-component intermediates for diagnostic harvest.

**Components:**
- `src/loss_functions/loss.py` — add `return_intermediates: bool = False` parameter to `FLPULoss.call`; with `True`, return `(loss, {"positive_risk", "negative_risk", "correction_triggered"})`. Backward-compatible default.
- `src/model_setup/heads.py` — `ClassificationHead.call` requests intermediates from its loss when diagnostics are enabled (signaled via a new constructor flag, e.g., `expose_loss_components: bool = False`) and stores them on `self.last_components`. Read-after-call contract.
- `tests/test_flpu_loss.py` — add tests: `return_intermediates=False` is back-compat (scalar return, no behavior change); `return_intermediates=True` returns `(scalar, dict)` with expected keys; the scalar loss matches across both modes (computation is identical).
- `tests/test_heads.py` — add tests: after `ClassificationHead.call` with `expose_loss_components=True`, `last_components` is populated with expected keys; with the flag off, intermediates are not requested; multiple consecutive calls update `last_components` to the latest.

**Dependencies:** Phase 1 (tracker contract knowable).

**Done when:** All new contract tests pass; existing FLPU and head tests still pass; the loss scalar is bit-identical between `return_intermediates=True` and `=False` paths.

### Phase 4: Train-step integration

**Goal:** Wire diagnostic dispatch into `LayerLRModel.train_step`, with hard regression coverage for the no-trackers path.

**Components:**
- `src/model_setup/layer_lr_model.py` — `LayerLRModel.__init__` gains `diagnostic_trackers` and `diagnostic_head_refs` optional parameters; `train_step` adds the three-category dispatch loop between `tape.gradient` and the multiplier scaling; `metrics` property is overridden to include per-step diagnostic trackers so Keras handles their reset and integration with `logs`.
- `tests/test_layer_lr_model.py` — **critical regression test added first**: with `diagnostic_trackers=None`, train_step behavior is identical to today (existing loss-tracking regression, sparse-gradient regression, metric-update tests all still pass). Same red-green-refactor pattern Tier 2 used after the review found silent train_step issues.
- New tests for the dispatch path: with trackers configured, after one training step, all expected scalars appear in `logs`; values match tape-computed ground truth on a tiny model; `metrics` property exposes trackers correctly; Keras resets them at epoch boundary.

**Dependencies:** Phases 1, 2, 3 (trackers, factory, head contract).

**Done when:** Regression test passes (existing train_step behavior unchanged when no trackers); dispatch tests pass; `LayerLRModel`'s 13 existing tests still pass.

### Phase 5: Periodic diagnostic + callback

**Goal:** Implement `PeriodicDiagnostic` base, `PredictionDistributionDiagnostic`, and `DiagnosticsCallback`.

**Components:**
- `src/diagnostics/periodic.py` — `PeriodicDiagnostic` abstract base with `run(model, reference_batch) -> dict[str, float]`; `PredictionDistributionDiagnostic` concrete subclass (applies sigmoid, computes summary stats).
- `src/diagnostics/callback.py` — `DiagnosticsCallback(periodic_diagnostics, reference_batch_source, update_freq, update_n)`. Loads reference batch in `on_train_begin`; dispatches periodic diagnostics at configured intervals; merges results into `logs`.
- `tests/test_diagnostics_callback.py` — `on_train_begin` loads reference batch from path or default-N-of-val; `on_epoch_end` fires periodic diagnostics under `"epoch"` mode; `on_train_batch_end` fires at correct stride under `"every_n_batches"` mode; multiple periodic diagnostics merge correctly.
- Extend `tests/test_diagnostics_trackers.py` with `PredictionDistributionDiagnostic` unit test (tiny model + fixed batch → expected summary stats; sigmoid applied; returned dict has expected keys).

**Dependencies:** Phases 1, 2 (trackers + factory must construct the periodic list).

**Done when:** Callback tests pass; periodic diagnostic test passes; `DiagnosticsCallback` correctly injects scalars into `logs` that `CSVLogger` captures.

### Phase 6: Assembly wiring and smoke test

**Goal:** Wire diagnostics through `build_endpoint_model` and extend the smoke test to exercise the full round-trip.

**Components:**
- `src/model_setup/assembly.py` — `build_endpoint_model` builds the diagnostic bundle via `build_trackers(config.diagnostics, ...)` after model assembly; passes bundle + head refs into `LayerLRModel.__init__`. `build_inference_model` is unchanged (inference is diagnostic-free).
- `src/run_cca_classification.py` — construct `DiagnosticsCallback` from the config, add to the callback list alongside existing `CSVLogger` / `TensorBoard`.
- `scripts/smoke_test_integrated_stack.py` — add a `DiagnosticsConfig` to the synthetic run; verify after fit that the CSV log has the expected diagnostic columns; verify the periodic-emission row appears at the epoch boundary; verify save/load round-trip leaves trackers reinitialized cleanly.
- `tests/test_assembly.py` — extend with `build_endpoint_model` diagnostic-wiring tests: bundle is correctly passed; head refs are correctly registered; `build_inference_model` does not wire diagnostics.

**Dependencies:** Phases 1–5.

**Done when:** Smoke test passes including diagnostic column verification; assembly tests pass; `run_cca_classification.py` runs to completion on synthetic data with diagnostics enabled.

### Phase 7: Local stress test (level 1 + level 2)

**Goal:** Run the level 1 (mechanical) and level 2 (numerical, local-only checks) stress tests on real cached CCA data.

**Components:**
- Configure `run_cca_classification.py` with the diagnostic-enabled `DEFAULT_CCA_CONFIG` plus the corrected prior (π=0.02).
- Short run (a few hundred steps) on real `cca_set/`: verify mechanical pass criteria (no crash, no NaN, save/load round-trip works on real data).
- Full-length local run: verify numerical pass criteria (loss decreases, encoder grad zero, head grad bounded, no NaN/Inf across the run, FLPU components within expected ranges, correction trigger rate plausible).
- Document any unexpected diagnostic readings in a run-notes file (e.g., `docs/notes/tier5-stress-test-notes.md`).

**Dependencies:** Phases 1–6 (full diagnostic surface available).

**Done when:** Both local runs complete; level 1 and level 2 (local-portion) pass criteria from DoD are met and documented.

### Phase 8: Cluster stress test (level 2 acceptance bar)

**Goal:** Validate `mixed_float16` behavior and any cluster-specific issues, ending in the full-length cluster run that is the level-2 acceptance bar.

**Components:**
- One short cluster run with the same diagnostic-enabled config: verify mechanical pass on cluster (cluster paths, CUDA, `mixed_float16` dtype policy applied), check `grad_overflow_rate` and other mixed-precision-sensitive diagnostics behave plausibly; triage any cluster-specific issues if they surface (potentially additional short runs).
- Full-length cluster run: this is the level-2 acceptance bar. Verify the full numerical pass criteria including bounded `grad_overflow_rate` and stable training dynamics under mixed-precision over the full run.
- Update `docs/notes/tier5-stress-test-notes.md` with the cluster-run diagnostic summary; this becomes input to the level-3 retrain workstream that begins after Tier 5 closes.

**Dependencies:** Phase 7 (local runs validate the stack before committing cluster time).

**Done when:** Cluster acceptance run completes; full level-2 pass criteria met; diagnostic outputs documented for the level-3 handoff.

## Additional Considerations

**Tier 3 / Tier 4 deferred items addressed:**

- **I2 (smoke-test backbone-validation-path coverage gap)** — addressed by Phase 6's smoke-test extension and Phase 7's real-data run (the synthetic-stand-ins boundary case from `engineering-patterns.md` becomes a real-data case).
- **I8 full version (shared train/eval config through `cca_config.py`)** — partially advanced by `DiagnosticsConfig` being a `RunConfig` sub-config; full version still deferred to a later tier focused on train/eval config unification.

**Forward compatibility:**

- **Multi-head:** every per-head tracker (`LossComponentTracker`, `BatchLabelBalanceTracker`, `PredictionDistributionDiagnostic`) takes `head_name` as a first-class init parameter. Adding the immigration/US/ICA heads later is "instantiate trackers per new head," no class changes.
- **ALUM:** loss-component contract is a dict; `LossComponentTracker` subscribes to keys, not positions. ALUM extends the dict with new keys (`adversarial_term`, etc.); new tracker instances reuse the same class. A `ForwardPassTracker` category (for adversarial perturbation magnitudes observed during the second forward pass) is deferred — YAGNI applies until ALUM lands and we know what its diagnostic surface should look like.
- **Other loss types (BCE, etc.):** factory introspects each head's loss for the `return_intermediates` parameter via `inspect.signature`; if absent, loss-component trackers for that head are skipped (or factory raises with a clear message — decide during implementation; either is defensible).

**Periodic-frequency / logger-frequency alignment:** `DiagnosticsConfig.periodic_update_freq` must align with `CSVLogger` / `TensorBoard` `update_freq` for clean output. `DiagnosticsConfig.__post_init__` cannot validate this (different callback instance), but the training script's setup code should sanity-check at construction and raise if mismatched. Implementation detail for Phase 6.

**No `with_resolved`-style binding for `DiagnosticsConfig`.** Unlike `LRScheduleConfig.with_resolved`, there's no downstream consumer that needs group names pre-resolved at the config level. Group names are derived from the model at factory build time; the empirical record of "what was tracked" lives in the CSV/TensorBoard column names produced by the run. This is the conscious choice from brainstorming, not an oversight.
