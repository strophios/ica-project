# Engineering Patterns

This document catalogs CS-specific engineering patterns validated
through the Tier 2 / Tier 3 / Tier 4 work in this project.
Patterns here are about *how the code is shaped* — validation
boundaries, config conventions, model-sharing approaches — and
are largely project-and-language-specific.

**Audience:** the user, and Claude on future sessions in this
project. **Purpose:** discoverability when looking for "do we
have an established way to do X in this codebase?"; proactive
suggestion when designing new features.

Companion document: `docs/notes/process-patterns.md` covers
content-agnostic process patterns (workflow shapes, review
disciplines, doc structures). Engineering patterns tend to be
project-specific; process patterns transfer more easily.

## Project-local validation scope

Section membership (Validated vs Developing) reflects evidence
**in this project**. The per-pattern `Validation status` field
may carry cross-project or literature context, but section
placement follows the project-local promotion rule below.

## Promotion rule

Patterns start in **Developing**. A pattern graduates to
**Validated** when it has been used successfully in **≥2
independent applications in this project** *and* we can
articulate **at least one boundary condition** (where the pattern
doesn't apply, or where its limits are unclear).

The boundary-condition requirement is the discipline that prevents "validated" from becoming a self-reinforcing assertion.

## Patterns

### Validated
- [Boundary-inventory pattern](#boundary-inventory-pattern) — validate at every layer where data crosses a boundary
- [Synthetic stand-ins for heavyweight dependencies in fast tests](#synthetic-stand-ins-for-heavyweight-dependencies-in-fast-tests) — minimal fakes matching real API shape
- [Wrapped-vs-flat forward-compat for config sub-objects](#wrapped-vs-flat-forward-compat-for-config-sub-objects) — wrap related fields in sub-objects anticipating future variants

### Developing
- [Pattern A vs Pattern 2 model sharing](#pattern-a-vs-pattern-2-model-sharing) — in-process Layer-instance sharing vs cross-process load-by-name
- [Empirical investigation before committing to design](#empirical-investigation-before-committing-to-design) — run a small experiment when library behavior is uncertain

---

## Validated patterns

### Boundary-inventory pattern

**Validation status**: Used in Tier 3 Piece 1 (preprocessor dual-boundary validation, commit `79ab31c`), Tier 3 closeout (I1 schema validation + I7 dtype check), and Tier 4 Piece 1 (M4 dual head-name validation, commit `a3505ae`). Three+ applications in this project with consistently positive results. Cross-project context: the house-style `defense-in-depth` skill articulates the same animating insight ("validate at every layer; each catches what others miss; turn bugs from 'vigilance keeps us safe' into 'the structure prevents them'").

**First used**: 2026-05-08, Tier 3 Piece 1 (commit `79ab31c`).

**Last used**: 2026-05-12, Tier 4 Piece 1 M4 (commit `a3505ae`).

**Known boundary conditions**: Applies to **data-flow boundaries** specifically — places where data crosses from one trust domain or representation into another. The house-style `defense-in-depth` skill's web-app-flavored 4-layer carving (entry-point / business-logic / environment-guards / debug-instrumentation) is only partial fit for this codebase; Layers 1 (entry-point) and 2 (business-logic) map cleanly, but Layers 3 and 4 are web-app-specific and don't transfer. The pattern doesn't apply to internal helper functions that operate on already-validated data.

The boundary-inventory pattern starts with a question: where do data or configuration cross boundaries in the system? A boundary is any point where one piece of code hands off to another — at a method call, across a config file, between processes. Once you inventory the boundaries, you ask what each boundary should validate. The `ClassifierPreprocessor` has two: one at `__init__` (configuration in, object out) and one at `__call__` (data dict in, model tensors out). The `__init__` boundary catches internal-config bugs — e.g., `endpoint_model=False` with empty `label_keys` is nonsensical regardless of input data. The `__call__` boundary catches config-vs-data mismatches — the configured `text_key` column doesn't exist in this batch. Neither boundary can detect what the other one misses; together, they're more robust.

Applied in Tier 3 Piece 1: the preprocessor's dual-boundary validation caught two different failure modes. Applied in Tier 4 Piece 1 M4: head-name validation appears at two boundaries — `ClassificationHead.__init__` requires explicit naming (catches "I forgot to name this"); `build_endpoint_model` asserts unique names across all heads (catches "I named two things the same"). The pattern's power comes from the fact that each boundary checks exactly what it can see. A constructor can't see how many other heads are being built; an assembly function can't check internal dataclass consistency. Validating at both points is redundant in the happy path but catches different failure modes.

When designing a multi-part system (preprocessor + model; data loading + training), inventory the handoff points and decide what each should validate. Don't try to do all validation in one place — you'll either over-validate at one boundary (expensive) or under-validate at others (dangerous).

### Synthetic stand-ins for heavyweight dependencies in fast tests

**Validation status**: Used in `scripts/smoke_test_integrated_stack.py` (fake backbone) and pervasively in `tests/` (synthetic row counts, mock datasets). Two+ applications in this project with positive results in terms of test-suite speed. The canonical boundary case (I2, see below) confirms the pattern's limits are non-trivial — promoted to Validated despite n=2 because the boundary condition is explicit and the value is clear.

**First used**: 2026-04-27, smoke test introduction (commit `079deff` "Add Tier 2 integration smoke test").

**Last used**: 2026-05-09, Tier 3 Piece 4 (commit `96ea283` "Tier 3 Piece 4: original-scope test coverage"), which added data-loading tests using synthetic dataframes.

**Known boundary conditions**: **Stand-ins must match the real dependency's API shape, not just its value-level interface.** The canonical case is **I2** (smoke-test backbone-validation gap, inherited from Tier 3 closeout): the fake backbone in the smoke test exposes `hidden_dim` as an instance attribute, but the real `keras_hub.models.Backbone.from_preset(...)` exposes it as a class property. The smoke test's `validate_against_backbone` call passes against the fake but doesn't catch property-vs-attribute drift in the real surface. The smoke test docstring (`scripts/smoke_test_integrated_stack.py:28-40`) acknowledges this gap; closure would require either a real-keras_hub backbone variant (heavyweight; env-gated or cluster-only) or accepting the gap as the cost of fast smoke tests.

The pattern uses minimal fakes that match the real dependency's API surface for tests that would otherwise be slow due to heavyweight dependencies. The smoke test needs a backbone-shaped object to exercise the loading and config-building paths. A real RoBERTa backbone loads from disk or network and takes seconds; a fake that returns a simple object with a `hidden_dim` attribute and a `predict` method runs in milliseconds. This speed gain is why the smoke test is valuable for CI/CD (tight feedback loop on integration correctness). The risk is that the fake's surface doesn't match the real object's surface — and you only discover this mismatch when you run against the real object.

The I2 case is instructive. The fake was implemented as a simple class: `class FakeBackbone: hidden_dim = 768; def predict(self, x): return np.zeros(...)`. This works for instance-attribute access (`backbone.hidden_dim`). But when the real backbone's interface changes (or was always) to expose `hidden_dim` as a class property (accessed via `Backbone.from_preset(...).hidden_dim` but defined on the class), the smoke test can't catch the difference. The synthetic stand-in works at the value level (both return 768) but fails at the shape level (attribute vs property).

Apply this pattern to tests where the real dependency adds >1 second of overhead. Construct the fake by mocking at the class level (using `unittest.mock.MagicMock` or a simple dataclass) when the real object's interface uses class properties, and by creating a simple instance subclass when the real object uses instance attributes. When constructing a fake, trace the real object's interface and match it exactly — don't just implement the subset you think you need.

### Wrapped-vs-flat forward-compat for config sub-objects

**Validation status**: Used in Tier 3 Piece 3a (RunConfig with four sub-configs: `FLPULossConfig`, `OptimizerConfig`, `LRScheduleConfig`, `RatioBatchConfig`, commit `d9c0348`) and Tier 4 Piece 2 (`ResolvedSteps` nested in `LRScheduleConfig`, commit `017dffe`). Two independent applications with clear positive results — both cases anticipated future variants and the wrapping shape paid off.

**First used**: 2026-05-09, Tier 3 Piece 3a (commit `d9c0348`).

**Last used**: 2026-05-12, Tier 4 Piece 2 (commit `017dffe`).

**Known boundary conditions**: Pattern applies when (a) future variants of the wrapped object are anticipated (e.g., `FLPULossConfig` anticipates ALUM and BCE variants per `docs/notes/pinned-questions.md`), OR (b) the wrapped fields form a coherent semantic group (e.g., `ResolvedSteps` wraps three fields that all come from the same train-time computation). Not justified for ad-hoc collections of unrelated fields — the extra nesting cost in JSON and code reads doesn't pay back without one of these justifications. Premature wrapping is its own anti-pattern.

When designing a frozen dataclass config, the question arises: should related fields be grouped in a sub-object or flattened into the parent? The wrapped-vs-flat choice has tradeoffs. Flat is simpler — fewer indentation levels, fewer class definitions — but mixes concerns; wrapped carries semantic clarity and forward-compat but costs nesting. The pattern says: wrap when the wrapped fields will plausibly evolve as a unit.

Tier 3 Piece 3a chose wrapped for the loss config: `FLPULossConfig` groups the fields specific to FLPU loss (prior, alpha, gamma). A future ALUM loss would have different fields (temperature, vat-epsilon, etc.). By wrapping, the `RunConfig` stays stable even as loss variants change. If fields were flat (`run_config.prior`, `run_config.alpha`, `run_config.gamma`, `run_config.temperature`, ...), adding a new loss variant means adding new optional fields to `RunConfig`, which gets unwieldy. The same principle applied to optimizer config (SGD, Adam, Adagrad each have different tuning parameters) and schedule config (warmup-then-decay looks different from cyclical).

Tier 4 Piece 2 applied the pattern again: `ResolvedSteps` groups `warmup_steps`, `decay_steps`, and `steps_per_epoch` inside `LRScheduleConfig`. These three fields are computed together at train time and are useless separately. If future schedule types (cyclical, polynomial decay) add more train-time-computed fields, `ResolvedSteps` grows as a unit. The wrapped shape makes it clear: "this is the computed record of what the schedule needs"; it's separate from the factor-input (`warmup_steps_factor`, `decay_steps_factor`).

When designing config objects, ask: will these fields evolve together? Will future variants add different fields? Do they form a coherent semantic group? If yes to any of these, wrap. If the fields are unrelated (field A for loss, field B for optimizer, field C for logging), keep them flat. The cost-benefit only works if the nesting carries meaning.

---

## Developing patterns

### Pattern A vs Pattern 2 model sharing

**Validation status**: Used in Tier 2 Piece 4c only (n=1 architecturally, commit `06e161c`, though both variants exercised). Pattern A in `run_cca_classification.py` (in-process Layer-instance sharing between train and post-train predict). Pattern 2 in `eval_cca_classifier.py` (cross-process: load weights into a freshly-built model). Boundary conditions were further explored in Tier 3 Piece 2 (commit `4243c63`) via the `.weights.h5` load-by-structure vs load-by-name finding. Promotion to Validated waits for one more architectural use case to confirm the patterns generalize.

**First used**: 2026-04-27, Tier 2 Piece 4c (commit `06e161c`).

**Last used**: 2026-05-08, Tier 3 Piece 2 (commit `4243c63`, explored related territory).

**Known boundary conditions**: Pattern A requires same-process train and inference (Layer instances are Python objects, not serializable across processes). Pattern 2 requires the inference-side model to be reconstructable independently — which means head names, hidden_dim, dropout, etc. must round-trip through the sidecar (this is what made the sidecar-as-single-source-of-truth work in Tier 3 Piece 3 and Tier 4 Piece 2 important). Tier 3 Piece 2's empirical finding (that `.weights.h5` keys variables by layer-class + positional index, not user-given name) reframed Pattern 2 from load-by-name to load-by-structure; the head-name contract is enforced at call sites (Keras `compile` routing), not at weight load.

Pattern A shares Layer instances between the training and post-train inference models. The training model is a `keras.Model` that includes the input tensors and target tensors, wired through the backbone and head layers. After training, you build a fresh inference model that reuses the same head Layer instances — since they're shared Python objects, the weights are automatically available. This works within a single process but breaks across process boundaries (Layer instances aren't serializable).

Pattern 2 rebuilds the model fresh in a new process (or later session) and loads weights from disk. The training script saves weights; the eval script constructs a fresh model (backbone + head) with the same architecture and loads the saved weights. This requires the architecture to be reproducible from metadata (which is why the sidecar with head name, hidden_dim, dropout became critical).

Tier 2 Piece 4c implemented both patterns: the training script uses Pattern A (construct training model, train, then predict post-train using the same Layer instances); the eval script uses Pattern 2 (fresh model, load weights). Tier 3 Piece 2 explored when the patterns work via a round-trip test: Pattern A saves weights, Pattern 2 loads them fresh, predictions should match bitwise. The interesting finding was that Keras's `.weights.h5` save format keys variables by layer-class + positional index (not by user-given name), which means renaming a head doesn't break load — the pattern is load-by-structure, not load-by-name. This reframed the understanding of Pattern 2 and made the explicit head-name contract (enforced at `compile` call sites) more important.

Open questions: Does Pattern 2 generalize to multi-head systems cleanly? Can you load a different head configuration than what was saved? Do we need a Pattern 3 for more complex model-sharing scenarios (e.g., transfer learning where you swap out one head but keep others)? This pattern will likely graduate to Validated when multi-head work lands and we need to extend the sharing scheme — probably discovering edge cases that clarify the pattern's boundaries.

### Empirical investigation before committing to design

**Validation status**: Used in Tier 3 Piece 2 only (n=1, commit `4243c63`). The pattern worked — the experiment caught a wrong assumption about Keras's `.weights.h5` keying behavior before it became baked into the test design and eval-script design — but boundary conditions (when is this worth doing? what magnitude of uncertainty justifies the investigation?) are unclear.

**First used**: 2026-05-08, Tier 3 Piece 2 (commit `4243c63` or an earlier experimental script).

**Last used**: same (2026-05-08).

**Known boundary conditions**: The cost-benefit framing is missing. The Tier 3 Piece 2 case was a clear win because the wrong assumption (Keras saves weights by user-given layer name) would have shaped multiple files (test design, eval-script design, smoke-test design); catching it before commit saved rework. But the pattern as currently used doesn't articulate when the experiment is worth running vs when reading docs or source code is sufficient. The question is: what threshold of uncertainty (or surface area affected) triggers "write a script to check"?

When library or external dependency behavior is non-obvious, you can respond in three ways: (1) read the docs, (2) read the source code, (3) write a small empirical test script. The pattern describes option 3. Tier 3 Piece 2 faced a question about Keras's `.weights.h5` saving behavior — does it key variables by user-given name (so renaming a layer breaks load) or by structure (so renaming doesn't matter)? The docs weren't clear. Reading Keras source was possible but tedious (the saving code is deep in TensorFlow). Writing a quick script (`scripts/experiment_endpoint_inference_evaluate.py` or similar) that creates a model, saves, renames a layer, loads, and checks if it worked took 15 minutes and answered the question definitively. This finding cascaded: it reframed the Pattern 2 understanding, shaped what the I5 round-trip test should check, and influenced whether renaming a head would be safe.

Promotion to Validated waits for a second use case where the experiment-vs-research tradeoff is more visible. The key unknowns: what magnitude of surface area (how many downstream files affected by the wrong assumption) justifies writing an experiment? Can you usually answer the question via docs + source? When is an experiment necessary vs just time-consuming?

Apply this pattern when a library's behavior is uncertain AND the wrong assumption would reshape significant design surface. If the assumption affects only one line of code, reading the docs is sufficient. If it cascades (the I5 test design depends on the answer, the eval script structure depends on it, the smoke test strategy depends on it), write the experiment.
