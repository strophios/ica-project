# Audit and Refactor: Tiers, Checkpoints, and Process

*Last updated: 2026-05-08 (Tier 3 Piece 1 done: dual-boundary input validation on `ClassifierPreprocessor`; M2 retired from Tier 4 list; suite at 87 → 101 tests).*

This doc exists so that a new session — whether picked up by you after
weeks away, by a fresh LLM collaborator, or by someone else entirely —
can quickly reconstruct what is happening in the current phase of work,
why specific choices were made, and what the next concrete action is.

It pairs with (but does not duplicate):

- `CLAUDE.md` — the orientation doc for the project as a whole.
- `docs/notes/pinned-questions.md` — substantive questions deliberately
  deferred for deeper engagement, with enough context to re-engage.
- `scratchpad.md` at the repo root — the user's working notes, including
  the original audit findings.

# The frame

We are working through a multi-tier audit and refactor. The goal is not
just correct code today; it is a **solid foundation for building the
planned multi-head classifier** (CCA head + immigrant involvement head +
combined ICA head, possibly with ALUM, possibly with per-layer LRs and
partial encoder unfreezing). The audit is the runway for that future
work; the refactor is the runway after that.

The work is organized into four tiers, each with a checkpoint at its end
before proceeding:

- **Tier 1 — Substantive correctness.** Math, semantics, test
  infrastructure. Get what's already in the repo actually right.
- **Tier 2 — Structural refactor.** Reshape the code (model setup,
  preprocessor, paths/config, data pipeline) so it supports the multi-
  head future naturally rather than fighting it.
- **Tier 3 — Robustness.** Expand test coverage, add missing invariants,
  harden error-handling boundaries.
- **Tier 4 — Hygiene.** Scratch files, dead code, scratch imports,
  naming. Low-severity but cumulative; worth the sweep.

Each tier ends in an adversarial review checkpoint. The checkpoint's
intent is to catch things we've rationalized past; the review's success
is measured by what it finds wrong, not by what it validates. A final
adversarial pass over the whole happens after Tier 4.

# Where we are

**Tier 1 complete.** Five commits on `main`:

1. `1431797` — Fix FLPU and DEDPUL correctness; document composition
   reasoning. FLPU rewrite, DEDPUL C1 fix, `pinned-questions.md` seed
   entry, `scripts/compare_dedpul_logit_vs_prob.py`.
2. `d32c536` — Clean up critical landmines and fragile code from audit.
   Small-fix batch (path typo, KeyError landmine, torch import, dead
   code, null handling, id-uniqueness, metrics, seeds, etc.).
3. `59b5e96` — Add pytest infrastructure and initial invariant tests.
   22 tests (FLPU + data split).
4. `8685c47` — Address adversarial review: correct α framing, strengthen
   tests, fix DEDPUL attribution. α reasoning corrected; DEDPUL
   attribution corrected via four-variant re-run; test suite expanded
   to 32.
5. `cdc3472` — Update CLAUDE.md to reflect Tier 1 state.

**Tier 1 review:** performed by an adversarial opus general-purpose
agent after the first three commits. Found several real issues
(see `8685c47` commit message). Review transcript lives in session
history; the substantive findings are addressed in `8685c47`.

**Tier 2 complete.** All four pieces, integration smoke test, adversarial review, and post-review fixes done.

**Tier 3 in progress.** Boundary-enforcement spine for the multi-head future. See `docs/notes/tier3-design.md` for overall framing (boundary-inventory pattern: each finding's data/config boundaries are inventoried, validation is added at each, defense-in-depth principle without forcing the four-layer carving) and per-piece design reasoning. Pieces planned: 1 (I3, preprocessor input validation), 2 (I5, Pattern-2 serialization round-trip test), 3 (I4, train/eval config coupling), 4 (original-scope test coverage for label construction and missing-value handling). Pieces 2–4 are pending design discussion; Piece 1 has landed.

  - **Piece 1 done** (this commit). Dual-boundary input validation on `ClassifierPreprocessor`. `__init__` checks for *internal-config-validity* bugs (raises `ValueError` with informative messages naming the bad value): `text_key` non-empty string, `label_keys` is a dict, `target_dtype` is a Keras-recognized dtype string, and standard-mode (`endpoint_model=False`) requires non-empty `label_keys` (the empty-`label_keys` configuration is only valid in endpoint mode for predict-only flow). `__call__` checks for *config-vs-data-mismatch* bugs (raises `KeyError` with the missing column set, the configured expectation, and the batch's actual keys, enumerating *all* missing columns rather than failing fast). Retires M2 from the Tier 4 deferred list — `target_dtype` validation lives naturally in the construction-validation block. 14 new tests across two new test classes (`TestConstructionValidation`, `TestCallTimeInputValidation`); suite at 87 → 101 passing. Smoke test re-run: still passes (Pattern A vs. Pattern 2 max-diff 0.00e+00).

- `789d88c` — Piece 1: `ClassificationHead` for multi-head refactor.
  Head-as-Layer abstraction, supports both standard mode (loss via
  outer `compile`) and endpoint mode (loss via `add_loss`). 11 tests
  for construction, forward-pass shape, endpoint-mode contract, and
  weight structure. Lives in `src/model_setup/heads.py`.
- `ad0f94b` — Piece 2: `LayerLRModel` for per-layer learning rates.
  `keras.Model` subclass overriding `train_step` to scale gradients
  by per-variable multipliers before optimizer apply. Supports
  discriminative fine-tuning (fixed geometric multipliers) and
  gradual unfreezing (callback-updated multipliers). 10 tests
  including full numerical verification against a numpy reference
  for uniform-multiplier, multiplier=0, and multiplier=0.5 cases.
  Lives in `src/model_setup/layer_lr_model.py`.
- `e3dda6a` — Piece 3: `ClassifierPreprocessor` refactored for
  multi-head targets. Takes `label_keys: dict[str, str]` mapping
  output-dict-key → source-column-name; emits a model-inputs-shaped
  dict (endpoint mode) or `(features, targets_dict)` tuple (standard
  mode). Targets cast to a stable `target_dtype` (default `float32`)
  at preprocess time so cached datasets carry predictable dtype;
  losses still cast `y_true` to `y_pred.dtype` at the loss boundary
  for mixed-precision robustness — see Piece 3 design doc for the
  layered-dtype framing. Positional-input fallback dropped (datasets
  are now required to be dict-valued, which `run_cca_classification.py`
  already produces). 12 tests covering construction, single- and
  multi-head shape contracts in both modes, source→output routing,
  and dtype casting. Lives in `src/preproc/preprocessor.py`; tests at
  `tests/test_preprocessor.py`.
- `4d8cba9` — Piece 4a: `src/config.py` introduced as the
  single source of truth for platform-conditional values (paths +
  dtype policy). File-existence detection on `/projects/ahd` with
  `ICA_ENV` env-var override; granular paths exposed as
  `pathlib.Path` constants (`PROJECT_ROOT`, `CCA_SET_DIR`,
  `DAPT_BACKBONE_WEIGHTS`, etc.); `DTYPE_POLICY` is `mixed_float16`
  on cluster CUDA, `float32` locally. `src/data_setup/dapt_data.py`
  renamed to `src/data_setup/data.py` (the file's contents weren't
  DAPT-specific — they handle parquet I/O, classification splits,
  and tf.data pipelines used across DAPT and classification).
  All callers updated: `dapt.py`, `run_cca_classification.py`,
  `eval_cca_classifier.py`, `run_prior_estimate.py`,
  `prior_estimation/lu_classifier.py`, scratch files
  (`test_module.py`, `test_script.py`), and `tests/test_data_splits.py`
  import. Test suite: 65 passing (no test changes — rename is
  mechanical, paths logic is environment-dependent and not
  test-mockable without sacrificing the test's value).

- `2f069c4` — Piece 4b: `src/model_setup/backbone.py` and
  `src/model_setup/assembly.py` introduced. `load_dapt_backbone`
  is the weights-only DAPT-checkpoint loader (legacy
  full-saved-model path dropped — was scratch-only).
  `build_endpoint_model` returns a `LayerLRModel` with `token_ids`,
  `padding_mask`, and one `"<head>_targets"` `keras.Input` per head
  (the suffix avoids op-name collision with the head Layer itself,
  which was the wrinkle that surfaced during implementation —
  Keras requires unique op names within a Functional graph).
  `build_inference_model` returns a `keras.Model` without target
  inputs; sharing head Layer instances between the two models gives
  Pattern A in-process weight sharing. The Pattern A safety question
  was settled empirically: `scripts/experiment_endpoint_inference_evaluate.py`
  (kept as a permanent fixture) demonstrates that Keras 3 filters
  losses by graph reachability, so the head's training-graph
  add_loss tensor doesn't contaminate inference-model `evaluate()`
  even when head instances are shared. 13 integration tests in
  `tests/test_assembly.py` cover construction, forward pass,
  training step, `freeze_encoder`, and Pattern A weight sharing.
  Plus a sparse-gradient (`tf.IndexedSlices`) handling fix in
  `LayerLRModel.train_step`: `Embedding` layers produce sparse
  gradients, and the prior `multiplier * grad` form failed for
  them; `tf.math.scalar_mul` handles both dense and sparse cases.
  This was a latent bug from Piece 2 surfaced by 4b's integration
  tests; one regression test added to `tests/test_layer_lr_model.py`.
- `06e161c` — Piece 4c: training and eval scripts rewritten
  end-to-end to use the Tier 2 abstractions; `classifier_from_dapt_checkpoint`
  and `src/model_setup/classification_setup.py` retired. Substantive
  changes: `ClassificationHead` extended with a `metrics` parameter
  (symmetric with `loss_fn` — both fire only when targets supplied;
  metric instances are renamed to be prefixed with the head's name to
  avoid multi-head collisions); `run_cca_classification.py` builds
  train + inference models sharing the head and backbone (Pattern A);
  `eval_cca_classifier.py` builds a fresh inference model and loads
  weights by name (Pattern 2); two preprocessor instances split fit/eval
  (with `cca_targets`) from predict (no targets) explicitly;
  `LossScaleOptimizer` wraps AdamW conditional on `IS_CLUSTER`; FLPU
  prior 0.03 → 0.02; the pre-existing test-predict bug
  (`steps=validation_steps` + `.repeat()` producing duplicate predictions)
  is fixed by building finite predict datasets manually. 6 new tests
  added to `tests/test_heads.py` for the metrics extension.

**Tier 2 pieces complete.** All four pieces (and their sub-pieces
4a/4b/4c) are landed. The Tier 2 abstractions are wired into the
training and eval paths; the legacy `classifier_from_dapt_checkpoint`
+ `classification_setup.py` are gone.

**Tier 2 closeout items:**

- *(this commit)* — Integration smoke test:
  `scripts/smoke_test_integrated_stack.py`. Exercises the full
  Tier 2 pipeline end-to-end on synthetic data (preprocessor →
  dataset_create → build_endpoint_model → build_inference_model
  (Pattern A) → fit → save_weights → rebuild → load_weights
  (Pattern 2) → predict). Uses a fake backbone (Embedding +
  multiply-by-padding-mask) so it runs locally without DAPT
  weights or cached cca_set/ data; runtime ~30s. Verified all
  eight pipeline steps succeed; Pattern A and Pattern 2 produce
  bitwise-identical predictions (max-diff 0.00e+00); backbone
  weights load correctly across the rebuild boundary. One small
  observation flagged for follow-on: fit's progress-bar "loss"
  field showed 0.00e+00 even though training was happening
  (metrics moved, weights changed) — most likely a Keras display
  artifact when compile-time loss is None and add_loss provides
  the loss; substantive at-runtime loss-monitoring should be
  verified during the actual cluster run before relying on the
  loss curve for monitoring / early stopping.
- *(this commit)* — Adversarial review of Tier 2: dispatched
  `code-reviewer` subagent against the cumulative diff
  (`5ddc330..079deff`, 12 commits). Returned **2 Critical, 4
  Important, 5 Minor**. The architectural shape — endpoint-layer
  pattern, Pattern A/2 split, naming conventions, metrics-in-head
  extension, deletion of `classification_setup.py` — got a clean
  bill. The blockers were localized regressions in
  `LayerLRModel.train_step` (C1) and an incomplete migration of two
  Phase-2 caller scripts (C2). Both Critical issues + Important
  issues I1 and I2 fixed in this commit; review notes and remaining
  deferrals in `docs/notes/tier2-design.md` "Post-review corrections"
  section.

  - **C1 fix**: `LayerLRModel.train_step` rewritten to mirror stock
    Keras `train_step` (TF backend) line-for-line plus the existing
    multiplier-scaling step. Adds `_compute_loss(... training=True)`
    in place of `compute_loss`, `_loss_tracker.update_state(loss,
    sample_weight=batch_size)`, and `optimizer.scale_loss(loss)`
    inside the `GradientTape` context. The smoke-test `loss=0`
    symptom flagged in the integration-pass commit was, as the
    review correctly diagnosed, a real bug rather than a display
    artifact: `_loss_tracker` was never being updated. Re-running
    the smoke test after the fix shows real loss values
    (`loss: 0.1865 → 0.1799` across batches). Bonus: the
    `optimizer.scale_loss` omission means `LossScaleOptimizer`
    (which Piece 4c wraps `AdamW` in conditional on `IS_CLUSTER`)
    was a no-op pre-fix; cluster mixed-precision training would
    have silently degraded vs. local. Now fixed.

  - **I1 fix**: `tests/test_layer_lr_model.py` gained a
    `TestLossTracking` class with two regression tests
    (`test_fit_history_records_nonzero_loss` and
    `test_fit_history_loss_close_to_evaluate_loss`). Both went
    red→green on the C1 fix. They explicitly assert against
    `history.history["loss"]` content rather than just structural
    weight changes — the missing assertion shape that allowed C1 to
    ship. Test suite: 85 → 87.

  - **I2 fix**: rolled into the C1 fix (`_compute_loss` substitution).

  - **C2 fix**: `run_prior_estimate.py` and
    `prior_estimation/lu_classifier.py` updated to the new
    `label_keys: dict[str, str]` preprocessor signature (broken on
    disk since `e3dda6a` / Piece 3, missed in 4a's caller-update
    sweep and 4c's training/eval rewrite). Both scripts now parse
    and import cleanly; runtime verification will happen at the
    next prior-estimation run on the cluster.

- **Deferred from review to Tier 3 (robustness)**: I3 (preprocessor
  source-column validation), I4 (test-eval coupling to training
  preprocessor), I5 (Pattern-2 serialization-format invariant
  pinning).
- **Deferred from review to Tier 4 (hygiene)**: M1, M3, M4
  (scratch-file raise placement, `_default_group_fn` separator,
  default head-name collision risk). **M2** (`target_dtype`
  validation) was retired into Tier 3 Piece 1 (2026-05) where it
  landed alongside the other construction-time validation checks
  on `ClassifierPreprocessor.__init__`. M5 (stale Piece 3
  design-doc paragraph) fixed in the Tier-2-review-corrections
  commit since it was a one-line edit.

Test suite after Piece 4c: **85 tests passing** (32 from Tier 1 + 17
for `ClassificationHead` (11 original + 6 metrics) + 11 for `LayerLRModel`
(10 original + 1 sparse-gradient regression) + 12 for
`ClassifierPreprocessor` + 13 for `assembly`).

**Deferred empirical checks** (to be batched when environment handling
is settled — touched in Piece 4):

- Smoke training run with the updated FLPU + corrected prior (≈ 0.02).
- Confirm training dynamics are reasonable under mixed_float16 + the
  removed-α FLPU.
- Sensitivity sweep on Ratio Batch (currently 1:10, much more aggressive
  than Ji 2023's prescription).

**Pedagogical framing in practice.** Starting from Tier 2 Piece 1, the
implementation mode shifted: Claude generates designs, test stubs, and
skeletons with detailed TODO recipes; the user hand-writes the core
implementation bodies; Claude comes back to harmonize comments and
commit. This is to keep the user's understanding of Keras internals
developing — Layer subclassing for Piece 1, `Model.train_step` override
for Piece 2. Candidates for the same treatment in remaining pieces:
the `ClassifierPreprocessor` rewrite in Piece 3, where the multi-head
target routing is the learning-rich part.

# Tier 2: Structural refactor (in progress)

**Intent:** reshape the code so that the multi-head future is a natural
extension rather than a rewrite. Tier 2 is *about shape*, not about math
or correctness — those were Tier 1. If Tier 2 touches the math, that is
a signal we designed the shape wrong.

**Scope, with status markers:**

1. **[DONE — commit `789d88c`] Model setup separation: heads as
   Layers.** Piece 1. `ClassificationHead` in
   `src/model_setup/heads.py`. Supports endpoint mode (loss via
   `add_loss`) and standard mode. Design and reasoning in
   `docs/notes/tier2-design.md` Piece 1.
2. **[DONE — commit `ad0f94b`] Per-layer learning rates / selective
   unfreezing.** Piece 2. `LayerLRModel` in
   `src/model_setup/layer_lr_model.py`. Custom `train_step` override
   applies per-variable multipliers before optimizer apply. Design
   and reasoning in `docs/notes/tier2-design.md` Piece 2.
3. **[DONE — commit `e3dda6a`] Preprocessor refactor for multi-label
   targets.** Piece 3. `ClassifierPreprocessor` takes
   `label_keys: dict[str, str]` mapping output-dict-key →
   source-column-name; emits multi-head-shaped output in both
   endpoint and standard modes. Targets cast to `target_dtype`
   at preprocess time (loss still casts at its boundary —
   layered-dtype framing). Positional-input fallback dropped.
   Design and reasoning in `docs/notes/tier2-design.md` Piece 3.
4. **[IN PROGRESS] Paths / config + data pipeline rename, and
   integration.** Piece 4. Subdivided into 4a/4b/4c.
   - **4a [DONE — commit `4d8cba9`]**: `src/config.py` introduced
     (platform detection + paths + dtype policy);
     `data_setup/dapt_data.py` renamed to `data_setup/data.py`;
     all callers updated. Mechanical; design and reasoning in
     `docs/notes/tier2-design.md` Piece 4a.
   - **4b [DONE — commit `2f069c4`]**: `src/model_setup/backbone.py`
     (`load_dapt_backbone`) and `src/model_setup/assembly.py`
     (`build_endpoint_model`, `build_inference_model`,
     `_default_group_fn`). Pattern A safety verified empirically
     via `scripts/experiment_endpoint_inference_evaluate.py`.
     Sparse-gradient fix in `LayerLRModel.train_step`. 13
     integration tests in `tests/test_assembly.py` + 1 regression
     test in `tests/test_layer_lr_model.py`. Design and reasoning
     in `docs/notes/tier2-design.md` Piece 4b.
   - **4c [DONE — commit `06e161c`]**: training and eval scripts rewritten
     end-to-end on the new abstractions; `ClassificationHead`
     extended with `metrics` parameter (head-internal, name-prefixed);
     `classifier_from_dapt_checkpoint` and
     `model_setup/classification_setup.py` deleted. Pattern A in the
     training script (in-process Layer-instance sharing); Pattern 2
     in the eval script (cross-process weight loading by name).
     Conditional `LossScaleOptimizer` wrap on `IS_CLUSTER`. FLPU
     prior 0.03 → 0.02. Test-predict bug fixed via finite predict
     datasets. Design and reasoning in `docs/notes/tier2-design.md`
     Piece 4c.
5. **[DONE — this commit] Integration pass.** End-to-end smoke test
   of the composed stack on synthetic data.
   `scripts/smoke_test_integrated_stack.py`. All eight pipeline
   steps verified; Pattern A vs Pattern 2 predictions match
   bitwise. Loss-display question flagged for follow-on.
6. **[PENDING] Adversarial review of Tier 2.** Likely the
   `code-reviewer` subagent this time (plan-alignment /
   architecture framing fits Tier 2 better than the opus
   general-purpose agent used for Tier 1).
7. **[ONGOING] Tests follow the refactor.** The tests-as-spec
   property from Tier 1 means existing tests should keep passing
   across the refactor. Where they fail, they are either flagging a
   regression or flagging a test that was testing implementation
   rather than behavior. Adjust accordingly (and add new tests for
   new invariants the refactor introduces).

**Process commitments (stated up front, still in effect):**

- **Design pass first.** A concrete design section in
  `docs/notes/tier2-design.md` before any code, explaining the
  intended structure, the patterns used, and the trade-offs chosen.
  The doc's job is to leave the user (and any future reader) with a
  mental model of what changed and why — not to be a spec for
  Claude to execute in isolation.
- **Staged implementation.** Not one giant refactor commit. Each
  piece preceded by a short "here's what we're about to do and the
  pattern you'll see" explainer. Pause points for check-in.
- **Named patterns.** New patterns (endpoint layers with `add_loss`,
  `Model.train_step` override, etc.) get explicit names and brief
  walkthroughs before use, so the user accumulates mental vocabulary.
- **Explanatory comments in code.** Denser than production norm,
  especially on Keras/TF idioms the user has not implemented before.
  Trim later once the pattern is internalized.
- **"You implement it" mode by request.** Offered explicitly when
  the implementation itself carries learning the user wants to own.
  Used for Piece 1 (`ClassificationHead`) and Piece 2
  (`LayerLRModel.train_step`); anticipated for Piece 3's
  `ClassifierPreprocessor` rewrite.

# Tier 3: Robustness

In progress. **Intent**: harden the boundaries the multi-head future
will lean on. Tier 2 reshaped the code; Tier 3 enforces the contracts
that reshape created. See `docs/notes/tier3-design.md` for the
overall framing (boundary inventory replacing the four-layer
defense-in-depth carving) and per-piece design reasoning.

**Scope, with status markers:**

1. **[DONE — Piece 1 (this commit)] I3 — preprocessor input
   validation.** Dual-boundary validation on `ClassifierPreprocessor`:
   `__init__` for internal-config-validity, `__call__` for
   config-vs-data-mismatch. Retires M2 from Tier 4 inheritance.
   14 new tests; suite at 87 → 101.
2. **[PLANNED] I5 — Pattern-2 serialization round-trip test.**
   Round-trip Pattern A (in-process) → save → fresh-build → load →
   Pattern 2 produces bitwise-identical predictions. Tightens
   `eval_cca_classifier.py` to call `load_weights(...,
   skip_mismatch=False)` so future variable-name drift fails loudly.
3. **[PLANNED] I4 — train/eval config coupling.** A config object
   shared between training and eval scripts (Option B: serialized
   alongside weights, or Option C: static module + serialized run
   config). Per pinned question #3, the config should not encode
   the train/predict distinction.
4. **[PLANNED] Original-scope test coverage.** Label-construction
   tests for the cca/immig/descriptor boolean combinations in
   `create_classifier_data`; missing-value (`fill_null("")`)
   coverage; any preprocessor shape-contract gaps surfaced by
   Piece 1.

**Deferred (not Tier 3): evaluation harness with calibration.**
Originally listed in the Tier 3 plan, but it's a research deliverable
(threshold selection on a hand-labeled PN test set, possibly Platt
scaling or isotonic regression), not foundation work. Punt to a
separate piece of work after Tier 4 hygiene.

The S6 looping issue (`steps=validation_steps` on test predict
producing duplicate predictions) was fixed during Tier 2 Piece 4c
and doesn't need Tier 3 treatment.

# Tier 4: Hygiene

Planned, not started. Scope includes:

- Move scratch files (`test_module.py`, `test_script.py`,
  `endpoint_layer_test.py`) to `scratch/` or `scripts/scratch/`. Extract
  the useful "save DAPT backbone weights" logic from `test_module.py`
  into `model_setup/` as a real function.
- Remove commented-out dead code from `preprocessor.py`, `dapt_data.py`,
  and the exploration-log block in `dapt.py`.
- Consider archiving or deleting `ramaswamy2016.py` (confirmed dropped).
- Minor style / import cleanups.

# How to pick up mid-work

Expected: you (or a fresh session) finishes a conversation mid-tier and
comes back later. To reconstruct state:

1. `git log --oneline -10` — the last several commits, with their titles
   summarizing substantive changes.
2. Read the most recent commit message for detail on what was just done.
3. `cat docs/notes/tiers-and-checkpoints.md` (this file) — knows the
   current tier and what is pending.
4. `cat docs/notes/pinned-questions.md` — knows the substantive deferred
   questions.
5. `cat scratchpad.md` — the original audit findings and user working
   notes.
6. `uv run pytest tests/ -q` — confirm the test suite still passes.

If a design doc exists for the current tier
(e.g., `docs/notes/tier2-design.md`), read that next.

# Process notes worth preserving

- Brainstorm-first, commit-later. Significant changes come with design
  reasoning captured in commits or `docs/notes/` before code lands.
- Tests as spec. Structural invariants get tests before refactors that
  touch them. A failed test during a refactor is diagnostic, not
  catastrophic.
- Pinned-questions for deliberate deferral. Anything we are choosing
  not to engage with deeply *now* but that we do not want to lose goes
  into `pinned-questions.md` with enough context to pick up later.
- Adversarial review at checkpoints, not continuously. Review is
  expensive and has diminishing returns at every commit; strategic
  checkpoints are where it earns its cost.
- Pedagogical framing: the user is developing their own understanding
  alongside this work. Code generation is accompanied by pattern-level
  explanation, named patterns, and explicit unfamiliarity-flags for
  things the user has not implemented before. See process commitments
  under Tier 2 above.

---
