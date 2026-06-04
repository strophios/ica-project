# Audit and Refactor: Tiers, Checkpoints, and Process

*Last updated: 2026-06-04 (Tier 5 subagent-executable portion complete: diagnostic instrumentation + empirical-run runbooks. Phases 1–6 + Phase 7 Tasks 1–2 + Phase 8 Task 1 committed; Phase 7 Tasks 3–5 and Phase 8 Tasks 2–4 are HUMAN-OPERATED and not yet performed. 374 tests passing.).*

This doc exists so that a new session — whether picked up by you after
weeks away, by a fresh LLM collaborator, or by someone else entirely —
can quickly reconstruct what is happening in the current phase of work,
why specific choices were made, and what the next concrete action is.

It pairs with (but does not duplicate):

- `CLAUDE.md` — the orientation doc for the project as a whole.
- `docs/notes/pinned-questions.md` — substantive questions deliberately
  deferred for deeper engagement, with enough context to re-engage.
- `docs/notes/process-patterns.md` and `docs/notes/engineering-patterns.md`
  — catalogs of process and engineering patterns validated through
  Tier 2/3/4 work, each with per-pattern metadata (validation status,
  first/last used, known boundary conditions) plus prose.
- `scratchpad.md` at the repo root — the user's working notes, including
  the original audit findings.

# The frame

We are working through a multi-tier audit and refactor. The goal is not
just correct code today; it is a **solid foundation for building the
planned multi-head classifier** (CCA head + immigrant involvement head +
combined ICA head, possibly with ALUM, possibly with per-layer LRs and
partial encoder unfreezing). The audit is the runway for that future
work; the refactor is the runway after that.

The audit/refactor work was organized into four tiers, each with a
checkpoint at its end before proceeding (a fifth tier, post-audit
diagnostic instrumentation, was added later):

- **Tier 1 — Substantive correctness.** Math, semantics, test
  infrastructure. Get what's already in the repo actually right.
- **Tier 2 — Structural refactor.** Reshape the code (model setup,
  preprocessor, paths/config, data pipeline) so it supports the multi-
  head future naturally rather than fighting it.
- **Tier 3 — Robustness.** Expand test coverage, add missing invariants,
  harden error-handling boundaries.
- **Tier 4 — Hygiene.** Scratch files, dead code, scratch imports,
  naming. Low-severity but cumulative; worth the sweep.
- **Tier 5 — Diagnostic instrumentation + empirical stress test.**
  Post-audit, not part of the original four-tier plan: permanent
  observability for the deferred training runs, plus the
  human-operated runbooks for those runs. Added once the foundation
  was solid.

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

  - **Piece 1 done** (commit `79ab31c`). Dual-boundary input validation on `ClassifierPreprocessor`. `__init__` checks for *internal-config-validity* bugs (raises `ValueError` with informative messages naming the bad value): `text_key` non-empty string, `label_keys` is a dict, `target_dtype` is a Keras-recognized dtype string, and standard-mode (`endpoint_model=False`) requires non-empty `label_keys` (the empty-`label_keys` configuration is only valid in endpoint mode for predict-only flow). `__call__` checks for *config-vs-data-mismatch* bugs (raises `KeyError` with the missing column set, the configured expectation, and the batch's actual keys, enumerating *all* missing columns rather than failing fast). Retires M2 from the Tier 4 deferred list — `target_dtype` validation lives naturally in the construction-validation block. 14 new tests across two new test classes (`TestConstructionValidation`, `TestCallTimeInputValidation`); suite at 87 → 101 passing.
  - **Piece 2 done** (commit `4243c63`, plus follow-on `8e58350` filling in the hash and a follow-on documenting the format-choice decision). Pattern 2 serialization invariant pinned in `tests/test_assembly.py` via `TestPatternTwoSerialization` (2 tests): `test_round_trip_predictions_match_bitwise` (Pattern A → save → fresh-build → load → bitwise-identical predictions, with a pre-load-difference assertion to break symmetry) and `test_load_weights_raises_on_shape_mismatch` (different `hidden_dim` → `skip_mismatch=False` raises `ValueError`). Production-path `load_weights` calls in `src/eval_cca_classifier.py`, `src/model_setup/backbone.py`, and `scripts/smoke_test_integrated_stack.py` gain explicit `skip_mismatch=False` to pin the load-strict discipline. **Empirical finding during implementation** (see `tier3-design.md` Piece 2 "Empirical finding" subsection): Keras 3's `.weights.h5` save format keys variables by *layer-class + positional index*, NOT by user-given name. Switching to legacy `.h5` format would enable `by_name=True` strict matching but was deliberately rejected (deprecation risk + redundant with call-site routing protection); decision criteria for revisiting captured in `tier3-design.md` "Decision: stay with `.weights.h5`" subsection. The originally-planned name-mismatch test was reframed as shape-mismatch. Implication: Pattern 2 is load-by-*structure*, not load-by-*name*; the head-name contract is enforced at call sites by Keras's `compile(loss={head: ...})` routing (a separate concern from weight loading). Suite: 101 → 103 passing.
  - **Piece 3a done** (this commit). `src/cca_config.py` introduces frozen-dataclass `RunConfig` capturing the architectural and research-dimension parameters of a CCA training run, with JSON sidecar serialization, `validate_against_backbone(backbone)` for hidden_dim cross-validation, `DEFAULT_CCA_CONFIG` as the canonical starting point, and a CLI helper for ad-hoc sidecar creation. Wrapped sub-configs (`FLPULossConfig`, `HeadConfig`, `RatioBatchConfig`, `LRScheduleConfig`, `OptimizerConfig`) — the `loss` wrapping is pre-namespaced for the planned ALUM piece (pinned question #1); the others are organizationally wrapped (lifts fields from flat to wrapped, makes future type-discrimination non-breaking — see `tier3-design.md` Piece 3 "wrapped vs. flat" reasoning). Per-dataclass `__post_init__` validation; cross-object validation (head names unique) at RunConfig; external-context validation (`validate_against_backbone`) as an explicit method. JSON forward-compat: ignores unknown fields with warning, fails loud on missing required fields. 64 new tests in `tests/test_cca_config.py`. Suite: 103 → 167 passing.
  - **Piece 3b done** (commit `c034597`, plus follow-on `ba1fc3d` filling in the hash). `src/run_cca_classification.py` builds from `cca_config.DEFAULT_CCA_CONFIG` (preprocessor, head, assembly, LR schedule, optimizer, ratio batch all driven from run_config), validates against the loaded backbone, writes the sidecar via `RunConfig.to_json(config_path_for_weights(...))` after fit. `src/eval_cca_classifier.py` loads the sidecar at startup via `RunConfig.from_json(...)`, validates against the loaded backbone, constructs its inference model from the same values. `scripts/smoke_test_integrated_stack.py` exercises the full RunConfig → fit → save (weights + sidecar) → load (sidecar + weights) → predict round-trip on synthetic data; the fake backbone now exposes a `hidden_dim` attribute matching the real keras_hub Backbone's contract. Smoke test re-run: still passes (Pattern A vs Pattern 2 max-diff 0.00e+00). Suite still at 167 (no new tests; the script integration is exercised by the smoke test).
  - **Piece 4 done** (commit `96ea283`). Original-scope test coverage. `tests/test_data_splits.py` extended with `TestLabelConstruction` (11 tests covering all four boolean combinations of `(cca, cca_descriptor) → cca_label` and `(immig, immig_descriptor) → immig_label`, plus integer-dtype assertions and a per-row independence test). `tests/test_data_loading.py` is new — 11 tests across `TestParquetMissingValueHandling` (null and `"NA"` substitution in `headline`/`lead_paragraph`) and `TestHeadlineWithLeadConcatenation` (the `headline + "</s>" + lead` build across all empty/normal combinations). Tests use pytest's `tmp_path` fixture to write small parquet files into temporary directories and call `data_from_parquet(tmp_path, ...)` — no production data needed. Three style observations flagged on the data-loading code (Python list comprehension for `headline_with_lead` rather than vectorized polars expression; list comprehension used for side-effect on `cols_to_select`; implicit ordering between `fill_null` and `"NA"` substitution); not bugs, deferred to Tier 4 hygiene if useful. Implemented via subagent delegation; output verified before commit. Suite: 167 → 189 passing.
  - **Closeout done** (this commit). Adversarial review of cumulative Tier 3 (commits `76c353f..96ea283`) returned 1 Critical + 8 Important + 7 Minor. Fixed in this commit: **C1** (round-trip test reframed to explicitly pin no-op-load-protection — same-seeded backbones + `freeze_encoder=True`, so post-load match is achievable only via actual head-weight load); **I1** (`expected_columns` wired into train/eval scripts + smoke test, making the 3-layer validation hierarchy real); **I3** (eval-side metrics dropped — empirically verified metric state vars aren't in `head.weights`); **I5** (finite test dataset for `evaluate()`, removing the `test_steps = validation_steps` approximation); **I6** (multi-head metric distinctness test added); **I7** (Layer-2 string-dtype check on `inputs[text_key]` in preprocessor `__call__`); **I8 interim** (metrics factored into `src/cca_metrics.py::make_cca_metrics()`; both production scripts use it; eval doesn't construct metrics at all). Deferred with explicit notes: **I2** (smoke-test exercises instance-attribute path, not class-property path — documented in smoke test docstring); **I4** (LR schedule resolution gap — sidecar carries factors, not resolved step counts; documented in design doc with revisit criteria); **I8 full** (metrics in RunConfig with JSON serialization — pinned for proper engagement when metric configurations become a research dimension). Tier 4 hygiene scope: M1–M7 (cosmetic / dead-comment / minor enforcement gaps). Suite: 189 → 192 passing. Smoke test: still passes (Pattern A vs Pattern 2 max-diff 0.00e+00). See `docs/notes/tier3-design.md` "Tier 3 closeout" subsection for the full triage and review-shape lessons.

**Tier 3 complete.** All four implementation pieces + closeout done.

**Tier 4 complete.** Three pieces (hygiene + I4 LR resolution + lessons docs), each landing as main + closeout commits, plus a final cross-phase review and a post-Tier-4 lessons-docs revision pass. See `docs/notes/tier4-design.md` for design reasoning.

  - **Piece 1 done** (commits `a3505ae` + `44b1faf`). Hygiene fixes from the Tier 2/3 inherited deferred list: M1 (`src/test_script.py` second-half deletion, lines 188-209 referencing the retired `classifier_from_dapt_checkpoint`), M3 (`HeadConfig.__post_init__` rejects head names containing `/`, which is Keras's variable-path separator used by `_default_group_fn` for discriminative-LR grouping), M4a (`ClassificationHead.__init__` requires explicit `name` — keyword-only, no default; `None` explicitly rejected with a ValueError), M4b (`build_endpoint_model` asserts unique head names — same invariant at construction-site as M4a, applying the boundary-inventory pattern at two boundaries). Closeout commit tightened a too-loose regex in the M4b test. Suite: 192 → 196 passing.
  - **Piece 2 done** (commits `017dffe` + `9d92c17`). I4 LR schedule resolution via nested `ResolvedSteps`. New frozen dataclass `ResolvedSteps(warmup_steps, decay_steps, steps_per_epoch)` in `src/cca_config.py`; `LRScheduleConfig` extended with `resolved: ResolvedSteps | None = None` field, `with_resolved(steps_per_epoch)` method using `math.floor(factor * steps_per_epoch)`, and `_from_dict` reconstruction with backward compat for older sidecars missing the key. `src/run_cca_classification.py` rewired: calls `with_resolved` after computing `steps_per_epoch`; `keras.optimizers.schedules.CosineDecay` reads `resolved.warmup_steps` and `resolved.decay_steps` rather than multiplying factors inline. Sidecar is now self-sufficient for LR schedule reconstruction (closes I4 from the Tier 3 closeout deferred list). Numerical note: explicit `math.floor` is a deliberate change from the prior float-passing behavior (effect e.g. 571.75 → 571; well below training noise floor; project doesn't rely on byte-exact reproduction). Closeout commit documented the deliberate validation duplication between `with_resolved` (boundary-condition error-message clarity) and `ResolvedSteps.__post_init__`; also added a type-narrowing assert in one test. Suite: 196 → 220 passing (24 new tests, several from `@pytest.mark.parametrize` expansion).
  - **Piece 3 done** (commits `ffa6efc` + `8476848`). Lessons docs. `docs/notes/process-patterns.md` (4 Validated + 1 Developing content-agnostic process patterns: Pedagogical pattern, Adversarial review after implementation, Design-doc-per-tier as living working doc + Post-review corrections section, Deferred-with-explicit-notes discipline; Skill-orchestrated design workflow as Developing). `docs/notes/engineering-patterns.md` (3 Validated + 2 Developing CS-specific patterns: Boundary-inventory pattern, Synthetic stand-ins for heavyweight dependencies, Wrapped-vs-flat forward-compat for config sub-objects; Pattern A vs Pattern 2 model sharing, Empirical investigation before committing to design as Developing). I2 dual-captured as the canonical boundary-condition example in the Synthetic-stand-ins entry. Closeout commit fixed 6 factual-accuracy issues against git history (dates, commit hashes, the Tier 2 review finding count) plus added a missing rationale sentence to engineering-patterns' Promotion rule. Suite unchanged (docs only); 220 passing.
  - **Cross-phase review done** (commit `76d569a`). Final cross-phase review across the whole Tier 4 closeout surfaced one Minor: `src/test_script.py`'s post-Piece-1 docstring misattributed the file's content as using the Tier 2 abstractions when in fact lines 1-185 use pre-Tier-2 inline shape (raw `keras_hub.models.Backbone.from_preset`, ad-hoc `EndpointLayer`, plain `keras.Model`). Closeout commit rewrote the docstring to accurately describe the file's actual shape and direct readers to the production scripts for examples of the Tier 2 abstractions in use.
  - **CLAUDE.md update done** (commit `f536153`, via project-claude-librarian subagent). 8 targeted edits: freshness date, test count, per-file test counts, M4a/M4b contracts, new `ResolvedSteps` API, Piece-by-piece status, residual deferrals. Replaced an earlier scope-creep CLAUDE.md attempt (`0188f39`) that was reverted (`4143c3a`) because of factual errors — Question 4 in `pinned-questions.md` captures the workflow observation that came out of that.
  - **Post-Tier-4 lessons-docs revision pass done** (commits `2b0feec` + `9b50c8a` + `e3f601f`). A read-through identified substance gaps in the original rapid-implementation drafts of the Piece 3 lessons docs. The Pedagogical-pattern entry was structurally correct but analytically thin — missed the actual *pedagogical* point (the user learns by being made to engage with substance dialogically, not just by having reasoning preserved for retrospective reading). Revision rewrote the entry, added a sub-pattern (human implementation at key code points with scaffolding), and added two new Validated entries to `process-patterns.md` (Investigator-subagent pattern; Implementation plans with file:line specificity). Four metadata-restating closing paragraphs across both lessons docs were trimmed. A new Question 4 in `pinned-questions.md` captured the n=1 "scope-creep correction via revert" observation. A new boundary condition was added to the Skill-orchestrated entry (substantive-prose work breaks the skill chain's TDD-equivalent forcing function). CLAUDE.md refreshed via the librarian. Suite still 220.

**Tier 5 — Empirical Stress Test + Diagnostic Instrumentation.
Subagent-executable portion complete** (range `10f6e01..b893644`,
46 commits, suite 220 → 374). Permanent diagnostic instrumentation
across 8 planned phases. Phases 1–6 + Phase 7 Tasks 1–2 + Phase 8
Task 1 are committed, reviewed, and tested. Design reasoning (incl.
the periodic-subsystem supersession note) in
`docs/notes/tier5-design.md`; per-phase landing summary in the
dedicated "Tier 5" section below.

  - **Done (committed):** new `src/diagnostics/` module —
    `trackers.py` (four per-step `keras.metrics.Metric` subclasses:
    `PerGroupGradNormTracker`, `GradientFiniteTracker`,
    `LossComponentTracker`, `BatchLabelBalanceTracker`), `factory.py`
    (`build_trackers` → `DiagnosticBundle`), `distribution_metrics.py`
    (per-head prediction-distribution metrics riding the head's
    `metrics=` path). New `DiagnosticsConfig` sub-config on `RunConfig`
    (10th field, back-compat default_factory). New contract surfaces:
    `FLPULoss.call(return_intermediates=)`,
    `ClassificationHead(expose_loss_components=)`,
    `LayerLRModel(diagnostic_trackers=, diagnostic_head_refs=)` +
    `metrics` override, `build_endpoint_model(diagnostics=)`.
    `run_cca_classification.py` refactored to importable
    `main(run_config=None, max_steps=None)` + `__main__` guard, plus a
    `CSVLogger`. `scripts/tier5_short_run.py` (reproducible short run)
    and `scripts/tier5_cluster.sbatch` (parameterized SLURM template).
  - **NOT done — HUMAN-OPERATED, handed off:** Phase 7 Tasks 3–5
    (level-1 short + level-2 full **real-data** runs on local
    `cca_set/`, then create `docs/notes/tier5-stress-test-notes.md`)
    and Phase 8 Tasks 2–4 (short + full **cluster** `mixed_float16`
    runs on Explorer, then the level-3 π=0.03-vs-0.02 research
    handoff). The empirical runs have **not** been performed;
    `tier5-stress-test-notes.md` does not yet exist. Runbooks:
    `docs/notes/tier5-implementation-plan/phase_07.md`, `phase_08.md`.

**Whole-project final review:** Line 43 of this doc anticipated "a final adversarial pass over the whole" after Tier 4. What actually happened: each tier received its own cumulative adversarial review (Tier 1's at `8685c47`, Tier 2's at `079deff`, Tier 3's at `987a8c0`, Tier 4's at `76d569a`). A single whole-project review covering Tier 1+2+3+4 cumulatively was not run. Whether the per-tier reviews collectively suffice, or whether a single end-state pass is still worth doing, is a decision point for a future session.

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
2. **[DONE — Piece 2 (this commit)] I5 — Pattern-2 serialization
   round-trip test.** Round-trip Pattern A (in-process) → save →
   fresh-build → load → Pattern 2 produces bitwise-identical
   predictions; shape-mismatch raises `ValueError`. Tightens
   `eval_cca_classifier.py`, `backbone.py`, and the smoke test to
   call `load_weights(..., skip_mismatch=False)`. Empirical
   finding (captured in tier3-design.md): Keras `.weights.h5`
   matches by structure not name — head renames don't break load,
   shape changes do.
3. **[DONE — Pieces 3a + 3b (this commit)] I4 — train/eval
   config coupling.** Option C (static config module + serialized
   run config sidecar). Piece 3a (commit `d9c0348`):
   `src/cca_config.py` with dataclasses, JSON I/O, validation,
   CLI helper. Piece 3b (this commit): training/eval scripts and
   smoke test wired through. The I3/I4/I5 boundary-enforcement
   spine is complete; only Piece 4 (original-scope test coverage)
   remains in Tier 3.
4. **[DONE — Piece 4 (this commit)] Original-scope test coverage.**
   Label-construction tests for the cca/immig/descriptor boolean
   combinations in `create_classifier_data` (`TestLabelConstruction`
   in `tests/test_data_splits.py`, 11 tests). Missing-value handling
   tests for `data_from_parquet` (`tests/test_data_loading.py`,
   11 tests across two classes). 22 new tests; suite 167 → 189.

**Deferred (not Tier 3): evaluation harness with calibration.**
Originally listed in the Tier 3 plan, but it's a research deliverable
(threshold selection on a hand-labeled PN test set, possibly Platt
scaling or isotonic regression), not foundation work. Punt to a
separate piece of work after Tier 4 hygiene.

The S6 looping issue (`steps=validation_steps` on test predict
producing duplicate predictions) was fixed during Tier 2 Piece 4c
and doesn't need Tier 3 treatment.

# Tier 4: Closeout (hygiene + I4 + lessons docs)

**Done.** Scope expanded beyond the original "Hygiene" framing
(see `docs/notes/tier4-design.md` for the design reasoning) to
three pieces: selected hygiene fixes from the Tier 2/3 inherited
deferred list, the I4 LR-resolution carry-over from Tier 3, and
new lessons-docs work. Per-piece breakdown is in the "Where we
are" section above; for the design reasoning see
`docs/notes/tier4-design.md`.

**Items from the original Tier 4 placeholder scope still pending**
(low-priority cumulative sweep work; pick up when the cost-benefit
warrants):

- Move scratch files (`test_module.py`, `test_script.py`,
  `endpoint_layer_test.py`) to `scratch/` or `scripts/scratch/`,
  with the useful "save DAPT backbone weights" logic from
  `test_module.py` extracted into `model_setup/` as a real
  function.
- Remove commented-out dead code from `preprocessor.py`,
  `data.py` (formerly `dapt_data.py`), and the exploration-log
  block in `dapt.py`.
- Archive or delete `ramaswamy2016.py`.
- Minor style / import cleanups.

**Note on pre-Tier-2 sandbox state in `test_script.py`:** Tier 4
Piece 1 deleted the file's dead second half (lines 188-209,
referencing the retired `classifier_from_dapt_checkpoint`), but
the kept first half (lines 1-185) still uses pre-Tier-2 inline
shape (raw `keras_hub.models.Backbone.from_preset`, ad-hoc
`EndpointLayer`, plain `keras.Model`). The docstring was updated
in the cross-phase-review closeout (`76d569a`) to accurately
describe this state. Reshape into the Tier 2 abstractions is a
candidate follow-up if the file's pedagogical / sandbox value
justifies the work; not a current priority since the production
stack is covered by `run_cca_classification.py`,
`eval_cca_classifier.py`, and the integrated smoke test.

# Tier 5: Empirical Stress Test + Diagnostic Instrumentation

**Subagent-executable portion done.** Intent: make the deferred
empirical training runs *observable* before running them — permanent
diagnostic instrumentation (gradient norms, overflow rate, FLPU loss
components, batch label balance, prediction-distribution collapse
signals) plus the operator runbooks for the runs themselves. Range
`10f6e01..b893644` (46 commits); suite 220 → 374. Design reasoning and
the full implementation plan: `docs/notes/tier5-design.md` and
`docs/notes/tier5-implementation-plan/phase_01.md … phase_08.md`.

**Phases, with status markers:**

1. **[DONE] Tracker module foundation.** `src/diagnostics/trackers.py`
   — `PerGroupGradNormTracker`, `GradientFiniteTracker`,
   `LossComponentTracker`, `BatchLabelBalanceTracker`. `hypothesis`
   added as a dev dependency for property-based tracker tests.
2. **[DONE] Config and factory.** Frozen `DiagnosticsConfig` sub-config
   on `RunConfig` (back-compat default_factory: missing/`null` → all
   enabled); `src/diagnostics/factory.py` `build_trackers →
   DiagnosticBundle`. Aggregation-constant duplication between config
   and trackers is pinned equal by `TestDiagnosticsAggregationConstantSync`.
3. **[DONE] Loss-component harvest path.**
   `FLPULoss.call(return_intermediates=True)` →
   `(loss, {positive_risk, negative_risk, correction_triggered})`; the
   loss scalar is bit-identical between the two flag paths.
   `ClassificationHead(expose_loss_components=True)` stashes the
   components dict on `last_components` for the train-step dispatch.
4. **[DONE] Train-step integration.** `LayerLRModel` gains
   `diagnostic_trackers` + `diagnostic_head_refs`, a `metrics` property
   override, and `_dispatch_diagnostics` observing **pre-scaling** grads
   + loss components + batch targets each step. Strict no-op
   (byte-identical to pre-Tier-5) when `diagnostic_trackers is None`.
5. **[DONE — original design superseded] Prediction-distribution
   metrics.** `src/diagnostics/distribution_metrics.py`
   (`PredictionMeanMetric`, `PredictionStdMetric` [float64 accumulation],
   `PredictionFracAboveMetric`) ride the head's `metrics=` path, NOT the
   originally-designed periodic-callback/reference-batch path. The
   `DiagnosticBundle["periodic"]` slot is a permanently-empty
   forward-compat stub with no current consumer. Supersession note in
   `tier5-design.md` and the `distribution_metrics.py` header.
6. **[DONE] Assembly wiring + smoke test.**
   `build_endpoint_model(diagnostics=)` gathers trainable variables
   (placed **after** the `freeze_encoder` block, so a frozen encoder
   yields no spurious backbone grad tracker) and builds the bundle.
   `run_cca_classification.py` wires distribution metrics +
   `expose_loss_components` + `diagnostics` + a `CSVLogger`.
7. **[Tasks 1–2 DONE; Tasks 3–5 HUMAN-OPERATED] Local stress test.**
   Done: `run_cca_classification.py` → importable `main()`;
   `scripts/tier5_short_run.py` (1-epoch / 200-step capped run). NOT
   done: the level-1 short + level-2 full **real-data** runs on
   `cca_set/`, and the resulting `docs/notes/tier5-stress-test-notes.md`.
   Runbook: `docs/notes/tier5-implementation-plan/phase_07.md`.
8. **[Task 1 DONE; Tasks 2–4 HUMAN-OPERATED] Cluster stress test.**
   Done: `scripts/tier5_cluster.sbatch` parameterized SLURM template
   (`short`/`full` mode arg, `<PLACEHOLDER_*>` operator fields). NOT
   done: the short + full cluster `mixed_float16` runs on Explorer and
   the level-3 π=0.03-vs-0.02 research handoff. Runbook:
   `docs/notes/tier5-implementation-plan/phase_08.md`.

**Cross-phase boundary inventory (final-review Minors, commit
`b893644`).** Post-range closeout: an `FLPULoss`-component-keys sync
test pinning the loss-intermediate key set against its consumers, and a
`build_endpoint_model` `key == head.name` guard. These are the Tier 5
instances of the boundary-inventory pattern (validate the same
invariant at both the producing and consuming site).

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
5. `cat docs/notes/process-patterns.md` and
   `docs/notes/engineering-patterns.md` — catalogs of process and
   engineering patterns validated through Tier 2/3/4 work; useful for
   "do we have an approach for X?" lookups and pattern-suggestion when
   designing.
6. `cat scratchpad.md` — the original audit findings and user working
   notes.
7. `uv run pytest tests/ -q` — confirm the test suite still passes.

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
