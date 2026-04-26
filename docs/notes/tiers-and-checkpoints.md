# Audit and Refactor: Tiers, Checkpoints, and Process

*Last updated: 2026-04-26 (Piece 3 landed; Pieces 1, 2, 3 done).*

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

**Tier 2 in progress.** Three of four pieces landed.

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

The new abstractions exist **alongside** the existing
`classifier_from_dapt_checkpoint` function; they are not yet wired
into any training script. The wiring is Piece 4's job.

**Tier 2 pieces remaining:**

- **Piece 4: Paths / config + data pipeline rename, and integration.**
  Consolidate scattered `path_prefix` blocks into a `paths.py` with
  platform detection; rename `data_setup/dapt_data.py` (already flagged
  in-code as mis-scoped); wire the refactored preprocessor + head +
  `LayerLRModel` into the training scripts; retire
  `classifier_from_dapt_checkpoint`. With Pieces 1–3 in place, this
  is now pure integration work — no new abstractions required.
- **Integration pass.** End-to-end smoke test of the composed stack
  on dummy data. Follows Piece 4.
- **Adversarial review at Tier 2 end.** Likely the `code-reviewer`
  subagent this time (plan-alignment / architecture framing fits
  Tier 2 better than the opus general-purpose one used for Tier 1).

Test suite after Piece 3: **65 tests passing** (32 from Tier 1 + 11
for `ClassificationHead` + 10 for `LayerLRModel` + 12 for
`ClassifierPreprocessor`).

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
4. **[PENDING] Paths / config + data pipeline rename, and
   integration.** Piece 4. Consolidate scattered `path_prefix`
   blocks into a `paths.py` with platform detection; rename
   `data_setup/dapt_data.py`; wire the refactored preprocessor +
   head + `LayerLRModel` into the training scripts; retire
   `classifier_from_dapt_checkpoint`.
5. **[PENDING] Integration pass.** End-to-end smoke test of the
   composed stack on dummy data.
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

Planned, not started. Scope includes:

- Tests for the data pipeline shape contract (preprocessor inputs/outputs
  under both standard and endpoint-layer modes).
- Tests for label construction (validates the cca/immig/descriptor
  boolean combinations).
- Missing-value handling: test coverage for the `fill_null("")` path in
  `data_from_parquet`.
- Evaluation harness: proper metrics aggregation on test set;
  calibration utilities.
- Fix the S6 looping issue (`steps=validation_steps` used on test
  predict calls causing duplicate predictions) with a finite,
  correctly-sized test dataset.

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
