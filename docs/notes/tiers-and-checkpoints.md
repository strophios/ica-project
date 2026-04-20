# Audit and Refactor: Tiers, Checkpoints, and Process

*Last updated: 2026-04-20 (end of Tier 1).*

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

**Tier 1 is complete.** Five commits on `main`:

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

Test suite: 32 tests passing, covering FLPU invariants (construction,
output structure, easy/adversarial, order invariance, non-negativity,
prior sensitivity, numpy-reference numerical correctness, clawback
branches, edge-case batches, mixed_float16 handling, and the
reduction='none' shape contract) plus data-split invariants
(determinism, mutual exclusion, coverage, ratios, id uniqueness).

**Deferred empirical checks** (to be batched into one milestone when
the environment story is settled, probably as part of Tier 2):

- Smoke training run with the updated FLPU + corrected prior (≈ 0.02).
- Confirm training dynamics are reasonable under mixed_float16 + the
  removed-α FLPU.
- Sensitivity sweep on Ratio Batch (currently 1:10, much more aggressive
  than Ji 2023's prescription).

# Tier 2: Structural refactor (next)

**Intent:** reshape the code so that the multi-head future is a natural
extension rather than a rewrite. Tier 2 is *about shape*, not about math
or correctness — those were Tier 1. If Tier 2 touches the math, that is
a signal we designed the shape wrong.

**Current scope** (subject to revision in the design pass):

1. **Model setup separation.** Split `classifier_from_dapt_checkpoint`
   into (a) "load backbone" and (b) "attach head(s)." Heads become
   composable. The endpoint-layer pattern becomes the default path for
   heads needing custom losses (FLPU, eventually ALUM).
2. **Per-layer learning rates / selective encoder unfreezing**, as a
   first-class training capability rather than an afterthought. Likely
   needs a custom training loop or multi-optimizer setup.
3. **Preprocessor refactor.** `ClassifierPreprocessor` takes a list or
   dict of label keys rather than a single `label_key`. Enables multi-
   label / multi-head batching natively.
4. **Paths and config.** A single `paths.py` (or similar) with platform
   detection replaces the scattered `path_prefix` blocks in every
   script. Environment handling clarified for local-vs-cluster.
5. **Data pipeline rename and reorganization.** `data_setup/dapt_data.py`
   is already flagged in-code as needing to rename since it handles both
   DAPT and classifier data.
6. **Tests: follow the refactor.** The tests-as-spec property from
   Tier 1 means existing tests should mostly still pass. Where they
   fail, they are either flagging a regression or flagging a test that
   was testing implementation rather than behavior. Adjust accordingly
   (and add new tests for new invariants the refactor introduces).

**Process commitments for Tier 2**:

- **Design pass first.** A concrete design doc (likely
  `docs/notes/tier2-design.md`) before any code, explaining the intended
  structure, the patterns used, and the trade-offs chosen. The doc's job
  is to leave the user (and any future reader) with a mental model of
  what changed and why — not to be a spec for Claude to execute in
  isolation.
- **Staged implementation.** Not one giant refactor commit. Each stage
  preceded by a short "here's what we're about to do and the pattern
  you'll see" explainer. Pause points for check-in.
- **Named patterns.** New patterns (multi-optimizer training, endpoint
  layers with `add_loss`, etc.) get explicit names and brief walkthroughs
  before use, so the user accumulates mental vocabulary.
- **Explanatory comments in code.** Denser than production norm,
  especially on Keras/TF idioms the user has not implemented before.
  Trim later once the pattern is internalized.
- **Adversarial review at the end.** Likely the `code-reviewer` subagent
  this time, because Tier 2 is architectural and plan-oriented (unlike
  Tier 1, which was substantive-reasoning-oriented and a better fit for
  a general-purpose opus adversary).

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
