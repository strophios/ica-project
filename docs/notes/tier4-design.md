# Tier 4 Design

*Living document — appended piece-by-piece as each design discussion lands.*
*Started: 2026-05-11.*

This captures the design decisions made during Tier 4 of the audit/refactor.
Like the Tier 2 and Tier 3 design docs, it is not a spec drafted up-front;
each section is the outcome of a dialogic design discussion with the user.
Record the decision plus enough reasoning that someone reading it later
knows *why* we landed where we did.

See `docs/notes/tiers-and-checkpoints.md` for the overall frame,
`docs/notes/tier3-design.md` for the prior tier (with the "Tier 3
closeout" section pointing at the Tier 3 review findings inherited
into Tier 4), and `docs/notes/pinned-questions.md` for deferred
substantive questions.

---

## Overall framing

**Intent:** close out the foundation-laying work that Tiers 2 and 3
opened, so the project is ready for empirical stress-testing of the
training architecture (mechanical + numerical correctness levels)
without lingering hygiene debt blocking interpretation of results.
Tier 4 is the "finish the foundation" tier — small fixes, formalize
known gaps, and capture the process and engineering patterns the
preceding tiers validated.

**The work splits into three flavors:**

1. **Hygiene fixes.** Tier 2/3 review findings small enough to land
   without architectural deliberation — `test_script.py` second-half
   deletion (absorbs M1), M3 (`/`-separator validation in head
   names), M4 (dual head-name-collision validation). Mechanical;
   correctness-preserving; the kind of work that should not block
   anything else.
2. **Architectural fix.** I4 (LR schedule resolution gap) is the
   most architectural item in scope — extending `LRScheduleConfig`
   to carry resolved step counts alongside the factor inputs so the
   sidecar becomes self-sufficient. Small in code size but requires
   a deliberate decision about how the resolved-values shape
   interacts with the existing wrapped-vs-flat sub-config pattern.
3. **Documentation.** Two new lessons-learned docs
   (`docs/notes/process-patterns.md` and
   `docs/notes/engineering-patterns.md`) capturing the patterns
   validated through the Tier 2/3 work, structured with explicit
   validation-status tagging so neither user nor future-Claude
   misreads a once-used pattern as well-established.

The first two are code work; the third is process work. The lessons
docs draw their content largely from the prior tiers' design docs
and from this one once it's complete.

---

## Definition of Done

**Primary deliverables:**

- Five code changes:
  - `test_script.py` second-half deletion (lines 188-209), keeping
    the working first-half as a wiring test for the endpoint-layer
    pattern. Absorbs M1.
  - M3: `HeadConfig.__post_init__` validates that head names do
    not contain `/`, preventing collision with the Keras variable
    path separator used by `_default_group_fn`.
  - M4: dual validation — `ClassificationHead.__init__` rejects
    `name=None`, and `build_endpoint_model` asserts unique head
    names across the supplied dict. Catches the collision at
    construction-site and at call-site (boundary-inventory
    pattern in action).
  - I4: `LRScheduleConfig` extended with resolved-step fields
    (`warmup_steps`, `decay_steps` or similar), populated by the
    training script after `steps_per_epoch` is computed, and
    serialized into the sidecar. Eval can reconstruct the
    schedule from sidecar alone, no script-local dependency.

- Two new lessons docs:
  - `docs/notes/process-patterns.md` — content-agnostic process
    patterns (pedagogical, adversarial-review-after-implementation,
    design-doc-per-tier with Post-review corrections, deferred-
    with-explicit-notes discipline)
  - `docs/notes/engineering-patterns.md` — CS-specific patterns
    (boundary-inventory, wrapped-vs-flat forward-compat, Pattern
    A vs Pattern 2 model sharing)
  - Both with two-section structure (Validated / Developing),
    explicit promotion rule between sections, and per-pattern tag
    block (validation status, first/last used, known boundary
    conditions). I2 (smoke-test backbone-validation gap) captured
    as a deferred gap entry, not fixed.

- This design doc (`docs/notes/tier4-design.md`), completed
  piece-by-piece as Tier 2/3 doc convention.

**Success criteria:**

- All 192 existing tests still pass.
- New validation tests for M3 and M4 pass with the fixes and
  would fail without them (TDD-style RED-GREEN).
- A fresh training run produces a self-sufficient sidecar
  containing resolved step counts; round-trip from sidecar to
  reconstructed schedule no longer requires `steps_per_epoch`
  from the training script.
- Both lessons docs are committed, with each pattern carrying a
  non-trivial validation-status sentence and at least one stated
  boundary condition.

**Explicit exclusions:**

- I8-full metric factoring (interim `src/cca_metrics.py` remains
  sufficient).
- I2 closure via test code (captured as a deferred gap in the
  engineering-patterns doc, not fixed).
- All `pinned-questions.md` content (four-layer FLPU framing,
  multi-class extension, preprocessor train/predict shape
  refactor).
- Any other items in the Tier 3 deferred lists not enumerated
  above.

---

## Pieces (anticipated)

Three pieces, compact structure (vs. a per-item 5-piece split). The
compact choice reflects Tier 4's smaller architectural surface
relative to Tier 2 and Tier 3: the hygiene items are individually
trivial (each is "one validation + one test" or smaller), and
forcing them into separate pieces would impose Tier 2/3 piece
ceremony on changes that don't warrant it.

1. **Piece 1: Hygiene fixes.** `test_script.py` second-half
   deletion (absorbs M1), M3 head-name `/`-validation, M4 dual
   head-name validation. One commit. Low risk; fast wins; tests
   the workflow on simple cases first.
2. **Piece 2: I4 — LR schedule resolution.** Extend
   `LRScheduleConfig` with a nested `ResolvedSteps` sub-object;
   wire `with_resolved` into the training script after
   `steps_per_epoch` is computed; remove the in-script
   resolution lines in favor of reads from the config. One
   commit. Most architectural item in scope.
3. **Piece 3: Lessons docs.** Two new files
   (`docs/notes/process-patterns.md`,
   `docs/notes/engineering-patterns.md`) capturing patterns
   validated through T2/T3/T4 work, with explicit
   validation-status tagging. One or two commits (TBD at
   implementation time based on cross-reference tightness).
   Last because the docs draw from this design doc once
   complete.

The piece ordering puts low-risk wins first (hygiene), then the
architectural piece while design is fresh (I4), then docs after
the design doc is full (lessons). Lessons-first was rejected
because the docs would be written with stale data; I4-first
was rejected because there's no concrete advantage to starting
at the highest-risk point.

---

## Piece 1: Hygiene fixes

### Decision

Single commit bundling three independent changes plus tests:

- **`test_script.py` second-half deletion** (absorbs M1). Delete
  lines 188-209 — the block referencing the retired
  `classifier_from_dapt_checkpoint` function (replaced in
  Tier 2 Piece 4c by `load_dapt_backbone` + `ClassificationHead`
  + `build_endpoint_model`). Includes the blocking `raise
  RuntimeError(...)` at line 200 (the M1 "raise placement"
  item). Lines 1-185 stay as-is. Add a top-of-file docstring
  documenting what the file exercises (endpoint-layer wiring)
  and what it deliberately does not (standard-mode training,
  covered in `tests/test_heads.py`).
- **M3: head-name `/`-validation.**
  `HeadConfig.__post_init__` in `src/cca_config.py` raises
  `ValueError` if `"/"` in `name`, with message naming the
  conflicting separator and pointing at `_default_group_fn`
  (which splits Keras variable paths on `/` to group
  variables by head).
- **M4: dual head-name validation.**
  `ClassificationHead.__init__` in `src/model_setup/heads.py`
  removes the `name=None` default; the parameter becomes
  required, with `ValueError` raised if `name=None` is passed.
  `build_endpoint_model` in `src/model_setup/assembly.py`
  asserts unique head names at function entry.

### Reasoning

**Why bundle into one piece.** Each individual change is too
small to warrant its own piece — M3 is "add one line + write
a test"; M4 is two validations + tests; `test_script.py` is a
deletion. The per-piece review pattern (code review subagent
at piece boundaries) still works under bundling — review just
covers three small things at once, which is still
well-bounded.

**Why M1 absorbs into `test_script.py` cleanup.** The M1
"raise placement" item was always a symptom of the broader
file-cleanup question, not a separate issue. The `raise
RuntimeError(...)` at line 200 exists *because* the
second-half block references the retired API and would
otherwise execute and fail. Once the dead code goes, the
raise goes with it. Treating M1 as separate would inflate
the inventory without adding signal.

**Why M3 validates head names rather than changing the
separator.** Considered (a) changing `_default_group_fn` to
use a different separator. Rejected: `/` is Keras's
variable-path convention; `_default_group_fn` should match
Keras's structure, not invent its own. Considered (b)
documenting the constraint without enforcing. Rejected:
silent breakage (a head named `"cca/v2"` would group as
`"cca"`, breaking discriminative fine-tuning) is the exact
failure mode boundary-inventory is meant to prevent.

**Why M4 removes the `name=None` default rather than keeping
it as a Keras-idiomatic fallback.** Keras's auto-generated
names (`classification_head_1`, etc.) would silently collide
if two heads were built without explicit names — exactly the
M4 latent failure mode. Removing the default forces explicit
naming at construction time. The cost is slight divergence
from Keras Layer base-class idiom; the gain is structural
prevention of a silent multi-head bug.

**Why M4 is dual-validated (construction-site + call-site).**
Boundary-inventory pattern: the same invariant (unique head
names) is enforced at two boundaries. Construction-site
(`ClassificationHead.__init__` requires explicit name)
catches "forgot to name." Call-site (`build_endpoint_model`
asserts uniqueness across the supplied dict/collection)
catches "named two things the same." Each catches what the
other misses. In practice `heads` is a `dict` so duplicate
keys can't occur structurally today, but the call-site
assert is a forward-compat guard against future API drift
(e.g., `heads` becoming a list of `(name, head)` pairs).

### Layout

- `src/test_script.py` — delete lines 188-209; add top-of-file
  docstring; lines 1-185 unchanged.
- `src/cca_config.py` — `HeadConfig.__post_init__` gains a
  `/`-check on `name`.
- `src/model_setup/heads.py` — `ClassificationHead.__init__`
  signature loses the `name=None` default; body gains a
  `name=None` rejection.
- `src/model_setup/assembly.py` — `build_endpoint_model` gains
  a unique-names assertion at function entry.

### Test coverage anticipated

- `tests/test_cca_config.py` — `HeadConfig(name="cca/v2")`
  raises `ValueError`; `HeadConfig(name="cca")` passes.
- `tests/test_heads.py` — `ClassificationHead(name=None)`
  raises `ValueError`; `ClassificationHead()` (no name) fails
  by signature.
- `tests/test_assembly.py` — `build_endpoint_model` with
  duplicate head names raises `ValueError` listing duplicates.
  (Implementation note: with `heads` as `dict`, structurally
  triggering duplicates requires either mocking or accepting
  the assert is a forward-compat guard only. Decide at
  implementation time.)

Net delta: +3-5 tests. All 192 existing tests should still
pass (only adding new validation, not changing existing
behavior).

### Patterns introduced

- **Boundary-inventory** pattern, applied again (T3 Piece 1,
  T3 closeout I1+I7, now T4 P1 M4). The pattern entry in
  `engineering-patterns.md` (Piece 3) will reference this as
  one of the applications.

### Open / deferred

- None anticipated for this piece. Implementation-time
  discoveries may surface review-tier findings; those land
  in the closeout section if Tier 4 gets an adversarial
  review.

---

## Piece 2: I4 — LR schedule resolution

### Decision

Make the sidecar self-sufficient for LR schedule
reconstruction by extending `LRScheduleConfig` with a nested
`ResolvedSteps` sub-object populated at training time.

- New frozen dataclass `ResolvedSteps` in `src/cca_config.py`
  with fields `warmup_steps`, `decay_steps`, `steps_per_epoch`
  (the last as provenance). `__post_init__` validates all
  three are positive ints.
- `LRScheduleConfig` extended with `resolved: ResolvedSteps |
  None = None` and a method `with_resolved(steps_per_epoch:
  int) -> LRScheduleConfig` returning a new instance with
  `resolved` populated.
- Training script (`src/run_cca_classification.py`): after
  `steps_per_epoch` is computed (around line 214),
  `dataclasses.replace` the `RunConfig`'s `lr_schedule` with
  the resolved version. Existing factor-resolution at lines
  286-291 is removed; the code reads from
  `run_config.lr_schedule.resolved` instead.
- Eval script (`src/eval_cca_classifier.py`): no changes —
  eval doesn't reconstruct the schedule, per the Tier 3
  closeout deferred reasoning.
- Smoke test (`scripts/smoke_test_integrated_stack.py`):
  exercise the `with_resolved` path before save; assert the
  loaded `RunConfig` has `resolved` populated and matches the
  saved values.

### Reasoning

**Why this is the I4 fix at all (revisiting Tier 3 closeout
deferral).** The Tier 3 closeout note marked I4 as "not a
current bug — eval doesn't reconstruct the schedule." That
remains true. But the gap (sidecar can't reproduce the
schedule independent of train-time `steps_per_epoch`) is real
and violates the principle that the sidecar should be the
single source of truth for run state. Closing it now is
cheap insurance against a future HP-search workflow where
`BATCH_SIZE` varies and silent schedule drift would be hard
to detect. Three candidate fixes were laid out in the Tier 3
note; this piece picks the simplest (record resolved counts
alongside factors).

**Why nested `ResolvedSteps` (alternative B) rather than flat
optional fields (alternative A).** A would add
`warmup_steps`, `decay_steps`, `steps_per_epoch` as flat
optional fields on `LRScheduleConfig` alongside the factors.
Simpler JSON. Cost: mixes input-spec (factors) with
computed-record (resolved counts) at the same level; reader
has to know which is which. B costs an extra layer of JSON
nesting but: (1) structurally separates input-spec from
computed-record; (2) "is this sidecar self-sufficient?"
becomes a single boolean check (`lr_schedule.resolved is not
None`); (3) matches the wrapped-vs-flat forward-compat
pattern Tier 3 Piece 3a already established for
`FLPULossConfig`, `OptimizerConfig`, etc. B chosen because
the structural distinction is real and the wrapped pattern
is project-validated.

**Why `with_resolved` returns a new instance rather than
mutating in place.** `LRScheduleConfig` is a frozen
dataclass per project convention (matches `RunConfig` and
all sibling sub-configs); the immutability is load-bearing
for safe JSON serialization and easier reasoning about
config state. `dataclasses.replace` is the idiomatic Python
way to "update" a frozen dataclass.

**Why remove the existing in-script resolution lines rather
than keep them for redundancy.** Keeping both ("factors
resolved in-script for use" + "factors resolved via
`with_resolved` for sidecar") would mean two computation
paths that could drift. Single source of truth: the config
holds the resolved values; the training script reads from
the config. Drift becomes structurally impossible.

**Backward compat.** Older sidecars (no `resolved` key) parse
via the `None` default. No migration needed. Eval doesn't
currently use the resolved values, so missing-resolved
doesn't cause runtime errors today. A future
`LRScheduleConfig.assert_resolved()` method can loudly fail
if eval (or HP search) ever needs to require resolution.

### Layout

- `src/cca_config.py` — new `@dataclass(frozen=True)
  ResolvedSteps` with `__post_init__`; `LRScheduleConfig`
  gains `resolved` field and `with_resolved` method.
- `src/run_cca_classification.py` — after `steps_per_epoch`
  computation (~line 214), `dataclasses.replace` the
  `run_config`'s `lr_schedule`. Lines 286-291 (existing
  factor resolution) removed; replaced by reads from
  `run_config.lr_schedule.resolved`.
- `scripts/smoke_test_integrated_stack.py` — smoke-test
  exercise of `with_resolved` + post-load assertion.

### Contracts

```python
@dataclass(frozen=True)
class ResolvedSteps:
    warmup_steps: int
    decay_steps: int
    steps_per_epoch: int  # provenance

    def __post_init__(self) -> None: ...  # positive-int validation


@dataclass(frozen=True)
class LRScheduleConfig:
    warmup_steps_factor: float
    decay_steps_factor: float
    resolved: ResolvedSteps | None = None

    def with_resolved(self, steps_per_epoch: int) -> "LRScheduleConfig":
        """Return new instance with resolved counts.

        Computation must match existing logic at
        run_cca_classification.py:286-291 exactly (floor vs round
        vs int-truncation verified at implementation time)."""
```

Sidecar JSON shape:

```json
"lr_schedule": {
  "warmup_steps_factor": 0.25,
  "decay_steps_factor": 3.0,
  "resolved": {
    "warmup_steps": 1250,
    "decay_steps": 15000,
    "steps_per_epoch": 5000
  }
}
```

### Test coverage anticipated

`tests/test_cca_config.py` additions:

- `ResolvedSteps` construction with valid positive ints.
- `ResolvedSteps.__post_init__` rejects zero, negative, non-int.
- `LRScheduleConfig.with_resolved(N)` correctness — matches
  the in-script computation for known factor and N values.
- `with_resolved` returns a new instance (frozen-dataclass
  immutability check).
- JSON round-trip with `resolved` populated.
- JSON round-trip with `resolved=None` (backward compat with
  pre-Piece-2 sidecars).
- RunConfig-level propagation:
  `dataclasses.replace(run_config, lr_schedule=updated)`
  produces a valid RunConfig that JSON-serializes correctly.

Net delta: +6-8 tests.

### Patterns introduced

- **Wrapped-vs-flat forward-compat for config sub-objects**,
  applied again (T3P3a + now T4P2). This is the second
  independent application, which graduates the pattern from
  Developing to Validated in the Piece 3 engineering-patterns
  doc.

### Open / deferred

- **Resolved-config requirement enforcement.** The current
  fix doesn't *require* that `lr_schedule.resolved` is
  populated before save — it's optional. A future
  `LRScheduleConfig.assert_resolved()` or a stricter
  `RunConfig`-level check would tighten this, but is
  premature: nothing currently breaks if a sidecar lacks
  resolved values, and adding the assert would couple the
  config validation to a workflow we don't yet have (HP
  search).
- **Other LR schedule parameters that depend on
  train-time state.** The current factors are the only
  schedule parameters that get resolved against
  `steps_per_epoch`. If future schedule types (e.g.,
  cyclical) add more train-time-dependent parameters,
  `ResolvedSteps` will need to grow — handle when it
  becomes pressing.

---

## Piece 3: Lessons docs

### Decision

Two new working documents capturing patterns validated
through Tier 2 / Tier 3 / Tier 4 work, with explicit
per-pattern validation-status tagging:

- `docs/notes/process-patterns.md` — content-agnostic process
  patterns (workflow shapes, review disciplines, doc
  structures).
- `docs/notes/engineering-patterns.md` — CS-specific patterns
  (validation shapes, config patterns, model-sharing
  approaches).

Each follows a shared shape (intro + project-local
validation scope note + promotion rule + index + Validated
section + Developing section), with per-pattern entries
using a bold-labeled metadata block + prose description.

### Reasoning

**Why two docs rather than one.** The user's earlier
observation noticed a real distinction: content-agnostic
process patterns (transferable across projects) vs.
CS-specific engineering patterns (mostly project-or-language
specific). Mixing them in one file means every lookup scans
past the wrong half; separating preserves the promotion
path (process patterns are candidates for eventual lifting
to global guidance; engineering patterns belong with the
project) and respects the different evolution cadences
(process patterns accumulate slowly across projects;
engineering patterns shift with the codebase).

**Why a two-section structure (Validated / Developing)
rather than per-pattern-tag-only.** The user's discussion
acknowledged real variation in validation level — not just
"validated vs candidate" but gradations within each. Sections
give quick-browse structure (show me only the solid stuff);
per-pattern tags give nuance (but here's exactly what
"solid" or "developing" means for this entry). The
combination prevents the failure mode the user flagged:
coming away thinking a pattern is well-validated when it
isn't. Internet research on pattern-maturity tagging (ADR
status fields, W3C standards stages, VA Design System badges,
Wikipedia citation tagging) confirmed hybrid section+tag
structures as the most effective format, with metadata
staleness and unclear promotion rules as the dominant
failure modes — both addressed by our explicit promotion
rule and per-pattern fields.

**Why the promotion rule is "≥2 independent applications
in this project + ≥1 stated boundary condition."** Two
applications are the minimum for "this works repeatedly,
not just once." The boundary condition forces honest
articulation of where the pattern's limits are unclear or
where it doesn't apply — without this, "validated" risks
becoming a self-reinforcing assertion. The research found
that most pattern catalogs lack explicit promotion criteria;
our rule is clearer than most found in the wild and is
worth preserving as-is.

**Why bold-labeled fields (alternative I) rather than
table-style header block (alternative II).** Both keep
metadata visually distinct from prose, mitigating
staleness. ADR-style tables are more formal but struggle
with multi-sentence field values; the boundary-conditions
field is exactly the one most likely to need long-form
text. Bold-labels read as continuous-with-prose, which
suits a working knowledge doc better than a published
architectural record.

**Why validation-status field carries cross-project
evidence (option α) rather than adding a third section
("Validated elsewhere, developing here").** The
skill-orchestrated design workflow is the canonical case:
extensively used in the user's other projects with
consistently positive results, but n=1 in this project.
Option α (fold into the validation-status field) keeps
the section structure clean (two sections, project-local
evidence), uses the per-pattern field for the kind of
nuance it was designed to carry, and avoids adding ceremony
for what may be a small case load. An intro paragraph makes
the scope explicit (section membership = project-local;
field content = full evidence picture) to prevent misreading.

**Why dual capture for I2.** I2 (smoke-test
backbone-validation gap) is two distinct things bundled
together: (1) a generalizable pattern — "use synthetic
stand-ins for heavyweight dependencies in fast tests" — with
a stated boundary condition; (2) a specific deferred item in
this project. Capturing only the pattern loses the
project-tracking fact; capturing only the deferred item
loses the teachable lesson. Both, in their natural homes
(pattern in `engineering-patterns.md`; deferred-item note in
this design doc's closeout), preserves both purposes.

### Layout

- `docs/notes/process-patterns.md` — new file.
- `docs/notes/engineering-patterns.md` — new file.

Both files share the same top-level structure:

```markdown
# [Process|Engineering] Patterns

[Intro: audience, twofold purpose, doc-landscape fit.]

## Project-local validation scope

[Explicit note: section membership reflects evidence in THIS
project; validation-status field may carry cross-project
evidence as additional context.]

## Promotion rule

[≥2 independent applications + ≥1 stated boundary condition.
Patterns start in Developing.]

## Patterns

### Validated
- [Pattern name](#anchor) — one-line summary

### Developing
- [Pattern name](#anchor) — one-line summary

---

## Validated patterns

### Pattern name

**Validation status**: [project-local evidence; may include
cross-project context]
**First used**: [date, tier/piece, commit]
**Last used**: [date, tier/piece, commit]
**Known boundary conditions**: [where it doesn't apply or
limits unclear]

[Prose: what it is, why we use it, what worked, what didn't,
when to apply / when not to.]

---

## Developing patterns
...
```

### Contracts

Each pattern entry contains the bold-labeled metadata fields
above plus prose. The validation-status field is the
primary carrier of evidence detail — including cross-project
evidence where applicable. The boundary-conditions field is
the primary carrier of "where this pattern's edges are."

### Seed content (Piece 3 deliverable)

`process-patterns.md`:

*Validated:*

- **Pedagogical pattern.** Dialogic per-piece design
  discussion with reasoning recorded inline; design doc
  captures decisions + alternatives + rationale, not just
  outcomes. Used T2, T3, T4. Boundary: works when work is
  design-amenable; lighter version applies even to
  mechanical pieces.
- **Adversarial review after implementation.** End-of-tier
  review surfaces Critical/Important/Minor findings; closeout
  commits address selected items with explicit deferrals.
  Used T2 (8 findings) and T3 (16 findings). Boundary:
  review value scales with surface area changed; not worth
  running for single-line fixes.
- **Design-doc-per-tier as living working doc + Post-review
  corrections section.** Per-tier design doc grows
  piece-by-piece; closeout records what review caught and
  fixed vs deferred. Used T2, T3, T4. Boundary: requires
  work that's naturally tier-able (cohesive scope, finite
  duration).
- **Deferred-with-explicit-notes discipline.** Items
  deferred with stated reasoning and tracking location, not
  silently dropped. Used pervasively in T2/T3/T4 design
  docs + `pinned-questions.md`. Boundary: requires periodic
  revisiting of the deferred list, or items accumulate
  unbounded.

*Developing:*

- **Skill-orchestrated design workflow.** Using
  `starting-a-design-plan` + `brainstorming` +
  `asking-clarifying-questions` skills to structure the
  design phase. T4 only (n=1 in this project).
  Validation-status notes extensive cross-project use
  elsewhere with consistently positive results; local fit
  not yet established.

`engineering-patterns.md`:

*Validated:*

- **Boundary-inventory pattern.** Validate at every layer
  where data crosses a boundary; each catches what others
  miss. Used T3P1 (preprocessor dual-boundary), T3 closeout
  (I1 + I7), T4P1 (M4 dual head-name validation). Boundary:
  applies to data-flow boundaries; the house-style
  `defense-in-depth` 4-layer carving is partial fit (Layers
  1/2 map cleanly; Layers 3/4 are web-app-flavored).
- **Synthetic stand-ins for heavyweight dependencies in
  fast tests.** Use minimal fakes that match the real
  dependency's API surface. Used in
  `scripts/smoke_test_integrated_stack.py` (fake backbone)
  and pervasively in `tests/`. Boundary: **stand-ins must
  match the real dependency's API shape, not just its
  value-level interface.** I2 is the canonical case where
  this slipped — fake exposes `hidden_dim` as instance
  attribute; real `keras_hub` exposes it as class property;
  validation passes against the fake but doesn't catch
  property-vs-attribute drift.
- **Wrapped-vs-flat forward-compat for config sub-objects.**
  Wrap related fields in a sub-object rather than flattening
  into the parent, anticipating future variants. Used
  T3P3a (4 sub-configs in `RunConfig`), T4P2 (`ResolvedSteps`
  nested in `LRScheduleConfig`). Boundary: pattern applies
  when (a) future variants of the wrapped object are
  anticipated, or (b) wrapped fields form a coherent
  semantic group; not justified for ad-hoc collections.

*Developing:*

- **Pattern A vs Pattern 2 model sharing.** Pattern A:
  share `Layer` instances between train and inference
  models for in-process weight propagation. Pattern 2:
  rebuild model fresh, load weights cross-process. Used
  T2P4c only — Pattern A in `run_cca_classification.py`,
  Pattern 2 in `eval_cca_classifier.py`. Boundary
  conditions partially explored in T3P2
  (`.weights.h5` load-by-structure vs load-by-name).
  n=1 architecturally; want one more use case before
  promoting.
- **Empirical investigation before committing to design.**
  When library behavior is uncertain, run a small
  experiment before designing around assumed behavior.
  T3P2 used this for the `.weights.h5` finding that
  reframed the I5 test. n=1; pattern works but the
  question of "when is this worth doing? what magnitude of
  uncertainty justifies it?" is unclear.

### Test coverage anticipated

No automated tests — these are prose documents. Verification
is qualitative: read-through confirms each pattern entry has
all four metadata fields, the validation-status field carries
real evidence (not just "validated"), and the
boundary-conditions field articulates something specific.

### Patterns introduced

This piece doesn't introduce patterns; it *documents* the
patterns introduced in prior tiers. The doc structure itself
(per-pattern metadata block + prose + project-local section
membership + cross-project field content) is novel to this
piece but won't be cataloged as a meta-pattern until it
shows up again somewhere (≥2-applications rule applies
recursively).

### Open / deferred

- **Whether the doc structure itself should eventually
  become a pattern entry.** Right now it's the structure of
  the doc, not a pattern in the doc. If a future project
  adopts the same structure, it'd graduate to a candidate
  pattern in `process-patterns.md`.
- **How the docs get maintained.** Implicit current policy:
  update at piece close when a pattern is used again
  (changes Last used field) or when boundary conditions
  shift. No explicit cadence; risk of staleness if not
  revisited. Adversarial-review-like periodic audit could
  address but premature now.

---

## Tier 4 closeout / deferred items

*Section appended after implementation (mirrors the
"Post-review corrections" sections in `tier2-design.md` and
`tier3-design.md`). Initial entries reflect items
deliberately deferred during design; will grow if Tier 4
gets an adversarial review.*

- **I2** (smoke-test backbone-validation-path gap, inherited
  from Tier 3 closeout). Captured dually: as the canonical
  boundary-condition example for the "Synthetic stand-ins
  for heavyweight dependencies" pattern in
  `engineering-patterns.md` (the generalizable lesson), and
  here as a project-tracking deferred item (the specific
  fact). Closure path when revisited: add a real-keras_hub
  backbone test in `tests/` (heavyweight; consider env-gated
  or cluster-only execution).
- **I8 (full)** metric factoring (inherited from Tier 3
  closeout). Interim `src/cca_metrics.py` helper from Tier 3
  closeout remains sufficient; full factoring deferred until
  multi-head work creates pressure for a richer factoring.
- **Resolved-config requirement enforcement** (Piece 2
  open/deferred). Current I4 fix makes `resolved` optional
  on `LRScheduleConfig`; a strict assertion is premature.
  Revisit when HP search workflow is built.
- **Other LR schedule parameters that depend on train-time
  state** (Piece 2 open/deferred). `ResolvedSteps` currently
  covers only the two factor-based parameters. Grow when
  new schedule types add more train-time-dependent
  parameters.
