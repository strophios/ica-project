# Tier 4 Phase 3: Lessons Docs Implementation Plan

**Goal:** Create two new working documents capturing patterns validated through Tier 2/3/4 work, with explicit per-pattern validation-status tagging.

**Architecture:** Two new files in `docs/notes/`:
- `process-patterns.md` — content-agnostic process patterns (4 Validated + 1 Developing entries seeded)
- `engineering-patterns.md` — CS-specific patterns (3 Validated + 2 Developing entries seeded)

Each file uses a shared shape: intro + project-local validation scope note + promotion rule + index + per-section pattern entries with bold-labeled metadata blocks + prose description. The seed content (which patterns and their metadata) is fully specified in `docs/notes/tier4-design.md` "Piece 3 — Seed content" subsection — implementation expands each bullet into a prose entry following the agreed format.

**Tech Stack:** Markdown. No code; no tests (verification is qualitative — read-through confirms each pattern entry has all four metadata fields).

**Scope:** 3 of 3 phases. Design reference: `docs/notes/tier4-design.md` "Piece 3: Lessons docs".

**Codebase verified:** 2026-05-11.

---

<!-- START_TASK_1 -->
### Task 1: Write process-patterns.md

**Files:**
- Create: `docs/notes/process-patterns.md`

**Context.** Content-agnostic process patterns: workflow shapes, review disciplines, doc structures. Per-pattern format uses bold-labeled metadata fields (Validation status, First used, Last used, Known boundary conditions) plus prose.

**Step 1: Read the seed content from the design doc**

Run:
```bash
sed -n '/^### Seed content/,/^### Test coverage anticipated/p' docs/notes/tier4-design.md
```

This shows the seed content subsection of Piece 3, which lists:
- Validated patterns: pedagogical pattern, adversarial review after implementation, design-doc-per-tier with Post-review corrections, deferred-with-explicit-notes discipline
- Developing pattern: skill-orchestrated design workflow

Each entry has a one-line summary and the agreed metadata (where used, boundary conditions, etc.).

**Step 2: Read the format spec from the design doc**

Run:
```bash
sed -n '/^### Layout$/,/^### Contracts$/p' docs/notes/tier4-design.md | head -60
```

This shows the shared doc shape (intro + project-local-scope note + promotion rule + index + per-pattern entries).

**Step 3: Determine concrete commit/date references**

For each Validated pattern, the metadata needs concrete first-used and last-used references. Use `git log` to find tier-marker commits:

Run:
```bash
git log --oneline --all | grep -E "Tier [234]"
```

This produces a list of all tier-related commits. Reference the relevant ones in the metadata fields (e.g., "First used: 2026-04-XX, T2 design opening, commit XYZ").

For dates within Tier 2/3 timeframes, the existing design docs at `docs/notes/tier2-design.md` and `docs/notes/tier3-design.md` have piece-by-piece dates (see preamble + section headers). Use those as authoritative.

**Step 4: Draft the file**

Create `docs/notes/process-patterns.md` with the following structure (expand seed bullets to full prose per the format):

```markdown
# Process Patterns

This document catalogs content-agnostic process patterns validated
through the Tier 2 / Tier 3 / Tier 4 work in this project. Patterns
here are about *how the work is done* — workflow shapes, review
disciplines, doc structures — independent of the technical content.

**Audience:** the user, and Claude on future sessions in this
project. **Purpose:** discoverability of validated patterns when
looking for "do we have an approach for X?"; proactive suggestion
during future work ("you may want to reuse the pedagogical pattern
documented here").

Companion document: `docs/notes/engineering-patterns.md` covers
CS-specific patterns (validation shapes, config conventions, etc.).
The two docs are separated because process patterns are largely
transferable across projects, while engineering patterns tend to
be project-and-language-specific.

## Project-local validation scope

Section membership (Validated vs Developing) reflects evidence
**in this project**. The per-pattern `Validation status` field
may carry cross-project evidence as additional context (e.g.,
"used extensively in user's other projects with consistently
positive results"), but section placement follows the
project-local promotion rule below.

This is deliberate. Cross-project evidence is real and worth
recording, but conflating "validated here" with "validated
somewhere" would let patterns sneak in based on external success
without local fit ever being tested.

## Promotion rule

Patterns start in **Developing**. A pattern graduates to
**Validated** when it has been used successfully in **≥2
independent applications in this project** *and* we can
articulate **at least one boundary condition** (where the pattern
doesn't apply, or where its limits are unclear).

The boundary-condition requirement is the discipline that
prevents "validated" from becoming a self-reinforcing assertion.

## Patterns

### Validated
- [Pedagogical pattern](#pedagogical-pattern) — dialogic per-piece design discussion with reasoning recorded inline
- [Adversarial review after implementation](#adversarial-review-after-implementation) — end-of-tier review with structured findings + closeout commits
- [Design-doc-per-tier as living working doc + Post-review corrections section](#design-doc-per-tier-as-living-working-doc--post-review-corrections-section) — per-tier doc growing piece-by-piece, with explicit corrections record
- [Deferred-with-explicit-notes discipline](#deferred-with-explicit-notes-discipline) — items deferred with stated reasoning and tracking location

### Developing
- [Skill-orchestrated design workflow](#skill-orchestrated-design-workflow) — using starting-a-design-plan, brainstorming, and asking-clarifying-questions skills to structure the design phase

---

## Validated patterns

### Pedagogical pattern

**Validation status**: Used in Tier 2, Tier 3, and Tier 4 with consistently positive results. Each tier's design doc captures decisions + alternatives + rationale rather than just outcomes, which has made later sessions and future-Claude able to reconstruct *why* decisions were made.
**First used**: 2026-04-XX, Tier 2 design opening. (Confirm commit hash via `git log` during implementation.)
**Last used**: 2026-05-11, Tier 4 design (this tier).
**Known boundary conditions**: Works when work is design-amenable (multiple alternatives worth exploring, non-trivial decisions to justify). For purely mechanical fixes — e.g., a one-line null-check addition — the pattern compresses to "decision: do X; reasoning: it fixes the bug." A lighter version of the pattern still applies even there.

[Prose description: 2-4 paragraphs explaining what the pattern is in practice (dialogic Q&A flow, decisions captured at full fidelity, alternatives explicitly considered), why it works (preserves reasoning for future readers; surfaces edge cases through Socratic questioning), what worked specifically in T2/T3/T4 (examples of decisions captured: nested vs flat sub-configs in T3P3a; Pattern A vs Pattern 2 in T2P4; α option for cross-project evidence in T4 lessons docs), and when to apply (any design discussion with genuine alternatives). Reference the relevant design doc sections for concrete examples.]

### Adversarial review after implementation

**Validation status**: Used at end of Tier 2 (8 findings) and Tier 3 (16 findings); closeout commits addressed selected items with explicit deferrals for the rest. Both reviews caught real bugs and design gaps that wouldn't have been found during in-flight piece-by-piece review.
**First used**: 2026-04-XX (T2 closeout review). (Confirm commit via `git log`.)
**Last used**: 2026-05-09 (T3 closeout).
**Known boundary conditions**: Review value scales with surface area changed. Not worth running for single-line fixes (you'll get findings, but the cost-benefit favors just landing the fix). Most useful at tier or major-piece boundaries.

[Prose description: explain the pattern (end-of-tier review by an outside reviewer subagent, structured findings categorized Critical/Important/Minor, closeout commits triage by impact), how it integrates with the project (closeout commits land alongside the tier; deferred items get tracked explicitly), what worked (specific examples: T2 review caught the sparse-gradient handling issue in LayerLRModel; T3 review caught the I4 LR resolution gap that became T4P2), and when to apply (tier or major-piece completion with substantive design surface area).]

### Design-doc-per-tier as living working doc + Post-review corrections section

**Validation status**: Used for Tier 2, Tier 3, and Tier 4. The "living document — appended piece-by-piece" framing matches how the work actually unfolds (decisions made dialogically, not drafted up-front). The Post-review corrections section explicitly records what the adversarial review caught and which items got fixed vs deferred — the doc itself becomes auditable.
**First used**: 2026-04-XX (T2 design doc).
**Last used**: 2026-05-11 (T4 design doc, this tier).
**Known boundary conditions**: Requires work that's naturally tier-able — cohesive scope, finite duration, identifiable closing point. Doesn't fit indefinite or open-ended work streams (e.g., ongoing maintenance, exploratory research). For those, a different doc shape (running notes, decision log) probably fits better.

[Prose description: explain the doc structure (Overall framing + Pieces anticipated + per-piece sections with Decision/Reasoning/Layout/Contracts/Test coverage/Patterns introduced/Open or deferred + Tier closeout/deferred items section), why it works (each piece's decision and reasoning is captured at full resolution; later readers can trace the design evolution), what worked in T2/T3/T4 (per-piece subsection hierarchy makes finding specific decisions easy), when to apply (tier-shaped work).]

### Deferred-with-explicit-notes discipline

**Validation status**: Used pervasively across T2, T3, T4 design docs and `pinned-questions.md`. The discipline is that items are deferred with stated reasoning (why deferred), tracking location (where to find when revisited), and closure path (what would close them) — never silently dropped.
**First used**: 2026-04-XX (T2 design, multiple instances).
**Last used**: 2026-05-11 (T4 design, I2 and I8-full deferrals).
**Known boundary conditions**: Requires periodic revisiting of the deferred list, or items accumulate unbounded. Currently no explicit cadence — relies on tier-boundary review surfacing whether deferred items have become pressing. If the deferred list grows beyond ~10-15 items, the lack of cadence will start to bite.

[Prose description: explain the pattern (deferred items get a stated reason, a tracking location like pinned-questions.md or a tier-design closeout section, and a closure path; e.g., I4's Tier 3 closeout note said "three candidate fixes; none warranted now" with explicit conditions for revisiting), why it works (institutional memory of why something wasn't done; prevents the "we should fix X someday" diffuse intentions that never land), what worked specifically (I4 got revisited and fixed in T4 because the deferred note made the conditions concrete), when to apply (any review or design decision that produces "we could do X but won't right now"; pinned-questions.md is the canonical home for substantive deferrals).]

---

## Developing patterns

### Skill-orchestrated design workflow

**Validation status**: Used in Tier 4 only (n=1 in this project). The user reports extensive cross-project use of this workflow (`starting-a-design-plan` → `brainstorming` → `asking-clarifying-questions` → `writing-design-plans` → `starting-an-implementation-plan` → `writing-implementation-plans`) with consistently positive results elsewhere. Local fit not yet established: does the rhythm suit ML-research design? Do the skill outputs shape well to this project's pedagogical preferences?
**First used**: 2026-05-11 (T4 design and implementation planning, this tier).
**Last used**: same.
**Known boundary conditions**: Unknown for this project. Cross-project evidence suggests it works well for general software design but the ML-research context — with its emphasis on numerical correctness, hyperparameter sensitivity, and experimental validation — may surface different demands than typical web-app design work. Watch for: skill-imposed structure fighting the work's actual shape; over-ceremony for small mechanical changes (the T4 hygiene piece used skills but the simple items could have skipped formal design).

[Prose description: explain what the workflow looks like in practice (skill-by-skill orchestration, branch creation, per-phase code review), why the user adopted it (cross-project track record), what was tested in T4 (full skill chain from `starting-a-design-plan` through `writing-implementation-plans`), open questions (does it scale down to mechanical work? does it scale up to genuinely architectural work? how does it interact with the pedagogical pattern?). Will be promoted to Validated after one more tier of use *plus* identification of clear boundary conditions.]
```

**Step 5: Verify the file**

Run:
```bash
wc -l docs/notes/process-patterns.md
grep -c "^### " docs/notes/process-patterns.md
```

Expected: file exists with reasonable line count (200-400); 5 `### ` pattern entries (4 Validated + 1 Developing).

**Step 6: Confirm all metadata fields are present per pattern**

Run:
```bash
grep -A 4 "^### " docs/notes/process-patterns.md | head -60
```

Expected: each pattern entry has the four bold-labeled metadata fields (Validation status, First used, Last used, Known boundary conditions) before its prose description.

No commit yet.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Write engineering-patterns.md

**Files:**
- Create: `docs/notes/engineering-patterns.md`

**Context.** CS-specific patterns: validation shapes, config conventions, model-sharing approaches. Same shared shape as `process-patterns.md`. Per the dual-capture decision in the design doc, the "Synthetic stand-ins for heavyweight dependencies" pattern includes I2 as its canonical boundary-condition example.

**Step 1: Reference the design doc seed content**

The seed content for engineering-patterns lives in `docs/notes/tier4-design.md` "Piece 3 — Seed content" subsection (same as Task 1 — re-read if needed). Patterns:
- Validated: boundary-inventory, synthetic stand-ins (with I2 as boundary case), wrapped-vs-flat forward-compat
- Developing: Pattern A vs Pattern 2 model sharing, empirical investigation before committing to design

**Step 2: Determine concrete commit/date references**

Same approach as Task 1 — use `git log` to find piece-marker commits for accurate first-used and last-used metadata. Engineering-patterns will reference T3P1 (boundary-inventory), T3P2 (empirical investigation), T2P4c (Pattern A/2), T3P3a + T4P2 (wrapped-vs-flat).

**Step 3: Draft the file**

Create `docs/notes/engineering-patterns.md` with the structure below. Section bodies use the same bold-labeled-metadata + prose format established in Task 1. Companion-doc cross-reference at the top points to `process-patterns.md`.

```markdown
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
articulate **at least one boundary condition** (where the
pattern doesn't apply, or where its limits are unclear).

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

**Validation status**: Used in T3P1 (preprocessor dual-boundary), T3 closeout (I1 schema validation + I7 dtype check), T4P1 M4 (head-name validation at construction-site + call-site). Three+ applications in this project, with consistently positive results. Cross-project context: the house-style `defense-in-depth` skill articulates the same animating insight ("validate at every layer; each catches what others miss; turn bugs from 'vigilance keeps us safe' into 'the structure prevents them'").
**First used**: 2026-05-XX (T3 Piece 1; commit via `git log`).
**Last used**: 2026-05-11 (T4 Piece 1 M4 dual head-name validation).
**Known boundary conditions**: Applies to **data-flow boundaries** specifically — places where data crosses from one trust domain or representation into another. The house-style `defense-in-depth` skill's web-app-flavored 4-layer carving (entry-point / business-logic / environment-guards / debug-instrumentation) is only partial fit for this codebase; Layers 1 (entry-point) and 2 (business-logic) map cleanly, but Layers 3 and 4 are web-app-specific and don't transfer. The pattern doesn't apply to internal helper functions that operate on already-validated data.

[Prose description: explain the pattern (identify each boundary where data crosses; place validation appropriate to that boundary; each catches what others miss), why it works (silent corruption becomes structurally hard, not just procedurally avoided), examples from T3P1 (preprocessor `__init__` catches internal-config bugs; `__call__` catches input-batch mismatches), examples from T4P1 (ClassificationHead.__init__ catches "forgot to name"; build_endpoint_model catches "named two things the same"), when to apply (data-flow boundaries; situations where silent failure would be hard to debug).]

### Synthetic stand-ins for heavyweight dependencies in fast tests

**Validation status**: Used in `scripts/smoke_test_integrated_stack.py` (fake backbone) and pervasively in `tests/` (synthetic row counts, mock datasets). Two+ applications in this project, with positive results in terms of test-suite speed. The canonical boundary case (I2, see below) confirms the pattern's limits are non-trivial — promoted to Validated despite n=2 because the boundary condition is explicit and the value is clear.
**First used**: 2026-04-XX (smoke test introduction; commit via `git log`).
**Last used**: 2026-05-09 (T3 Piece 4 added data-loading tests using synthetic dataframes).
**Known boundary conditions**: **Stand-ins must match the real dependency's API shape, not just its value-level interface.** The canonical case is **I2** (smoke-test backbone-validation gap, inherited from Tier 3 closeout): the fake backbone exposes `hidden_dim` as an instance attribute, but the real `keras_hub.models.Backbone.from_preset(...)` exposes it as a class property. The smoke test's `validate_against_backbone` call passes against the fake but doesn't catch property-vs-attribute drift in the real surface. The smoke test docstring (`scripts/smoke_test_integrated_stack.py:28-40`) acknowledges this gap; closure would require either a real-keras_hub backbone variant (heavyweight; env-gated or cluster-only) or accepting the gap as the cost of fast smoke tests.

[Prose description: explain the pattern (use minimal fakes that match the real API surface for tests that would otherwise be slow due to heavyweight deps), why it works (test-suite speed enables tight feedback loops), the I2 case in detail (instance-attribute vs class-property mismatch), how to avoid the boundary-condition failure (when constructing a fake, prefer to mock at the class level rather than instance level when the real object's interface uses class properties), when to apply (tests where the real dependency adds >1s of overhead; not when the dependency's behavior is itself the thing under test).]

### Wrapped-vs-flat forward-compat for config sub-objects

**Validation status**: Used in T3P3a (RunConfig with four sub-configs: FLPULossConfig, OptimizerConfig, LRScheduleConfig, RatioBatchConfig) and T4P2 (ResolvedSteps nested in LRScheduleConfig). Two independent applications with clear positive results — both cases anticipated future variants and the wrapping shape paid off.
**First used**: 2026-05-XX (T3 Piece 3a; commit via `git log`).
**Last used**: 2026-05-11 (T4 Piece 2, this tier).
**Known boundary conditions**: Pattern applies when (a) future variants of the wrapped object are anticipated (e.g., FLPULossConfig anticipates ALUM and BCE variants per `pinned-questions.md`), OR (b) the wrapped fields form a coherent semantic group (ResolvedSteps wraps three related fields that travel together). Not justified for ad-hoc collections of unrelated fields — the extra nesting cost in JSON and code reads doesn't pay back without one of these justifications. Premature wrapping is its own anti-pattern.

[Prose description: explain the pattern (when designing a frozen-dataclass config, wrap related fields in a sub-object rather than flattening them into the parent, *if* the wrapped fields will plausibly evolve as a unit), why it works (avoids forced restructuring when new variants land; sub-config naming carries semantic clarity), examples from T3P3a (each loss type, optimizer type, schedule type would have different fields — wrapping makes the variant discrimination forward-compat), examples from T4P2 (ResolvedSteps groups three fields that all come from the same train-time computation; if future schedule types add more train-time-dependent fields, ResolvedSteps grows), when to apply (anticipated variants OR coherent semantic group), when not to (ad-hoc unrelated fields; speculative wrapping without a concrete justification).]

---

## Developing patterns

### Pattern A vs Pattern 2 model sharing

**Validation status**: Used in T2 Piece 4c only (n=1 architecturally, though both variants exercised). Pattern A in `run_cca_classification.py` (in-process Layer-instance sharing between train and post-train predict). Pattern 2 in `eval_cca_classifier.py` (cross-process: load weights into a freshly-built model). Boundary conditions were further explored in T3 Piece 2 via the `.weights.h5` load-by-structure vs load-by-name finding. Promotion to Validated waits for one more architectural use case to confirm the patterns generalize.
**First used**: 2026-05-XX (T2 Piece 4c; commit via `git log`).
**Last used**: 2026-05-XX (T2P4c; T3P2 explored related territory but didn't introduce a new use of either pattern).
**Known boundary conditions**: Pattern A requires same-process train and inference (Layer instances are Python objects, not serializable across processes). Pattern 2 requires the inference-side model to be reconstructable independently — which means head names, hidden_dim, dropout, etc. must round-trip through the sidecar (this is what made the sidecar-as-single-source-of-truth work in T3P3 and T4P2 important). T3P2's empirical finding (that `.weights.h5` keys variables by layer-class + positional index, not user-given name) reframed Pattern 2 from load-by-name to load-by-structure; the head-name contract is enforced at call sites (Keras `compile` routing), not at weight load.

[Prose description: explain both patterns (A: in-process layer sharing; 2: cross-process load-by-structure), why each is used where (training script does post-train predict in-process; eval script is cross-process via a separate invocation), known limits (process-locality for A; structural matching requirements for 2), open questions (does Pattern 2 generalize to multi-head cleanly? do we need a Pattern 3 for cross-process Layer-instance reuse?). This pattern will likely graduate to Validated when multi-head heads land and we need to extend the sharing scheme.]

### Empirical investigation before committing to design

**Validation status**: Used in T3 Piece 2 (`.weights.h5` by-name vs by-structure discovery via `scripts/experiment_endpoint_inference_evaluate.py` or similar). n=1; pattern worked (the experiment caught a wrong assumption before it became a baked-in design constraint), but boundary conditions — *when* is this worth doing? what *magnitude* of uncertainty justifies the investigation? — are unclear.
**First used**: 2026-05-XX (T3 Piece 2; commit via `git log`).
**Last used**: same.
**Known boundary conditions**: The cost-benefit framing is missing. The T3P2 case was a clear win because the wrong assumption would have shaped multiple files (test design, eval-script design, smoke-test design); catching it before commit saved rework. But the pattern as currently used doesn't articulate when the experiment is worth running vs when reading docs / source is sufficient. Promotion to Validated waits for a second use case where the experiment-vs-research tradeoff is more visible.

[Prose description: explain the pattern (when a library or external dependency's behavior is uncertain or underdocumented, run a small empirical experiment script in `scripts/` to verify behavior before designing around assumed behavior), why it worked in T3P2 (Keras's `.weights.h5` saving behavior was non-obvious from docs; assumed by-name keying turned out to be by-structure), open questions (what threshold of uncertainty triggers the experiment? how do we avoid premature experimentation? when does reading library source code substitute for an experiment?), when (probably) to apply (significant design surface area that would be reshaped by the wrong assumption), when not (low-cost questions answerable by docs).]
```

**Step 4: Verify the file**

Run:
```bash
wc -l docs/notes/engineering-patterns.md
grep -c "^### " docs/notes/engineering-patterns.md
```

Expected: file exists; 5 `### ` pattern entries (3 Validated + 2 Developing).

**Step 5: Confirm metadata fields per pattern**

Run:
```bash
grep -A 4 "^### " docs/notes/engineering-patterns.md | head -60
```

Expected: each pattern entry has all four bold-labeled metadata fields before prose.

**Step 6: Verify I2 is captured as the canonical boundary case**

Run:
```bash
grep -A 2 "^### Synthetic stand-ins" docs/notes/engineering-patterns.md
grep "I2" docs/notes/engineering-patterns.md
```

Expected: the "Synthetic stand-ins" pattern entry explicitly references I2 as the canonical boundary-condition example.

No commit yet.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Commit (one or two commits)

**Context.** Two options: (a) one combined commit for both files; (b) two commits, one per file. Decision criteria: if drafting Task 1 and Task 2 surfaced cross-references between the two docs (e.g., one pattern entry referencing another doc's pattern), a single commit reads more naturally because the cross-references land together. Otherwise, two commits give cleaner per-doc history. Default to **one combined commit** for simplicity unless cross-references made the docs tightly coupled.

**Step 1: Confirm files are ready**

Run:
```bash
ls -la docs/notes/process-patterns.md docs/notes/engineering-patterns.md
git status
```

Expected: both files exist; both show as untracked in `git status`.

**Step 2: Verify no placeholder text remains**

Run:
```bash
grep -nE "(\[Prose description|XX|\bTBD\b|FIXME|2026-04-XX|2026-05-XX)" docs/notes/process-patterns.md docs/notes/engineering-patterns.md
```

Expected: **no matches**. If any match appears, the corresponding placeholder must be replaced before commit:
- `[Prose description:` blocks → expand into actual prose per the bracketed guidance (2-4 paragraphs explaining what the pattern is, why we use it, what worked, when to apply).
- `XX` / `TBD` / `FIXME` → fill in with the concrete date or commit hash from `git log` (Tasks 1-2 Step 3).

Re-run the grep after fixes; loop until empty.

**Step 3: Read each file end-to-end for qualitative verification**

Manually read each file. Verify:
- Each pattern entry has all four metadata fields (Validation status / First used / Last used / Known boundary conditions).
- Each `Validation status` field carries a non-trivial sentence (not just "validated" or "developing").
- Each `Known boundary conditions` field articulates something specific.
- Cross-references between docs (if any) actually resolve to the intended sections.
- The intro + project-local-scope + promotion rule sections are present and accurate.
- Each prose description block is actual prose (not the bracketed instruction template from this plan).

**Step 4: Stage and commit**

If using single-commit approach:

```bash
git add docs/notes/process-patterns.md docs/notes/engineering-patterns.md
git commit -m "$(cat <<'EOF'
Tier 4 Piece 3: lessons docs (process-patterns + engineering-patterns)

Two new working documents capturing patterns validated through
Tier 2 / Tier 3 / Tier 4 work. See docs/notes/tier4-design.md
"Piece 3" for the design.

- docs/notes/process-patterns.md: content-agnostic process
  patterns (4 Validated entries: pedagogical pattern, adversarial
  review after implementation, design-doc-per-tier with
  Post-review corrections, deferred-with-explicit-notes
  discipline; 1 Developing entry: skill-orchestrated design
  workflow).
- docs/notes/engineering-patterns.md: CS-specific patterns (3
  Validated: boundary-inventory, synthetic stand-ins with I2 as
  canonical boundary case, wrapped-vs-flat forward-compat; 2
  Developing: Pattern A vs Pattern 2 model sharing, empirical
  investigation before committing to design).

Each pattern uses bold-labeled metadata (validation status, first
and last used, known boundary conditions) plus prose description.
Section membership reflects project-local evidence; validation
status field carries cross-project context where applicable
(option α from the design discussion).

I2 dual-captured: the canonical boundary-condition example in
engineering-patterns.md "Synthetic stand-ins..." entry; also
listed as a deferred-item in docs/notes/tier4-design.md
closeout section.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If using two-commit approach, split the message accordingly:
- Commit 1: `Tier 4 Piece 3a: process-patterns lessons doc` with the process-patterns details.
- Commit 2: `Tier 4 Piece 3b: engineering-patterns lessons doc` with the engineering-patterns details, plus a note that this completes Piece 3.

**Step 5: Verify commit landed**

Run:
```bash
git log --oneline -3
git status
```

Expected: New HEAD commit(s); working tree clean.

**Step 6: Run pytest one final time**

Run:
```bash
pytest
```

Expected: All tests still pass (no test changes in this phase; this confirms we didn't accidentally break anything via documentation work).

**Step 7: Request final code review (per project Tier 2/3 convention)**

Dispatch the code-reviewer subagent for an overall Tier 4 review (cross-phase). Review prompt should reference `docs/notes/tier4-design.md` for the spec across all three pieces, plus the per-phase commits for the diffs.

If review surfaces findings, address via a closeout commit and append to `docs/notes/tier4-design.md` "Tier 4 closeout / deferred items" section.
<!-- END_TASK_3 -->
