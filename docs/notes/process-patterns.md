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

**Validation status**: Used in Tier 2 (started 2026-04-20), Tier 3 (started 2026-05-08), and Tier 4 (started 2026-05-11) with consistently positive results. Each tier's design doc captures decisions + alternatives + rationale rather than just outcomes, which has made later sessions and future-Claude able to reconstruct *why* decisions were made.

**First used**: 2026-04-20, Tier 2 design opening (commit `789d88c`).

**Last used**: 2026-05-11, Tier 4 design (commit `6bc897a`).

**Known boundary conditions**: Works when work is design-amenable — multiple alternatives worth exploring, non-trivial decisions with clear tradeoffs. For purely mechanical fixes (a one-line null-check, a forgotten docstring), the pattern compresses to "decision: do X; reasoning: it fixes the bug." A lighter version of the pattern still applies even in mechanical contexts, but the full dialogic exploration is overkill.

The pedagogical pattern means capturing design decisions dialogically: exploring alternatives, articulating tradeoffs, and recording the choice and its rationale *at the time of decision*. Tier 2 started this with the head composition and Layer-LR-model abstractions, where the design doc captured why Keras's functional API led us to endpoint-layer heads and why discriminative fine-tuning required a specialized `LayerLRModel`. Tier 3 continued the pattern through the boundary-inventory framing, explaining why each of I3, I4, and I5 required different validation strategies rather than a single belt-and-suspenders approach. Tier 4 applied it again for the wrapped-vs-flat decision on `ResolvedSteps`.

What made this pattern effective: the decisions weren't obvious upfront. The head composition discussion surfaced genuine alternatives (functions vs layers; standard vs endpoint modes). The I3/I4/I5 boundary-inventory framing only emerged through Socratic questioning — a reader skimming the Tier 3 doc immediately understands *why* three separate validations were needed, not just that they were added. When future-Claude revisits the wrapped-vs-flat question (whether to flatten `ResolvedSteps` into `LRScheduleConfig`), the Tier 4 design doc makes the cost-benefit reasoning transparent, shortcutting the whole discussion.

Apply this pattern when facing design decisions with genuine alternatives. When the choice is mechanical ("this code is wrong; here's the fix"), note the reasoning but don't force a full Socratic dialogue. The pattern is about preserving reasoning, not ceremony.

### Adversarial review after implementation

**Validation status**: Used at end of Tier 2 (8 findings, 2 critical) and Tier 3 (16 findings, 1 critical); closeout commits addressed selected items with explicit deferrals for the rest. Both reviews caught real bugs and design gaps that wouldn't have been found during in-flight piece-by-piece review.

**First used**: 2026-04-28, Tier 2 closeout review (commit `76c353f`).

**Last used**: 2026-05-09, Tier 3 closeout (commit `987a8c0` "Tier 3 closeout: address adversarial review").

**Known boundary conditions**: Review value scales with surface area changed. Not worth running for single-line fixes (you'll get findings, but the cost-benefit favors just landing the fix). Most useful at tier or major-piece boundaries where the accumulated design surface is non-trivial.

An adversarial review is a focused examination by an external reviewer (in this case, a code-review subagent with explicit adversarial instructions) looking for gaps the implementation team might miss. Tier 2's review surfaced a critical issue: the sparse-gradient handling in `LayerLRModel.train_step` was broken — the loss-scaling optimizer case wasn't handled, causing training to silently fail under `mixed_float16`. This was an implementation-level bug that in-flight review wouldn't have caught because the piece-review happened before the scaling code was wired in. Tier 3's review caught a more subtle architectural gap: the configuration passed to the training script was never validated to match what the eval script expected, creating a silent-drift risk across the train-eval boundary.

The pattern integrates with the project through closeout commits that land after the review, categorized as Critical (must fix before moving on), Important (fix in closeout or defer explicitly), or Minor (polish, usually deferred). This triage discipline prevents two failure modes: (1) trying to fix everything at once and bloating the scope, (2) fixing nothing and racking up debt. The review findings are recorded in the design doc's "Tier X closeout / Post-review corrections" section, making them auditable — future readers can see exactly what the review found, what was fixed when, and why remaining items were deferred.

Apply this pattern at tier completion or after major-piece work. Don't run it for incremental fixes or refactors under 100 lines. The cost (hiring a reviewer, addressing findings) is only justified when the surface area is substantial enough that the review will catch something real.

### Design-doc-per-tier as living working doc + Post-review corrections section

**Validation status**: Used for Tier 2 (started 2026-04-20), Tier 3 (started 2026-05-08), and Tier 4 (started 2026-05-11). The "living document — appended piece-by-piece" framing matches how the work actually unfolds (decisions made dialogically, not drafted up-front). The Post-review corrections section explicitly records what the adversarial review caught and which items got fixed vs deferred — the doc itself becomes auditable.

**First used**: 2026-04-20, Tier 2 design doc (commit `789d88c`).

**Last used**: 2026-05-11, Tier 4 design doc (commit `6bc897a`).

**Known boundary conditions**: Requires work that's naturally tier-able — cohesive scope, finite duration, identifiable closing point. Doesn't fit indefinite or open-ended work streams (e.g., ongoing maintenance, exploratory research, continuous refactoring). For those, a different doc shape (running notes, decision log, ADR directory) probably fits better.

The per-tier design doc is a working document that grows piece-by-piece as design discussions land, rather than a spec drafted upfront. Each tier's doc opens with an overall framing (intent, work flavors, definition of done), then appends piece sections as they're designed (decision, reasoning, layout, contracts, test coverage, patterns introduced, open/deferred). The Tier 2 doc captures the multi-head abstract shapes; the Tier 3 doc builds the boundary-inventory framing on top; Tier 4 adds the I4 decision to wrap resolved steps. A reader tracing through these sections sees the design evolve and the reasoning crystallize.

What made this shape work: each piece's decision and reasoning is captured at full resolution, not summarized. The Tier 3 doc's I3/I4/I5 section explains exactly why three separate validations were needed and how they interact, with enough detail that a future engineer can decide whether to revisit the design. The post-review corrections section turns the doc into an audit trail — "the review found C1 (sparse gradients), we fixed it in this commit, linked it here; the review found I2 (synthetic backbone), we deferred it with this reasoning."

Apply this pattern to tier-shaped work. Create a doc at the tier's opening with the overall framing. Append each piece's design section as it's discussed. After the tier's review, add the corrections section. The result is a document that's simultaneously a design spec, a decision log, and an audit trail.

### Deferred-with-explicit-notes discipline

**Validation status**: Used pervasively across Tier 2, Tier 3, and Tier 4 design docs and in `docs/notes/pinned-questions.md`. The discipline is that items are deferred with stated reasoning (why deferred), tracking location (where to find when revisited), and closure path (what would close them) — never silently dropped.

**First used**: 2026-04-20, Tier 2 design (commit `789d88c`, multiple deferred items in piece descriptions).

**Last used**: 2026-05-11, Tier 4 design (commit `6bc897a`, I2 and I8-full deferrals in the closeout section).

**Known boundary conditions**: Requires periodic revisiting of the deferred list, or items accumulate unbounded. Currently no explicit cadence — relies on tier-boundary reviews surfacing whether deferred items have become pressing. If the deferred list grows beyond ~10-15 items, the lack of cadence will start to bite.

A deferred item is a legitimate decision to *not* do something right now, paired with enough context that someone (future-Claude, the user, a teammate) can revisit it later without re-doing the original analysis. The pattern requires three pieces: (1) **reasoning** — why not do it now? (e.g., "I4 is not a current bug — eval doesn't reconstruct the schedule, so the sidecar-self-sufficiency gap is nice-to-have, not blocking"); (2) **tracking location** — where does the note live? (e.g., "I4 deferred note in tier3-design.md closeout section"; (3) **closure path** — what conditions would make it worth revisiting? (e.g., "if we build an HP search workflow where `BATCH_SIZE` varies, schedule drift becomes a risk; revisit then").

Tier 3 closed with a deferred list of three items (I2, I4, I8-full) — each with explicit reasoning and closure paths. When Tier 4 began and I4 became a priority (the sidecar-self-sufficiency principle was compelling enough to justify the cost), the deferred note made it clear what would need to happen: record resolved steps alongside factors. The pattern prevented the failure mode of "we deferred something and then forgot it existed" or worse, "we forgot why we deferred it and did a half-measure that doesn't actually solve the problem."

Apply this pattern to any review or design decision that produces "we could do X but won't right now." Record the reasoning and closure path. `docs/notes/pinned-questions.md` is the canonical home for substantive deferrals that cross tiers; tier-design-doc closeout sections are the home for tier-specific deferrals. Periodically (at tier boundaries) audit the deferred list to see if closure paths have been met.

---

## Developing patterns

### Skill-orchestrated design workflow

**Validation status**: Used in Tier 4 only (n=1 in this project). The user reports extensive cross-project use of this workflow (`starting-a-design-plan` → `brainstorming` → `asking-clarifying-questions` → `writing-design-plans` → `starting-an-implementation-plan` → `writing-implementation-plans`) with consistently positive results elsewhere. Local fit not yet established: does the rhythm suit ML-research design? Do the skill outputs shape well to this project's pedagogical preferences?

**First used**: 2026-05-11, Tier 4 design and implementation planning (commit `6bc897a`, this tier).

**Last used**: same (2026-05-11).

**Known boundary conditions**: Unknown for this project. Cross-project evidence suggests it works well for general software design but the ML-research context — with its emphasis on numerical correctness, hyperparameter sensitivity, and experimental validation — may surface different demands than typical web-app design work. Watch for: skill-imposed structure fighting the work's actual shape; over-ceremony for small mechanical changes (the Tier 4 hygiene pieces used skills but most items were simple fixes that could have skipped formal design phases). Another boundary: does the workflow scale down to small bug fixes? Does it scale up to genuinely architectural decisions (like the multi-head future's structure)?

The skill-orchestrated design workflow uses a structured sequence of Claude skills to move from rough ideas through validated designs to implementation planning. The user has built and refined this workflow across projects: starting with `starting-a-design-plan` to gather context and clarify requirements; moving through `brainstorming` to explore alternatives systematically; using `asking-clarifying-questions` to surface ambiguities; then `writing-design-plans` to document the chosen direction; and finally `writing-implementation-plans` to decompose into tasks. In Tier 4, this workflow took a heuristic spec (three pieces of work: hygiene fixes, I4 resolution, lessons docs) and produced a detailed implementation plan with per-task acceptance criteria and verification steps.

What was tested in Tier 4: does the full skill chain produce outputs that integrate smoothly with the project's existing pedagogical practices (design-doc-per-tier, adversarial review, deferred-with-explicit-notes)? Answer so far: yes, but incompletely. The `writing-design-plans` and `writing-implementation-plans` skills produced structured documents that fit well with the existing pattern. The `brainstorming` skill worked for exploring alternatives (the I4 wrapped-vs-flat question, the I2 synthetic-vs-real backbone tradeoff). But the workflow's emphasis on "get full clarity upfront before starting work" sits slightly uncomfortably with the project's "start design, implement piece, append to design doc" cadence — the two aren't incompatible, but they're not perfectly aligned either.

Will be promoted to Validated after one more tier of use *plus* identification of clear boundary conditions. The key unknowns: (1) does the workflow scale down gracefully to mechanical work without imposing ceremony? (2) does the pedagogical-pattern (design discussion with reasoning recorded) emerge naturally from the skill outputs, or does it require additional hand-work? (3) does the skill-orchestrated approach outperform the previous ad-hoc discussion model, or do they produce equivalent results with different costs?
