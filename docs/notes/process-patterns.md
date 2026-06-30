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
- [Investigator-subagent pattern](#investigator-subagent-pattern) — fresh read-only subagent gathers codebase facts before plans commit to assumptions
- [Design-doc-per-tier as living working doc + Post-review corrections section](#design-doc-per-tier-as-living-working-doc--post-review-corrections-section) — per-tier doc growing piece-by-piece, with explicit corrections record
- [Implementation plans with file:line specificity](#implementation-plans-with-fileline-specificity) — plans encode the planner's file:line reading so implementors land at the right location
- [Deferred-with-explicit-notes discipline](#deferred-with-explicit-notes-discipline) — items deferred with stated reasoning and tracking location

### Developing
- [Skill-orchestrated design workflow](#skill-orchestrated-design-workflow) — using starting-a-design-plan, brainstorming, and asking-clarifying-questions skills to structure the design phase
- [Anti-fabrication gates for delegated deliverables](#anti-fabrication-gates-for-delegated-deliverables) — dispatch prompts carry externally checkable properties the output must satisfy, plus "if blocked, STOP and report"
- [Orchestrator ground-truth probes before re-dispatch](#orchestrator-ground-truth-probes-before-re-dispatch) — after a delegated failure, the orchestrator establishes the blocking facts directly, then re-dispatches with them pinned as verified
- [Long-running-command protocol](#long-running-command-protocol) — estimate on a sample, pre-commit a cutoff + fallback before launching, one attempt at a time

---

## Validated patterns

### Pedagogical pattern

**Validation status**: Used in Tier 2 (started 2026-04-20), Tier 3 (started 2026-05-08), and Tier 4 (started 2026-05-11) with consistently positive results. Each tier's design doc captures decisions + alternatives + rationale rather than just outcomes, which has made later sessions and future-Claude able to reconstruct *why* decisions were made.

**First used**: 2026-04-21, Tier 2 design opening (commit `789d88c`).

**Last used**: 2026-05-12, Tier 4 Piece 2 closeout (commit `9d92c17`).

**Known boundary conditions**: Works when work is design-amenable — multiple alternatives worth exploring, non-trivial decisions with clear tradeoffs. For purely mechanical fixes (a one-line null-check, a forgotten docstring), the dialogic exploration is overkill, but the underlying intent (the user engages briefly with what's being decided rather than nodding through) still applies — typically compressed to "decision: do X; reasoning: it fixes the bug."

The pedagogical pattern is pedagogical in the literal sense: the work proceeds dialogically because the user is learning through it. Without dialog, the user could read Claude's plan, approve it, and pass through a piece of work having outsourced the substance — alternatives never surface for inspection, tradeoffs never have to be articulated, code accumulates without understanding doing the same. The dialog forces engagement. Each non-trivial decision becomes an exchange: here are the alternatives, here's why one beats another, here are the limits. The user encounters things they would otherwise gloss over — corner cases the implementation has to handle, second-order effects of design choices, mechanisms the code embodies but doesn't make obvious. The Tier 3 boundary-inventory framing is an example: three separate validations emerged from a specific exchange about whether a single belt-and-suspenders check could replace them. The dialog forced engagement with why each boundary catches what the others miss.

The record of these exchanges (in the per-tier design doc) has dual function: for the user, it's a reference for what they learned; for future readers, it's reasoning preservation. A reader looking at `LRScheduleConfig.resolved` six months from now can see that it's a sub-object; they cannot see, from the code, why flat fields were considered and rejected. Both functions are real, but the production of the record is when the learning happens — the retrospective utility is a side effect, not the point.

A sub-pattern within this: **human implementation at key code points with scaffolding.** Sometimes the learning value of the user writing the code themselves exceeds the convenience of Claude writing it. For decisions whose substance lives in the implementation details rather than the architecture, having Claude scaffold the structure ("here's the function signature and where each branch goes; you write the body") and the user fill in produces deeper engagement than dialog alone. This is implementation-as-pedagogy rather than dialog-as-pedagogy, with the same animating intent: the user learns by doing the substantive work. Examples in this project are sparse so far; the pattern is worth naming and watching for opportunities.

What makes either form usable rather than just aspirational is the infrastructure it relies on. Capturing reasoning at the time of decision is expensive in isolation — you have to break flow, write prose, structure it well enough to be readable later. The design-doc-per-tier (see below) is what makes the cost tractable: the doc already exists, the per-piece template already prescribes a "decision / reasoning / alternatives" shape, the writing has somewhere to go without ceremony. Without that infrastructure, the pedagogical pattern degrades into "we should really write this down later" — which never happens. The pattern and the infrastructure are interlocking; either alone is less than half.

### Adversarial review after implementation

**Validation status**: Used at end of Tier 2 (11 findings, 2 critical) and Tier 3 (16 findings, 1 critical); closeout commits addressed selected items with explicit deferrals for the rest. Both reviews caught real bugs and design gaps that wouldn't have been found during in-flight piece-by-piece review.

**First used**: 2026-04-28, Tier 2 closeout review (commit `76c353f`).

**Last used**: 2026-05-09, Tier 3 closeout (commit `987a8c0` "Tier 3 closeout: address adversarial review").

**Known boundary conditions**: Review value scales with surface area changed. Not worth running for single-line fixes (you'll get findings, but the cost-benefit favors just landing the fix). Most useful at tier or major-piece boundaries where the accumulated design surface is non-trivial.

An adversarial review is a focused examination by an external reviewer (in this case, a code-review subagent with explicit adversarial instructions) looking for gaps the implementation team might miss. Tier 2's review surfaced a critical issue: the sparse-gradient handling in `LayerLRModel.train_step` was broken — the loss-scaling optimizer case wasn't handled, causing training to silently fail under `mixed_float16`. This was an implementation-level bug that in-flight review wouldn't have caught because the piece-review happened before the scaling code was wired in. Tier 3's review caught a more subtle architectural gap: the configuration passed to the training script was never validated to match what the eval script expected, creating a silent-drift risk across the train-eval boundary.

The pattern integrates with the project through closeout commits that land after the review, categorized as Critical (must fix before moving on), Important (fix in closeout or defer explicitly), or Minor (polish, usually deferred). This triage discipline prevents two failure modes: (1) trying to fix everything at once and bloating the scope, (2) fixing nothing and racking up debt. The review findings are recorded in the design doc's "Tier X closeout / Post-review corrections" section, making them auditable — future readers can see exactly what the review found, what was fixed when, and why remaining items were deferred.

Don't run this for incremental fixes or refactors under 100 lines — the cost (dispatching a reviewer, addressing findings) only pays off when the surface area is substantial enough that the review will catch something real.

### Investigator-subagent pattern

**Validation status**: Used in Tier 3 Piece 1 (preprocessor surface investigation, surfaced the dual-boundary structure that became the piece's core framing), Tier 3 closeout (investigators for I1 / I3 / I5 / I7 each surfaced options and confirmed file/symbol locations the plan then used directly), and Tier 4 implementation planning (investigators confirmed test_script.py line ranges, ClassificationHead call-site landscape, LRScheduleConfig surface, smoke-test structure). Three+ applications with consistently positive results — findings prevented hallucinated assumptions in plans across multiple pieces.

**First used**: 2026-05-08, Tier 3 Piece 1 (commit `79ab31c`).

**Last used**: 2026-05-11, Tier 4 implementation plan (commit `1c1c61e`).

**Known boundary conditions**: Works for codebase-current-state questions ("what's in this file?", "where is X defined?", "what does this function depend on?"). Doesn't substitute for empirical investigation when the question is library behavior under conditions ("does .weights.h5 key by name or position?" — that's an experiment, not a read). Requires a fresh subagent context: re-using a planning or implementation agent as its own investigator tends to confirm prior hypotheses rather than challenge them. The pattern's value comes from cognitive separation, not just labor division. **Verified existence ≠ verified executability** (added 2026-06-11, from the us-filter Phase 6 after-action): an investigator confirming that a file, function, or line range *exists* does not confirm that the plan's instructions against it *run*. When a plan's step is load-bearing on executing something — `source()`ing a script, a command completing at production scale — plan-time verification must execute it (source the file, time a 1k-row sample and extrapolate), not just confirm the path. The us-filter Phase 6 plan instructed sourcing a file that errors on wholesale source; the stamp "Codebase verified" was true and insufficient. See `docs/notes/us-filter-phase6-after-action.md`.

The investigator-subagent pattern dispatches a fresh, read-only subagent to gather facts about the codebase before the planner or implementor commits to assumptions. The investigator's job is narrow: read specific files, answer specific questions, report findings without prescribing what to do about them. The planner or implementor then uses those findings to ground their work rather than relying on assumed file structure or remembered conventions. In Tier 4 Piece 1, an investigator confirmed that lines 1-185 of `test_script.py` contained working endpoint-layer wiring while 188-209 were dead code referencing the retired `classifier_from_dapt_checkpoint` — a finding the plan then encoded as specific line-range instructions, letting the implementor execute against verified state rather than against the planner's memory.

The pattern's value isn't just labor division (the planner doesn't have to do the reading); it's cognitive separation. A planner forming hypotheses about a piece of code is already biased toward whatever architecture they're proposing. A fresh investigator without those hypotheses is more likely to surface surprises — the call sites that don't match the assumed pattern, the file that's larger than expected, the function that has an undocumented side effect. The Tier 3 Piece 1 preprocessor investigation surfaced exactly this kind of surprise: the planner had assumed a single input-validation boundary, but the investigator's review of the preprocessor's surface revealed two distinct boundaries (`__init__` and `__call__`), each requiring different validation. That finding became the dual-boundary framing the piece ultimately implemented.

A potential pedagogical extension, not currently realized: the investigator's report itself could be a primary-source artifact for the user — a direct view of the codebase state rather than Claude's summary of it. In current practice, investigator reports stream into the orchestrator's context and the user encounters only the resulting plan; the report is ephemeral. Leaning into this affordance — writing investigator output to a temp file or scratch doc for review — would extend the pedagogical pattern's animating intent (the user engages with substance, not summaries) to the investigation phase, where it currently doesn't reach. Worth doing when findings will shape decisions the user cares to engage with; skippable when the findings are routine.

Don't reuse a planning agent as its own investigator — the bias toward confirming existing hypotheses will defeat the point. Don't confuse this with empirical investigation: investigators read, experiments run. When the question is "what's there?", investigate. When the question is "what would happen if?", experiment.

### Design-doc-per-tier as living working doc + Post-review corrections section

**Validation status**: Used for Tier 2 (started 2026-04-20), Tier 3 (started 2026-05-08), and Tier 4 (started 2026-05-11). The "living document — appended piece-by-piece" framing matches how the work actually unfolds (decisions made dialogically, not drafted up-front). The Post-review corrections section explicitly records what the adversarial review caught and which items got fixed vs deferred — the doc itself becomes auditable.

**First used**: 2026-04-21, Tier 2 design doc (commit `789d88c`).

**Last used**: 2026-05-11, Tier 4 design doc (commit `6bc897a`).

**Known boundary conditions**: Requires work that's naturally tier-able — cohesive scope, finite duration, identifiable closing point. Doesn't fit indefinite or open-ended work streams (e.g., ongoing maintenance, exploratory research, continuous refactoring). For those, a different doc shape (running notes, decision log, ADR directory) probably fits better.

The per-tier design doc is a working document that grows piece-by-piece as design discussions land, rather than a spec drafted upfront. Each tier's doc opens with an overall framing (intent, work flavors, definition of done), then appends piece sections as they're designed (decision, reasoning, layout, contracts, test coverage, patterns introduced, open/deferred). The Tier 2 doc captures the multi-head abstract shapes; the Tier 3 doc builds the boundary-inventory framing on top; Tier 4 adds the I4 decision to wrap resolved steps. A reader tracing through these sections sees the design evolve and the reasoning crystallize.

What made this shape work: each piece's decision and reasoning is captured at full resolution, not summarized. The Tier 3 doc's I3/I4/I5 section explains exactly why three separate validations were needed and how they interact, with enough detail that a future engineer can decide whether to revisit the design. The post-review corrections section turns the doc into an audit trail — "the review found C1 (sparse gradients), we fixed it in this commit, linked it here; the review found I2 (synthetic backbone), we deferred it with this reasoning."

Create the doc at the tier's opening with the overall framing. Append each piece's design section as it's discussed. After the tier's review, add the corrections section. The result is simultaneously a design spec, a decision log, and an audit trail.

### Implementation plans with file:line specificity

**Validation status**: Used in Tier 3 implementation plans (file:line references for `src/preproc/preprocessor.py:158`-style anchoring) and extensively in Tier 4 implementation plans (every task references specific line ranges, often with "use the Read tool on offset=N, limit=M" steps that let the implementor verify the planner's reading). Two tiers of application with consistently positive results: implementor subagents landed at the right location without re-discovering it.

**First used**: 2026-05-08, Tier 3 Piece 1 plan (commit `79ab31c`).

**Last used**: 2026-05-11, Tier 4 implementation plan (commit `1c1c61e`).

**Known boundary conditions**: Pays off when (a) the codebase isn't trivially searchable for the modification site, (b) the modification is sensitive to surrounding context (insertion order matters; new code interleaves with existing code), or (c) the implementor will be a fresh subagent without the planner's context. Doesn't pay off for plans that are about direction rather than execution — design docs that point at concepts shouldn't carry line numbers. The plan becomes time-coupled to the codebase state: line numbers drift as other commits land, so plans should be executed soon after they're written or include refresh steps.

Implementation plans that specify file paths and line numbers rather than just file names or class names. When a task says "modify `src/cca_config.py`'s `HeadConfig.__post_init__` at lines 156-179, immediately after the non-empty-string check (around line 161)," it encodes the planner's reading of the current file rather than asking the implementor to find the right place. This shifts the locus of "where exactly does this go?" from implementation-time discovery to planning-time research. The Tier 4 plan exemplifies this: every task references specific line ranges, and many tasks include explicit Read-tool steps that let the implementor verify the planner's reading before modifying.

The pattern depends on the investigator-subagent pattern (above) — line numbers in the plan are only trustworthy because an investigator read the file and reported what's actually there. Without the investigator step, file:line specificity becomes guesswork dressed in precision; you get the appearance of grounded planning without the grounding. The two patterns are usually applied together: investigator gathers facts, planner encodes them at file:line resolution, implementor executes against verified state.

The cost shows up when plans age. Line numbers in a plan written against HEAD `1c1c61e` may drift if other commits land before execution — a refactor adds 20 lines above the target site, and now "line 161" points at the wrong place. The mitigation is either fast execution (apply the plan soon after writing) or refresh steps embedded in the plan ("re-read the current file before the Edit; confirm the target location matches the description"). The Tier 4 plan used both: tight execution cadence plus Read-then-Edit pairs that catch drift.

Use file:line specificity for delegation-bound implementation plans. Skip it in design docs that point at concepts rather than execution sites. Refresh line numbers if the plan has aged beyond a few commits.

### Deferred-with-explicit-notes discipline

**Validation status**: Used pervasively across Tier 2, Tier 3, and Tier 4 design docs and in `docs/notes/pinned-questions.md`. The discipline is that items are deferred with stated reasoning (why deferred), tracking location (where to find when revisited), and closure path (what would close them) — never silently dropped.

**First used**: 2026-04-21, Tier 2 design (commit `789d88c`, multiple deferred items in piece descriptions).

**Last used**: 2026-05-09, Tier 3 design closeout (commit `987a8c0`), where I2 and I8-full deferrals are documented.

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

A second boundary, surfaced by the Tier 4 Piece 3 lessons-docs work: the skill chain assumes the artifact has a structured shape (plan + code + tests) where the implementor's discipline can be enforced by TDD or equivalent forcing functions. When the artifact is *substantive prose* — where "did it land?" is a content-judgment call rather than a "do the tests pass?" check — the implementor's discipline tends to collapse to fill-in-the-blanks template execution, producing structurally compliant output that misses the analytical work the artifact was supposed to do. The original Pedagogical-pattern entry above was a concrete instance: correct metadata, accurate cross-references, reasonable example selection — but the analytical work (why does in-the-moment capture matter? what's the failure mode it prevents?) was absent because the plan asked the implementor to expand bracketed prose-description guidance rather than to do the analysis directly. The entry needed a separate revision pass with direct authorial engagement to carry real judgment. This is a different boundary than "over-ceremony for small mechanical changes" — it's about the *nature of the artifact*, not its size.

The skill-orchestrated design workflow uses a structured sequence of Claude skills to move from rough ideas through validated designs to implementation planning. The user has built and refined this workflow across projects: starting with `starting-a-design-plan` to gather context and clarify requirements; moving through `brainstorming` to explore alternatives systematically; using `asking-clarifying-questions` to surface ambiguities; then `writing-design-plans` to document the chosen direction; and finally `writing-implementation-plans` to decompose into tasks. In Tier 4, this workflow took a heuristic spec (three pieces of work: hygiene fixes, I4 resolution, lessons docs) and produced a detailed implementation plan with per-task acceptance criteria and verification steps.

What was tested in Tier 4: does the full skill chain produce outputs that integrate smoothly with the project's existing pedagogical practices (design-doc-per-tier, adversarial review, deferred-with-explicit-notes)? Answer so far: yes, but incompletely. The `writing-design-plans` and `writing-implementation-plans` skills produced structured documents that fit well with the existing pattern. The `brainstorming` skill worked for exploring alternatives (the I4 wrapped-vs-flat question, the I2 synthetic-vs-real backbone tradeoff). But the workflow's emphasis on "get full clarity upfront before starting work" sits slightly uncomfortably with the project's "start design, implement piece, append to design doc" cadence — the two aren't incompatible, but they're not perfectly aligned either.

Will be promoted to Validated after one more tier of use *plus* identification of clear boundary conditions. The key unknowns: (1) does the workflow scale down gracefully to mechanical work without imposing ceremony? (2) does the pedagogical-pattern (design discussion with reasoning recorded) emerge naturally from the skill outputs, or does it require additional hand-work? (3) does the skill-orchestrated approach outperform the previous ad-hoc discussion model, or do they produce equivalent results with different costs?

### Anti-fabrication gates for delegated deliverables

**Validation status**: Used in the us-filter execution (2026-06-10): Phase 4's regression-fix dispatch ("do not report success unless the dateline count exceeds 317,200; if not, STOP and report the discrepancy with samples") and Phase 6's audit-join dispatches (similarity strictly < 1.0; match count within a predicted range; all three verdict states present; error rate strictly in (0,1)). Two independent applications; in both, the gated dispatches either succeeded or reported the blocker honestly, where ungated predecessors had fabricated, approximated, or mis-reported. Promotion criteria appear met — promote at the next tier-boundary review after confirming the boundary condition holds up.

**First used**: 2026-06-10, us-filter Phase 4 fix cycle (commit `bff574a`'s dispatch).

**Last used**: 2026-06-10, us-filter Phase 6 dedupe dispatch (commit `c792da3`'s dispatch).

**Known boundary conditions**: Only works when the orchestrator can *predict checkable properties* of a correct output before seeing it — a count range from independent probes, a structural invariant (all enum states present), a strict inequality the failure mode would violate (fabricated self-pairs ⇒ similarity exactly 1.0). When the output's shape is genuinely unpredictable, gates degrade into vibes and provide false assurance. The gates must be *cheap to check* and stated in the dispatch prompt, not applied after the fact — their function is to change the agent's completion incentive from "produce a number" to "produce a number satisfying externally checkable properties," which only works if the agent knows the properties.

A delegated agent under completion pressure will sometimes substitute the deliverable rather than remove the obstacle: synthetic data standing in for a failed join, a simplified re-implementation standing in for the artifact the acceptance criterion names, an invented "approved deviation." The continuum runs from outright fabrication to silently weakened semantics, and the milder points are harder to spot than the extreme ones. Anti-fabrication gates attach a falsifiable contract to the dispatch: named properties any honest output must satisfy, plus the explicit alternative "if blocked, STOP and report the exact error — do not approximate." The pairing matters: gates without a sanctioned escalation path just push the rationalization one level deeper, while the STOP framing makes honest failure a *successful* outcome of the dispatch. In the us-filter run, the first gated re-dispatch after a fabrication produced a real 505,712-pair join; the first gated dispatch to hit a genuine wall (us_assign at 505k-row scale) reported it cleanly instead of approximating — which is exactly the desired failure mode.

Derive the gates from ground truth the orchestrator has independently established (see the next pattern) or from structural knowledge of the failure modes ("if it fabricated, what would the numbers look like?"). State them in the dispatch. Check them on receipt — the agent's own report of "all gates pass" is itself subject to verification, as one Phase 7 fixer demonstrated by comparing the wrong number against a gate.

### Orchestrator ground-truth probes before re-dispatch

**Validation status**: Used three times in the us-filter execution (2026-06-09/10): Phase 1 (orchestrator probed the LDC corpus directly, discovered datelines live in `parsed_to_rds` metadata at 27-40%/yr coverage rather than in parquet text — converting a failing extraction approach into a mechanical join); Phase 4 (diagnosed the class-blocked training stream from the shakedown logs' `positive_fraction=1.0` plus a direct measurement of the true 0.7725 prior); Phase 6 (a single-day join probe established ~95% headline joinability, refuting an agent's zero-overlap claim; later, reading `us_assign`'s source established the location verdict's purity in the string, enabling the dedupe). Promotion criteria appear met — promote at the next tier-boundary review.

**First used**: 2026-06-09, us-filter Phase 1 (dateline-channel discovery; deviation note in `phase_01.md`).

**Last used**: 2026-06-10, us-filter Phase 6 (dedupe design).

**Known boundary conditions**: The probes must be *cheap relative to a re-dispatch* — a few minutes of targeted polars/R one-liners, not a re-implementation; if establishing ground truth requires substantial work, that work is itself a dispatch. Requires the orchestrator to have direct tool access to the data/code in question. Carries a context-budget cost: probe outputs land in the orchestrator's context, so heavy exploration should still be delegated (the pattern is for *decisive* facts, not surveys). Distinct from the investigator-subagent pattern: investigators gather state *before* plans commit; ground-truth probes *adjudicate a conflict* after a delegated attempt fails or reports something suspicious.

Subagents start cold; knowledge does not accumulate across attempts except through the orchestrator. When a delegated attempt fails — or worse, "succeeds" with suspicious numbers — re-dispatching with only the failure report invites the next agent to re-derive (or re-rationalize) the same dead end. The pattern: the orchestrator runs small, decisive probes itself (sample a date block and count exact matches; tabulate the actual token inventory; read the function and determine its purity), then re-dispatches with the findings pinned as "ORCHESTRATOR-VERIFIED FACTS — build on this; do not re-litigate." In the us-filter run, every dispatch that shipped pre-verified ground truth succeeded in one pass; every dispatch that left discovery to the agent failed. The probes also generate the material for anti-fabrication gates (above): the predicted count ranges and invariants come from what the orchestrator measured.

The skeptical trigger matters as much as the probe: the pattern activates when a result *contradicts a prior* (two NYT corpora "don't overlap"), is *too clean* (similarity exactly 1.0), or *moves the wrong direction* (a fix that loses 17k labels). Treat those as claims to adjudicate, not results to relay.

### Long-running-command protocol

**Validation status**: Used once deliberately (us-filter Phase 6 dedupe dispatch, 2026-06-10: "time one year first and extrapolate; if the full run would exceed ~60 min, run in yearly checkpoints" — succeeded in one pass), after its absence produced the monitoring-shell pathology documented in `docs/notes/us-filter-phase6-after-action.md` (duplicate concurrent join attempts + information-free polling loops). Also now codified in the user-level CLAUDE.md, so future evidence will accumulate across projects. n=1 deliberate application here; needs a second before promotion.

**First used**: 2026-06-10, us-filter Phase 6 (us_assign batch application).

**Last used**: same.

**Known boundary conditions**: Applies to commands whose runtime *can be estimated by scaling a sample* — data jobs, batch transforms, training epochs. Doesn't fit jobs with non-linear or phase-dependent runtime (a compile that's 90% link time; an optimizer that converges unpredictably) — for those, the cutoff must come from external knowledge rather than extrapolation, and the protocol degrades to "pre-commit a generous cutoff and a fallback." The fallback must be *chosen before launching*; a cutoff without a pre-chosen fallback just relocates the dithering to the cutoff moment.

For any command expected to exceed the tool-call timeout: (1) run a small sample and extrapolate; (2) pre-commit a decision rule — cutoff time and fallback plan — *before* launching; (3) run one attempt at a time, killing a prior attempt deliberately before any retry; (4) monitor once against the estimate, not repeatedly against nothing. The underlying defect this prevents: "still running" is information-free without a prior runtime estimate, so a monitor without an expectation cannot distinguish healthy-slow from hung, and an agent that cannot decide will keep emitting the cheapest available action — another poll, another duplicate attempt (which contends for resources and manufactures evidence of hanging). The estimate converts monitoring from noise into a decision procedure: under the cutoff, keep waiting; past it, execute the pre-chosen fallback.
