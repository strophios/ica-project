# US/not-US Pre-Filter — Phase 8 Implementation Plan

**Goal:** Excellent, durable documentation of the project's first calibration work — `docs/notes/calibration-notes.md`.

**Architecture:** A single documentation deliverable produced through a **dialogic drafting session** between the user and the assistant. This is deliberately *not* an auto-generated artifact.

**Tech Stack:** Markdown prose (`docs/notes/` convention); the `writing-for-a-technical-audience` skill.

**Scope:** Phase 8 of 8.

**Codebase verified:** 2026-06-09 (`docs/notes/` convention is freshness-dated, reasoning-over-recipe; `calibration-notes.md` does not exist yet).

---

## Acceptance Criteria Coverage

Implements/satisfies **us-filter.AC7.2**:

### us-filter.AC7.2 Documentation
`docs/notes/calibration-notes.md` exists, was developed dialogically with the user, and is user-reviewed.

---

## ⚠️ Execution constraint — DO NOT AUTO-GENERATE

**This phase is human-in-the-loop.** The autonomous execution flow (task-implementor subagents) MUST NOT write `calibration-notes.md` solo. The user has explicitly deferred a conceptual calibration walkthrough to this drafting session as an iterative-development opportunity (design doc "Process note", and AC7.2).

When execution reaches Phase 8, the executor must **stop and surface to the user**:

> "Phase 8 is the dialogic calibration-notes session. It is not auto-generated — please pick it up live with the assistant so we can draft `docs/notes/calibration-notes.md` together."

Do not mark Phase 8 complete via an autonomous subagent. It completes only after the live session + user review.

---

<!-- START_TASK_1 -->
### Task 1: Conduct the dialogic drafting session → `docs/notes/calibration-notes.md`

**Files:**
- Create: `docs/notes/calibration-notes.md` (during the live session, not before)

**Session scaffold — concepts to develop *with* the user:**

1. **Platt scaling `σ(A·logit + B)`**
   - What `A` does (slope — sharpens/softens the logit) and what `B` does (intercept — absorbs class-balance skew).
   - Why Platt is strictly more flexible than temperature scaling for a single logit (temperature has no intercept).
   - Why isotonic regression was ruled out (needs ~1000+/class; unstable under imbalance).

2. **What ECE measures**
   - Predicted confidence vs observed frequency; equal-width binning; the weighted `|acc − conf|` average.
   - Relationship to the Brier score (proper scoring rule) and the reliability diagram.

3. **The imbalance / distribution interaction (load-bearing)**
   - Calibrated probabilities are calibrated *to a distribution*.
   - The labeled-val natural balance is a *proxy* for the dateline-less deployment distribution — an acknowledged, measured gap (tie to Phase 6's proxy-gap diagnostic).
   - Why fitting on natural-balance val (never the rebalanced training batches) is required — rebalancing skews `B` and degrades calibration.

4. **Cross-references**
   - The implemented `src/calibration/` module (`PlattCalibrator`, `calibration_report`, the `.calibration.json` sidecar).
   - The `fit_population` field as the record of *what distribution* the scores are calibrated to.

**Form:** `docs/notes/` convention — freshness-date header (e.g. `*Last updated: YYYY-MM-DD.*`), reasoning-over-recipe, research-scientist register. Apply the `writing-for-a-technical-audience` skill while drafting (avoid AI-writing tells; lead with what each concept *does*).

**Done when:** the notes exist, were developed dialogically (not auto-generated), cover the rationale + the load-bearing distribution concept, and are **user-reviewed**.
<!-- END_TASK_1 -->

---

## Phase 8 Done When

`docs/notes/calibration-notes.md` exists, developed dialogically with the user and user-reviewed, covering Platt's `A·logit+B`, what ECE measures, and the imbalance/distribution interaction.

Covers **us-filter.AC7.2**.

---

## Note on the remaining AC7 cross-cutting criteria

- **AC7.1 (scope):** satisfied structurally across the plan — the conservative R heuristic productionization and pre-1960/post-1995 application are explicitly out of scope (see the design's "Out of scope"); no task performs them.
- **AC7.3 (operational shakedown, local + cluster):** the local short-run shakedown is Phase 4 Task 5; the cluster `mixed_float16` shakedown is an operator-run step (flagged in Phase 4). These are operational gates, not documentation, and are tracked there.
