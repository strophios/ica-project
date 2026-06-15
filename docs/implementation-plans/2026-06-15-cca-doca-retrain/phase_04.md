# Phase 4: Gold set + CCA eval tooling

**Goal:** Produce a score-stratified CCA coding template (target 3,000, MVP floor 500) and the CCA
slice-eval tooling that turns hand-coded labels into precision/recall/F1.

**Codebase verified:** 2026-06-15 (validation tooling to re-read at execution:
`src/validation/build_coding_template.py`, `schema.py`, `slice_eval.py`, `doca_recall.py`).

---

## Acceptance Criteria Coverage
### cca-doca.AC4: Eval
- **cca-doca.AC4.1:** The coding template is `validate_gold_set`-conformant with null CCA labels and a
  prefix-stratified ordering; the first 500 rows are themselves approximately stratified.
- **cca-doca.AC4.2:** `evaluate_cca_slice` computes precision/recall/F1 at a threshold from a coded
  set, dropping null-label rows.

---

## Approach (lean — refine at execution)
- **Score the candidate pool.** Apply the trained CCA head (Phase 3) over a candidate set's cached
  embeddings to get `cca_score`. Candidates: US-restricted articles (and the existing `cca_event`
  column in `schema.py` is the label to be filled).
- **Score-stratified, prefix-stratified template.** Extend `build_coding_template.py` (or a CCA sibling)
  to sample by era × news_desk × **cca_score band** so the coded set has enough predicted-positives
  for precision and enough coverage for recall. Emit rows in a round-robin-over-strata order so any
  prefix (esp. the first 500) is itself approximately stratified. Target 3,000; `validate_gold_set`-conformant,
  `cca_event` null. Draw **once**.
- **CCA slice eval.** Add `apply_cca_model` (mirror `slice_eval.py:apply_us_model`, but features-mode
  on cached embeddings + the CCA weights/sidecar) and `evaluate_cca_slice` (mirror `evaluate_slice`,
  using `cca_event` + `cca_score`, dropping null labels). Reuse `doca_recall` as a secondary
  (topic-skewed) diagnostic over DoCA-matched rows.

## Vigilance
- Confirm the prefix-stratification property empirically (the first 500 rows' strata distribution
  ≈ the full 3,000's). Sanity-check score bands are populated (not all rows in one band).

## Tasks (to detail at execution)
1. `apply_cca_model` over cached embeddings → `cca_score`. **Verifies:** supports AC4.2.
2. Score-stratified, prefix-stratified CCA coding template (target 3,000). **Verifies:** AC4.1.
3. `evaluate_cca_slice` (P/R/F1 at threshold) + `doca_recall` wiring. **Verifies:** AC4.2.

## Human step (off the code critical path)
Hand-code the first ≥500 rows → run `evaluate_cca_slice` → the MVP precision/recall/F1 numbers for
the meeting. Coding more later walks down the same drawn list (no re-draw).

## Phase 4 done when
- Template drawn (schema-valid, prefix-stratified); `apply_cca_model` + `evaluate_cca_slice` exist
  with tests; `uv run pytest` green; `ruff` clean. (Metrics await ≥500 coded — human.)
