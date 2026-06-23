# Multi-Head ICA Assembly Implementation Plan — Phase 6

**Goal:** Apply the assembled `IcaModel` over the corpora — `api_corpus` 1960–1995 (in-period) and LDC 1996–2007 (out-of-DoCA expansion test) — producing per-head score outputs and ranked `ica_candidates`, with LDC gated gold-first on dateline `us_label`.

**Architecture:** Two embed re-runs prepare the inputs (finish the `full` API cache 1976–1995; build a `stripped_text` LDC 1996–2007 cache via a join). A pure `gold_first_us_gate` chooses gold dateline labels over the ML head where available. `src/apply_ica.py` (Imperative Shell) loads `IcaModel`, scores both caches features-mode, and writes the score dirs + ranked candidates.

**Tech Stack:** Python 3.12, `uv`, `polars`, Keras 3 + TF (embedding re-runs are backbone forward passes — GPU/cluster-scale), `pytest`.

**Scope:** Phase 6 of 6 from `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`. Depends on Phase 5 (`IcaModel`) and Phase 3 `US_FILTER_FULL_WEIGHTS`.

**Codebase verified:** 2026-06-23 (codebase-investigator).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### multi-head-ica-assembly.AC6: Apply and dataset expansion
- **multi-head-ica-assembly.AC6.1 Success:** Scoring `api_corpus` (1960–1995) writes `us_filter/api_us_scores/` + `cca_doca/api_cca_scores/` and a ranked `ica_candidates` parquet.
- **multi-head-ica-assembly.AC6.2 Success:** Scoring LDC 1996–2007 (`ldc_9507` cache) yields out-of-DoCA-period ICA candidates.
- **multi-head-ica-assembly.AC6.3 Edge:** LDC US gating prefers the gold dateline-derived `us_label` where available, ML US head (on the `stripped_text` re-embed) as fallback; CCA/relevance scoring uses the `stripped_text` re-embed so the dateline does not leak into those heads' inputs.

---

## Verified context (from investigation)

- Apply precedent `src/apply_us_filter.py:20-87` (per-year write; `config.US_FILTER_SCORES_DIR = us_filter/api_us_scores`). CCA scoring `src/score_cca_doca.py:29-77` (`scored_candidates.parquet` + `face_validity_*.csv`). `config.CCA_DOCA_SCORES_DIR = cca_doca/api_cca_scores` exists; **no ICA-candidates const**.
- `embed_corpus.py:347-392` CLI: `--years`, `--append`, `--corpus`, `--year-column`, `--lead-column`, `--source-pattern`, `--out-suffix`, `--stamp`, `--us-weights`. `full` part-1 = 1960–1975; part-2 = 1976–1995 via `--append`.
- `stripped_text` lives in `config.US_FILTER_LABELED_PARQUET` (`ldc_labeled.parquet`: `id`, `us_label` nullable, `label_source`, `stripped_text`; **no year**; 1987–2007). `ldc_corpus` is hive-partitioned on `publication_year`. So the LDC stripped cache needs a join (`ldc_corpus[1996-2007]` × `ldc_labeled`).
- LDC year select precedent: `cca_oos_eval.py:37-39` (`scan_parquet(..., hive_partitioning=True)`).
- Greedy-glob bug already fixed (`data_from_parquet` `pattern=`); `run_us_classification.py` is NOT on the apply path → nothing to fold in.
- Tests: `tests/test_apply_us_filter.py:111-270` (output schema, score range, threshold semantics); `tests/test_embed_corpus.py:67-92` (cache round-trip). FCIS: Imperative Shell apply.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: `ICA_CANDIDATES_DIR` config const

**Verifies:** None (infrastructure)

**Files:**
- Modify: `src/config.py` (add `ICA_CANDIDATES_DIR: Path = CCA_DOCA_DIR / "ica_candidates"`; apply writes per-corpus files under it: `api_1960_1995.parquet` and `ldc_1996_2007.parquet`, so the two corpora's candidates do not collide)

**Verification (operational):** `uv run python -c "import src.config as c; print(c.ICA_CANDIDATES_DIR)"` prints the path.

**Commit:** `chore(config): add ICA_CANDIDATES_DIR`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Embed re-runs (operational, compute-heavy)

**Verifies:** multi-head-ica-assembly.AC6.1, AC6.3 (input preparation)

**Files:**
- Create: `scripts/build_stripped_ldc_source.py` (`# pattern: Imperative Shell`) — join `ldc_corpus[1996-2007]` to `ldc_labeled[id, stripped_text, us_label]`, write a single source parquet.
- Operator embed commands documented.

**Implementation:**
- Finish `full` (API 1976–1995): `uv run python -m src.embed_corpus --full --years 1976-1995 --append --corpus api_corpus --out-suffix full --stamp <YYYYMMDD>`.
- Build the stripped LDC source (Task script), then embed: `uv run python -m src.embed_corpus --full --source-pattern <stripped_source.parquet> --lead-column stripped_text --no-year --out-suffix ldc_9607_stripped --stamp <YYYYMMDD>` (smoke-test with `--limit` first). The new suffix `ldc_9607_stripped` (1996–2007, dateline-stripped) is distinct from the existing raw `ldc_9507` cache — record the rename in the artifact map / data-map note.

**Note (cached `us_logit` is vestigial at apply):** `embed_corpus.py` has no `--us-weights` flag — the embed-time `us_logit` is hardcoded to the smoke-test weights. That is harmless here: the apply-time US gate is computed by the assembled `us` head from CLS features (Phase 5 `IcaModel`, which loads `us_classifier_full`), and the cached `us_logit` is NOT consumed at apply. So no re-embed-with-full-weights is required, and no `--us-weights` flag is passed.

**Verification (operational):** both caches exist; provenance records `lead_column=stripped_text` (LDC) and the full-head US weights; row counts sane vs the source.

**Commit:** `chore(embed): full part-2 + stripped LDC 1996-2007 caches`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Gold-first US gate helper

**Verifies:** multi-head-ica-assembly.AC6.3

**Files:**
- Modify: `src/preproc/us_location.py` (add pure `gold_first_us_gate(gold_label, ml_pass)`)
- Test: `tests/test_us_location.py` (extend)

**Implementation:** pure — elementwise: use `gold_label` where non-null, else `ml_pass`; also return the gold-coverage fraction.

**Testing:** gold present ⟹ gold overrides ML (both True→False gold wins and vice versa); gold null ⟹ ML fallback; coverage fraction correct.

**Verification:** `uv run pytest tests/test_us_location.py` — all pass.

**Commit:** `feat(us-gate): gold-first US gate for LDC apply`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: `src/apply_ica.py` + output-schema test

**Verifies:** multi-head-ica-assembly.AC6.1, AC6.2, AC6.3

**Files:**
- Create: `src/apply_ica.py` (`# pattern: Imperative Shell`)
- Test: `tests/test_apply_ica.py` (unit — output schema + LDC gating)

**Implementation:**
- Load `IcaModel`.
- **API path:** score the `full` cache features → per-head calibrated probs + composed ICA. The US gate is the assembled `us` head's calibrated score from CLS (NOT the cached `us_logit`). Write per-year `us_filter/api_us_scores/` and `cca_doca/api_cca_scores/`, and a ranked `ICA_CANDIDATES_DIR/api_1960_1995.parquet` (`id, year, us_score, cca_score, rel_score, ica_score, gated`). Optional `face_validity_*.csv` (mirror `score_cca_doca`).
- **LDC path:** score the `ldc_9607_stripped` cache; gate via `gold_first_us_gate` (join `ldc_labeled.us_label`), ML fallback (assembled `us` head on stripped CLS) otherwise; write `ICA_CANDIDATES_DIR/ldc_1996_2007.parquet`; log the gold-vs-fallback split.

**Testing:** output schema/dtypes + score ranges [0,1] on a fixture (mirror `test_apply_us_filter.py`); a fixture row with gold `us_label` has its gate decided by the gold label, not the ML score.

**Verification (operational):** runs on both caches → score dirs + ranked candidates with the right schema; gold-vs-fallback split logged for LDC.

**Commit:** `feat(apply): multi-head ICA apply over API + LDC corpora`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
