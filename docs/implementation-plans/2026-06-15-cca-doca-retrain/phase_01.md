# Phase 1: DoCA-labeled, US-restricted training table

**Goal:** Turn the embedding cache into a labeled, US-restricted PU training table — DoCA-confirmed
positives vs. US-restricted unlabeled — split deterministically for training.

**Codebase verified:** 2026-06-15.

---

## Acceptance Criteria Coverage

### cca-doca.AC1: Training table
- **cca-doca.AC1.1:** An API article whose `id` is in `cca_matches_good` (succeeded) is labeled
  positive; one that is not is unlabeled.
- **cca-doca.AC1.2:** After US restriction, the unlabeled pool contains only articles scoring US at
  the chosen threshold; positives (confirmed US via DoCA) are retained.
- **cca-doca.AC1.3:** `create_cca_doca_data` produces disjoint train/val/test splits (unique `id`, no
  leakage), PU-separated, deterministic under seed 200.

---

## Design notes
- **Positives via R export, then Python join.** DoCA matches are an RDS; follow the project pattern
  (R produces derived parquets, e.g., `r/dateline/build_labels.R` → `ldc_labeled.parquet`) rather than
  adding a Python RDS reader.
- **US restriction = unlabeled-only.** Positives are confirmed US (DoCA codes US collective action);
  do NOT drop a positive because the US model scores it low. Restrict only the *unlabeled* pool to
  `us_logit >= threshold`. Default threshold: logit `0.0` (prob 0.5), since the Platt calibration
  sidecar is missing — note this and revisit if we fit calibration. This is the defensible reading of
  the "apply US model, restrict to US" decision; record it in the design doc if it diverges.
- The embedding cache (Phase 0) is the join hub: it carries `id`, `year`, `us_logit`, and a row index
  into the CLS matrix. Labels and splits operate on a polars frame of
  `(id, year, us_logit, cca_label, emb_row)`, and feature arrays are gathered by `emb_row` per split.

---

<!-- START_TASK_1 -->
### Task 1: Export DoCA positives to parquet (R)

**Verifies:** cca-doca.AC1.1 (positive set)

**Files:** Create `r/doca/export_cca_positives.R`

**Implementation:** Read `config`-equivalent path to `cca_matches_good.rds`; filter
`match_quality == "succeeded"`; take **unique `article_id`**; write a parquet
`cca_doca_positives.parquet` with column `id` (= `article_id`) and optionally aggregated event-form
flags (`street`/`lawsuit`/`conventional`/`boycott` any-true) for later use. Print unique count
(expect ~13–14k of the 15,627 once failures dropped). Write under a new `cca/` data dir or alongside
`us_filter/` (pick and record in `src/config.py`).

**Verification:** `Rscript r/doca/export_cca_positives.R` then inspect row count + that all `id` match
the `nyt://article/...` form. Spot-check a few ids exist in `api_corpus`.

**Commit:** `feat(cca-doca): export DoCA-confirmed CCA positives to parquet`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Build the labeled, US-restricted table + split function

**Verifies:** cca-doca.AC1.1, cca-doca.AC1.2, cca-doca.AC1.3

**Files:**
- Modify: `src/data_setup/data.py` (add `create_cca_doca_data`, mirroring `create_us_filter_data:145-177`)
- Create: `src/build_cca_doca_table.py` (Imperative Shell) assembling the table from the cache + positives

**Implementation:**
- `build_cca_doca_table.py`: load the embedding-cache sidecar parquet(s) (`id`, `year`, `us_logit`,
  `emb_row`); left-join `cca_doca_positives.parquet` → `cca_label = id ∈ positives`; compute
  `us = us_logit >= threshold`. Persist the joined table (parquet) for reproducibility.
- `create_cca_doca_data(table, threshold=0.0)`: assert unique `id`; positives = `cca_label==1`
  (kept regardless of `us`); unlabeled = `cca_label==0 & us`. Split each group 90/5/5 with the
  `_split` helper pattern from `create_us_filter_data` (seed 200), within-split shuffle (seed 200).
  Return `{"train":{"pos","unl"}, "val":{...}, "test":{...}}` of polars frames carrying `emb_row`.
  Print pos/unl counts per split (mirrors `create_classifier_data:117-125`).

**Testing:** `tests/test_data_splits.py` — add cca-doca cases: synthetic table → splits are disjoint
by `id` (AC1.3), positives retained under US restriction while non-US unlabeled dropped (AC1.2),
labeling correct (AC1.1), deterministic under seed 200.

**Verification:** `uv run pytest tests/test_data_splits.py -q`. Expected: pass. Then run
`build_cca_doca_table.py` on the 250k cache; eyeball counts (positives ~13–14k pre-split; unlabeled =
US-restricted sample).

**Commit:** `feat(cca-doca): DoCA-labeled US-restricted PU split`
<!-- END_TASK_2 -->

---

## Phase 1 done when
- Positives parquet exists; the joined table builds from the cache; counts look sane.
- `create_cca_doca_data` enforces unique-id + no-leakage + PU separation; tests pass; `ruff` clean.
