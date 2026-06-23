# Multi-Head ICA Assembly Implementation Plan — Phase 1

**Goal:** Build a verification gate that confirms (or flags) the four data/artifact pre-flight facts the rest of the assembly depends on, before any retrain.

**Architecture:** A pure Functional-Core module (`src/preflight/checks.py`) of deterministic verdict functions, plus a thin Imperative-Shell script (`scripts/preflight_assembly.py`) that gathers real out-of-repo provenance/mtimes and prints a verdict table. The gate fixes nothing — it produces a recorded verdict + remediation pointers that route to later phases.

**Tech Stack:** Python 3.12, `uv`, `polars` (parquet/row counts), stdlib `json`/`pathlib`/`os.stat`, `pytest`.

**Scope:** Phase 1 of 6 from `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`.

**Codebase verified:** 2026-06-23 (codebase-investigator).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### multi-head-ica-assembly.AC7: Pre-flight verifications (cross-cutting)
- **multi-head-ica-assembly.AC7.1:** The cached `us_logit` is confirmed produced by `us_classifier_full`, not the deprecated smoke-test weights.
- **multi-head-ica-assembly.AC7.2:** CCA and relevance calibration sidecars are confirmed present (or fit if missing).
- **multi-head-ica-assembly.AC7.3:** DoCA freshness (`doca.csv`/`tmp.R` → match → positives) is confirmed propagated, or the edit is explicitly accepted as incidental.
- **multi-head-ica-assembly.AC7.4:** The `ldc_9507` cache's `us_logit` is confirmed computed on dateline-stripped LDC text.

---

## Verified context (from investigation)

- `src/embed_corpus.py:92-124` (`provenance_record()`) writes `provenance.NNN.json` per cache with `us_weights.{path,size,mtime}`, `text_channel`, `lead_column`, `n_rows`, `stamp`.
- Current cache provenance records `us_weights.path = "us_filter/us_classifier.weights.h5"` (the SMOKE-TEST weights) for `full`, `ldc_9507`, `relevance_pos`, `us_train_ldc`. The operative training-gate weights may differ if a table-build re-scored via `--us-weights` (`run_cca_doca.py` supports this) — the check inspects the table-build invocation, not just the cache.
- `src/calibration/sidecar.py:13-19` (`calibration_path_for_weights`). Sidecars present: `cca_doca/cca_doca.calibration.json`, `cca_doca/cca_doca_street.calibration.json`. MISSING: `relevance/relevance.calibration.json`.
- DoCA chain: `../../LDC2008T19/data/cca_matches_good.rds` (mtime 2026-05-08) → `r/doca/export_cca_positives.R` → `cca_doca/cca_doca_positives.parquet` (mtime 2026-06-15, 15,614 rows). `doca.csv`/`tmp.R` edited 2026-06-17 (newer than the RDS → match potentially stale). Counts are stdout-only; mtime ordering is the signal.
- `ldc_9507` provenance: `lead_column=None` → raw `lead_paragraph`, NOT `stripped_text`. LDC has datelines, so this is a US-head channel mismatch.
- LDC dateline-derived gold `us_label` lives in `us_filter/ldc_labeled.parquet` (built by `r/dateline/build_labels.R`, LDC 1987–2007); Phase 6 will prefer it over the ML head for LDC gating. Phase 1 reports its coverage over the LDC 1996–2007 apply ids so Phase 6 can size the gold-vs-fallback split.
- Conventions: `uv run pytest` (pythonpath=["."], 618 tests passing); `ruff check` pre-commit gate; one-shot scripts live in `scripts/`.

---

## Remediation routing (Phase 1 flags; later phases fix)

- AC7.1 (US weights) → Phase 3 retrain must gate with `us_classifier_full`. (At *apply* time the API US gate is computed by the assembled `us` head from CLS via the Phase 5 `IcaModel`, which loads `us_classifier_full`; the cached `us_logit` is not consumed at apply — so AC7.1 is closed for both training and apply, and the embed-time smoke-test `us_logit` is vestigial.)
- AC7.2 (relevance calibrator) → Phase 4 fits it.
- AC7.4 (`ldc_9507` raw text) → Phase 6 re-embeds with `stripped_text`; LDC gating prefers gold `us_label`.

---

<!-- START_SUBCOMPONENT_A (tasks 1) -->
<!-- START_TASK_1 -->
### Task 1: Pre-flight verdict functions + tests

**Verifies:** multi-head-ica-assembly.AC7.1, AC7.2, AC7.3, AC7.4

**Files:**
- Create: `src/preflight/checks.py` (`# pattern: Functional Core`)
- Test: `tests/test_preflight_checks.py` (unit)

**Implementation:**
Pure functions — no I/O — taking already-loaded inputs and returning a structured verdict. Define a small frozen `Verdict` dataclass (`name: str`, `status: Literal["PASS","WARN","FAIL"]`, `detail: str`, `remediation: str | None`). Functions:
- `us_weights_verdict(cache_provenance: dict, table_build_us_weights: str | None) -> Verdict` — FAIL when the operative training-gate weights are not `us_classifier_full` (i.e. `table_build_us_weights` is None AND the cache `us_weights.path` basename is the smoke-test `us_classifier.weights.h5`); PASS when `us_classifier_full` is the operative gate; WARN when undetermined.
- `calibration_presence_verdict(present: dict[str, bool]) -> Verdict` — keys `cca`, `cca_street`, `relevance`; FAIL if a CCA sidecar is missing; WARN (not FAIL) if only `relevance` is missing (Phase 4 fixes it); PASS if all present.
- `doca_freshness_verdict(mtimes: dict[str, float | None]) -> Verdict` — keys `doca_csv`, `rds`, `positives`; WARN if `doca_csv > rds` (match stale) or `rds > positives` (positives stale); PASS if monotone `doca_csv <= rds <= positives`; missing mtime → WARN with detail.
- `ldc_channel_verdict(provenance: dict) -> Verdict` — FAIL if `lead_column != "stripped_text"`; PASS otherwise.
- `ldc_gold_coverage_verdict(n_apply_ids: int, n_with_gold_label: int) -> Verdict` — informational: always PASS, `detail` carries the coverage fraction so Phase 6 can size the gold-vs-ML-fallback split.

**Testing:**
Tests must verify each AC case via synthetic inputs (no filesystem):
- AC7.1: smoke-test-only provenance + `table_build_us_weights=None` → FAIL; `table_build_us_weights="…us_classifier_full.weights.h5"` → PASS.
- AC7.2: relevance-missing → WARN; cca-missing → FAIL; all-present → PASS.
- AC7.3: `doca_csv>rds` → WARN; monotone → PASS; a None mtime → WARN.
- AC7.4: `lead_column=None` → FAIL; `lead_column="stripped_text"` → PASS.
- Plus `ldc_gold_coverage_verdict` returns PASS with the right fraction in `detail`.

**Verification:**
Run: `uv run pytest tests/test_preflight_checks.py`
Expected: all tests pass.

**Commit:** `feat(preflight): pure pre-flight verdict functions + tests`
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2) -->
<!-- START_TASK_2 -->
### Task 2: `scripts/preflight_assembly.py` gate

**Verifies:** multi-head-ica-assembly.AC7.1, AC7.2, AC7.3, AC7.4 (operationally)

**Files:**
- Create: `scripts/preflight_assembly.py` (`# pattern: Imperative Shell`)

**Implementation:**
Gather real inputs and call the Task 1 core:
- Read the latest `provenance.*.json` from `config.CCA_EMBED_CACHE_DIR / {full, ldc_9507, relevance_train, us_train_ldc}` (handle a missing cache dir as a WARN, not a crash).
- Resolve the operative training-gate US weights: inspect the CCA/relevance run sidecars (`*.config.json`) and/or the table-build invocation for a `--us-weights`/`us_classifier_full` reference; pass to `us_weights_verdict`.
- Stat the calibration sidecars via `calibration_path_for_weights` for `cca_doca`, `cca_doca_street`, `relevance` → `present` dict.
- Stat the DoCA chain mtimes (`doca.csv`, `cca_matches_good.rds`, `cca_doca_positives.parquet`) using config/data-map paths → `doca_freshness_verdict`.
- For LDC: read `ldc_9507` provenance → `ldc_channel_verdict`; load `us_filter/ldc_labeled.parquet` and the LDC 1996–2007 apply id set, compute gold-`us_label` coverage → `ldc_gold_coverage_verdict`.
- Print a verdict table (name / status / detail / remediation); exit nonzero if any `FAIL`.

**Verification (operational — reads out-of-repo artifacts; logic tested in Task 1):**
Run: `uv run python -m scripts.preflight_assembly`
Expected: prints five verdict rows; nonzero exit reflects current known FAILs (AC7.1 embed-time, AC7.4) and the relevance-calibrator WARN. Operator records the verdict and confirms the remediation routing.

**Commit:** `feat(preflight): assembly pre-flight gate script`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_B -->
