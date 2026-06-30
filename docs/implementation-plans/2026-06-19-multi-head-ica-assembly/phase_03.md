# Multi-Head ICA Assembly Implementation Plan — Phase 3

**Goal:** Retrain the CCA and relevance heads on a harmonized population (same fused US gate on the background, same `us_classifier_full` weights, same held-out clean-ICA id set excluded from both), with a runtime leakage guard, and rename the relevance head `cca`→`rel`.

**Architecture:** Extract the fused-US-gate logic into a shared pure helper both training paths call; add a `us_classifier_full` config const; add a runtime leakage-guard assertion; wire the fused gate into the CCA path; rename the relevance head; then execute both features-mode retrains with the Phase-2 holdout id set.

**Tech Stack:** Python 3.12, `uv`, `polars`, Keras 3 + TF (features-mode), `pytest`.

**Scope:** Phase 3 of 6 from `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`. Depends on Phase 2 (`holdout_ids.parquet`).

**Codebase verified:** 2026-06-23 (codebase-investigator).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### multi-head-ica-assembly.AC1: Harmonized retrain on a shared population
- **multi-head-ica-assembly.AC1.1 Success:** CCA and relevance heads are retrained from a single harmonized table that applies the *same* fused US gate (`us_location` + ML filter) to both.
- **multi-head-ica-assembly.AC1.2 Success:** The clean ICA eval ids are absent from both heads' train *and* val pools (verified: `eval_ids ∩ (train ∪ val) = ∅` for each head).
- **multi-head-ica-assembly.AC1.3 Edge:** ~30% of the 466 anchors are reserved into the eval slice and excluded from training positives in both heads.
- **multi-head-ica-assembly.AC1.4 Failure:** If any eval id leaks into either training pool, the retrain aborts rather than training on it.

---

## Verified context (from investigation)

- CCA gates dateline-only `us_logit >= threshold` (`src/build_cca_doca_table.py:24-35`); does NOT apply the fused gate. Relevance applies it inline (`src/run_relevance.py:103-116`).
- CCA `_rescore_us_restriction` recomputes `us_logit` from `--us-weights` + Platt (`run_cca_doca.py:82-98`). Relevance's `relevance_train` cache already used `us_classifier_full` (`build_relevance_table.py:34,101`).
- `compute_location_signals` is pure (`src/preproc/us_location.py:123-167`); the fused boolean is inline in run_relevance; no shared helper.
- Holdout is a filter, no assertion: `create_cca_doca_data:219`, `create_relevance_data:267` (`src/data_setup/data.py`). Split tests: `tests/test_cca_doca_data.py:81-94,112-130`. No relevance split test.
- `us_classifier_full.weights.h5` hardcoded in `run_us_features.py:63`, `calibrate_us_filter.py:46`, `build_relevance_table.py:34`; no config const. Smoke-test const is `config.US_FILTER_CLASSIFIER_WEIGHTS` (`config.py:117`).
- `HeadConfig` rejects names containing `/` only (`cca_config.py:175-182`); `"rel"` is valid. Pattern-2 loads by structure, so the rename does not break loading existing weights.
- Seed `keras.utils.set_random_seed(200)`; splits seed=200. Features-mode retrain ≈ minutes.

---

<!-- START_SUBCOMPONENT_A (tasks 1) -->
<!-- START_TASK_1 -->
### Task 1: Shared fused-US-gate helper

**Verifies:** multi-head-ica-assembly.AC1.1

**Files:**
- Modify: `src/preproc/us_location.py` — add pure `apply_fused_us_gate(table)` (`# pattern: Functional Core`) and a shared `load_location_signals(ids)` shell.
- Modify: `src/run_relevance.py` — replace the inline fused-gate + `_load_location_signals` with the shared helpers (behavior-preserving).
- Test: `tests/test_us_location.py` (extend)

**Implementation:** `apply_fused_us_gate(table)` returns the table with `us = us & ~(any_not_us & ~any_us)` (requires `us`, `any_us`, `any_not_us` columns). `load_location_signals(ids)` reads API-corpus rows for `ids` and calls `compute_location_signals`.

**Testing:** truth-table cases — US-only kept; clearly-foreign (`any_not_us ∧ ¬any_us`) dropped; diaspora (US location present) kept. Confirm relevance still produces the same gated counts on a fixture.

**Verification:** `uv run pytest tests/test_us_location.py` — all pass.

**Commit:** `refactor(us-gate): shared fused-US-gate helper for both heads`
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2-3) -->
<!-- START_TASK_2 -->
### Task 2: `US_FILTER_FULL_WEIGHTS` config const

**Verifies:** None (infrastructure)

**Files:**
- Modify: `src/config.py` — add `US_FILTER_FULL_WEIGHTS = US_FILTER_DIR / "us_classifier_full.weights.h5"`.
- Modify: `src/run_us_features.py`, `src/calibrate_us_filter.py`, `src/build_relevance_table.py` — use the const.

**Verification (operational):** `uv run python -c "import src.config as c; print(c.US_FILTER_FULL_WEIGHTS)"` prints the path; no remaining hardcoded literal (grep).

**Commit:** `chore(config): add US_FILTER_FULL_WEIGHTS const`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Runtime leakage-guard assertion

**Verifies:** multi-head-ica-assembly.AC1.4

**Files:**
- Modify: `src/data_setup/data.py` — add `assert_holdout_excluded(splits, holdout_ids)`.
- Test: `tests/test_data_splits.py` (extend) + a relevance-split mirror in `tests/test_cca_doca_data.py`.

**Implementation:** raises `ValueError` enumerating offending ids if any holdout id appears in any train/val pool (pos/neg/unl) for either head's split dict; no-op when clean.

**Testing:** planted-leak split → raises with the offending id; clean split → passes; cover both CCA (pos/unl) and relevance (pos/neg/unl) split shapes.

**Verification:** `uv run pytest tests/test_data_splits.py tests/test_cca_doca_data.py` — all pass.

**Commit:** `feat(data): runtime leakage-guard assertion for holdout ids`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 4-5) -->
<!-- START_TASK_4 -->
### Task 4: Apply the fused gate in the CCA path

**Verifies:** multi-head-ica-assembly.AC1.1

**Files:**
- Modify: `src/run_cca_doca.py` — after `label_and_restrict`, load location signals and `apply_fused_us_gate` to gate the unlabeled/background pool (DoCA positives kept by construction); call `assert_holdout_excluded` after split.
- Test: `tests/test_cca_doca_data.py` (extend)

**Implementation:** mirror relevance's background gating; canonical invocation passes `--us-weights=<US_FILTER_FULL_WEIGHTS>` so `us_logit` is the full head. Positive-handling unchanged (all DoCA positives retained).

**Testing:** with synthetic location signals, the CCA unlabeled pool drops clearly-foreign rows while positives are retained.

**Verification:** `uv run pytest tests/test_cca_doca_data.py` — all pass.

**Commit:** `feat(cca): apply fused US gate to CCA background (harmonized)`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Rename relevance head `cca` → `rel`

**Verifies:** (supports AC1.1; load-bearing for Phase 5 assembly head-dict key)

**Files:**
- Modify: `src/run_relevance.py` — construct the `HeadConfig`/`ClassificationHead` with `name="rel"`, so the retrained sidecar records `rel`.
- Test: `tests/test_run_relevance.py` (new or extend) — focused sidecar/name + Pattern-2 load test.

**Implementation:** override the `RunConfig`'s `HeadConfig.name` to `"rel"` via `dataclasses.replace` (the existing replace pattern at `run_relevance.py:86`), so BOTH the live `ClassificationHead` and the serialized sidecar record `"rel"`. Passing `name="rel"` only to the `ClassificationHead` constructor would desync the head from the sidecar (which serializes the `RunConfig`), and Phase 5 reconstructs the head by the sidecar-recorded name — reintroducing the `cca` collision. Verify a fresh `rel`-named head loads existing `relevance.weights.h5` by structure (Pattern 2).

**Testing:** retrained config sidecar has `head.name == "rel"`; Pattern-2 load of prior weights into a `rel` head succeeds (no shape/name error).

**Verification:** `uv run pytest tests/test_run_relevance.py` — all pass.

**Commit:** `feat(relevance): rename head cca→rel for multi-head assembly`
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_C -->

<!-- START_SUBCOMPONENT_D (tasks 6) -->
<!-- START_TASK_6 -->
### Task 6: Execute the harmonized retrains

**Verifies:** multi-head-ica-assembly.AC1.1, AC1.2, AC1.3, AC1.4 (operationally)

**Files:** none (runs existing entrypoints; operator commands documented)

**Implementation:** run both retrains with the Phase-2 `holdout_ids.parquet`:
- CCA: `uv run python -m src.run_cca_doca --prior 0.02 --threshold 0.5 --us-weights <US_FILTER_FULL_WEIGHTS> --holdout-ids <PLAN data>/holdout_ids.parquet` (both tracks: all-forms and `--form-filter` street).
- Relevance: `uv run python -m src.run_relevance --prior 0.05 --holdout-ids <same holdout_ids.parquet>`.
The Task-3 assertion runs inside both (AC1.4).

**Verification (operational):** both complete in minutes; the leakage assertion passes (would abort on leak); sidecars written; relevance head named `rel`; printed test-set logit separation is sane vs the prior runs. Operator records counts.

**Commit:** `chore(retrain): harmonized CCA+relevance retrain with clean holdout`
<!-- END_TASK_6 -->
<!-- END_SUBCOMPONENT_D -->
