# US/not-US Pre-Filter — Phase 7 Implementation Plan

**Goal:** Apply the settled, calibrated model to the full 1960–1995 API corpus and persist the reusable artifact triple.

**Architecture:** Reuse the Phase-6 `apply_us_model` (Pattern-2 cross-process load) to score the dateline-less API corpus, calibrate logits → `us_score`, threshold → `us`, and write a *derived* scored parquet (joinable to `api_corpus` by `id`) — consistent with the Phase-1 don't-mutate-shared-corpora decision. The durable artifact is the triple `*.weights.h5` + `*.config.json` + `*.calibration.json`; a reload check proves reproducibility.

**Tech Stack:** Python, keras, polars, pytest.

**Scope:** Phase 7 of 8.

**Codebase verified:** 2026-06-09 (`eval_cca_classifier.py` Pattern-2 apply; API schema; calibration sidecar from Phase 5).

---

## Acceptance Criteria Coverage

Implements/tests **us-filter.AC5**:

### us-filter.AC5: The artifact and output columns are produced and reusable
- **AC5.1 Success:** `us_score` (calibrated [0,1]) and `us` (boolean) columns are written to the 1960–1995 parquet.
- **AC5.2 Success:** the artifact triple (`*.weights.h5` + `*.config.json` + `*.calibration.json`) is saved; reload reproduces scores within fp tolerance.
- **AC5.3 Success:** a default `us` threshold is documented; the CCA-consumer recall-targeted threshold recipe is recorded.
- **AC5.4 Edge:** dateline-less API rows are handled consistently with the training input construction.

---

## Approach notes

- **Input construction (AC5.4):** API leads are already dateline-less. The model input is `headline + "</s>" + lead_paragraph` via `data_from_parquet(..., lead_column="lead_paragraph")` (the default) — the *same* concatenation the training path uses, just pointed at the API lead instead of `stripped_text`. No stripping, no residue guard needed on the API side (nothing to strip).
- **Derived output:** `us_filter/api_us_scores/{year}.parquet` with `id, us_score, us`. Consumers join to `api_corpus` by `id`. (Decision: derived, not in-place — reversible, doesn't mutate the shared corpus.)
- **`apply_us_model`** is built in Phase 6 Task 5: load `UsRunConfig` sidecar → fresh `us` `ClassificationHead` → `build_inference_model` → `load_weights(skip_mismatch=False)` → finite predict → `PlattCalibrator.transform` (from `.calibration.json`) → calibrated `us_score`.

---

<!-- START_TASK_1 -->
### Task 1: `src/apply_us_filter.py` — batch apply + write-back

**Verifies:** us-filter.AC5.1, AC5.4.

**Files:**
- Create: `src/apply_us_filter.py` (`# pattern: Imperative Shell`)
- Create: `tests/test_apply_us_filter.py`
- Modify: `src/config.py` (add `US_FILTER_SCORES_DIR = US_FILTER_DIR / "api_us_scores"`)

**Implementation:** `main(threshold=0.5)`:
- **Read the whole corpus once**: `df = data_from_parquet(config.PROJECT_ROOT, "api_corpus", addl_columns=["year"], lead_column="lead_paragraph")`. The glob `api_corpus/**/*.parquet` matches the flat per-year files directly under the folder (empirically verified). **Do NOT** use a `f"api_corpus/{year}"` sub-path — Phase 2 writes flat `{year}.parquet` files, so `api_corpus/{year}/**/*.parquet` matches nothing and raises `ComputeError`. (If memory is a concern over the full corpus, loop years reading each file by *exact path* `pl.scan_parquet(config.API_CORPUS_DIR / f"{year}.parquet")` and reproduce `data_from_parquet`'s null/"NA"-clean + `headline_with_lead` concat inline — but the single whole-corpus read is the default.)
- `scores = apply_us_model(df["headline_with_lead"])` → calibrated `us_score`; `us = us_score >= threshold`.
- Attach `id, us_score, us, year`; write **per-year** outputs by grouping on `year`: for each `y` in the distinct years, `df.filter(pl.col("year")==y).select(["id","us_score","us"]).write_parquet(US_FILTER_SCORES_DIR / f"{y}.parquet")`. Print per-year row counts and the US-positive fraction.
- **`id` dtype:** API `id` is `character` (str); preserve as str (no cast). Phase 7 only touches the API corpus, so no cross-dtype join occurs here (the API↔LDC id convention is pinned in Phase 6).

Reuse the finite-predict batching from `eval_cca_classifier.py` (`drop_remainder=False`, length-assert scores vs df).

**Testing** (`tests/test_apply_us_filter.py`, fake-backbone model per `test_assembly.py` + a stub calibrator): synthetic API parquet → output has `id, us_score, us`; `us_score ∈ [0,1]`; `us == (us_score >= threshold)`; row count and `id` preserved.

**Verification:** `uv run pytest tests/test_apply_us_filter.py` → pass. Operational full apply (`uv run python -m src.apply_us_filter`) is the operator step once the calibrated model is settled.

**Commit:** `feat(us-filter): batch-apply calibrated US filter to API corpus`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Artifact reload check + threshold recipe

**Verifies:** us-filter.AC5.2, AC5.3.

**Files:**
- Create: `src/validation/artifact_check.py` (or a test-only helper)
- Create: `tests/test_artifact_reload.py`
- Create/update: a short threshold-recipe note (in the artifact dir README or `docs/notes`)

**Implementation:**
- `reload_and_score(weights_path, logits_batch) -> us_scores`: loads the triple — `UsRunConfig.from_json(config_path_for_weights(weights_path))`, model + `load_weights`, `PlattCalibrator` from `calibration_path_for_weights(weights_path)` — and returns calibrated scores for a fixed input.
- Threshold docs: default `us` threshold = **0.5** on the calibrated probability. **CCA-consumer recall recipe:** to protect CCA recall, choose the largest threshold whose `doca_recall` (Phase 6) on the DoCA-matched set ≥ target (e.g. 0.98); record the chosen threshold + target alongside the artifact.

**Testing** (`tests/test_artifact_reload.py`): build a tiny model + calibrator, save the triple to `tmp_path`, reload via `reload_and_score`, assert calibrated scores match the pre-save scores within fp tolerance (AC5.2). Assert the documented default threshold (0.5) and that the recipe references `doca_recall`.

**Verification:** `uv run pytest tests/test_artifact_reload.py` → pass.

**Commit:** `feat(us-filter): artifact-triple reload check + threshold recipe`
<!-- END_TASK_2 -->

---

## Phase 7 Done When

- `us_score` + `us` materialized over 1960–1995 (derived `api_us_scores/` parquet), input built consistently with training (AC5.1/5.4).
- The artifact triple reloads and reproduces scores within fp tolerance (AC5.2).
- A default threshold + CCA-consumer recall recipe are documented (AC5.3).

Covers **us-filter.AC5**. The full real-corpus apply is operator-invoked once the calibrated model is settled; the synthetic-data tests verify the path.
