# Multi-Head ICA Assembly Implementation Plan — Phase 4

**Goal:** Calibrate all three heads, pick the US gate threshold from the recall recipe, select the fusion combiner (calibrated-AND vs ≤3-param LR) on the clean eval set under a pre-registered 1-SE margin rule, validate the composed score's calibration, and persist a declarative `fusion.json`.

**Architecture:** A new `calibrate_relevance.py` (mirrors `calibrate_us_filter.py`); a new `src/fusion/` module (FCIS: pure `combiner.py` + `FusionConfig`, `sidecar.py` for `fusion.json` I/O); a `pick_us_threshold` helper on the recall recipe; and an Imperative-Shell `src/fit_fusion.py` that scores the clean eval, runs the AND-vs-LR CV with the 1-SE rule, checks composed calibration, and writes `fusion.json`.

**Tech Stack:** Python 3.12, `uv`, `polars`, `numpy`, `scikit-learn>=1.7.2` (`LogisticRegression`, `StratifiedKFold`), Keras 3 + TF (features-mode inference), `pytest`.

**Scope:** Phase 4 of 6 from `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`. Depends on Phase 2 (clean eval set + `apply_relevance_model`) and Phase 3 (retrained `rel`/CCA heads).

**Codebase verified:** 2026-06-23 (codebase-investigator).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### multi-head-ica-assembly.AC3: Per-head and composed calibration
- **multi-head-ica-assembly.AC3.1 Success:** Each head has a Platt calibrator fit on natural-balance data; all three `*.calibration.json` sidecars are present with A/B recorded.
- **multi-head-ica-assembly.AC3.2 Success:** The gate threshold `τ_us` is the largest threshold meeting the target anchor/DoCA recall (the recall recipe).
- **multi-head-ica-assembly.AC3.3 Success:** The composed ICA score's calibration is reported (reliability / ECE / Brier) on the clean eval set; a final 2-param Platt is fit if mis-calibrated.

### multi-head-ica-assembly.AC4: Empirical fusion selection
- **multi-head-ica-assembly.AC4.1 Success:** Calibrated-AND baseline and the ≤3-param logistic challenger are both evaluated by cross-validation on the clean eval set.
- **multi-head-ica-assembly.AC4.2 Success:** The chosen combiner is recorded in `fusion.json` (gate threshold, calibrator refs, combine rule, score space).
- **multi-head-ica-assembly.AC4.3 Decision:** The LR ships only if it beats AND by more than the pre-registered CV-noise margin; otherwise the parameter-free AND ships.

**Pre-registered margin rule (AC4.3):** LR is selected iff the cross-validated mean of (LR − AND) on PR-AUC exceeds **one standard error** of the paired CV difference; otherwise AND. Fixed before viewing results.

---

## Verified context (from investigation)

- `src/calibration/calibrator.py:14,30,58` (`platt_fit`, `platt_transform`, `PlattCalibrator.fit(…, fit_population=, sample_weight=)`); `report.py:9` (`calibration_report → {ece,brier,reliability}`); `sidecar.py:13,22,28` (`calibration_path_for_weights`, `save/load_calibration`). Payload `{method,A,B,fit_population,n}`.
- `src/calibrate_us_filter.py:43-76` — natural-balance val fit pattern to mirror. `src/calibrate_cca.py` — IPW-weighted gold variant (reference).
- `src/validation/doca_recall.py:39` (`doca_recall(scored_df{doca_id,us_score}, threshold) → {recall,n}`); recipe `docs/notes/us-filter-threshold-recipe.md`. No programmatic threshold-pick.
- No fusion module exists. `scikit-learn>=1.7.2` (`pyproject.toml:16`), already imported in `calibrator.py:11`.
- `embed_corpus.load_cache_meta`/`load_cache` give `id`→`emb_row`→`cls[emb_row]`. `apply_cca_model` (`cca_slice_eval.py:28-55`), `apply_relevance_model` (Phase 2), `apply_us_model` (`slice_eval.py:28-101`).
- Tests to mirror: `tests/test_calibration_{platt,report,sidecar}.py`. FCIS: pure combiner math vs I/O/CV shell.

---

<!-- START_SUBCOMPONENT_A (tasks 1) -->
<!-- START_TASK_1 -->
### Task 1: `src/calibrate_relevance.py`

**Verifies:** multi-head-ica-assembly.AC3.1

**Files:**
- Create: `src/calibrate_relevance.py` (`# pattern: Imperative Shell`, mirror `calibrate_us_filter.py`)

**Implementation:** load the `relevance_train` cache, take `create_relevance_data(meta)["val"]` (natural-balance, seed=200), apply the retrained `rel` head to val features, `PlattCalibrator.fit(val_logits, val_y, fit_population="relevance_train_val_natural_balance")`, report before/after ECE/Brier via `calibration_report`, `save_calibration(cal, calibration_path_for_weights(RELEVANCE_WEIGHTS))`.

**Verification (operational):** `uv run python -m src.calibrate_relevance` runs; `relevance/relevance.calibration.json` exists with A/B; printed ECE drops or holds.

**Commit:** `feat(calibration): relevance-head Platt calibrator`
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2-3) -->
<!-- START_TASK_2 -->
### Task 2: `src/fusion/combiner.py` + `FusionConfig`

**Verifies:** multi-head-ica-assembly.AC4.1

**Files:**
- Create: `src/fusion/combiner.py` (`# pattern: Functional Core`)
- Test: `tests/test_fusion_combiner.py` (unit)

**Implementation:** pure `combine_and(p_cca, p_rel)` (elementwise product); `fit_logistic_combiner(scores, labels)` and `apply_logistic_combiner(model_or_coefs, scores)` using sklearn `LogisticRegression` over ≤3 features (`z_cca`, `z_rel`, optional `z_us`); a frozen `FusionConfig` dataclass (`gate_threshold: float`, `combine: Literal["product","logreg"]`, `coefs: list[float] | None`, `score_space: Literal["prob","logit"]`, `includes_us: bool`).

**Testing:** AND product correctness + monotonicity; LR fit determinism (fixed `random_state`); coefficient count ≤3 (≤4 with soft-US); `FusionConfig` validation (rejects unknown `combine`).

**Verification:** `uv run pytest tests/test_fusion_combiner.py` — all pass.

**Commit:** `feat(fusion): pure combiners + FusionConfig`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `src/fusion/sidecar.py` (`fusion.json` I/O)

**Verifies:** multi-head-ica-assembly.AC4.2

**Files:**
- Create: `src/fusion/sidecar.py`
- Test: `tests/test_fusion_sidecar.py` (unit)

**Implementation:** `save_fusion(cfg, path)` / `load_fusion(path) -> FusionConfig` JSON round-trip; `fusion_path_for_weights(weights_path)` mirroring `calibration_path_for_weights`. The payload also records the per-head calibrator references it composes.

**Testing:** round-trip equality (product and logreg configs); malformed payload → `ValueError`.

**Verification:** `uv run pytest tests/test_fusion_sidecar.py` — all pass.

**Commit:** `feat(fusion): fusion.json sidecar I/O`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 4-5) -->
<!-- START_TASK_4 -->
### Task 4: `pick_us_threshold` helper

**Verifies:** multi-head-ica-assembly.AC3.2

**Files:**
- Modify: `src/validation/doca_recall.py` (add `pick_us_threshold(scored_df, target_recall, thresholds)`)
- Test: `tests/test_doca_recall.py` (extend)

**Implementation:** evaluate `doca_recall` over a threshold grid; return the largest threshold whose recall ≥ `target_recall`; documented behavior when none qualify (return the lowest threshold + a warning flag).

**Testing:** synthetic scored frame → returns the correct largest-qualifying threshold; none-qualify edge handled.

**Verification:** `uv run pytest tests/test_doca_recall.py` — all pass.

**Commit:** `feat(validation): pick_us_threshold from recall recipe`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: `src/fit_fusion.py` — selection + composed calibration

**Verifies:** multi-head-ica-assembly.AC3.2, AC3.3, AC4.1, AC4.2, AC4.3

**Files:**
- Create: `src/fit_fusion.py` (`# pattern: Imperative Shell`)
- Test: `tests/test_fit_fusion.py` (the pure decision rule)

**Implementation:** load the clean Phase-2 eval set + held-out anchors; score calibrated US/CCA/rel over those ids (via `load_cache_meta` id→emb_row, `apply_*_model`, per-head calibrators); set `τ_us` via `pick_us_threshold`; on the gated survivors run AND vs LR by `StratifiedKFold` CV (PR-AUC), apply the **1-SE pre-registered rule** (a pure `select_combiner(cv_and, cv_lr) -> "product"|"logreg"`) to choose; compute composed-score calibration via `calibration_report` and optionally fit a final 2-param Platt; `save_fusion(FusionConfig(...))`; emit a metrics JSON (chosen combiner, τ_us, per-combiner CV PR-AUC ± SE, composed ECE/Brier).

**Label-budget note:** the optional composed-score Platt (AC3.3) and the LR combiner draw on the *same* scarce labels (~466 anchors, ~30% held out, + coded boundary positives). Per EPV/Harrell, fit the composed Platt only if the held-out positive count supports its 2 extra parameters on top of the LR; otherwise report composed calibration (ECE/Brier/reliability) without refitting, and record the decision.

**Testing:** unit-test `select_combiner` — LR mean improvement >1 SE → `"logreg"`; ≤1 SE → `"product"`; tie/degenerate → `"product"`. (End-to-end fit operationally verified.)

**Verification (operational):** `uv run python -m src.fit_fusion …` runs on the clean eval → writes `fusion.json` + metrics; prints chosen combiner, τ_us, composed ECE/Brier.

**Commit:** `feat(fusion): empirical AND-vs-LR selection + composed calibration`
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_C -->
