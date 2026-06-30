# Multi-Head ICA Assembly Implementation Plan — Phase 5

**Goal:** Assemble one inference artifact — frozen shared encoder + `{us, cca, rel}` heads + the `fusion.json` composition — that maps a row (cached CLS features, optionally raw text) to per-head calibrated probabilities and one composed ICA score, and prove cross-process reload.

**Architecture:** An `IcaModel` wrapper (`src/assemble_ica.py`, Imperative Shell) loads the three head configs, transfers each head's trained weights via a temporary single-head feature-inference model (Pattern 2), assembles them into one `build_feature_inference_model` (Pattern A in-process sharing), and applies the gate + Phase-4 combiner on the calibrated outputs. A `reload_and_score_ica` in `artifact_check.py` reproduces the composed score from disk in a fresh construction.

**Tech Stack:** Python 3.12, `uv`, Keras 3 + TF, `numpy`, `scikit-learn` (combiner), `pytest`.

**Scope:** Phase 5 of 6 from `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`. Depends on Phase 3 (retrained heads) and Phase 4 (`src/fusion/`, calibrators, `fusion.json`).

**Codebase verified:** 2026-06-23 (codebase-investigator).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### multi-head-ica-assembly.AC5: Assembled multi-head artifact
- **multi-head-ica-assembly.AC5.1 Success:** A single inference model builds with the frozen backbone + `{us, cca, rel}` heads loaded by structure (Pattern 2).
- **multi-head-ica-assembly.AC5.2 Success:** The artifact scores cached features (and raw text) to three logits + the composed ICA score.
- **multi-head-ica-assembly.AC5.3 Success:** Cross-process reload reproduces scores within tolerance (the `artifact_check` analogue).
- **multi-head-ica-assembly.AC5.4 Failure:** Head-name collision or a missing head weight raises at assembly time, not silently.

---

## Verified context (from investigation)

- `build_feature_inference_model(heads, hidden_dim)` (`src/model_setup/assembly.py:324-333`) accepts `dict[str, ClassificationHead]`, emits per-head logits dict, asserts unique names. Token-mode `build_inference_model(backbone, heads, seq_length)` (`:336-392`) for the optional text path.
- **US head is features-mode 768-d CLS** (`run_us_features.py:111-122`, `build_feature_endpoint_model`) — same cache as CCA/rel; a shared-CLS multi-head model is coherent.
- **3-head load mechanism (no per-head Keras API):** per head, build a temp single-head `build_feature_inference_model({name: head})`, `load_weights(skip_mismatch=False)` → the head instance now holds trained weights; then assemble all three into one model (Pattern A). Proven by `tests/test_assembly.py` (TestPatternAWeightSharing :353-398; TestPatternTwoSerialization :414-703; shape-mismatch raise :646-703).
- Composition is Python arithmetic (gate + combiner), reusing Phase 4 `src/fusion/combiner.py`; precedent wrapper `apply_us_model` (`slice_eval.py:28-107`).
- `artifact_check.py:reload_and_score:18-38` (US-only today) — extend with `reload_and_score_ica`. `load_dapt_backbone` (`backbone.py:22-56`) for the token path; same DAPT backbone that produced the cache.
- Head configs: US `UsRunConfig.from_json(config_path_for_weights(...))`; CCA/rel `cca_config.RunConfig.from_json(...)`. Calibrators via `load_calibration(calibration_path_for_weights(...))`.

---

<!-- START_SUBCOMPONENT_A (tasks 1) -->
<!-- START_TASK_1 -->
### Task 1: `src/assemble_ica.py` — `IcaModel`

**Verifies:** multi-head-ica-assembly.AC5.1, AC5.2, AC5.4

**Files:**
- Create: `src/assemble_ica.py` (`# pattern: Imperative Shell`)
- Test: `tests/test_assemble_ica.py` (unit)

**Implementation:** `IcaModel`:
- Constructor loads the three head config sidecars (US `UsRunConfig`; CCA/rel `RunConfig`), constructs each `ClassificationHead` with the recorded `hidden_dim`/`name` (loss_fn per config; inference ignores it), and transfers trained weights via a temp single-head `build_feature_inference_model({name: head})` + `load_weights(skip_mismatch=False)` per head.
- Assembles `self.model = build_feature_inference_model({"us": us_head, "cca": cca_head, "rel": rel_head}, hidden_dim=768)`.
- Loads the three `PlattCalibrator`s and the `FusionConfig` (`load_fusion`).
- `predict_ica_from_features(features) -> dict` returning per-head calibrated probs + `ica_score`: gate `calib_us ≥ τ_us`, then `combine_*` (Phase 4) on survivors, 0.0 elsewhere.
- Optional `predict_ica_from_text(texts)`: `load_dapt_backbone` + token-mode `build_inference_model`, same heads, same composition.

**Testing:** 3-head assembly emits `{us,cca,rel}`; per-head assembled scores equal standalone single-head scores on a fixture (weights transferred); duplicate head name → `ValueError`; a missing/shape-mismatched head weight → `ValueError` (mirror the Pattern-2 shape-mismatch test); composed `ica_score` is 0 for gated-out rows and in [0,1] for survivors.

**Verification:** `uv run pytest tests/test_assemble_ica.py` — all pass.

**Commit:** `feat(assembly): IcaModel multi-head inference artifact`
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2) -->
<!-- START_TASK_2 -->
### Task 2: `reload_and_score_ica` cross-process proof

**Verifies:** multi-head-ica-assembly.AC5.3

**Files:**
- Modify: `src/validation/artifact_check.py` (add `reload_and_score_ica`)
- Test: `tests/test_artifact_check.py` (extend)

**Implementation:** `reload_and_score_ica(us_weights, cca_weights, rel_weights, fusion_path, features)` — fresh construction (no shared instances): reload the artifact set (3× weights+config+calibration + `fusion.json`), assemble, and return the composed ICA score. Validates each config against the backbone where applicable.

**Testing:** `reload_and_score_ica` output matches `IcaModel.predict_ica_from_features` within tolerance (bitwise on the frozen-feature path) on a fixture artifact set written to `tmp_path`, proving the artifact set reproduces ICA scores cross-process.

**Verification:** `uv run pytest tests/test_artifact_check.py` — all pass.

**Commit:** `feat(validation): cross-process ICA artifact reload proof`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_B -->
