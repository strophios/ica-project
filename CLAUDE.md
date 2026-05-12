# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*Last updated: 2026-05-12 (Tier 4 Piece 2 closeout: Phase 2 review Minors M1 + M2 addressed; all 220 tests passing).*

## Project Overview

This project builds an ML system to identify **Immigrant Collective Action (ICA)** events in New York Times articles, enabling expansion of protest event datasets (like the Dynamics of Collective Action / DoCA) without massive manual coding effort. The approach uses **Positive-Unlabeled (PU) learning** with a RoBERTa backbone, built on Keras 3 + TensorFlow, targeting both local macOS (tensorflow-metal) and the HPC cluster "Explorer" (tensorflow + CUDA, paths like `/projects/ahd`).

### The Research Problem

The "haystack" problem: finding the tiny fraction of NYT articles reporting ICA events among ~1.8M articles. This is complicated by three factors:
1. **Positive-Unlabeled data**: we have labeled positives (from DoCA + NYT indexer tags) but zero confirmed negatives. Unlabeled articles may contain unreported positives (Hanna 2017 found 62% of "false positives" were actually correct).
2. **Severe class imbalance**: ICA articles are a vanishingly small minority.
3. **The project assumes SCAR** (Selected Completely At Random) for the labeling mechanism — labeled positives are treated as a random sample of all positives. This is a simplifying assumption; alternative PU approaches exist if it proves inadequate.

### Planned Architecture (not yet fully implemented)

The end goal is a **multi-headed classifier**: a shared DAPT RoBERTa encoder feeding separate classification heads for:
- CCA event identification (implemented)
- Immigrant involvement (not yet implemented)
- US events (not yet implemented)
- Combined ICA output head, fed by the above three + the shared encoder representation (not yet implemented)

This decomposition lets us leverage the larger labeled datasets for CCA and immigration separately, making the sparse ICA labeling problem tractable. Currently only the CCA head exists.

## Development Setup

- **Python 3.12**, managed with `uv`
- Activate environment: `source .venv/bin/activate`
- Install dependencies: `uv sync` (runtime), `uv sync --group dev` (adds pytest)
- No `__init__.py` files exist (implicit namespace packages)
- All scripts must be run from the **project root** (imports use `src.*` paths, e.g., `import src.model_setup.dapt_setup`)
- **Configuration**: `src/config.py` is the single source of truth for platform-conditional values. Detects cluster vs. local via `Path("/projects/ahd").exists()` (override with `ICA_ENV=cluster|local`); exports `IS_CLUSTER`, `PROJECT_ROOT`, granular paths (`CCA_SET_DIR`, `DAPT_BACKBONE_WEIGHTS`, `LDC_CORPUS`, etc.), and `DTYPE_POLICY` (`mixed_float16` on cluster, `float32` locally — MPS mixed-precision support is patchy). Scripts apply the dtype policy explicitly: `keras.config.set_dtype_policy(config.DTYPE_POLICY)`.
- **Tests**: pytest is configured (`pyproject.toml [tool.pytest.ini_options]`, `pythonpath = ["."]`). Run with `pytest` from the project root. Current coverage: 220 tests passing — invariant tests for `FLPULoss` (`tests/test_flpu_loss.py`); the train/val/test split logic + label-construction + id-uniqueness in `create_classifier_data` (`tests/test_data_splits.py`, including 11 Tier 3 Piece 4 label-construction tests added 2026-05); missing-value handling + headline-with-lead concatenation in `data_from_parquet` (`tests/test_data_loading.py`, 11 tests across two classes — Tier 3 Piece 4, added 2026-05); construction/shape/contract tests for the Tier 2 `ClassificationHead` (`tests/test_heads.py`, 17 tests including 6 covering the `metrics` parameter added in 4c); `LayerLRModel` (`tests/test_layer_lr_model.py`, 13 tests including a sparse-gradient regression and two loss-tracking regressions added post-Tier-2-review); `ClassifierPreprocessor` (`tests/test_preprocessor.py`, 26 tests including the 14 Tier 3 Piece 1 dual-boundary validation tests added in 2026-05); `RunConfig`/`HeadConfig`/sub-configs (`tests/test_cca_config.py`, 89 tests covering per-dataclass construction validation, derived properties, JSON round-trip, forward-compat, backbone validation, path helper, resolved steps, and LR schedule resolution — Tier 3 Piece 3a expanded in Tier 4 Piece 2); and integration tests for the assembled stack (`tests/test_assembly.py`, 15 tests including 2 Tier 3 Piece 2 Pattern-2-serialization tests covering the round-trip-bitwise invariant and the `skip_mismatch=False` shape-mismatch fail-loud invariant).
- **Reproducibility**: training scripts call `keras.utils.set_random_seed(200)` to match the `seed=200` used by the polars `.sample()` splits in `src/data_setup/data.py`.

## Architecture: Three-Phase ML Pipeline

The project implements a three-phase pipeline, each with dedicated training scripts. Phase 1 is complete; phases 2 and 3 are implemented but need revision.

### Phase 1: Domain-Adaptive Pre-Training (DAPT) — DONE
- **Script:** `src/dapt.py`
- Fine-tunes a RoBERTa masked language model on the full LDC news corpus (~1.16M headline+lede pairs)
- Model setup: `src/model_setup/dapt_setup.py` — loads `roberta_base_en` backbone + `MaskedLMHead`, manually loads pre-trained LM head weights from a `.npy` file via layer index access (`model.layers[4]`)
- Produces DAPT backbone weights (`dapt_backbone.weights.h5`) used by downstream phases
- `src/test_module.py` extracts and saves the backbone weights from a DAPT checkpoint
- Rationale: adapts RoBERTa's general English understanding to the NYT headline/lede domain (Gururangan 2020)

### Phase 2: Class Prior Estimation — IMPLEMENTED, NEEDS REVISION
- **Scripts:** `src/prior_estimation/lu_classifier.py` (train L/U classifier), `src/run_prior_estimate.py` (estimate prior)
- Trains a linear classifier (frozen DAPT backbone + single Dense layer) to distinguish labeled vs. unlabeled samples
- Feeds predictions into **DEDPUL** (Ivanov 2020) EM algorithm to estimate the positive class prior. Current best estimate is π_pos ≈ 0.02 for CCA (robust across `kde_mode` and bandwidth choices); `src/run_cca_classification.py` still passes `prior=0.03` for continuity with existing trained models and is flagged for update on the next CCA retrain.
- `src/prior_estimation/dedpul_em.py` and `dedpul_utils.py` are adapted from the [DEDPUL repo](https://github.com/dimonenka/DEDPUL/). DEDPUL expects the *probability* of being unlabeled (its convention is 0 = labeled, 1 = unlabeled); `run_prior_estimate.py` applies `sigmoid` + `1 - p` to convert the L/U classifier's logit output. An earlier version fed logits directly (and attributed the effect of the fix to the sigmoid — the bulk of the observed shift was actually DEDPUL's bandwidth grid being mis-calibrated for logit-scale inputs; see `scripts/compare_dedpul_logit_vs_prob.py` for the four-variant attribution table).
- `src/prior_estimation/ramaswamy2016.py` is an alternative kernel-based prior estimation method (requires `cvxopt`, currently commented out in dependencies)

### Phase 3: CCA Classification — IMPLEMENTED (Tier 2 refactor done)
- **Script:** `src/run_cca_classification.py` (train), `src/eval_cca_classifier.py` (evaluate)
- Binary classifier built from Tier 2 abstractions: `src/model_setup/backbone.py:load_dapt_backbone` (DAPT-finetuned RoBERTa), `src/model_setup/heads.py:ClassificationHead` (named `"cca"`, with FLPU loss and per-head metrics), wired together by `src/model_setup/assembly.py:build_endpoint_model` (training, with `cca_targets` Input) and `build_inference_model` (predict, no target Inputs). The training script uses Pattern A (in-process Layer-instance sharing between train and inference models); the eval script uses Pattern 2 (cross-process: fresh head, weights loaded by name).
- Uses **FLPULoss** (`src/loss_functions/loss.py`): focal cross-entropy (Lin 2020, γ=2) wrapped in non-negative PU learning (Kiryo 2017), parameterized by the estimated class prior (currently 0.02, the corrected estimate from `run_prior_estimate.py`). The focal-loss `alpha` knob has been removed (see the FLPU docstring and `docs/notes/pinned-questions.md` for the cost-sensitive-nnPU interpretation and the reasoning behind defaulting α=off). `docs/notes/pinned-questions.md` also lays out a four-layer composition framing (risk definition / PU estimation / sample allocation / optimization-level regularization) that should guide any future changes to how mechanisms stack.
- Currently trains with **frozen encoder** and classification head only; per-layer learning rates and encoder unfreezing are forward-compatible via `LayerLRModel` (the type returned by `build_endpoint_model`) — set `freeze_encoder=False` and pass `layer_multipliers={...}` when ready.
- `src/test_script.py` is a sandbox for local testing; partially broken pending Tier 4 hygiene cleanup (some references to retired `classification_setup.py` were stub-commented in 4c).
- Current quality: better than chance, but hand review shows the NYT indexer tag definitions are too generous (highest-leverage improvement is refining label definitions). Empirical retraining run with the corrected prior + new abstractions is one of the deferred items at the end of Tier 2.

**Tier 2 abstractions, now on the training path (as of 4c):**
- `src/model_setup/heads.py` — `ClassificationHead`, a `keras.layers.Layer` subclass supporting both standard mode (loss handled by outer `compile()`) and endpoint mode (loss registered via `add_loss`, needed for FLPU and eventual ALUM). The `metrics` parameter (added in 4c, symmetric with `loss_fn`) holds per-head metric instances that update inside `call()` when targets are provided; metric names are prefixed with the head's name to avoid multi-head collisions. Tier 2 Piece 1 (commit `789d88c`); metrics extension landed in Piece 4c.
- `src/model_setup/layer_lr_model.py` — `LayerLRModel`, a `keras.Model` subclass overriding `train_step` to scale gradients by per-variable multipliers before optimizer apply. Supports discriminative fine-tuning (fixed geometric multipliers) and gradual unfreezing (callback-updated multipliers). Added in Tier 2 Piece 2 (commit `ad0f94b`); a sparse-gradient (`tf.IndexedSlices`) handling fix landed in 4b. The `train_step` implementation was rewritten post-review to mirror stock Keras `train_step` (TF backend) line-for-line plus the multiplier scaling step — the original version was missing `_loss_tracker.update_state` (which broke `history.history["loss"]` and TensorBoard logging) and `optimizer.scale_loss` (which silently no-op'd `LossScaleOptimizer` under `mixed_float16`). See `docs/notes/tier2-design.md` "Post-review corrections" section.
- `src/preproc/preprocessor.py` — `ClassifierPreprocessor` refactored to multi-head shape: `label_keys: dict[str, str]` mapping output-dict-key → source-column-name; emits multi-head-shaped output in both endpoint and standard modes; casts targets to `target_dtype` (default `float32`) at preprocess time. Tier 2 Piece 3 (commit `e3dda6a`). Dual-boundary input validation added in Tier 3 Piece 1 (2026-05): `__init__` validates internal-config validity (`text_key` non-empty string, `label_keys` is dict, `target_dtype` is a valid Keras dtype, standard-mode requires non-empty `label_keys`) raising `ValueError`; `__call__` validates input-batch keys against the configured `text_key` + `label_keys` source columns (enumerate-all-missing), raising `KeyError` with the configured expectation and the batch's actual keys.
- `src/model_setup/backbone.py` — `load_dapt_backbone(weights_path)`, the DAPT-checkpoint loading split out of the now-retired `classifier_from_dapt_checkpoint`. Tier 2 Piece 4b (commit `2f069c4`).
- `src/model_setup/assembly.py` — `build_endpoint_model` and `build_inference_model`, the wiring functions that combine backbone + heads into full Keras models. Training model is a `LayerLRModel` with target Inputs named `"<head>_targets"` (suffixed to avoid op-name collision with the head Layer itself); inference model is a plain `keras.Model` with no target inputs. Sharing head Layer instances between the two gives Pattern A in-process weight sharing (verified safe in Keras 3 via `scripts/experiment_endpoint_inference_evaluate.py`). Tier 2 Piece 4b.
- **Wired into `run_cca_classification.py` (Pattern A) and `eval_cca_classifier.py` (Pattern 2) in Piece 4c.** `classifier_from_dapt_checkpoint` and `src/model_setup/classification_setup.py` are deleted.

## Key Design Decisions

- **Models output logits** (no final activation), so all losses use `from_logits=True`
- **Mixed precision** (`mixed_float16`) is enabled in all training scripts for GPU efficiency; local macOS may not fully support this
- **Data pipeline** uses `tf.data.Dataset` with `sample_from_datasets()` to handle PU class imbalance via weighted sampling (e.g., 1:9 pos:unl for CCA training, 1:5 for L/U classifier)
- **Re-balanced batches** are central to the PU + class imbalance strategy: every batch is guaranteed to contain labeled positive signal
- Preprocessed datasets are cached to disk (`cca_set/` directory) because `from_tensor_slices()` takes minutes on the full corpus
- Two preprocessor classes in `src/preproc/preprocessor.py`: `CustomPreprocessor` (for DAPT/MLM tasks with masking) and `ClassifierPreprocessor` (for classification tasks, multi-head-aware as of Tier 2 Piece 3 — takes a `label_keys: dict[str, str]` mapping output-key → source-column, supports both standard and endpoint-layer patterns, casts targets to `target_dtype` at preprocess time)

## Data

- **Sources**: Dynamics of Collective Action (DoCA, 1960-1995, 23k+ events) + NYT Annotated Corpus (LDC, 1987-2007, 1.8M articles). Training data comes from the overlap period where both exist.
- **Text input**: `headline + "</s>" + lead_paragraph` (RoBERTa separator token). Headlines + ledes (not full articles) because the NYT Archive API provides these freely, enabling eventual expansion to the full NYT (1851-present).
- **Labels from NYT indexer descriptors** (first pass, needs refinement):
  - CCA: "demonstrations and riots", "hunger strikes", "picketing", "civil disobedience", "strikes", "boycotts"
  - Immigration: "immigration", "immigration and emigration", "asylum (political)", "migrant labor", "deportation", "refugees and expatriates", "illegal aliens", "visas", "illegal immigrants", "immigration and refugees", "citizenship", and several agency names (Border Patrol, ICE, etc.)
- Labels stored as `cca`/`cca_descriptor` and `immig`/`immig_descriptor` boolean columns; combined into binary `cca_label`
- Train/val/test split: 90/5/5, applied separately to labeled and unlabeled subsets
- Counts: ~20k CCA articles, ~11k immigration articles, ~661 ICA articles (the overlap); ~18k positive and ~1M unlabeled in training

## Current Status and Open Work

**Done:**
- DAPT on headline/lede pairs
- CCA classifier (single-head) with FLPU loss
- Class prior estimation via DEDPUL

**Priority open items:**
- **Tier 2 complete.** All four pieces (1, 2, 3, 4a/b/c), integration smoke test, adversarial review, and post-review fixes have landed. See `docs/notes/tiers-and-checkpoints.md` for full status and `docs/notes/tier2-design.md` for design decisions (with a "Post-review corrections" section documenting what the review found and which issues were fixed vs. deferred).
- **Tier 3 in progress.** Tier 3 (robustness) is the boundary-enforcement spine for the multi-head future. See `docs/notes/tier3-design.md` for the overall framing (boundary-inventory pattern: validate at every boundary; each catches what others miss) and per-piece design reasoning. Pieces planned: 1 (I3, preprocessor input validation), 2 (I5, Pattern-2 serialization round-trip test), 3 (I4, train/eval config coupling — the most architectural), 4 (original-scope test coverage for label construction and missing-value handling).
  - **Piece 1 done.** Dual-boundary validation on `ClassifierPreprocessor`: `__init__` checks for internal-config-validity bugs (raises `ValueError`), `__call__` checks for config-vs-data-mismatch bugs (raises `KeyError`, enumerates all missing columns). Retires M2 from the Tier 4 deferred list (`target_dtype` validation lives naturally in the construction-validation block). 14 new tests; suite at 87 → 101 passing.
  - **Piece 2 done.** Pattern 2 serialization round-trip + shape-mismatch tests in `tests/test_assembly.py`. Production-path `load_weights` calls (`eval_cca_classifier.py`, `model_setup/backbone.py`, smoke test) tightened with explicit `skip_mismatch=False`. Empirical finding during implementation: Keras's `.weights.h5` save format keys variables by *layer-class + positional index*, NOT by user-given name — so renaming a head doesn't break load (the original mismatch test was reframed from name-mismatch to shape-mismatch). Pattern 2 is therefore load-by-*structure*, not load-by-*name*; the head-name contract is enforced at call sites (Keras `compile(loss={head: ...})` routing) rather than at weight load. Switching to legacy `.h5` format would enable `by_name=True` strict matching but was deliberately rejected (legacy format, deprecation risk; protection redundant with call-site routing). Captured in `docs/notes/tier3-design.md` Piece 2 "Empirical finding" + "Decision: stay with `.weights.h5`" subsections, with explicit decision criteria for revisiting. 2 new tests; suite at 101 → 103 passing.
  - **Piece 3a done.** `src/cca_config.py` introduces frozen-dataclass `RunConfig` (with sub-configs `HeadConfig`, `FLPULossConfig`, `RatioBatchConfig`, `LRScheduleConfig`, `OptimizerConfig`) capturing the architectural and research-dimension parameters of a CCA training run. Includes JSON sidecar serialization (`.weights.h5` → `.config.json` via `config_path_for_weights`), `validate_against_backbone(backbone)` for cross-object hidden_dim validation, `DEFAULT_CCA_CONFIG` as the canonical starting point, and a CLI helper (`python -m src.cca_config write_default <weights_path>`). Forward-compat design: `loss` field is wrapped in `FLPULossConfig` to anticipate ALUM/BCE discrimination (pinned question #1); `optimizer` and `lr_schedule` are wrapped sub-configs to lift their fields above flat (the wrapped-vs-flat forward-compat distinction; see `docs/notes/tier3-design.md` Piece 3 reasoning). Validation hierarchy: each dataclass's `__post_init__` validates its own fields; cross-object invariants (head names unique) live at RunConfig; external-context invariants (backbone match) live in `validate_against_backbone`. 64 new tests across 11 test classes; suite 103 → 167.
  - **Piece 3b done.** `run_cca_classification.py` builds from `cca_config.DEFAULT_CCA_CONFIG`, validates against the loaded backbone, and writes the sidecar via `RunConfig.to_json(config_path_for_weights(weights_path))` after fit. `eval_cca_classifier.py` loads the sidecar at startup via `RunConfig.from_json(...)` and constructs its inference model from the same values. `scripts/smoke_test_integrated_stack.py` exercises the full RunConfig → fit → save (weights + sidecar) → load (sidecar + weights) → predict round-trip on synthetic data. No new tests (script integration is exercised by the smoke test). Suite still at 167.
  - **Piece 4 done.** Original-scope test coverage. `tests/test_data_splits.py` extended with `TestLabelConstruction` (11 tests covering all four boolean combinations of `(cca, cca_descriptor) → cca_label` and `(immig, immig_descriptor) → immig_label`, plus integer-dtype and per-row-independence checks). `tests/test_data_loading.py` added (new file, 11 tests across `TestParquetMissingValueHandling` covering null/`"NA"` substitution in headline and lead_paragraph and `TestHeadlineWithLeadConcatenation` covering the `headline + "</s>" + lead` build across all empty/normal combinations). 22 new tests; suite 167 → 189 passing.
  - **Closeout done.** Adversarial review (returned 1 Critical + 8 Important + 7 Minor) addressed: C1 (round-trip test now explicitly pins no-op-load-protection via same-seeded backbones + `freeze_encoder=True`); I1 (`expected_columns` wired into train/eval scripts + smoke test as Layer-1 schema validation, making the 3-layer hierarchy real rather than aspirational); I3 (eval-side metrics dropped — empirically verified metric state vars aren't in `head.weights`); I5 (finite test dataset for `evaluate()`, removing the `test_steps = validation_steps` approximation); I6 (multi-head metric distinctness test added); I7 (Layer-2 dtype check on `text_key` in preprocessor `__call__`); I8 interim (metrics factored into `src/cca_metrics.py`'s `make_cca_metrics()` helper, both production scripts use the same source). I2 (smoke-test backbone-validation-path) + I4 (LR schedule resolution gap) + I8 full version deferred with explicit notes. Suite: 189 → 192 passing. See `docs/notes/tier3-design.md` "Tier 3 closeout" subsection for the full triage.
- **Tier 3 complete.** All four implementation pieces done; adversarial review done with closeout fixes done; deferrals documented. Next session can move to Tier 4 (hygiene) or to the deferred empirical / research items (smoke training run with corrected prior, ALUM, etc.).
- **Tier 4 Piece 1 done.** M4b resolved (tightened regex in default head-name collision test). Suite 219 → 220 passing. Commit 44b1faf.
- **Tier 4 Piece 2 done.** Phase 2 code review Minors M1 + M2 addressed: M1 documents the intentional validation duplication in `LRScheduleConfig.with_resolved` (clearer public-API errors); M2 adds type-narrowing assert in test to align with surrounding test idiom. No behavioral change. Suite: 220 passing. Commit 9d92c17.
- **Tier 4 remaining**: Piece 3 (Lessons docs — cross-project learnings from Tier 3 boundary-inventory design pattern).
- **Deferred empirical checks** (batched until Piece 4 when environment handling is settled): smoke training run with updated FLPU + corrected prior (≈ 0.02); confirm training dynamics under `mixed_float16` + removed-α FLPU; sensitivity sweep on Ratio Batch (currently 1:10, more aggressive than Ji 2023).
- Implement ALUM/VAT (Virtual Adversarial Training); incomplete sketch exists in `src/test_module.py`
- Benchmarking and comparative testing (FLPU vs. ALUM, etc.)
- Refine NYT indexer tag definitions for CCA and immigration labels (current definitions are too generous)
- Pull DoCA article headlines (1960-1984) via NYT Archive API to expand training data
- Build additional classification heads (immigration, US, combined ICA)
- Hyperparameter search
- Output calibration (decision threshold, not just raw .5)
- Hand-label a small PN test set (~200-1000 articles) for proper evaluation
- Retrain CCA classifier with the corrected prior (π_pos ≈ 0.02 rather than 0.03)

**Handoff docs for picking up mid-work:**
- `docs/notes/tiers-and-checkpoints.md` — tier plan, commit-level status, and "how to pick up" instructions. Operational doc — read at the start of each piece, updated at piece close.
- `docs/notes/tier2-design.md` — per-piece design reasoning for Tier 2.
- `docs/notes/tier3-design.md` — per-piece design reasoning for Tier 3, with overall boundary-inventory framing for the I3/I4/I5 inherited findings (Piece 1 implemented; Pieces 2–4 anticipated).
- `docs/notes/pinned-questions.md` — deliberately deferred substantive questions, currently covering (1) composing nnPU + α + γ + ALUM across a four-layer framing, (2) extension to multi-class heads, and (3) `ClassifierPreprocessor` train/predict shape — config-vs-call-mode design smell, with a candidate `for_training` / `for_prediction` method-split refactor sketched and forward-pressure on Tier 3 Piece 3 (I4) captured.

## Key References (cited in code and memo)

- **Ji2023**: Main paper this project builds on (PU learning + ALUM for text classification)
- **Kiryo2017**: Non-negative PU learning (nnPU) loss formulation
- **Ivanov2020 (DEDPUL)**: Prior estimation via density estimation
- **Lin2020**: Focal loss (via Keras `BinaryFocalCrossentropy`)
- **Gururangan2020**: Domain-adaptive pretraining
- **Liu2020**: Adversarial training for robustness (ALUM)
- **Hanna2017**: Prior work on ML for protest event identification; defines the haystack/coding task decomposition
- **Oliver2023**: Report-event dyad complexity in protest event analysis
- **Voss2024**: The ICA dataset this project extends
- **Deep Learning with Python** (Chollet 2025): Architecture/training guidance
- **Detailed project memo**: `ml_memo/ml_memo.qmd` — comprehensive explanation of the problem, methods, and plans
