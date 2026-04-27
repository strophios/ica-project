# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*Last updated: 2026-04-27 (Tier 2 Piece 4c landed; tier closeout pending integration smoke test + adversarial review).*

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
- **Tests**: pytest is configured (`pyproject.toml [tool.pytest.ini_options]`, `pythonpath = ["."]`). Run with `pytest` from the project root. Current coverage is narrow: 85 tests passing — invariant tests for `FLPULoss` (`tests/test_flpu_loss.py`) and the train/val/test split logic in `create_classifier_data` (`tests/test_data_splits.py`), plus construction/shape/contract tests for the Tier 2 `ClassificationHead` (`tests/test_heads.py`, 17 tests including 6 covering the `metrics` parameter added in 4c), `LayerLRModel` (`tests/test_layer_lr_model.py`, 11 tests including a sparse-gradient regression), `ClassifierPreprocessor` (`tests/test_preprocessor.py`, 12 tests), and integration tests for the assembled stack (`tests/test_assembly.py`, 13 tests covering `build_endpoint_model`, `build_inference_model`, training-step behavior, and Pattern A weight sharing).
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
- `src/model_setup/layer_lr_model.py` — `LayerLRModel`, a `keras.Model` subclass overriding `train_step` to scale gradients by per-variable multipliers before optimizer apply. Supports discriminative fine-tuning (fixed geometric multipliers) and gradual unfreezing (callback-updated multipliers). Added in Tier 2 Piece 2 (commit `ad0f94b`); a sparse-gradient (`tf.IndexedSlices`) handling fix landed in 4b (gradient scaling uses `tf.math.scalar_mul`).
- `src/preproc/preprocessor.py` — `ClassifierPreprocessor` refactored to multi-head shape: `label_keys: dict[str, str]` mapping output-dict-key → source-column-name; emits multi-head-shaped output in both endpoint and standard modes; casts targets to `target_dtype` (default `float32`) at preprocess time. Tier 2 Piece 3 (commit `e3dda6a`).
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
- **Tier 2 refactor, all four pieces landed (4a, 4b, 4c).** See `docs/notes/tiers-and-checkpoints.md` for full status and `docs/notes/tier2-design.md` for design decisions. The Tier 2 abstractions are now wired into the training and eval scripts; `classifier_from_dapt_checkpoint` and `classification_setup.py` are gone. Remaining Tier 2 closeout:
  - **Integration smoke test** of the composed stack on dummy or real data — verifies fit → save → load → predict at runtime. The 85-test suite covers individual abstractions; the smoke test verifies they cohere.
  - **Adversarial review of Tier 2** (likely the `code-reviewer` subagent) — check the cumulative shape of Pieces 1–4 against the design doc. Particular attention to: the endpoint-pattern decisions, the Pattern A vs. Pattern 2 split, the naming-convention subtleties (`<head>_targets` Input name vs. head Layer name vs. output name), the metrics-in-head extension, and the deletion of `classification_setup.py`.
  - **Integration pass.** End-to-end smoke test on dummy data.
  - **Adversarial review of Tier 2.** Likely the `code-reviewer` subagent (plan/architecture framing).
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
- `docs/notes/tiers-and-checkpoints.md` — tier plan, commit-level status, and "how to pick up" instructions.
- `docs/notes/tier2-design.md` — per-piece design reasoning for Tier 2 (Pieces 1 and 2 marked implemented; Piece 3 not yet added).
- `docs/notes/pinned-questions.md` — deliberately deferred substantive questions, currently covering (1) composing nnPU + α + γ + ALUM across a four-layer framing and (2) extension to multi-class heads.

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
