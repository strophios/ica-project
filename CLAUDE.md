# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*Last updated: 2026-04-20 (after Tier 1 audit and cleanup).*

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
- **Path handling**: scripts switch between local (`~/immigration_project/...`) and cluster (`/projects/ahd`) paths; look for `path_prefix` at the top of each script
- **Tests**: pytest is configured (`pyproject.toml [tool.pytest.ini_options]`, `pythonpath = ["."]`). Run with `pytest` from the project root. Current coverage is narrow — invariant tests for `FLPULoss` and for the train/val/test split logic in `create_classifier_data` (32 tests, all passing). See `tests/test_flpu_loss.py` and `tests/test_data_splits.py`.
- **Reproducibility**: training scripts call `keras.utils.set_random_seed(200)` to match the `seed=200` used by the polars `.sample()` splits in `src/data_setup/dapt_data.py`.

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

### Phase 3: CCA Classification — IMPLEMENTED, NEEDS REVISION
- **Script:** `src/run_cca_classification.py` (train), `src/eval_cca_classifier.py` (evaluate)
- Binary classifier: `src/model_setup/classification_setup.py` builds `backbone → [CLS] token → dropout → dense(hidden_dim, relu) → dropout → dense(1, no activation)`
- Uses **FLPULoss** (`src/loss_functions/loss.py`): focal cross-entropy (Lin 2020, γ=2) wrapped in non-negative PU learning (Kiryo 2017), parameterized by the estimated class prior. The focal-loss `alpha` knob has been removed (see the FLPU docstring and `docs/notes/pinned-questions.md` for the cost-sensitive-nnPU interpretation and the reasoning behind defaulting α=off). `docs/notes/pinned-questions.md` also lays out a four-layer composition framing (risk definition / PU estimation / sample allocation / optimization-level regularization) that should guide any future changes to how mechanisms stack.
- Currently trains with **frozen encoder** and classification head only; per-layer learning rates and encoder unfreezing are planned
- `src/test_script.py` is a sandbox for local testing with validation-only data and the endpoint layer pattern
- Current quality: better than chance, but hand review shows the NYT indexer tag definitions are too generous (highest-leverage improvement is refining label definitions)

## Key Design Decisions

- **Models output logits** (no final activation), so all losses use `from_logits=True`
- **Mixed precision** (`mixed_float16`) is enabled in all training scripts for GPU efficiency; local macOS may not fully support this
- **Data pipeline** uses `tf.data.Dataset` with `sample_from_datasets()` to handle PU class imbalance via weighted sampling (e.g., 1:9 pos:unl for CCA training, 1:5 for L/U classifier)
- **Re-balanced batches** are central to the PU + class imbalance strategy: every batch is guaranteed to contain labeled positive signal
- Preprocessed datasets are cached to disk (`cca_set/` directory) because `from_tensor_slices()` takes minutes on the full corpus
- Two preprocessor classes in `src/preproc/preprocessor.py`: `CustomPreprocessor` (for DAPT/MLM tasks with masking) and `ClassifierPreprocessor` (for classification tasks, supports both standard and endpoint-layer patterns)

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
- **Tier 2 refactor** (next focus, still being designed): per-layer learning rates + encoder unfreezing, and the structural groundwork for the planned multi-head classifier.
- Implement ALUM/VAT (Virtual Adversarial Training); incomplete sketch exists in `src/test_module.py`
- Benchmarking and comparative testing (FLPU vs. ALUM, etc.)
- Refine NYT indexer tag definitions for CCA and immigration labels (current definitions are too generous)
- Pull DoCA article headlines (1960-1984) via NYT Archive API to expand training data
- Build additional classification heads (immigration, US, combined ICA)
- Hyperparameter search
- Output calibration (decision threshold, not just raw .5)
- Hand-label a small PN test set (~200-1000 articles) for proper evaluation
- Retrain CCA classifier with the corrected prior (π_pos ≈ 0.02 rather than 0.03)

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
