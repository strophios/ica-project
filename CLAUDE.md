# CLAUDE.md

> ⚠️ **PARTIALLY RECONCILED (2026-06-26).** The "Planned Architecture" and "Current
> Status and Open Work" sections below were updated for the `cca-doca-retrain` arc:
> the multi-head `IcaModel` now exists and produces ICA candidates (CCA + relevance
> `rel` heads, the fused US gate, fusion, and apply are all real). The **rest** of
> this file (the older single-head CCA / standalone US-filter narrative, the data
> counts, label sources) still carries 2026-06-10 content and is stale on the
> retrain specifics. **For authoritative current state: `docs/notes/project-state-
> and-data-map.md` (data/artifact map) and `docs/notes/roadmap.md` (what's next /
> deferred).** A full line-by-line rewrite of the remaining sections is still
> deferred.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*Last updated: 2026-06-10. Orientation doc for the project as a whole — current state and contracts. For tier-by-tier history (what landed when, commit-level), see `docs/notes/tiers-and-checkpoints.md`; for per-tier design reasoning, the `docs/notes/tier{2,3,4,5}-design.md` docs. For the US/not-US pre-filter (the second classification head, plus its R dateline-labeling pipeline), see the "US/not-US Pre-Filter" section below and `r/CLAUDE.md`.*

## Project Overview

This project builds an ML system to identify **Immigrant Collective Action (ICA)** events in New York Times articles, enabling expansion of protest event datasets (like the Dynamics of Collective Action / DoCA) without massive manual coding effort. The approach uses **Positive-Unlabeled (PU) learning** with a RoBERTa backbone, built on Keras 3 + TensorFlow, targeting both local macOS (tensorflow-metal) and the HPC cluster "Explorer" (tensorflow + CUDA, paths like `/projects/ahd`).

### The Research Problem

The "haystack" problem: finding the tiny fraction of NYT articles reporting ICA events among ~1.8M articles. This is complicated by three factors:
1. **Positive-Unlabeled data**: we have labeled positives (from DoCA + NYT indexer tags) but zero confirmed negatives. Unlabeled articles may contain unreported positives (Hanna 2017 found 62% of "false positives" were actually correct).
2. **Severe class imbalance**: ICA articles are a vanishingly small minority.
3. **The project assumes SCAR** (Selected Completely At Random) for the labeling mechanism — labeled positives are treated as a random sample of all positives. This is a simplifying assumption; alternative PU approaches exist if it proves inadequate.

### Architecture: the assembled multi-head ICA model (implemented 2026-06-26)

The system is a **multi-head ICA classifier** over a shared frozen DAPT RoBERTa
encoder. It is assembled in `src/assemble_ica.py` (`IcaModel`) from three heads
that share the 768-d CLS feature, each trained features-mode on cached embeddings:
- **US** (`us`) — the US/not-US pre-filter (BCE), used as a hard gate.
- **CCA** (`cca`) — collective-action event identification (FLPU / focal-nnPU).
- **relevance** (`rel`) — immigrant relevance (FLPU). This is the head the older
  text below calls "immigration"; it was built, then renamed `cca`→`rel` in the
  Phase 3 harmonized retrain (its sidecar records `head.name=="rel"`).

`IcaModel.predict_ica_from_features((n,768))` returns calibrated `{us, cca, rel}`
probabilities plus an `ica_score`, composed as: **US gate** (`calib_us ≥ τ_us`, or a
`gate_override` mask) → **combine** calibrated CCA·rel (product-AND, or a ≤3-param LR
chosen empirically by a 1-SE rule — `src/fusion/`) → **composed Platt** →
`ica_score` (0.0 for gated-out rows). The fusion is persisted as
`cca_doca/ica_fusion.fusion.json`. `src/apply_ica.py` runs the assembled model over
the API (1960–1995, ML gate) and LDC (1996–2007, gold-first gate) corpora to produce
`cca_doca/ica_candidates/{api_1960_1995,ldc_1996_2007}.parquet`.

This decomposition leverages the larger CCA and relevance labeled datasets
separately, making the sparse ICA problem tractable. The standalone US/not-US filter
(documented below) is the same head, now consumed as the assembled gate rather than a
separate model. For full current state see `docs/notes/project-state-and-data-map.md`;
the multi-head design/assembly reasoning is in the docs linked under "Current Status"
below. **Known ceiling:** the US head misses diaspora collective action and caps
system recall — a scoped-but-deferred retrain (`docs/notes/us-head-retrain-plan.md`).

## Development Setup

- **Python 3.12**, managed with `uv` (this is a uv project — `uv.lock` is checked in)
- **Run commands with `uv run`; do not activate the venv.** `uv run <command>` (e.g. `uv run pytest`, `uv run python -m src.cca_config write_default <weights_path>`) resolves and auto-syncs the environment each time. Avoid `source .venv/bin/activate` — it is unnecessary in a uv project and only adds an activation/permission step with no benefit. The `dev` dependency group (pytest, hypothesis) installs by default, so `uv run pytest` needs no extra flag.
- Install/sync dependencies: `uv sync` (runtime + default dev group). Add a dependency with `uv add <pkg>` — never `pip install` into the venv.
- No `__init__.py` files exist (implicit namespace packages)
- All scripts must be run from the **project root** (imports use `src.*` paths, e.g., `import src.model_setup.dapt_setup`)
- **Configuration**: `src/config.py` is the single source of truth for platform-conditional values. Detects cluster vs. local via `Path("/projects/ahd").exists()` (override with `ICA_ENV=cluster|local`); exports `IS_CLUSTER`, `PROJECT_ROOT`, granular paths (`CCA_SET_DIR`, `DAPT_BACKBONE_WEIGHTS`, `LDC_CORPUS`, the US-filter `US_FILTER_*` family, `API_CORPUS_DIR`, `VALIDATION_DIR`, etc.), and `DTYPE_POLICY` (`mixed_float16` on cluster, `float32` locally — MPS mixed-precision support is patchy). Scripts apply the dtype policy explicitly: `keras.config.set_dtype_policy(config.DTYPE_POLICY)`.
- **Reproducibility**: training scripts call `keras.utils.set_random_seed(200)` to match the `seed=200` used by the polars `.sample()` splits in `src/data_setup/data.py`.

### Tests

pytest is configured (`pyproject.toml [tool.pytest.ini_options]`, `pythonpath = ["."]`). Run with `uv run pytest` from the project root (no venv activation needed). `hypothesis` (dev dependency) backs property-based tests in the diagnostics suite. **Current coverage: 618 Python tests passing** (the US-filter subsystem added the `test_us_*`, `test_calibration_*`, `test_dateline_guard`, `test_apply_us_filter`, `test_artifact_reload`, and `test_*` validation suites). The R dateline pipeline has its own testthat suite — `Rscript r/tests/run_tests.R` from the project root, currently 79 passing assertions (see `r/CLAUDE.md`). Suites:
- `tests/test_flpu_loss.py` — `FLPULoss` invariants + loss-component correctness (the `return_intermediates` path)
- `tests/test_data_splits.py` — train/val/test split, label construction, id-uniqueness in `create_classifier_data`
- `tests/test_data_loading.py` — missing-value handling + headline-with-lead concatenation in `data_from_parquet`
- `tests/test_heads.py` — `ClassificationHead` construction/shape/contract, incl. the `expose_loss_components` flag-on contract
- `tests/test_layer_lr_model.py` — `LayerLRModel` incl. guarded-diagnostic-dispatch, metrics-override, pre-scaling-invariant, endpoint-target-extraction
- `tests/test_preprocessor.py` — `ClassifierPreprocessor` construction + call-time validation
- `tests/test_cca_config.py` — `RunConfig`/`HeadConfig`/sub-configs incl. `DiagnosticsConfig` + agg-constant-sync
- `tests/test_assembly.py` — assembled stack integration, Pattern-2 serialization round-trip, diagnostics wiring
- `tests/test_diagnostics_trackers.py` / `test_diagnostics_factory.py` / `test_diagnostics_distribution_metrics.py` — the diagnostics module

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
- Feeds predictions into **DEDPUL** (Ivanov 2020) EM algorithm to estimate the positive class prior. Current best estimate is π_pos ≈ 0.02 for CCA (robust across `kde_mode` and bandwidth choices). The CCA classifier now uses `prior=0.02`; an earlier `0.03` was kept for continuity with previously-trained models and is being phased out on retrain.
- `src/prior_estimation/dedpul_em.py` and `dedpul_utils.py` are adapted from the [DEDPUL repo](https://github.com/dimonenka/DEDPUL/). DEDPUL expects the *probability* of being unlabeled (its convention is 0 = labeled, 1 = unlabeled); `run_prior_estimate.py` applies `sigmoid` + `1 - p` to convert the L/U classifier's logit output. An earlier version fed logits directly; the bulk of the observed shift on fixing this was DEDPUL's bandwidth grid being mis-calibrated for logit-scale inputs (see `scripts/compare_dedpul_logit_vs_prob.py` for the four-variant attribution table).
- `src/prior_estimation/ramaswamy2016.py` is an alternative kernel-based prior estimation method (requires `cvxopt`, currently commented out in dependencies)

### Phase 3: CCA Classification — IMPLEMENTED
- **Scripts:** `src/run_cca_classification.py` (train), `src/eval_cca_classifier.py` (evaluate).
- `run_cca_classification.py` is an importable module: `main(run_config=None, max_steps=None)` (defaults to `DEFAULT_CCA_CONFIG`; `max_steps` caps `steps_per_epoch`) guarded by `if __name__ == "__main__": main()`. Importing it does not trigger training. `scripts/tier5_short_run.py` calls `main()` with a 1-epoch / 200-step capped config for a reproducible short run; `scripts/tier5_cluster.sbatch` is a parameterized SLURM template (operator fills `<PLACEHOLDER_*>`; `short`/`full` mode arg).
- Binary classifier built from the model-construction abstractions (below): `load_dapt_backbone` (DAPT-finetuned RoBERTa) + a `ClassificationHead` named `"cca"` (FLPU loss + per-head metrics), wired by `build_endpoint_model` (training, with a `cca_targets` Input) and `build_inference_model` (predict, no target Inputs).
- Uses **FLPULoss** (`src/loss_functions/loss.py`): focal cross-entropy (Lin 2020, γ=2) wrapped in non-negative PU learning (Kiryo 2017), parameterized by the estimated class prior. The focal-loss `alpha` knob has been removed (see the FLPU docstring and `docs/notes/pinned-questions.md` for the cost-sensitive-nnPU interpretation and the four-layer composition framing that should guide future changes to how mechanisms stack).
- Trains with a **frozen encoder** and classification head only. Per-layer learning rates and encoder unfreezing are forward-compatible via `LayerLRModel` (the type returned by `build_endpoint_model`) — set `freeze_encoder=False` and pass `layer_multipliers={...}` when ready.
- Current quality: better than chance, but hand review shows the NYT indexer tag definitions are too generous (highest-leverage improvement is refining label definitions). An empirical retraining run with the corrected prior is one of the deferred items below.
- `src/test_script.py` is a pre-Tier-2 sandbox exercising the endpoint-layer training pattern with inline shapes (raw `keras_hub` backbone, ad-hoc `EndpointLayer`, plain `keras.Model`); it is not the production path. The production path is covered by `tests/test_heads.py` and `tests/test_assembly.py`.

## US/not-US Pre-Filter (second classification head)

A standalone **US/not-US** binary classifier, built on the same frozen-DAPT-backbone + single-`ClassificationHead` spine as the CCA classifier but as **plain supervised PN with BCE** (no FLPU/prior/nnPU coupling). Its purpose is to pre-filter the full NYT Archive API corpus (1960–1995) down to plausibly-US events before downstream CCA/ICA classification. Phases 1–7 of the implementation plan (`docs/implementation-plans/2026-06-06-us-filter/phase_01.md … phase_08.md`) have landed; **Phase 8 (a dialogic calibration-notes drafting session) is deliberately open.** Operator-gated items (full training run, cluster shakedown, gold-set hand-coding + slice eval / DoCA recall / escalation decision, full-corpus apply) are described in the phase files and not yet executed.

### Labels: the R dateline pipeline (`r/`)

US labels are derived **outside Python**, in the `r/` tree (see `r/CLAUDE.md` for its contracts). Key discovery driving the design (phase_01.md execution-deviation note, the authoritative amendment): **datelines do not live in the LDC parquet text** — the parquet-feeding CSV pipeline never extracted the NITF `dateline` element. They survive in the parallel `parsed_to_rds/{year}.rds` metadata (27–40% coverage). So:
- **Label channel = rds join** (`r/dateline/build_labels.R`): per-year rds `dateline` field joined onto LDC parquet by `id`, resolved structure-first via `resolve_dateline_field` (states/countries/AP-city gazetteers in `r/dateline/gazetteers/`), fused with desk/section signals (`r/dateline/label_policy.R`).
- **Text channel = hygiene only**: a strict, **conditional** dateline-stripper produces a leakage-proof `stripped_text` column (strips a caps-block prefix only if it has a date field, a recognized state/country qualifier, or is a bare AP-list city — emphasis-caps ledes like `"PILOBOLUS - that dance troupe…"` are NOT stripped).
- **Derived-parquet convention**: `build_labels.R` writes `us_filter/ldc_labeled.parquet` (the training source: `us_label` bool + `label_source` + `stripped_text`). The API corpus (`api_corpus/`, produced by `r/api_ingest/rds_to_parquet.R`) is **read-only**; `us_filter/api_us_scores/` is **derived** (apply output). These are gitignored data products, not checked in.

### Python training + apply path

- **`src/us_config.py` — `UsRunConfig`** (frozen dataclass) + `UsHeadConfig`. Deliberately parallel to `cca_config.RunConfig` but with **no FLPU/prior**; REUSES the shared sub-configs (`LRScheduleConfig`, `OptimizerConfig`, `DiagnosticsConfig`, `ResolvedSteps`) and MIRRORS the property surface (`label_keys`, `expected_columns`, `validate_against_backbone`, `to_json`/`from_json`, `config_path_for_weights` re-export). `DEFAULT_US_CONFIG` is the canonical starting point. Carries **escalation knobs** for forward-compatible fine-tuning: `freeze_encoder` (default `True`), `unfreeze_top_n` (validated `[0, 12]`), `layer_multipliers`; sidecars predating these knobs back-compat-default to frozen-probe. Convergence path: unifies with `RunConfig` via a loss-type discriminated union on the head config when the multi-head config is built.
- **`src/us_metrics.py` — `make_us_metrics()`**: canonical binary metrics (logits-space thresholds at 0.0, PR-AUC for imbalance), mirroring `make_cca_metrics`.
- **`src/run_us_classification.py`**: importable `main(run_config=None, max_steps=None)` guarded by `if __name__ == "__main__"`. Reads `us_filter/` via `data_from_parquet(..., lead_column="stripped_text")`, splits via `create_us_filter_data`, and **asserts no dateline residue** on every split (`assert_no_dateline_residue`, AC2.2 leakage guard) before training. `scripts/us_short_run.py` is the reproducible level-1 short run (1 epoch / 200 steps, local float32).
- **`src/preproc/dateline_guard.py` — `has_dateline_prefix` / `assert_no_dateline_residue`**: the Python port of the R extractor's *conditional-strip detection* half. **Boundary-inventory pair** with `r/dateline/resolve_dateline.R`: any change to the R credit-line / caps-block / delimiter / conditional-stripping logic MUST be mirrored here and vice versa. Mirrors the conditional "would-strip" semantics exactly (so unstripped emphasis-caps ledes do not register as residue).
- **`src/apply_us_filter.py` — `main(threshold=0.5)`**: batch-applies the calibrated model to the full API corpus, writes per-year parquets to `US_FILTER_SCORES_DIR` with `id` / `us_score` (calibrated, in [0,1]) / `us` (bool). Default threshold 0.5; the CCA-consumer recall recipe (pick the largest threshold whose DoCA recall ≥ target) is in `docs/notes/us-filter-threshold-recipe.md`.

### Calibration (`src/calibration/`)

Platt scaling as a post-hoc seam between logits and probabilities. `calibrator.py` (Functional Core: `platt_fit` / `platt_transform`, `PlattCalibrator`), `report.py` (pure ECE / Brier / reliability), `sidecar.py` (Imperative Shell: `*.calibration.json` I/O, `calibration_path_for_weights`). This extends Pattern-2 reload to an **artifact triple**: `*.weights.h5` + `*.config.json` + `*.calibration.json` — proven sufficient for cross-process reproduction by `src/validation/artifact_check.py` (`reload_and_score`). **Calibration-fit rule**: the calibrator is fit on **natural-balance** (un-rebalanced) data only, so the learned A/B map to the real prior rather than the training-time rebalanced one.

### Validation instruments (`src/validation/`)

Tooling for the operator-gated gold-set evaluation (mostly not yet run on real data):
- `schema.py` — durable gold-set schema (`validate_gold_set`), shared across model iterations.
- `build_coding_template.py` — stratified (era-bucket × news_desk) hand-coding candidate sampler with CLI entry point; emits schema-conforming rows with null labels.
- `slice_eval.py` — `apply_us_model` (the canonical model-application helper, reused by apply + artifact_check), `evaluate_slice`, `proxy_gap` (dateline-vs-event-location gap for pre-1986 transfer eval).
- `doca_recall.py` — `doca_recall(scored_df, threshold)` recall diagnostic over DoCA-matched articles (input must carry `doca_id` + `us_score`).
- `escalation.py` — `top_n_group_fn` (per-layer discriminative-LR grouping for unfreezing the top-N RoBERTa layers) + `escalation_decision` (frozen-probe → fine-tune decision logic).
- `free_audit.py` — heuristic free-audit metrics (error rate vs dateline labels, lead similarity).
- `artifact_check.py` — the artifact-triple reload proof.

### Data-pipeline changes supporting the US filter

- **`src/data_setup/data.py`**: `data_from_parquet` gained a `lead_column` parameter (default `"lead_paragraph"`; the US filter passes `"stripped_text"`) so the same loader serves both the CCA (raw lede) and US (dateline-stripped) text channels. New `create_us_filter_data(dataset)`: drops null `us_label` rows, splits the `True`/`False` groups separately 90/5/5 (seed=200), then **shuffles each split deterministically** (seed=200) — the within-split shuffle prevents class-blocking when `from_tensor_slices` preserves row order and `SHUFFLE_BUFFER` is smaller than the full split.

### Model-construction abstractions

These modules are the multi-head-ready spine that Phase 3 is built on. Contracts described here are the *current* state.

- **`src/model_setup/heads.py` — `ClassificationHead`** (a `keras.layers.Layer`). Supports standard mode (loss handled by outer `compile()`) and endpoint mode (loss registered via `add_loss`, needed for FLPU and eventual ALUM). Contracts:
  - `name` is **keyword-only and required**; `name=None` raises `ValueError` (prevents the silent Keras auto-name fallback that would collide across heads). Construction-site half of a boundary-inventory pair with `build_endpoint_model`'s call-site uniqueness check.
  - `metrics` (symmetric with `loss_fn`): per-head metric instances updated inside `call()` when targets are provided; metric names are prefixed with the head's name to avoid multi-head collisions.
  - `expose_loss_components=False` (keyword-only): when `True`, `call()` invokes `loss_fn.call(..., return_intermediates=True)`, stashes the components dict on the `last_components` attr (read by `LayerLRModel._dispatch_diagnostics`), and `add_loss(loss)`. Construction-time guard: `expose_loss_components=True` with a `loss_fn` lacking a `return_intermediates` parameter raises `ValueError`. The flag-off path is byte-unchanged.

- **`src/model_setup/layer_lr_model.py` — `LayerLRModel`** (a `keras.Model`). Overrides `train_step` to scale gradients by per-variable multipliers before optimizer apply (discriminative fine-tuning + gradual unfreezing). Contracts/gotchas:
  - `train_step` mirrors stock Keras `train_step` (TF backend) line-for-line plus the multiplier-scaling step. It **must** retain `_loss_tracker.update_state` (otherwise `history.history["loss"]` and TensorBoard logging break) and `optimizer.scale_loss` (otherwise `LossScaleOptimizer` silently no-ops under `mixed_float16`). Handles sparse gradients (`tf.IndexedSlices`, produced by `Embedding`) via `tf.math.scalar_mul`.
  - `diagnostic_trackers` (a `DiagnosticBundle`) and `diagnostic_head_refs` (list of `ClassificationHead`): a `metrics` property override appends the per-step trackers so Keras logs/resets them; `_dispatch_diagnostics` observes the **pre-scaling** (computed, not applied) gradients + loss components + batch targets each step. In endpoint mode (`y is None`, `x` a dict) targets are extracted from `x` by configured head name (`f"{head}_targets"`, hardened via `_head_refs_by_name`). **Strict no-op (byte-identical to the pre-diagnostics path) when `diagnostic_trackers is None`.**

- **`src/preproc/preprocessor.py` — `ClassifierPreprocessor`** (multi-head shape). `label_keys: dict[str, str]` maps output-dict-key → source-column-name; emits multi-head-shaped output in both endpoint and standard modes; casts targets to `target_dtype` (default `float32`) at preprocess time. Dual-boundary input validation: `__init__` validates internal-config validity (`text_key` non-empty string, `label_keys` is dict, `target_dtype` a valid Keras dtype, standard-mode requires non-empty `label_keys`) raising `ValueError`; `__call__` validates input-batch keys against the configured `text_key` + `label_keys` source columns (enumerating *all* missing), raising `KeyError`, and checks `text_key` dtype is string. (The companion `CustomPreprocessor` in the same file handles DAPT/MLM masking.)

- **`src/model_setup/backbone.py` — `load_dapt_backbone(weights_path)`**: the weights-only DAPT-checkpoint loader. Production `load_weights` calls pass explicit `skip_mismatch=False`.

- **`src/model_setup/assembly.py` — `build_endpoint_model` / `build_inference_model`**: wiring functions combining backbone + heads. Training model is a `LayerLRModel` with target Inputs named `"<head>_targets"` (suffix avoids op-name collision with the head Layer itself); inference model is a plain `keras.Model` with no target inputs. Sharing head Layer instances between the two gives **Pattern A** in-process weight sharing (verified safe in Keras 3 — losses filtered by graph reachability — via `scripts/experiment_endpoint_inference_evaluate.py`). `build_endpoint_model` asserts unique head names at the call site (`ValueError` listing duplicates). Optional `diagnostics=None` (a `DiagnosticsConfig`): when provided, gathers constituent trainable variables (**after** the `freeze_encoder` block, so a frozen encoder yields no spurious backbone grad tracker), calls `build_trackers`, and passes the bundle + head refs into the `LayerLRModel`.
  - **Pattern A vs. Pattern 2**: `run_cca_classification.py` uses Pattern A (in-process Layer-instance sharing between train and inference models); `eval_cca_classifier.py` uses Pattern 2 (cross-process: fresh head, weights loaded). Keras `.weights.h5` keys variables by *layer-class + positional index*, not by user-given name — so Pattern 2 is load-by-*structure*; the head-name contract is enforced at call sites (Keras `compile(loss={head: ...})` routing), not at weight load.

- **`src/cca_config.py` — `RunConfig`** (frozen dataclass) + sub-configs (`HeadConfig`, `FLPULossConfig`, `RatioBatchConfig`, `LRScheduleConfig`, `OptimizerConfig`, `DiagnosticsConfig`). Captures the architectural and research-dimension parameters of a CCA training run. JSON sidecar serialization (`.weights.h5` → `.config.json` via `config_path_for_weights`); `validate_against_backbone(backbone)` for cross-object hidden_dim validation; `DEFAULT_CCA_CONFIG` as the canonical starting point; CLI helper (`uv run python -m src.cca_config write_default <weights_path>`). Validation hierarchy: each dataclass's `__post_init__` validates its own fields; cross-object invariants (head names unique) at `RunConfig`; external-context invariants (backbone match) in `validate_against_backbone`. `HeadConfig` rejects head names containing `/` (Keras's variable-path separator used by `_default_group_fn` — a `/` would silently mis-group variables and break discriminative LR).
  - **LR schedule resolution**: `ResolvedSteps(warmup_steps, decay_steps, steps_per_epoch)` (frozen) carries the integer step counts the LR schedule consumes; `LRScheduleConfig.resolved: ResolvedSteps | None` + `with_resolved(steps_per_epoch)` populate it via `math.floor(factor * steps_per_epoch)`. `run_cca_classification.py` calls `with_resolved` after computing `steps_per_epoch` and reads `resolved.warmup_steps` / `resolved.decay_steps` when constructing `CosineDecay`, so the sidecar records concrete step counts. Validation is intentionally duplicated between `with_resolved` (outer-boundary error clarity) and `ResolvedSteps.__post_init__` — any input-rule change must touch both.
  - **`DiagnosticsConfig`** (frozen, 7 defaulted fields) is `RunConfig`'s 10th field via `dataclasses.field(default_factory=DiagnosticsConfig)`. Pre-diagnostics sidecars are back-compat: missing/`null` → `DiagnosticsConfig()` (all-enabled). Module constants `_VALID_GRADIENT_AGGREGATIONS` / `_VALID_SUMMARY_STATS` deliberately duplicate `trackers._VALID_AGGREGATIONS` (config must not import the machinery it configures) and are pinned equal by `TestDiagnosticsAggregationConstantSync` — change both together.

### Diagnostic instrumentation (`src/diagnostics/`)

Permanent observability for training runs (the tooling for the deferred empirical stress tests). The factory is the single source of truth for what a run instantiates, mirroring `src/cca_metrics.py:make_cca_metrics`.

- **`trackers.py`** — four `keras.metrics.Metric` subclasses observed inside `LayerLRModel.train_step`: `PerGroupGradNormTracker` (per-group L2 grad norm, mean/max; dense + sparse; empty group e.g. frozen encoder reports 0.0 because nothing was computed), `GradientFiniteTracker` (`grad_overflow_rate`; ≈0 under local float32, the **active** diagnostic under `mixed_float16` / `LossScaleOptimizer`), `LossComponentTracker` (aggregates an FLPU intermediate by key; `KeyError` on absent key = loss/consumer mismatch signal), `BatchLabelBalanceTracker` (running positive-class fraction; `KeyError` on absent head target). Categories are enforced by registration in `factory.py`, not by inheritance.
- **`factory.py`** — `build_trackers(config, *, group_fn, heads, trainable_variables) → DiagnosticBundle` (a `TypedDict` with `per_step: dict[str, list[Metric]]` and a permanently-empty `periodic: list` forward-compat slot). `_loss_exposes_intermediates` is the introspection guard: heads present but no loss exposing `return_intermediates` → `ValueError`; the empty-`heads` case is a deliberate no-op.
- **`distribution_metrics.py`** — `PredictionMeanMetric` / `PredictionStdMetric` / `PredictionFracAboveMetric` + `make_distribution_metrics(config)`. These ride the **head's `metrics=` path** (standard `update_state(y_true, y_pred)`, `y_true` ignored), NOT the `DiagnosticBundle` / `train_step` dispatch — computed for train and val phases per epoch with no extra forward pass. `PredictionStdMetric` accumulates in float64 to avoid float32 catastrophic cancellation in `E[s^2] - E[s]^2` (load-bearing for the std≈0 distribution-collapse signal).
- **Loss-component contract**: `FLPULoss.call(y_true, y_pred, return_intermediates=False)` — when `True`, returns `(loss, {positive_risk, negative_risk, correction_triggered})`; the loss scalar is bit-identical between the two flag paths and `get_config` is unchanged. `correction_triggered` is the nnPU clip / Kiryo active-recovery firing indicator.
- New files in this module carry FCIS `# pattern:` headers: `factory.py` is Functional Core; `trackers.py`/`distribution_metrics.py` are Mixed (unavoidable — Keras `Metric` state); the new scripts are Imperative Shell.
- **Superseded design note**: the original diagnostics design had a periodic-diagnostic / `DiagnosticsCallback` / reference-batch path; it was replaced — per-head distribution metrics ride the head's `metrics=` path instead. `DiagnosticBundle["periodic"]` is a permanently-empty forward-compat slot with no current consumer. See the supersession note in `docs/notes/tier5-design.md`.

## Key Design Decisions

- **Models output logits** (no final activation), so all losses use `from_logits=True`
- **Mixed precision** (`mixed_float16`) is enabled in all training scripts for GPU efficiency; local macOS may not fully support this
- **Data pipeline** uses `tf.data.Dataset` with `sample_from_datasets()` to handle PU class imbalance via weighted sampling (e.g., 1:9 pos:unl for CCA training, 1:5 for L/U classifier)
- **Re-balanced batches** are central to the PU + class imbalance strategy: every batch is guaranteed to contain labeled positive signal
- Preprocessed datasets are cached to disk (`cca_set/` directory) because `from_tensor_slices()` takes minutes on the full corpus

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

> **The sections below this banner describe the pre-`cca-doca-retrain` state and are
> retained for the Tier-1–5 / US-filter-phase history. They are NOT current on the
> multi-head model.** As of 2026-06-26 the multi-head `IcaModel` is assembled and
> producing ICA candidates (see "Architecture" above), the CCA + relevance (`rel`)
> heads were retrained features-mode on a harmonized population, all three heads are
> Platt-calibrated, the fusion is fit and persisted, and `apply_ica.py` has run on
> the API and LDC caches. Authoritative current state + open work:
> `docs/notes/project-state-and-data-map.md` and `docs/notes/roadmap.md`.

**Done (multi-head ICA assembly, Phases 1–6, 2026-06-26):**
- `IcaModel` assembled (`src/assemble_ica.py`); fusion module `src/fusion/`; apply
  (`src/apply_ica.py`) → `cca_doca/ica_candidates/`. CCA + relevance (`rel`)
  harmonized retrain; three Platt calibrators; clean hand-coded eval set
  (`validation/ica_coding_template_coded.csv`, 214 ICA positives).
- Two captured findings: the US head's diaspora-recall ceiling
  (`docs/notes/us-head-retrain-plan.md`, a deferred retrain) and the apply results +
  raw-vs-stripped channel correction + cluster runbook
  (`docs/notes/ica-apply-results-and-cluster-runbook.md`).

**Done (pre-retrain history):**
- DAPT on headline/lede pairs
- CCA classifier (single-head) with FLPU loss
- Class prior estimation via DEDPUL
- **Tiers 1–5 (audit/refactor + diagnostic instrumentation) — subagent-executable portion complete.** Tier 1 (correctness), Tier 2 (structural refactor — the abstractions above), Tier 3 (robustness / boundary enforcement), Tier 4 (hygiene + LR-schedule resolution + lessons docs), and Tier 5 (diagnostic instrumentation) have all landed, with per-tier adversarial review. Commit-level status is in `docs/notes/tiers-and-checkpoints.md`; per-tier design reasoning is in the `tier{2,3,4,5}-design.md` docs.
- **US/not-US pre-filter — Phases 1–7 complete** (see "US/not-US Pre-Filter" above). R dateline-labeling pipeline, US-filter config/metrics/training/apply, Platt calibration + artifact triple, and validation instruments all landed with per-phase adversarial review. **Phase 8 (dialogic calibration-notes drafting) is deliberately open**, and the operator-gated runs below are not yet performed.

**Pending — HUMAN-OPERATED (handed off, not yet performed):**
- US filter: full local training run + cluster `mixed_float16` shakedown; gold-set hand-coding (`build_coding_template.py`) → slice eval / DoCA recall / escalation decision; full-corpus apply (`apply_us_filter.py`) over `api_corpus/`. Runbooks: `docs/implementation-plans/2026-06-06-us-filter/phase_*.md` (phase_01/phase_03 carry authoritative execution-deviation amendments).
- Tier 5 Phase 7 Tasks 3–5: level-1 short + level-2 full **real-data** runs on local `cca_set/`, then create `docs/notes/tier5-stress-test-notes.md` (does not yet exist). Runbook: `docs/notes/tier5-implementation-plan/phase_07.md`.
- Tier 5 Phase 8 Tasks 2–4: short + full **cluster** `mixed_float16` runs on Explorer, then the level-3 π=0.03-vs-0.02 research handoff. Runbook: `docs/notes/tier5-implementation-plan/phase_08.md`.
- These runs are the intended coverage for the smoke-test backbone-validation gap (the synthetic stand-in's canonical boundary case). The diagnostic instrumentation above is the observability for them.

**Deferred empirical checks** (the subject of the pending Phase 7/8 runs): smoke/short + full training runs with updated FLPU + corrected prior (≈ 0.02) on real data; confirm training dynamics under `mixed_float16` + removed-α FLPU on the cluster; sensitivity sweep on Ratio Batch (currently 1:10, more aggressive than Ji 2023).

**Other open items:**
- Implement ALUM/VAT (Virtual Adversarial Training); incomplete sketch exists in `src/test_module.py`
- Benchmarking and comparative testing (FLPU vs. ALUM, etc.)
- Refine NYT indexer tag definitions for CCA and immigration labels (current definitions are too generous)
- Pull DoCA article headlines (1960-1984) via NYT Archive API to expand training data
- ~~Build additional classification heads (immigration, combined ICA); unify the standalone US filter into the shared-encoder multi-head~~ — **DONE 2026-06-26**: the relevance (`rel`) head and the combined ICA composition exist; the US filter is consumed as the assembled gate (see "Architecture" above).
- Hyperparameter search
- Output calibration (decision threshold, not just raw .5) — **partly done**: all three heads + the composed ICA score are now Platt-calibrated; τ_us picked via `doca_recall.pick_us_threshold`.
- Hand-label a small PN test set (~200-1000 articles) for proper evaluation — **partly done**: a 214-positive hand-coded ICA eval set exists (`validation/ica_coding_template_coded.csv`); a 500-row CCA gold set also exists.
- ~~Retrain CCA classifier with the corrected prior (π_pos ≈ 0.02 rather than 0.03)~~ — **DONE**: CCA retrained features-mode at π=0.02 on the DoCA-matched population.

**Handoff docs for picking up mid-work:**
- `docs/notes/tiers-and-checkpoints.md` — tier plan, commit-level status, and "how to pick up" instructions. Operational doc — read at the start of each piece, updated at piece close.
- `docs/notes/tier{2,3,4,5}-design.md` — per-tier design reasoning (Tier 2 structural refactor; Tier 3 boundary-inventory framing; Tier 4 hygiene + LR resolution + lessons docs; Tier 5 diagnostics, incl. the periodic-subsystem supersession note and the real/synthetic stress-test framing).
- `docs/notes/tier5-implementation-plan/phase_01.md … phase_08.md` — the Tier 5 implementation plan. Phases 1–6 + Phase 7 Tasks 1–2 + Phase 8 Task 1 are done; Phase 7 Tasks 3–5 and Phase 8 Tasks 2–4 are HUMAN-OPERATED runbooks not yet executed (these are the plan; do not rewrite them).
- `docs/notes/process-patterns.md` — content-agnostic process patterns (pedagogical, adversarial-review, investigator-subagent, design-doc-per-tier, implementation-plans-with-file:line-specificity, deferred-with-explicit-notes, skill-orchestrated-workflow), with Validated/Developing tiers and promotion criteria.
- `docs/notes/engineering-patterns.md` — CS-specific patterns (boundary-inventory, synthetic-stand-ins, wrapped-vs-flat-forward-compat, Pattern-A-vs-Pattern-2, empirical-investigation-before-design).
- `docs/notes/pinned-questions.md` — deliberately deferred substantive questions: (1) composing nnPU + α + γ + ALUM across the four-layer framing, (2) extension to multi-class heads, (3) `ClassifierPreprocessor` train/predict shape design smell (with a candidate `for_training`/`for_prediction` method-split refactor), (4) "scope-creep correction via revert" as a candidate process pattern.

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
