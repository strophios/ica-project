# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*Last updated: 2026-08-19. Orientation doc for the project as a whole — current state and contracts. This is the full line-by-line reconciliation for the `cca-doca-retrain` arc (merged via PR #1) and the multi-head assembly that followed; it supersedes the earlier partially-reconciled version. Reconciled 2026-07-30 for the encoder-unfreeze / tuned-cache arc (the deployed `IcaModel` is unchanged; a validated mixed-stack rel candidate now exists — see Architecture), and 2026-08-11 for the branched-encoder decision (`docs/notes/branched-encoder-strategy.md` — the experiment ladder that reordered roadmap §A1; no code or artifact changes). For the live data/artifact map and model-state snapshot, see `docs/notes/project-state-and-data-map.md`; for what's next / deferred, `docs/notes/roadmap.md` (the single live next-steps list). For the US/not-US label pipeline (R), see `r/CLAUDE.md`.*

## Project Overview

This project builds an ML system to identify **Immigrant Collective Action (ICA)** events in New York Times articles, enabling expansion of protest event datasets (like the Dynamics of Collective Action / DoCA) without massive manual coding effort. The approach uses **Positive-Unlabeled (PU) learning** with a RoBERTa backbone, built on Keras 3 + TensorFlow, targeting both local macOS (tensorflow-metal) and the HPC cluster "Explorer" (tensorflow + CUDA, paths like `/projects/ahd`).

### The Research Problem

The "haystack" problem: finding the tiny fraction of NYT articles reporting ICA events among millions. This is complicated by three factors:
1. **Positive-Unlabeled data**: we have labeled positives (now DoCA-matched articles; see Data) but zero confirmed negatives. Unlabeled articles may contain unreported positives (Hanna 2017 found 62% of "false positives" were actually correct).
2. **Severe class imbalance**: ICA articles are a vanishingly small minority.
3. **The project assumes SCAR** (Selected Completely At Random) for the labeling mechanism — labeled positives are treated as a random sample of all positives. This is a simplifying assumption; alternative PU approaches exist if it proves inadequate.

## Architecture: the assembled multi-head ICA model

The system is a **multi-head ICA classifier** over a shared frozen DAPT RoBERTa encoder. It is assembled in `src/assemble_ica.py` (`IcaModel`) from three heads that share the 768-d CLS feature, each trained **features-mode** on cached embeddings (see below):
- **US** (`us`) — the US/not-US pre-filter (BCE), used as a hard gate.
- **CCA** (`cca`) — collective-action event identification (FLPU / focal-nnPU), DoCA-trained.
- **relevance** (`rel`) — immigrant relevance (FLPU). Built as the "immigration" head, then renamed `cca`→`rel` in the Phase 3 harmonized retrain (its sidecar records `head.name=="rel"`).

`IcaModel.predict_ica_from_features((n,768))` returns calibrated `{us, cca, rel}` probabilities plus an `ica_score`, composed as: **US gate** (`calib_us ≥ τ_us`, or a `gate_override` mask) → **combine** calibrated CCA·rel (product-AND, or a ≤3-param LR chosen empirically by a 1-SE CV rule — `src/fusion/`) → **composed Platt** → `ica_score` (0.0 for gated-out rows). The fusion is persisted as `cca_doca/ica_fusion.fusion.json`. `src/apply_ica.py` runs the assembled model over the API (1960–1995, ML gate) and LDC (1996–2007, gold-first gate) corpora to produce `cca_doca/ica_candidates/{api_1960_1995,ldc_1996_2007}.parquet`. Reload-proof in `src/validation/artifact_check.py:reload_and_score_ica`.

The decomposition leverages the larger CCA and relevance labeled datasets separately, making the sparse ICA problem tractable. **Text-mode `predict_ica_from_text` is not implemented** — features-mode (cached CLS embeddings) is the path.

**Known ceiling (reframed 2026-07-24):** the US head under-scores diaspora collective action (US-soil action about foreign matters) because its dateline labels encode *filing* location, not *event* location — but at the *deployed* lenient gate (τ_us=0.02) the recall ceiling is modest (~5/26 diaspora anchors dropped; the oft-quoted 27% figure was measured at τ=0.3). The gate's real deployed weakness is the **foreign leak at high recall** (~27–31% of known-foreign rejected at ≥0.98 US recall). A v1 retrain (nnPNU on DoCA + dateline-negative labels) was executed and evaluated 2026-07-24 — **decision: no swap** (the current head wins the gate regime); results + keeper findings in `docs/notes/us-head-retrain-plan.md`.

### Model quality (as of 2026-07-10)

On the boundary-enriched hand-coded ICA eval (1,131 rows, 214 ICA positives, ~19% base rate) the composed `ica_score` separates ICA from non-ICA at **ROC-AUC 0.80** / PR-AUC 0.61. Decomposed vs the ICA label: rel 0.78 (the workhorse ranker), CCA 0.62, US 0.38 (below chance *as a ranker* by design — ICA is not monotone in US-ness). Per-head **own-terms** eval (each head vs its own hand-coded dimension) inverts that ranking: own-terms ROC US 0.925 / CCA 0.927 / rel 0.829 — CCA discriminates collective action well on its own terms; rel, the best ICA ranker, is the weakest head at its own job. Precision numbers from the enriched eval *overstate* corpus precision (no random-sample gold set exists yet). Full numbers and caveats: `ml_memo/ica_model_state_2026-06.md` (June) and `ml_memo/ica_model_update_2026-07.md` (July); artifact `cca_doca/experiments/eval_heads_own_terms.json`.

### Validated mixed-stack candidate (2026-07-30, NOT yet productionized)

The deployed `IcaModel` is unchanged (all-frozen-DAPT). The encoder-unfreeze arc produced a **validated but un-deployed** improvement: fine-tuning the DAPT encoder with the **rel** head's loss only (rel-first sequential, top-N unfreeze + discriminative LR; `docs/notes/encoder-unfreeze-strategy.md`) genuinely lifts rel (own-terms 0.83→0.85, diaspora recall 0.38→0.66) but **negatively transfers** to CCA (0.93→0.74) and US (0.93→0.83) — caught by the pre-registered re-eval, so no wholesale encoder swap. A hand-composed **mixed stack** (tuned rel head `relevance/relevance_tuned.weights.h5` reading a tuned-backbone CLS cache; production CCA/US on the production DAPT cache; production fusion/calibrations) lifts **composed ICA ROC 0.80→0.82** and diaspora recall@0.10 0.221→0.250 — a real system gain from the rel swap alone. **Two 2026-08-12 corrections** (`docs/notes/metal-execution-findings.md`): (1) the July tuned-head artifacts are **tensorflow-metal-execution-bound** (trained on metal; `relevance_tuned` scores 0.853 vs-ICA under metal and 0.386 under correct CPU/CUDA math — do not deploy or re-score them as portable); correct-math CPU retrains confirm the rel gain (vs-ICA 0.852–0.855, diaspora 0.662) and soften the transfer magnitudes (CCA 0.928→0.795, US F1 0.97→0.951; verdict direction unchanged), while the composed 0.82 awaits re-measurement at the fusion refit. (2) The branched-encoder **graft test passed** (ladder stage 1, `docs/notes/branched-encoder-strategy.md` execution record): the entire rel gain lives in `transformer_layer_11`, so the mixed stack is a **K=1 branched model** (shared pristine trunk + two top-layer variants, ~1.08× one encoder pass, not 2×). **The ladder completed 2026-08-19: all four stages run, and the branched architecture won** — the joint CCA+rel sweep's best cell only tied the branched composed-proxy bar (0.808 vs 0.8064) with both per-head guardrails failed, so joint retired as a clean negative result (keeper finding: joint training makes the heads genuinely *complementary* — composition adds instead of subtracts — but at unaffordable per-head cost; full verdict in the strategy note's execution record). **Active work: branched productionization** — per-head-features `IcaModel`, the pristine-trunk + July-layer-12 graft artifact chain, fusion refit via the now-parameterized `fit_fusion.py`, swap decision (roadmap §A1).

## Development Setup

- **Python 3.12**, managed with `uv` (this is a uv project — `uv.lock` is checked in)
- **Run commands with `uv run`; do not activate the venv.** `uv run <command>` (e.g. `uv run pytest`, `uv run python -m src.cca_config write_default <weights_path>`) resolves and auto-syncs the environment each time. Avoid `source .venv/bin/activate` — it is unnecessary in a uv project and only adds an activation/permission step with no benefit. The `dev` dependency group (pytest, hypothesis) installs by default, so `uv run pytest` needs no extra flag.
- Install/sync dependencies: `uv sync` (runtime + default dev group). Add a dependency with `uv add <pkg>` — never `pip install` into the venv.
- No `__init__.py` files exist (implicit namespace packages)
- All scripts must be run from the **project root** (imports use `src.*` paths, e.g., `import src.model_setup.dapt_setup`)
- **Configuration**: `src/config.py` is the single source of truth for platform-conditional values. Detects cluster vs. local via `Path("/projects/ahd").exists()` (override with `ICA_ENV=cluster|local`); exports `IS_CLUSTER`, `PROJECT_ROOT`, granular paths (`CCA_SET_DIR`, `DAPT_BACKBONE_WEIGHTS`, `LDC_CORPUS`, the US-filter `US_FILTER_*` family incl. `US_FILTER_FULL_WEIGHTS`, `RELEVANCE_DOCA_WEIGHTS`, `ICA_CANDIDATES_DIR`, `API_CORPUS_DIR`, `VALIDATION_DIR`, etc.), and `DTYPE_POLICY` (`mixed_float16` on cluster, `float32` locally — MPS mixed-precision support is patchy). Scripts apply the dtype policy explicitly: `keras.config.set_dtype_policy(config.DTYPE_POLICY)`.
- **Reproducibility**: training scripts call `keras.utils.set_random_seed(200)` to match the `seed=200` used by the polars `.sample()` splits in `src/data_setup/data.py`.

### Tests

pytest is configured (`pyproject.toml [tool.pytest.ini_options]`, `pythonpath = ["."]`). Run with `uv run pytest` from the project root (no venv activation needed). `hypothesis` (dev dependency) backs property-based tests in the diagnostics suite. **Current coverage: 1168 Python tests passing** (measured 2026-07-30 via `uv run pytest`). The suite now spans the CCA/US spines, the diagnostics module, calibration, the multi-head assembly (`test_assemble_ica`, `test_feature_assembly`, `test_artifact_check_ica`), fusion (`test_fusion_combiner`, `test_fusion_sidecar`, `test_fit_fusion`), the DoCA-retrain data path (`test_cca_doca_data`, `test_embed_corpus`, `test_build_ica_eval_set`, `test_ica_eval`), the relevance head (`test_run_relevance`, `test_relevance_slice_eval`), the encoder-unfreeze / tuned-cache arc (`test_artifact_guard`, `test_tuned_cache_knobs`, `test_tuned_calibration_knobs`, `test_eval_heads_own_terms`, `test_run_us_pnu`), and preflight checks. The R side has its own testthat suite — `Rscript r/tests/run_tests.R` from the project root, **161 passing assertions as of 2026-07-24** (dateline pipeline + the archive-pull transforms; see `r/CLAUDE.md`). Representative Python suites:
- `tests/test_flpu_loss.py` — `FLPULoss` invariants + loss-component correctness (the `return_intermediates` path)
- `tests/test_data_splits.py` / `test_us_data_splits.py` — train/val/test split, label construction, id-uniqueness
- `tests/test_data_loading.py` — missing-value handling + headline-with-lead concatenation in `data_from_parquet`
- `tests/test_heads.py` — `ClassificationHead` construction/shape/contract, incl. the `expose_loss_components` flag-on contract
- `tests/test_layer_lr_model.py` — `LayerLRModel` incl. guarded-diagnostic-dispatch, metrics-override, pre-scaling-invariant, endpoint-target-extraction
- `tests/test_preprocessor.py` — `ClassifierPreprocessor` construction + call-time validation
- `tests/test_cca_config.py` / `test_us_config.py` — `RunConfig`/`UsRunConfig`/sub-configs incl. `DiagnosticsConfig` + agg-constant-sync
- `tests/test_assembly.py` / `test_assemble_ica.py` / `test_feature_assembly.py` — assembled stack integration, Pattern-2 serialization round-trip, diagnostics wiring, the multi-head `IcaModel`
- `tests/test_diagnostics_*` — the diagnostics module

## The ML pipeline

The project is a three-phase pipeline. Phase 1 (DAPT) is unchanged. Phases 2–3 have been re-run for the DoCA retrain; the **current** training path for all three heads is **features-mode on cached CLS embeddings**, not live text forward passes.

### Phase 1: Domain-Adaptive Pre-Training (DAPT) — DONE
- **Script:** `src/dapt.py`
- Fine-tunes a RoBERTa masked language model on the full LDC news corpus (~1.16M headline+lede pairs)
- Model setup: `src/model_setup/dapt_setup.py` — loads `roberta_base_en` backbone + `MaskedLMHead`, manually loads pre-trained LM head weights from a `.npy` file via layer index access (`model.layers[4]`)
- Produces the DAPT backbone weights (`dapt_backbone.weights.h5`) used by downstream phases (loaded via `src/model_setup/backbone.py:load_dapt_backbone`)
- Rationale: adapts RoBERTa's general English understanding to the NYT headline/lede domain (Gururangan 2020)

### Phase 2: Class Prior Estimation — π = 0.02 SETTLED
- **Scripts (current):** `src/run_cca_doca_prior.py` (DEDPUL re-estimate on the DoCA-matched population) → `cca_doca/prior_estimate.json`; `src/run_relevance_prior.py` (DEDPUL on the **relevance** population) → `relevance/prior_estimate.json`, **π̂ = 0.02** — supersedes the old 0.05-by-analogy default (a load-bearing correction: the wrong prior was the dominant cause of the text-mode rel-unfreeze collapse, see `docs/notes/encoder-unfreeze-strategy.md`). **Original path:** `src/prior_estimation/lu_classifier.py` + `src/run_prior_estimate.py`.
- Trains a linear L/U classifier (frozen DAPT backbone + single Dense) and feeds predictions into **DEDPUL** (Ivanov 2020) EM to estimate the positive class prior.
- **π = 0.02** is the settled operating value: DEDPUL re-estimated for the DoCA-matched population and **swept over {0.01, 0.02, 0.03, 0.05}, found to be an operating-point knob, not a quality knob** (the nnPU logit non-identifiability shifts scale, not the precision/recall frontier). The older π=0.03 is retired.
- DEDPUL details: `src/prior_estimation/dedpul_em.py` and `dedpul_utils.py` are adapted from the [DEDPUL repo](https://github.com/dimonenka/DEDPUL/). DEDPUL expects the *probability* of being unlabeled (0 = labeled, 1 = unlabeled); the prior scripts apply `sigmoid` + `1 - p` to convert L/U logits (see `scripts/compare_dedpul_logit_vs_prob.py` for the attribution table). `src/prior_estimation/ramaswamy2016.py` is an alternative kernel-based method (requires `cvxopt`, currently out of dependencies).

### Phase 3: head training — CURRENT PATH is features-mode on cached embeddings

All three heads are trained on **cached frozen-DAPT CLS embeddings (768-d)**, not live forward passes — which is why each head trains in minutes. The pipeline:

1. **`src/embed_corpus.py`** — runs the frozen DAPT encoder over a corpus once, caches CLS features to `cca_doca/embed_cache/<suffix>/` (`train250k`, `full`, `ldc_9507`, `us_train_ldc`, etc.). This is the only expensive (GPU/cluster) step. Gained a `--backbone-weights` override (records the producing backbone in the cache provenance fields — load-bearing for token-mode eval consumers like `apply_us_model`) used to re-embed under a *tuned* encoder. **Ordering gotcha:** the `us_weights` load must not clobber the `--backbone-weights` override (a 2026-07-29 bug silently reproduced DAPT embeds through a whole retrain; now fixed and guarded by a **mandatory Step 0b cosine check** that any "tuned" re-embed actually moved the representation — `docs/notes/tuned-retrain-runbook.md`).
2. **`src/build_cca_doca_table.py`** / **`src/build_relevance_table.py`** — assemble the labeled training tables (positives + US-restricted unlabeled background) for CCA and rel.
3. **CCA:** `src/run_cca_doca.py` (features-mode FLPU on `train250k`); prior from `run_cca_doca_prior.py`. Two parallel definitions carried forward per a collaborator scoping call: **all-forms** (`cca_doca.weights.h5`) and **street-only** (`cca_doca_street.weights.h5`). Scored via `src/score_cca_doca.py`.
4. **relevance:** `src/run_relevance.py` (features-mode FLPU, η=0, fused-gated). Positives = curated immigration-content descriptors ∪ ICA anchors, US-restricted.
5. **US:** `src/run_us_features.py` (features-mode BCE on the `us_train_ldc` cache) → `us_classifier_full.weights.h5`, held-out test F1 0.97. (The old `us_classifier.weights.h5` is a 200-step smoke test — do not use it.)
6. **Calibration:** `src/calibrate_cca.py` / `src/calibrate_relevance.py` / `src/calibrate_us_filter.py` — Platt scaling, fit on natural-balance (un-rebalanced) data, IPW-weighted to corpus base rate. All three heads carry a `.calibration.json` sidecar (the artifact triple, below).
7. **Fusion:** `src/fit_fusion.py` → `cca_doca/ica_fusion.fusion.json` (+ `ica_fusion_metrics.json`).
8. **Assemble + apply:** `src/assemble_ica.py` (`IcaModel`) → `src/apply_ica.py` → `cca_doca/ica_candidates/`.

The **Phase 3 harmonized retrain** trained CCA and rel on a *shared* population: same `us_classifier_full` weights, the same **fused** US gate (`src/preproc/us_location.apply_fused_us_gate`), and the same clean-ICA holdout excluded from both (leakage guard `src/data_setup/data.assert_holdout_excluded`, which spans four consumers: US head, CCA head, rel head, ICA eval).

**Encoder-unfreeze / tuned-cache machinery (2026-07).** To fine-tune the shared encoder, the rel head is trained **text-mode with top-N unfreeze** via `src/run_relevance_text.py` (escalation knobs; `--eta`/`--prior`/`--peak-lr`/`--graded-decay`/`--weights-out`; a 3-stream text Ratio-Batch), over the text-bearing population table from `src/build_relevance_text_table.py`; `src/extract_tuned_backbone.py` then pulls the tuned encoder out of the trained artifact so the corpora can be re-embedded under it. The whole tuned retrain-and-compare sequence (re-embed → features-retrain US/CCA on the tuned cache → recalibrate → refit fusion → gold re-eval, without touching any production artifact) is the runbook `docs/notes/tuned-retrain-runbook.md`. **`src/artifact_guard.py` — `check_no_production_overwrite`** (Functional Core) is wired into all six trainers/calibrators: passing a non-default `--suffix`/cache while leaving the weights path at its production default fails loudly rather than clobbering a production artifact. **Known gap:** `src/fit_fusion.py` is fully hardcoded (no CLI, no per-head weights/cache params) — a tuned fusion refit needs it parameterized first; `output_dir` is the sole escape hatch and silently defaults to the production fusion path (runbook Step 6 / roadmap A2).

**Older text-mode path (reference, still tested, NOT how current heads were trained):** `src/run_cca_classification.py` (train) and `src/eval_cca_classifier.py` (eval) build a text-mode single-head CCA classifier — `load_dapt_backbone` + a `ClassificationHead` named `"cca"`, wired by `build_endpoint_model` / `build_inference_model`, FLPU loss, live text preprocessing. This remains the reference implementation for the endpoint-model training pattern (covered by `tests/test_heads.py`, `tests/test_assembly.py`) and the importable-`main()` convention (`scripts/tier5_short_run.py`, `scripts/tier5_cluster.sbatch`), but the deployed heads are the features-mode artifacts above. `src/test_script.py` / `src/endpoint_layer_test.py` are pre-Tier-2 sandboxes exercising the endpoint pattern with inline shapes; not the production path.

## US/not-US Pre-Filter (the `us` head)

The US head is a binary US/not-US classifier on the same frozen-DAPT-backbone + single-`ClassificationHead` spine, trained as **plain supervised PN with BCE** (no FLPU/prior/nnPU coupling). It pre-filters plausibly-US events before CCA/rel/ICA scoring and is consumed as the assembled model's hard gate. The current deployed head is the features-mode `us_classifier_full` (F1 0.97, Platt-calibrated A=1.03/B=−0.22, ECE 0.007→0.004); its DoCA-recall sweep gives 0.96 @ calib 0.5, and the recipe threshold 0.25 reaches 0.98.

**v1 diaspora-recall retrain (2026-07-24, decision: NO SWAP).** A nnPNU retrain on event-location labels (DoCA + section-derived reliable P/N) was built and evaluated but the current head wins the deployed-gate regime; details + keeper findings in `docs/notes/us-head-retrain-plan.md`. Its machinery is **permanent infrastructure** for any future US-head experiment: `src/build_us_pnu_table.py` (the P/N/U corpus), `src/build_holdout_ids_ldc.py` + `validation/ica_holdout_ids_ldc.parquet` (the dual-id-space eval-anchor holdout), `src/run_us_pnu.py` (nnPNU trainer; `create_us_pnu_data`), and eval `scripts/eval_us_retrain.py`. Finding: pure nnPU (η=0) collapses structurally at this population's high prior (π̂≈0.83) — the PN term is load-bearing.

### Labels: the R dateline pipeline (`r/`)

US labels are derived **outside Python**, in the `r/` tree (see `r/CLAUDE.md` for its contracts — do not duplicate here). Key facts:
- **Datelines do not live in the LDC parquet text** — the CSV pipeline never extracted the NITF `dateline`. They survive in the parallel `parsed_to_rds/{year}.rds` metadata (27–40% coverage). So the **label channel is the rds join** (`r/dateline/build_labels.R`), resolved structure-first against gazetteers (`r/dateline/gazetteers/`) and fused with desk/section signals (`r/dateline/label_policy.R`).
- **Text channel = hygiene only**: a strict, **conditional** dateline-stripper produces a leakage-proof `stripped_text` column (strips a caps-block prefix only if it has a date field, a recognized state/country qualifier, or is a bare AP-list city — emphasis-caps ledes are NOT stripped).
- **Derived-parquet convention**: `build_labels.R` writes `us_filter/ldc_labeled.parquet` (`us_label` bool + `label_source` + `stripped_text`). The API corpus (`api_corpus/`) is **read-only**; `us_filter/api_us_scores/` is **derived**. These are gitignored data products.

### Python config, guard, calibration, validation

- **`src/us_config.py` — `UsRunConfig`** (frozen dataclass) + `UsHeadConfig`. Parallel to `cca_config.RunConfig` but with **no FLPU/prior**; REUSES the shared sub-configs (`LRScheduleConfig`, `OptimizerConfig`, `DiagnosticsConfig`, `ResolvedSteps`) and MIRRORS the property surface. `DEFAULT_US_CONFIG` is canonical. Carries escalation knobs (`freeze_encoder` default `True`, `unfreeze_top_n` validated `[0,12]`, `layer_multipliers`); sidecars predating these back-compat-default to frozen-probe.
- **`src/us_metrics.py` — `make_us_metrics()`**: canonical binary metrics (logits-space thresholds at 0.0, PR-AUC), mirroring `make_cca_metrics`.
- **`src/preproc/dateline_guard.py` — `has_dateline_prefix` / `assert_no_dateline_residue`**: Python port of the R extractor's *conditional-strip detection* half. **Boundary-inventory pair** with `r/dateline/resolve_dateline.R`: any change to the credit-line / caps-block / delimiter / conditional-stripping logic MUST be mirrored in both. Mirrors the conditional "would-strip" semantics exactly.
- **`src/preproc/us_location.py`** — fuses the dateline ML filter with a location signal (`any_us`/`any_not_us`): `apply_fused_us_gate` (the fused gate — halves the foreign-event leak; `us = us_ml & ~(any_not_us & ~any_us)`, so it can only make the ML gate more restrictive), `gold_first_us_gate` (trust an authoritative label — DoCA match / dateline `us_label` — over the ML head, used for the LDC apply).
- **`src/run_us_classification.py`** — the text-mode US training entry (importable `main`); reads `us_filter/` via `data_from_parquet(..., lead_column="stripped_text")`, and **asserts no dateline residue** on every split (AC2.2 leakage guard) before training. `src/apply_us_filter.py` — `main(threshold=0.5)` batch-applies the calibrated head to the API corpus. **Known bug:** `run_us_classification.py` still reads `us_filter/**/*.parquet` via a greedy glob that pulls in `audit/api_ldc_matched.parquet` (no `id` column → crash); apply the additive `pattern=` fix (as done for `data_from_parquet`) when that path is next touched.

### Calibration (`src/calibration/`)

Platt scaling as a post-hoc seam between logits and probabilities. `calibrator.py` (Functional Core: `platt_fit` / `platt_transform`, `PlattCalibrator`), `report.py` (pure ECE / Brier / reliability), `sidecar.py` (Imperative Shell: `*.calibration.json` I/O). This extends Pattern-2 reload to an **artifact triple**: `*.weights.h5` + `*.config.json` + `*.calibration.json` — proven sufficient for cross-process reproduction by `src/validation/artifact_check.py`. **Calibration-fit rule**: the calibrator is fit on **natural-balance** (un-rebalanced) data only, so the learned A/B map to the real prior.

### Validation instruments (`src/validation/`)

- `schema.py` — durable gold-set schema (`validate_gold_set`).
- `ica_eval.py` — the ICA eval-set assembly (`assemble_eval_frame`, `apply_us_scope_to_ica`, `holdout_ids_from_template`, `reserve_anchor_holdout`); feeds the hand-coded `validation/ica_coding_template_coded.csv` (214 ICA positives).
- `doca_recall.py` — `doca_recall` + `pick_us_threshold` (the τ_us recall recipe, anchored on DoCA positives).
- `slice_eval.py` / `cca_slice_eval.py` / `relevance_slice_eval.py` — `apply_us_model` / `apply_*_model` helpers + slice metrics.
- `cca_oos_eval.py` — out-of-sample eval (LDC 1995–2007 ROC-AUC 0.89 all-forms / 0.90 street vs `cca_descriptor`).
- `escalation.py` — `top_n_group_fn` / `per_layer_group_fn` (per-layer discriminative-LR grouping) + `graded_multipliers` (ULMFiT-style graded per-layer LR) + `escalation_decision` (frozen-probe → fine-tune logic). **Fix (2026-07-27):** `top_n_group_fn` never matched real backbone variable paths until the `transformer_layer` naming was corrected — the grouping silently no-op'd before that.
- `build_coding_template.py` / `build_cca_coding_template.py` / `build_ica_coding_template.py` — stratified hand-coding samplers.
- `free_audit.py` — heuristic free-audit metrics. `artifact_check.py` — the artifact-triple reload proof (incl. `reload_and_score_ica`).

### Data-pipeline support

- **`src/data_setup/data.py`**: `data_from_parquet` gained a `lead_column` parameter (default `"lead_paragraph"`; the US filter passes `"stripped_text"`; CCA/rel use raw `headline_with_lead`) and a `pattern=` override (fixes the greedy-glob crash). `create_us_filter_data` drops null `us_label` rows, splits the `True`/`False` groups separately 90/5/5 (seed=200), then shuffles each split deterministically (prevents class-blocking). `assert_holdout_excluded` enforces the four-consumer clean-ICA holdout.

## Model-construction abstractions

These modules are the multi-head-ready spine. Contracts here are the *current* state.

- **`src/model_setup/heads.py` — `ClassificationHead`** (a `keras.layers.Layer`). Supports standard mode (loss handled by outer `compile()`) and endpoint mode (loss registered via `add_loss`, needed for FLPU and eventual ALUM). Contracts:
  - `name` is **keyword-only and required**; `name=None` raises `ValueError` (prevents the silent Keras auto-name fallback that would collide across heads). Construction-site half of a boundary-inventory pair with `build_endpoint_model`'s call-site uniqueness check.
  - `metrics` (symmetric with `loss_fn`): per-head metric instances updated inside `call()` when targets are provided; metric names are prefixed with the head's name to avoid multi-head collisions.
  - `expose_loss_components=False` (keyword-only): when `True`, `call()` invokes `loss_fn.call(..., return_intermediates=True)`, stashes the components dict on the `last_components` attr (read by `LayerLRModel._dispatch_diagnostics`), and `add_loss(loss)`. Construction-time guard: `expose_loss_components=True` with a `loss_fn` lacking a `return_intermediates` parameter raises `ValueError`. The flag-off path is byte-unchanged.

- **`src/model_setup/layer_lr_model.py` — `LayerLRModel`** (a `keras.Model`). Overrides `train_step` to scale gradients by per-variable multipliers before optimizer apply (discriminative fine-tuning + gradual unfreezing). Contracts/gotchas:
  - `train_step` mirrors stock Keras `train_step` (TF backend) line-for-line plus the multiplier-scaling step. It **must** retain `_loss_tracker.update_state` (otherwise `history.history["loss"]` and TensorBoard logging break) and `optimizer.scale_loss` (otherwise `LossScaleOptimizer` silently no-ops under `mixed_float16`). Handles sparse gradients (`tf.IndexedSlices`) via `tf.math.scalar_mul`.
  - **Loss-weighting vs stock Keras — RESOLVED 2026-07-24 (verified against installed Keras 3.12 stock trainer):** the flatten discrepancy was already fixed and the batch-size weighting is correct-by-design (partial-batch proportionality). Real drift found and fixed in the same check: the tracked loss now mirrors stock's replica unscaling under `tf.distribute` (identity on single-device, display-only). Encoder-unfreeze runs (roadmap thread A item 3) are clear on this axis.
  - `diagnostic_trackers` (a `DiagnosticBundle`) and `diagnostic_head_refs`: a `metrics` property override appends the per-step trackers; `_dispatch_diagnostics` observes the **pre-scaling** gradients + loss components + batch targets each step. In endpoint mode targets are extracted from `x` by configured head name (`f"{head}_targets"`). **Strict no-op (byte-identical to the pre-diagnostics path) when `diagnostic_trackers is None`.**

- **`src/preproc/preprocessor.py` — `ClassifierPreprocessor`** (multi-head shape). `label_keys: dict[str, str]` maps output-dict-key → source-column-name; emits multi-head-shaped output in both endpoint and standard modes; casts targets to `target_dtype` (default `float32`). Dual-boundary input validation: `__init__` validates internal-config validity (raising `ValueError`); `__call__` validates input-batch keys against the configured columns (enumerating *all* missing, raising `KeyError`). (The companion `CustomPreprocessor` handles DAPT/MLM masking.)

- **`src/model_setup/backbone.py` — `load_dapt_backbone(weights_path)`**: the weights-only DAPT-checkpoint loader. Production `load_weights` calls pass explicit `skip_mismatch=False`.

- **`src/model_setup/assembly.py` — `build_endpoint_model` / `build_inference_model`**: wiring functions combining backbone + heads. Training model is a `LayerLRModel` with target Inputs named `"<head>_targets"`; inference model is a plain `keras.Model` with no target inputs. Sharing head Layer instances between the two gives **Pattern A** in-process weight sharing (verified safe in Keras 3 — losses filtered by graph reachability). `build_endpoint_model` asserts unique head names at the call site. Optional `diagnostics=None`: when provided, gathers constituent trainable variables (**after** the `freeze_encoder` block) and passes the bundle + head refs into the `LayerLRModel`.
  - **Pattern A vs. Pattern 2**: in-process runs (e.g. the text-mode `run_cca_classification.py`) use Pattern A (Layer-instance sharing); cross-process reload (`eval_cca_classifier.py`, and the features-mode heads assembled by `IcaModel`) uses Pattern 2 (fresh head, weights loaded by *structure* — Keras `.weights.h5` keys variables by layer-class + positional index, not by user-given name; the head-name contract is enforced at call sites, not at weight load).

- **`src/cca_config.py` — `RunConfig`** (frozen dataclass) + sub-configs (`HeadConfig`, `FLPULossConfig`, `RatioBatchConfig`, `LRScheduleConfig`, `OptimizerConfig`, `DiagnosticsConfig`). Captures a CCA training run. JSON sidecar (`.weights.h5` → `.config.json` via `config_path_for_weights`); `validate_against_backbone(backbone)` for hidden_dim validation; `DEFAULT_CCA_CONFIG` canonical; CLI helper (`uv run python -m src.cca_config write_default <weights_path>`). Validation hierarchy: each dataclass validates its own fields; cross-object invariants at `RunConfig`; external-context in `validate_against_backbone`. `HeadConfig` rejects head names containing `/` (Keras's variable-path separator — a `/` would silently mis-group variables and break discriminative LR).
  - **LR schedule resolution**: `ResolvedSteps(warmup_steps, decay_steps, steps_per_epoch)` (frozen) carries integer step counts; `LRScheduleConfig.resolved` + `with_resolved(steps_per_epoch)` populate it via `math.floor(...)`. Validation is intentionally duplicated between `with_resolved` and `ResolvedSteps.__post_init__` — any input-rule change must touch both.
  - **`DiagnosticsConfig`** (frozen, 7 defaulted fields) is `RunConfig`'s 10th field. Pre-diagnostics sidecars are back-compat (missing/`null` → `DiagnosticsConfig()`). Module constants `_VALID_GRADIENT_AGGREGATIONS` / `_VALID_SUMMARY_STATS` deliberately duplicate `trackers._VALID_AGGREGATIONS` and are pinned equal by `TestDiagnosticsAggregationConstantSync` — change both together.

## Fusion (`src/fusion/`)

The empirically-chosen combination of the calibrated CCA and rel probabilities into the ICA score, conditional on the US gate. `combiner.py` (Functional Core: `combine_and`, `fit_/apply_logistic_combiner`, `FusionConfig`); `sidecar.py` (Imperative Shell: `save_/load_fusion`, `fusion_path_for_weights`). The combiner choice (product-AND vs ≤3-param LR over calibrated CCA·rel) is made empirically by a pre-registered 1-SE CV rule on the conditional-on-US population (`src/fit_fusion.py`); a composed Platt calibrates the product. Persisted as `cca_doca/ica_fusion.fusion.json`. An unfrozen encoder is what would justify replacing product-AND with a learned combined head (roadmap thread A item 3).

## Diagnostic instrumentation (`src/diagnostics/`)

Permanent observability for training runs. The factory is the single source of truth for what a run instantiates, mirroring `src/cca_metrics.py:make_cca_metrics`.

- **`trackers.py`** — four `keras.metrics.Metric` subclasses observed inside `LayerLRModel.train_step`: `PerGroupGradNormTracker` (per-group L2 grad norm; empty group e.g. frozen encoder reports 0.0), `GradientFiniteTracker` (`grad_overflow_rate`; the active diagnostic under `mixed_float16` / `LossScaleOptimizer`), `LossComponentTracker` (aggregates an FLPU intermediate by key; `KeyError` = loss/consumer mismatch), `BatchLabelBalanceTracker` (running positive-class fraction). Categories are enforced by registration in `factory.py`, not by inheritance.
- **`factory.py`** — `build_trackers(config, *, group_fn, heads, trainable_variables) → DiagnosticBundle` (a `TypedDict` with `per_step` and a permanently-empty `periodic` forward-compat slot). `_loss_exposes_intermediates` is the introspection guard: heads present but no loss exposing `return_intermediates` → `ValueError`; empty-`heads` is a deliberate no-op.
- **`distribution_metrics.py`** — `PredictionMeanMetric` / `PredictionStdMetric` / `PredictionFracAboveMetric` + `make_distribution_metrics(config)`. These ride the **head's `metrics=` path** (standard `update_state`, `y_true` ignored), NOT the `train_step` dispatch — computed for train and val per epoch with no extra forward pass. `PredictionStdMetric` accumulates in float64 to avoid catastrophic cancellation in `E[s^2] - E[s]^2` (load-bearing for the std≈0 distribution-collapse signal).
- **Loss-component contract**: `FLPULoss.call(y_true, y_pred, return_intermediates=False)` — when `True`, returns `(loss, {positive_risk, negative_risk, correction_triggered})`; the loss scalar is bit-identical between the two flag paths. `correction_triggered` is the nnPU clip / Kiryo active-recovery firing indicator.
- New files carry FCIS `# pattern:` headers. **Superseded design note**: the original periodic-diagnostic / `DiagnosticsCallback` / reference-batch path was replaced by the head's `metrics=` path; `DiagnosticBundle["periodic"]` is a permanently-empty forward-compat slot (see `docs/notes/tier5-design.md`).

## Key Design Decisions

- **Models output logits** (no final activation), so all losses use `from_logits=True`
- **Mixed precision** (`mixed_float16`) is enabled for GPU efficiency on the cluster; local macOS uses `float32` (MPS mixed-precision support is patchy)
- **Data pipeline** uses `tf.data.Dataset` with `sample_from_datasets()` to handle PU class imbalance via weighted sampling (e.g., 1:9 pos:unl for CCA training, 1:5 for L/U classifier)
- **Re-balanced batches** are central to the PU + class imbalance strategy: every batch is guaranteed to contain labeled positive signal. **Calibration is fit on natural-balance data** so the A/B map to the real prior, not the rebalanced one.
- **Features-mode over cached CLS embeddings** is the current training path for all deployed heads — the frozen encoder is run once per corpus (`embed_corpus.py`), so each head trains in minutes.
- **Headline + lede as input** (`headline + "</s>" + lead_paragraph`), not full articles — captures the signal cheaply and enables eventual expansion to the full NYT (1851–present) via the free NYT Archive API.
- **The US head is a hard gate, not a fusion feature** — ICA is not monotone in US-ness; using it as a ranking signal would hurt.

## Data

- **Sources**: Dynamics of Collective Action (DoCA, 1960–1995, ~23.6k events) + two NYT corpora — the NYT Archive **API corpus** (parquet, 1960–1995, ~3.7M rows; `id, headline, lead_paragraph, abstract, keywords, year, news_desk, section_name`; **no dateline column**) and the **LDC corpus** (NYT Annotated Corpus, hive parquet partitioned on `publication_year`, 1987–2007, ~1.16M rows; carries `full_text` and the `cca`/`immig` descriptor labels, plus datelines in the parallel rds metadata).
- **Text input**: `headline + "</s>" + lead_paragraph` (RoBERTa separator token).
- **CCA positives are now DoCA-matched articles**, NOT the over-generous NYT indexer descriptors. Lineage: `doca.csv` → matched by `tmp.R` (fuzzy headline + exact `pub_date`) → `cca_matches_good.rds` (~15,627 unique `article_id`) → `r/doca/export_cca_positives.R` → `cca_doca/cca_doca_positives.parquet` → the `cca_label` (~13,742 in the training table after US-restriction). The unlabeled background is US-restricted articles (`us_logit ≥ 0`).
- **The NYT indexer descriptors remain in use**, but no longer as the CCA label: the immigration-content descriptors ∪ ICA anchors are the **rel head's** positive pool, and the LDC `cca`/`immig` descriptors serve as out-of-sample eval references (`cca_oos_eval.py`).
- **US labels**: `r/dateline/build_labels.R` → `us_filter/ldc_labeled.parquet` (`us_label` from datelines fused with desk/section; `stripped_text` the leakage-guarded channel).
- **Corpus expansion (2026-07, ongoing):** the resumable Archive-API pull (`r/api_ingest/pull_archive.R` + `archive_transform.R`, checkpointing to `nyt_archive_raw/`) now covers **1960–2025** (`api_corpus/` parquet, 66 files, ~6.37M rows — **991 duplicate ids, an unresolved data-quality flag**) plus a **1870–1959 raw-only skeleton** (rotating-month sampling, not yet transformed to parquet). The deployed heads were trained/applied on 1960–1995; applying to the expansion needs an era-audit first (**hard data cutovers found:** 2025 has no `lead_paragraph` at all, and pre-1970 is largely lead-empty — coalesce policy `headline + "</s>" + coalesce(lead_paragraph, abstract)` via a planned `lead_fallback_column` embed knob; historical abstracts are NYT-*Index* register, a channel shift to validate before use — see `docs/notes/roadmap.md`).
- **Eval assets** (hand-coded): a **500-row CCA gold set** (`validation/cca_coding_first500_coded.csv`; 169 positive for all-forms, 144 for street; leakage-held-out, IPW-reweighted to corpus base rate — all-forms at logit ≥ 1.0: reweighted precision ≈ 0.82, DoCA recall ≈ 0.35, vs a protest-keyword lexicon at 0.19 and random at 0.02) and a **1,131-row ICA eval set with 214 positives** (`validation/ica_coding_template_coded.csv`; boundary-enriched, score-stratified). All numbers trace to `docs/notes/project-state-and-data-map.md` and `cca_doca/experiments/`.
- Train/val/test split: 90/5/5, applied separately to labeled and unlabeled subsets (seed=200).

**Data location:** all large data and trained artifacts live *outside* this repo — in sibling (`../`, `api_corpus/`, `ldc_corpus/`, `cca_doca/`, `relevance/`, `us_filter/`, `validation/`) and grandparent (`../../`, `LDC2008T19/`, DoCA matches) directories. The repo holds code only. The full out-of-repo map is in `docs/notes/project-state-and-data-map.md`.

## Current Status and Open Work

**Landed major arcs:**
- **DAPT** (Phase 1); single-head CCA classifier + FLPU loss; DEDPUL prior estimation (text-mode originals).
- **Tiers 1–5** — audit/refactor + diagnostic instrumentation (the abstractions above), per-tier adversarial review. History: `docs/notes/tiers-and-checkpoints.md`, `tier{2,3,4,5}-design.md`.
- **US/not-US filter** — R dateline pipeline, config/metrics/training/apply, features-mode retrain (`us_classifier_full`, F1 0.97) + Platt calibration + validation.
- **CCA/DoCA retrain (2026-06)** — DoCA-matched positives, features-mode at π=0.02, two tracks, calibrated, out-of-sample eval.
- **Relevance head + fused US gate**; **multi-head ICA assembly** (2026-06-26, `IcaModel` + fusion + apply → candidates); **per-head own-terms eval** (2026-07-10).

**Pre-Aug-6 "refine-and-apply" arc — CLOSED at the 2026-07-30 write-up freeze (`docs/notes/roadmap.md` is authoritative).** Set by the 2026-07-10 team meeting (`docs/meetings/20260710_notes.md`): refine the model, generate post-1995 ICA candidates, and check recall against DoCA + the team's coded ICA events, write-up by Aug 1 (meeting Aug 6). Outcomes: US-head retrain executed → no swap; encoder unfreeze (rel-first sequential) executed → rel wins big, CCA/US negative transfer caught by the pre-registered check, mixed stack lifts composed ICA 0.80→0.82 (see Architecture); write-up drafted (`ml_memo/ica_model_update_2026-07.md`, operator reviewing, send Aug 1). Stretch items (VAT/ALUM, temporal signal) not promoted — the PU-collapse investigation consumed the window.

**Post-meeting arc — now active (`docs/notes/roadmap.md` §A1 is authoritative; do not rewrite its detail here; reordered 2026-08-11 by `docs/notes/branched-encoder-strategy.md`; the 2026-08-06 meeting produced no project feedback).** In priority order: (1) **parameterize `fit_fusion.py`** (needed by every path); (2) the **branched-encoder experiment ladder** — **COMPLETE 2026-08-19, branched won** (graft passed; head capacity ruled out; solo depth N=1; joint retired — best cell tied the bar with guardrails failed); **now productionizing branched** (per-head-features `IcaModel`, graft artifact chain, fusion refit, swap decision); (3) **apply to the expanded corpus** (needs the coalesce `lead_fallback_column` embed knob + a 1996–2025 embed, paid only after the encoder decision; era-sliced eval); (4) the pre-registered paired **1970s lead-vs-abstract** channel experiment before any historical coalesce; (5) backward densify pull (operator-side, ongoing). VAT is an unbundled post-ladder A/B; temporal signal stays evidence-gated.

**Known ceilings / open items:**
- **US-head diaspora recall** — reframed 2026-07-24 (see Architecture): at the deployed lenient gate the ceiling is modest; the deployed weakness is the foreign leak at high recall. Retrain executed, no swap (`docs/notes/us-head-retrain-plan.md`).
- **Frozen encoder** — the deployed heads all sit on a frozen DAPT encoder. Unfreezing (rel-first) is validated as a real quality lever but negatively transfers to CCA/US; the mixed-stack candidate (Architecture) captures the rel gain un-deployed. The branched-vs-joint decision ladder (`docs/notes/branched-encoder-strategy.md`) is the pre-registered route to deploying it.
- **No random-sample gold set** — enriched-eval precision overstates corpus precision; a random-sample draw is needed to pin absolute numbers.
- **`fit_fusion.py` is fully hardcoded** — blocks any fusion refit (branched, joint, or other retrain) until parameterized; first item in roadmap §A1.
- **991 duplicate ids** in the expanded `api_corpus/` (1960–2025) — unresolved data-quality flag.
- **`run_us_classification.py` greedy-glob bug** (above) — fix on next touch.

**Handoff docs for picking up mid-work:**
- `docs/notes/project-state-and-data-map.md` — the data/artifact map + model state (filesystem-verified). **Read this first for "where does X live / what state is it in".**
- `docs/notes/roadmap.md` — the single live next-steps/deferred list + index of all other notes. **Read this first for "what's next".**
- `ml_memo/ica_model_state_2026-06.md` (June) + `ml_memo/ica_model_update_2026-07.md` (July) — collaborator-facing model numbers (composed + decomposed + own-terms; the July memo covers the encoder-unfreeze arc).
- `docs/notes/encoder-unfreeze-strategy.md` — the rel-first-vs-joint decision, the literature survey, and the full execution findings (η=0 collapse, backbone-clobber bug, the transfer verdict). **Read this first for the encoder-unfreeze arc.**
- `docs/notes/branched-encoder-strategy.md` — the 2026-08-11 successor decision: the branched-vs-joint design space, the mixed-stack-is-a-K=1-branch reframe, and the pre-registered experiment ladder with decision rules (+ execution record: stage 1 passed 2026-08-12). **Read this first for the active model arc.**
- `docs/notes/metal-execution-findings.md` — the 2026-08-12 finding that the July tuned-head artifacts are metal-execution-bound, the corrected correct-math numbers, and the CPU-force / acceptance-check / calibration-path deployment rules. **Read before training or scoring any head locally.**
- `docs/notes/tuned-retrain-runbook.md` — exact command sequence to retrain US/CCA on a tuned cache and compare, plus the mandatory Step 0b representation-changed check and the `fit_fusion.py`/`run_relevance.py` gaps.
- `docs/notes/ica-apply-results-and-cluster-runbook.md` — apply results, the raw-vs-stripped channel correction, and outstanding cluster work.
- `docs/notes/us-head-retrain-plan.md` — the US-head retrain design + v1 outcome (no swap).
- `docs/notes/cca-model-characterization.md`, `docs/reports/cca-collaborator-memo.md` — CCA retrain characterization + memo.
- `docs/notes/tiers-and-checkpoints.md`, `tier{2,3,4,5}-design.md` — Tier 1–5 audit/refactor history + design reasoning.
- `docs/notes/process-patterns.md`, `engineering-patterns.md` — content-agnostic process patterns and CS-specific engineering patterns (boundary-inventory, synthetic-stand-ins, Pattern-A-vs-Pattern-2, empirical-investigation-before-design).
- `docs/notes/pinned-questions.md` — deliberately deferred substantive questions (nnPU + α + γ + ALUM composition; multi-class heads; `ClassifierPreprocessor` train/predict shape design smell).
- `r/CLAUDE.md` — the R dateline pipeline contracts.

## Key References (cited in code and memo)

- **Ji2023**: Main paper this project builds on (PU learning + ALUM for text classification)
- **Kiryo2017**: Non-negative PU learning (nnPU) loss formulation
- **Ivanov2020 (DEDPUL)**: Prior estimation via density estimation
- **Lin2020**: Focal loss (via Keras `BinaryFocalCrossentropy`)
- **Gururangan2020**: Domain-adaptive pretraining
- **Liu2020**: Adversarial training for robustness (ALUM)
- **Hanna2017**: Prior work on ML for protest event identification; defines the haystack/coding task decomposition
- **Oliver2023**: Report-event dyad complexity in protest event analysis
- **Voss2024**: The ICA dataset this project extends (Voss, Lauterwasser, & Bloemraad 2024, *Socius*)
- **Deep Learning with Python** (Chollet 2025): Architecture/training guidance
- **Detailed project memo**: `ml_memo/ml_memo.qmd` — comprehensive explanation of the problem, methods, and plans
- **Model-state snapshots for collaborators**: `ml_memo/ica_model_state_2026-06.md` (June — the multi-head classifier's numbers and known ceilings) and `ml_memo/ica_model_update_2026-07.md` (July — the encoder-unfreeze arc, mixed-stack result, PU-collapse methods finding, corpus expansion)
