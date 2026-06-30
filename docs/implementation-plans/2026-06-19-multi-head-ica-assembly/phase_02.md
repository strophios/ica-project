# Multi-Head ICA Assembly Implementation Plan — Phase 2

**Goal:** Assemble a clean, contamination-free joint-ICA evaluation set (held-out anchor positives + an immigrant-enriched boundary draw + cautiously-reused coded rows) and emit the held-out id list Phase 3 will exclude from both retrains.

**Architecture:** Pure Functional-Core label/reservation helpers (`src/validation/ica_eval.py`), a relevance-head inference helper (`relevance_slice_eval.py`), a composed-score boundary sampler (`build_ica_coding_template.py`), and an Imperative-Shell assembly script (`scripts/build_ica_eval_set.py`) that emits the coding template + `holdout_ids.parquet`. A documented operator step hand-codes the holistic `ica_event`.

**Tech Stack:** Python 3.12, `uv`, `polars`, Keras 3 + TF (features-mode inference), `pytest`.

**Scope:** Phase 2 of 6 from `docs/design-plans/2026-06-19-multi-head-ica-assembly.md`.

**Codebase verified:** 2026-06-23 (codebase-investigator).

---

## Acceptance Criteria Coverage

This phase implements and tests:

### multi-head-ica-assembly.AC2: Clean joint-ICA evaluation set
- **multi-head-ica-assembly.AC2.1 Success:** The eval set carries a joint-ICA label (US ∩ CCA ∩ immigrant-relevant) per row and validates against a documented schema.
- **multi-head-ica-assembly.AC2.2 Success:** Boundary candidates were drawn stratified near the current heads' decision boundary (selection-by-old-score does not contaminate the retrain).
- **multi-head-ica-assembly.AC2.3 Failure:** Rows missing any of the three component judgments are rejected by schema validation.

---

## Verified context (from investigation)

- `src/validation/schema.py:15-59` — durable gold schema already includes nullable `us_event`, `event_location`, `cca_event`, `event_type`, `immig_relevant`, `ica_event`, plus optional `cca_logit`/`cca_score`. **No schema extension needed.** `validate_gold_set` enforces required non-null columns + enums (`corpus`∈{api,ldc}, `sample_stratum`∈{…}).
- `validation/cca_coding_first500_coded.csv` — 500 coded rows with `us_event`, `cca_event`, `event_location`, `immig`(0/1)+`immig_conf`; **no `ica_event`**, column named `immig` not `immig_relevant`. Coded under a weaker US gate + idiosyncratic immig coding → treat as ADVISORY.
- `validation/cca_coding_template.parquet` — 2,553 schema-conformant rows, labels null.
- `relevance/ica_anchors.parquet` — 552 rows, ~466 unique `article_id` (multi-label by `event_type4`); US∩CCA∩immigrant by construction (`r/doca/export_ica_anchors.R`). Holdout unit = deduped `article_id`.
- Inference helpers: `src/validation/cca_slice_eval.py:28-55` (`apply_cca_model`), `slice_eval.py:28-107` (`apply_us_model` w/ Platt). **No relevance helper** — must add. Score-stratified sampling precedent: `src/validation/build_cca_coding_template.py:build_cca_template:52-107`.
- `_load_holdout_ids` (`run_cca_doca.py:68-79`) reads a parquet `id` column → `list[str]`.
- Tests precedent: `tests/test_validation_schema.py`, `tests/test_build_coding_template.py`, `tests/test_cca_eval.py`. Conventions: `uv run pytest`, `ruff check`.

---

## Label logic (corrected — protects the interaction test)

ICA logically requires US **and** CCA, so:
- `ica_event = False` is auto-set **iff `us_event == False` OR `cca_event == False`** (objective scope gates).
- `ica_event` is **left null for hand-coding on every `us_event == True ∧ cca_event == True` row**, regardless of `immig_relevant` — because marginal relevance is context-free and a true ICA can be relevant only *in context*. `immig_relevant` is recorded but is NOT dispositive and must NOT auto-derive `ica_event`.
- Anchors: `ica_event = True` by construction (confirmation-only, with provenance).

This lets the eval quantify true-ICA-outside-marginal-relevance (a Phase-4 interpretation + Approach-C / contextual-relevance signal).

---

<!-- START_SUBCOMPONENT_A (tasks 1) -->
<!-- START_TASK_1 -->
### Task 1: ICA-label + anchor-reservation helpers

**Verifies:** multi-head-ica-assembly.AC2.1, AC2.3

**Files:**
- Create: `src/validation/ica_eval.py` (`# pattern: Functional Core`)
- Test: `tests/test_ica_eval.py` (unit)

**Implementation:** pure functions, no I/O:
- `derive_ica_negatives(df) -> df` — sets `ica_event=False` where `us_event==False` OR `cca_event==False`; leaves `ica_event` null where `us_event==True ∧ cca_event==True` (the holistic-coding region). Never reads `immig_relevant` for this.
- `reconcile_immig_column(df) -> df` — maps legacy `immig`(0/1)→`immig_relevant`(bool) into a NEW column, preserving the original as `immig_advisory` and flagging provenance; does not overwrite a hand-coded `immig_relevant`.
- `reserve_anchor_holdout(anchor_df, frac=0.30, seed=200) -> (holdout_ids, train_eligible_ids)` — dedupe by `article_id`, deterministic split.
- `assemble_holdout_ids(*id_sets) -> list[str]` — union + dedupe.

**Testing:** synthetic frames verify: any-component-False ⟹ `ica_event False`; `us∧cca` True ⟹ `ica_event` null even when `immig_relevant=False`; `immig_relevant` never forces auto-False; immig reconcile mapping + advisory preservation; dedupe-by-article-id; deterministic reservation fraction; union dedupe. Assembled rows pass `validate_gold_set`.

**Verification:** `uv run pytest tests/test_ica_eval.py` — all pass.

**Commit:** `feat(validation): ICA eval label logic + anchor holdout reservation`
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2-3) -->
<!-- START_TASK_2 -->
### Task 2: Relevance-head inference helper

**Verifies:** (supports AC2.2; reused infra, no standalone AC)

**Files:**
- Create: `src/validation/relevance_slice_eval.py`
- Test: `tests/test_relevance_slice_eval.py` (unit — contract/shape)

**Implementation:** `apply_relevance_model(features, weights_path)` mirroring `apply_cca_model` — load the relevance `RunConfig` sidecar, construct a fresh `ClassificationHead`, `build_feature_inference_model`, `load_weights(skip_mismatch=False)`, predict logits over a feature matrix. Returns logits shape `(n,)`.

**Testing:** shape/contract test on a tiny synthetic feature batch (mirror the CCA inference test); assert output length and finiteness.

**Verification:** `uv run pytest tests/test_relevance_slice_eval.py` — all pass.

**Commit:** `feat(validation): relevance-head features-mode inference helper`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Composed-score boundary sampler

**Verifies:** multi-head-ica-assembly.AC2.2

**Files:**
- Create: `src/validation/build_ica_coding_template.py` (adapt `build_cca_template`)
- Test: `tests/test_build_ica_template.py` (unit)

**Implementation:** given cached embeddings + CCA/relevance weights, compute calibrated CCA & relevance scores, then stratify the candidate pool so the operator can find contextual ICA: bins over **CCA strength × relevance band**, deliberately INCLUDING high-CCA / low-marginal-relevance cells (where contextual ICA hides), not just high-relevance. Emit schema-conforming rows with null labels and a `sample_stratum` tag; exclude ids already in the anchor or coded-500 sets.

**Testing:** schema conformance (`validate_gold_set`), presence of low-relevance×high-CCA stratum, determinism (seed), exclusion correctness (mirror `test_cca_eval.py`).

**Verification:** `uv run pytest tests/test_build_ica_template.py` — all pass.

**Commit:** `feat(validation): composed-score ICA coding-template sampler`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 4) -->
<!-- START_TASK_4 -->
### Task 4: `scripts/build_ica_eval_set.py` + hand-coding handoff

**Verifies:** multi-head-ica-assembly.AC2.1, AC2.2, AC2.3 (operationally)

**Files:**
- Create: `scripts/build_ica_eval_set.py` (`# pattern: Imperative Shell`)

**Implementation:** orchestrate Tasks 1–3:
- Reserve the ~30% anchor holdout (deduped `article_id`); mark them `ica_event=True`.
- Reconcile the coded-500 into `immig_relevant`/`immig_advisory`; re-confirm their `us_event` by recomputing it from the current fused US gate / gold dateline `us_label`, and DROP any coded-500 row whose imported `us_event` disagrees with the recomputed value (the original gate was weaker — imports are advisory only); route the surviving `us∧cca` rows into the hand-coding worklist.
- Draw the composed-score boundary sample (Task 3).
- Merge all sources; apply `derive_ica_negatives`; write (a) the operator coding template (null `ica_event` across the `us∧cca` region) and (b) `holdout_ids.parquet` (`id` column) = union of reserved anchors + boundary draw + reused coded ids, for Phase 3.
- Print the hand-coding worklist size (count of `us∧cca` rows needing `ica_event`).

**★ Operator step (documented, not automated):** hand-code `ica_event` holistically for every `us_event∧cca_event` row; confirm anchors; re-run `validate_gold_set` on the completed file.

**Verification (operational):** script runs; emits a schema-valid template + `holdout_ids.parquet`; `validate_gold_set` passes on the assembled (pre-coding) frame; worklist count printed.

**Commit:** `feat(validation): assemble clean ICA eval set + holdout id list`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_C -->
