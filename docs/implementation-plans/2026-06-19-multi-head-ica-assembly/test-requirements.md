# Test Requirements — Multi-Head ICA Assembly

Maps each acceptance criterion (`multi-head-ica-assembly.AC1.1` … `AC7.4`) from
`docs/design-plans/2026-06-19-multi-head-ica-assembly.md` to either an **automated test**
(unit/integration, with the test file named in the implementation phase tasks) or a
**documented operational/human verification**.

This is an ML research project (Python 3.12 / uv / Keras 3 + TF). A large share of the
acceptance criteria are deliberately operational: they depend on out-of-repo embedding
caches, real features-mode retrains, Platt calibration fits over real val data, operator
hand-coding, and full-corpus apply. Those cannot be unit-tested without reproducing the
data and model artifacts they assert over. The rule applied throughout: **pure decision
logic is unit-tested; the real-data execution of that logic is operationally verified.**
Several criteria therefore appear in *both* columns — the logic half is automated, the
execution half is operational.

All automated tests run with `uv run pytest` from the project root (pythonpath=["."]).
All operational commands run from the project root with `uv run`.

---

## Summary table

| AC | Automated test | Operational / human verification |
|----|----------------|----------------------------------|
| AC1.1 | — (gate/rename logic in AC1.4, Task 1, Task 4, Task 5 tests) | ✅ harmonized retrain execution |
| AC1.2 | — (covered by AC1.4 guard tests) | ✅ retrain run, guard passes |
| AC1.3 | ✅ `tests/test_ica_eval.py` (reservation fraction) | ✅ confirmed in retrain |
| AC1.4 | ✅ `tests/test_data_splits.py`, `tests/test_cca_doca_data.py` | ✅ abort-on-leak in retrain |
| AC2.1 | ✅ `tests/test_ica_eval.py` (label logic + schema) | ✅ hand-coding `ica_event` |
| AC2.2 | ✅ `tests/test_build_ica_template.py` | ✅ boundary-draw execution |
| AC2.3 | ✅ `tests/test_ica_eval.py` (schema rejection) | — |
| AC3.1 | — (calibrator math already covered) | ✅ three Platt fits, sidecars present |
| AC3.2 | ✅ `tests/test_doca_recall.py` (`pick_us_threshold`) | ✅ τ_us chosen on real recall |
| AC3.3 | — (report math already covered) | ✅ composed calibration report |
| AC4.1 | ✅ `tests/test_fusion_combiner.py` | ✅ CV on real eval set |
| AC4.2 | ✅ `tests/test_fusion_sidecar.py` | ✅ `fusion.json` written by fit |
| AC4.3 | ✅ `tests/test_fit_fusion.py` (`select_combiner` 1-SE rule) | ✅ rule applied to real CV result |
| AC5.1 | ✅ `tests/test_assemble_ica.py` | — |
| AC5.2 | ✅ `tests/test_assemble_ica.py` | — |
| AC5.3 | ✅ `tests/test_artifact_check.py` | — |
| AC5.4 | ✅ `tests/test_assemble_ica.py` | — |
| AC6.1 | ✅ `tests/test_apply_ica.py` (output schema) | ✅ API full-corpus apply |
| AC6.2 | — (covered by AC6.1 schema test) | ✅ LDC 1996–2007 apply |
| AC6.3 | ✅ `tests/test_us_location.py`, `tests/test_apply_ica.py` (gold-first gate) | ✅ embed re-runs + real gold-vs-fallback split |
| AC7.1 | ✅ `tests/test_preflight_checks.py` | ✅ gate script run |
| AC7.2 | ✅ `tests/test_preflight_checks.py` | ✅ gate script run |
| AC7.3 | ✅ `tests/test_preflight_checks.py` | ✅ gate script run |
| AC7.4 | ✅ `tests/test_preflight_checks.py` | ✅ gate script run |

---

## AC1 — Harmonized retrain on a shared population

### AC1.1 — CCA and relevance retrained from one harmonized fused-US-gate table
**Automated (logic):** The constituent logic is covered indirectly:
- `tests/test_us_location.py` (Phase 3 Task 1) — the shared pure `apply_fused_us_gate(table)`
  helper: US-only kept; clearly-foreign (`any_not_us ∧ ¬any_us`) dropped; diaspora kept;
  relevance reproduces the same gated counts on a fixture (behavior-preserving refactor).
- `tests/test_cca_doca_data.py` (Phase 3 Task 4) — with synthetic location signals, the CCA
  unlabeled/background pool drops clearly-foreign rows while DoCA positives are retained.
- `tests/test_run_relevance.py` (Phase 3 Task 5) — the `cca`→`rel` rename takes effect in both
  the live head and the serialized sidecar.

**Operational (execution):** AC1.1 as a whole — *that the two heads are actually retrained on the
harmonized table* — cannot be unit-tested; it requires the real features-mode retrains over the
out-of-repo embedding cache (`cca_doca/embed_cache/relevance_train`).
- Justification: real model retrains over a multi-GB embedding cache; not reproducible in a unit test.
- Approach (Phase 3 Task 6): run
  `uv run python -m src.run_cca_doca --prior 0.02 --threshold 0.5 --us-weights <US_FILTER_FULL_WEIGHTS> --holdout-ids <plandata>/holdout_ids.parquet`
  (both tracks: all-forms and `--form-filter` street), and
  `uv run python -m src.run_relevance --prior 0.05 --holdout-ids <same holdout_ids.parquet>`.
- Confirms: both complete in minutes; sidecars written; the relevance head sidecar records
  `head.name == "rel"`; both runs gated the background on the same fused US gate with the same
  `us_classifier_full` weights. Operator records the gated counts.

### AC1.2 — Eval ids absent from both heads' train ∪ val (`eval_ids ∩ (train ∪ val) = ∅`)
**Automated (logic):** Covered by the AC1.4 leakage-guard tests (the guard *is* the
empty-intersection assertion) — `tests/test_data_splits.py` and `tests/test_cca_doca_data.py`.

**Operational (execution):** Confirmation that the property holds on the *real* splits is part of
the Phase 3 Task 6 retrain — the runtime guard (`assert_holdout_excluded`) runs inside both retrains
and passes (it would abort otherwise). Same command/condition as AC1.1.

### AC1.3 — ~30% of the 466 anchors reserved into the eval slice, excluded from training positives
**Automated (logic):** `tests/test_ica_eval.py` (Phase 2 Task 1) — `reserve_anchor_holdout(anchor_df,
frac=0.30, seed=200)` dedupes by `article_id` and produces a deterministic split at the specified
fraction.

**Operational (execution):** That the reserved anchors are actually excluded from both heads'
training positives is confirmed by the Phase 3 Task 6 retrain (the holdout id list is the
`--holdout-ids` input; the guard enforces exclusion). Same command/condition as AC1.1.

### AC1.4 — Leakage of any eval id into either training pool aborts the retrain
**Automated (logic):** `tests/test_data_splits.py` (Phase 3 Task 3) plus a relevance-split mirror in
`tests/test_cca_doca_data.py` — `assert_holdout_excluded(splits, holdout_ids)`:
- planted-leak split → raises `ValueError` enumerating the offending id;
- clean split → passes (no-op);
- covers both CCA (pos/unl) and relevance (pos/neg/unl) split shapes.

This is the leakage-guard logic that is explicitly unit-testable.

**Operational (execution):** The abort firing (or not firing) on the real splits is observed during
the Phase 3 Task 6 retrain — same command/condition as AC1.1.

---

## AC2 — Clean joint-ICA evaluation set

### AC2.1 — Holistically hand-coded `ica_event` per row, validates against schema
**Automated (logic):** `tests/test_ica_eval.py` (Phase 2 Task 1) — the label-derivation logic:
- `derive_ica_negatives`: any-component-False ⟹ `ica_event=False`; `us_event ∧ cca_event` True ⟹
  `ica_event` left null (the holistic-coding region) *even when `immig_relevant=False`*;
  `immig_relevant` never forces auto-False (this independence is what lets Phase 4 test the fusion);
- `reconcile_immig_column`: legacy `immig`(0/1)→`immig_relevant`(bool) into a new column, preserving
  `immig_advisory`, not overwriting hand-coded values;
- assembled rows pass `validate_gold_set`.

**Operational / human (execution):** The actual *hand-coding* of `ica_event` across the
`us_event ∧ cca_event` region is the one human bottleneck in the plan and cannot be automated.
- Justification: requires holistic human judgment of contextual ICA per article; no oracle exists.
- Approach (Phase 2 Task 4 ★ operator step): run `uv run python -m scripts.build_ica_eval_set` to
  emit the coding template (null `ica_event` across the `us∧cca` region) and the printed worklist
  size; the operator hand-codes `ica_event` for every `us_event ∧ cca_event` row, confirms anchors,
  then re-runs `validate_gold_set` on the completed file.
- Confirms: the completed eval file passes `validate_gold_set` with no null `ica_event` in the coded
  region.

### AC2.2 — Boundary candidates drawn stratified near the heads' decision boundary
**Automated (logic):** `tests/test_build_ica_template.py` (Phase 2 Task 3) — the composed-score
boundary sampler:
- schema conformance (`validate_gold_set`);
- presence of the low-relevance × high-CCA stratum (where contextual ICA hides);
- determinism (seed);
- exclusion correctness (ids already in the anchor or coded-500 sets are excluded).

Supporting infrastructure (no standalone AC): `tests/test_relevance_slice_eval.py` (Phase 2 Task 2)
— shape/contract test of `apply_relevance_model` on a tiny synthetic feature batch (output length,
finiteness).

**Operational (execution):** Drawing the real boundary sample requires the cached embeddings +
retrained CCA/relevance weights.
- Justification: depends on out-of-repo embedding cache and trained head weights.
- Approach: the Phase 2 Task 4 `scripts/build_ica_eval_set.py` run (same command as AC2.1)
  produces the real stratified draw as part of the coding template.

### AC2.3 — Rows missing any of the three component judgments rejected by schema validation
**Automated (logic):** `tests/test_ica_eval.py` (Phase 2 Task 1) — `validate_gold_set` rejects rows
missing any required component judgment (`us_event` / `cca_event` / the joint label) per the durable
schema (`src/validation/schema.py`). This is pure validation logic; fully unit-testable, no
operational counterpart.

---

## AC3 — Per-head and composed calibration

### AC3.1 — Each head has a Platt calibrator fit on natural-balance data; three sidecars present
**Automated (logic):** The Platt fit/transform/report/sidecar math is already covered by the
existing suite (`tests/test_calibration_{platt,report,sidecar}.py`); no new automated test is added
for AC3.1 — the new code (`src/calibrate_relevance.py`) is a thin Imperative-Shell mirror of
`calibrate_us_filter.py` and is verified operationally.

**Operational (execution):** Fitting the three calibrators on real natural-balance val data.
- Justification: requires real val-split features through the retrained heads; a calibration fit over
  real data, not unit logic.
- Approach (Phase 4 Task 1): `uv run python -m src.calibrate_relevance` (CCA tracks already have
  sidecars from prior work; US has one).
- Confirms: `relevance/relevance.calibration.json` exists with `A`/`B` recorded and
  `fit_population` tagged natural-balance; printed ECE drops or holds; all three
  `*.calibration.json` sidecars present (`cca_doca`, `cca_doca_street`, `relevance`).

### AC3.2 — `τ_us` is the largest threshold meeting the target anchor/DoCA recall (recall recipe)
**Automated (logic):** `tests/test_doca_recall.py` (Phase 4 Task 4) — `pick_us_threshold(scored_df,
target_recall, thresholds)`:
- synthetic scored frame → returns the correct largest-qualifying threshold;
- none-qualify edge handled (returns lowest threshold + warning flag).

**Operational (execution):** Picking the real `τ_us` over the DoCA-matched scored articles.
- Justification: requires real calibrated US scores over the DoCA-matched eval rows.
- Approach (Phase 4 Task 5): the `src/fit_fusion.py` run sets `τ_us` via `pick_us_threshold` and
  prints it; operator confirms it meets the target recall.

### AC3.3 — Composed ICA-score calibration reported; final 2-param Platt fit if mis-calibrated
**Automated (logic):** The reliability/ECE/Brier report math is covered by the existing
`tests/test_calibration_report.py`; no new automated test is added for the composed-score report
itself (it is a real-data computation).

**Operational (execution):** Reporting composed-score calibration on the clean eval set, and the
label-budget decision on whether to refit a 2-param Platt.
- Justification: a real-data calibration computation over the scarce held-out positives; the EPV/
  Harrell label-budget decision is operational judgment, not unit logic.
- Approach (Phase 4 Task 5): `uv run python -m src.fit_fusion …` prints composed ECE/Brier and the
  reliability summary; the composed Platt is fit only if the held-out positive count supports its 2
  extra parameters, and the decision is recorded in the emitted metrics JSON.

---

## AC4 — Empirical fusion selection

### AC4.1 — Calibrated-AND baseline and ≤3-param logistic challenger both evaluated by CV
**Automated (logic):** `tests/test_fusion_combiner.py` (Phase 4 Task 2) — the pure combiners:
- `combine_and(p_cca, p_rel)` product correctness + monotonicity;
- `fit_logistic_combiner` / `apply_logistic_combiner` determinism (fixed `random_state`);
- coefficient count ≤3 (≤4 with the optional soft-US term);
- `FusionConfig` validation (rejects unknown `combine`).

**Operational (execution):** Running the AND-vs-LR `StratifiedKFold` CV on the real clean eval set.
- Justification: requires the real clean eval set + calibrated head scores over its ids.
- Approach (Phase 4 Task 5): the `src/fit_fusion.py` run computes per-combiner CV PR-AUC ± SE and
  emits them to the metrics JSON.

### AC4.2 — Chosen combiner recorded in `fusion.json` (gate threshold, calibrator refs, rule, space)
**Automated (logic):** `tests/test_fusion_sidecar.py` (Phase 4 Task 3) — `save_fusion`/`load_fusion`
JSON round-trip equality for both product and logreg configs; malformed payload → `ValueError`;
`fusion_path_for_weights` mirrors `calibration_path_for_weights`; payload records per-head calibrator
references.

**Operational (execution):** `fusion.json` is actually written by the Phase 4 Task 5
`src/fit_fusion.py` run; operator confirms the file exists and records the selected combiner,
`τ_us`, calibrator refs, combine rule, and score space.

### AC4.3 — LR ships only if it beats AND by more than the pre-registered CV-noise margin
**Automated (logic):** `tests/test_fit_fusion.py` (Phase 4 Task 5) — the pure decision rule
`select_combiner(cv_and, cv_lr)`:
- LR mean improvement > 1 SE of the paired CV difference → `"logreg"`;
- improvement ≤ 1 SE → `"product"`;
- tie/degenerate → `"product"`.

The 1-SE margin is pre-registered (fixed before viewing results) in the Phase 4 plan header.

**Operational (execution):** Applying the rule to the real CV result.
- Justification: the *decision input* (real paired CV difference and its SE) comes from the
  real-data CV; only the rule itself is unit logic.
- Approach (Phase 4 Task 5): the `src/fit_fusion.py` run feeds the real CV statistics into
  `select_combiner`; the chosen combiner is printed and written to `fusion.json`.

---

## AC5 — Assembled multi-head artifact

All of AC5 is unit-testable with synthetic fixtures because assembly + reload are pure structural
operations over weights; no real corpus or retrain is needed (the synthetic fixtures stand in for
trained weights, per the project's synthetic-stand-in pattern).

### AC5.1 — Single inference model builds with frozen backbone + `{us, cca, rel}` heads (Pattern 2)
**Automated:** `tests/test_assemble_ica.py` (Phase 5 Task 1) — 3-head assembly via
`build_feature_inference_model({"us","cca","rel"})` emits the three-head output dict; heads loaded by
structure (Pattern 2) through the temp single-head transfer.

### AC5.2 — Artifact scores cached features (and raw text) to three logits + composed ICA score
**Automated:** `tests/test_assemble_ica.py` (Phase 5 Task 1) — per-head assembled scores equal
standalone single-head scores on a fixture (weights correctly transferred); composed `ica_score` is
0.0 for gated-out rows and in [0,1] for survivors.

### AC5.3 — Cross-process reload reproduces scores within tolerance (artifact_check analogue)
**Automated:** `tests/test_artifact_check.py` (Phase 5 Task 2) — `reload_and_score_ica` output matches
`IcaModel.predict_ica_from_features` within tolerance (bitwise on the frozen-feature path) on a
fixture artifact set written to `tmp_path` (3× weights+config+calibration + `fusion.json`), proving
cross-process reproduction.

### AC5.4 — Head-name collision or missing head weight raises at assembly time, not silently
**Automated:** `tests/test_assemble_ica.py` (Phase 5 Task 1) — duplicate head name → `ValueError`;
a missing / shape-mismatched head weight → `ValueError` (mirrors the Pattern-2 shape-mismatch test in
`tests/test_assembly.py`).

---

## AC6 — Apply and dataset expansion

### AC6.1 — Scoring `api_corpus` (1960–1995) writes US + CCA score dirs and ranked `ica_candidates`
**Automated (logic):** `tests/test_apply_ica.py` (Phase 6 Task 4) — output schema/dtypes and score
ranges [0,1] on a fixture (mirrors `tests/test_apply_us_filter.py`); the ranked-candidates schema
(`id, year, us_score, cca_score, rel_score, ica_score, gated`).

**Operational (execution):** The full-corpus apply itself, including the compute-heavy embed re-run
to finish the `full` cache.
- Justification: full-corpus scoring + a backbone forward-pass embed re-run over `api_corpus`
  1976–1995; GPU/cluster-scale, out-of-repo data.
- Approach (Phase 6 Task 2 + Task 4):
  - Finish the cache: `uv run python -m src.embed_corpus --full --years 1976-1995 --append --corpus api_corpus --out-suffix full --stamp <YYYYMMDD>`.
  - Apply: `uv run python -m src.apply_ica …` over the `full` cache.
- Confirms: `us_filter/api_us_scores/`, `cca_doca/api_cca_scores/`, and
  `ICA_CANDIDATES_DIR/api_1960_1995.parquet` exist with the right schema; row counts sane vs source.

### AC6.2 — Scoring LDC 1996–2007 yields out-of-DoCA-period ICA candidates
**Automated (logic):** Covered by the AC6.1 output-schema test in `tests/test_apply_ica.py` (the LDC
write path shares the candidate schema).

**Operational (execution):** The LDC apply, including the `stripped_text` re-embed.
- Justification: a `stripped_text` backbone re-embed of LDC 1996–2007 + full-corpus scoring;
  GPU/cluster-scale, out-of-repo data.
- Approach (Phase 6 Task 2 + Task 4):
  - Build the stripped source: `uv run python -m scripts.build_stripped_ldc_source`.
  - Embed: `uv run python -m src.embed_corpus --full --source-pattern <stripped_source.parquet> --lead-column stripped_text --no-year --out-suffix ldc_9607_stripped --stamp <YYYYMMDD>` (smoke-test with `--limit` first).
  - Apply: `uv run python -m src.apply_ica …` over the `ldc_9607_stripped` cache.
- Confirms: `ICA_CANDIDATES_DIR/ldc_1996_2007.parquet` exists with the right schema; the cache
  provenance records `lead_column=stripped_text`.

### AC6.3 — LDC US gating prefers gold dateline `us_label`, ML US head as fallback; CCA/rel use stripped re-embed
**Automated (logic):**
- `tests/test_us_location.py` (Phase 6 Task 3) — pure `gold_first_us_gate(gold_label, ml_pass)`: gold
  present overrides ML (both True→False-gold and False→True directions); gold null ⟹ ML fallback;
  coverage fraction correct.
- `tests/test_apply_ica.py` (Phase 6 Task 4) — a fixture row carrying a gold `us_label` has its gate
  decided by the gold label, not the ML score.

**Operational (execution):** The real gold-vs-fallback split over the LDC apply ids, and the
guarantee that CCA/relevance score the `stripped_text` re-embed (not the raw `ldc_9507`).
- Justification: depends on the real `us_filter/ldc_labeled.parquet` gold coverage over the LDC
  1996–2007 ids and the real `stripped_text` cache.
- Approach (Phase 6 Task 4): the `src/apply_ica.py` LDC run logs the gold-vs-fallback split.
- Confirms: the logged gold coverage matches the Phase 1 AC7 `ldc_gold_coverage_verdict` estimate;
  CCA/rel inputs are sourced from the `ldc_9607_stripped` cache.

---

## AC7 — Pre-flight verifications (cross-cutting)

All four AC7 verdicts have unit-tested verdict *logic* (`src/preflight/checks.py` pure functions) and
an operational *gate-script run* that feeds real out-of-repo provenance/mtimes into that logic.

### AC7.1 — Cached `us_logit` confirmed produced by `us_classifier_full`, not smoke-test weights
**Automated (logic):** `tests/test_preflight_checks.py` (Phase 1 Task 1) — `us_weights_verdict`:
smoke-test-only provenance + `table_build_us_weights=None` → FAIL;
`table_build_us_weights="…us_classifier_full.weights.h5"` → PASS; undetermined → WARN.

**Operational (execution):** `uv run python -m scripts.preflight_assembly` (Phase 1 Task 2) reads the
real cache `provenance.*.json` + the operative table-build US-weights reference and prints the
verdict; operator records it. (Per the Phase 1 remediation note, the embed-time `us_logit` is
vestigial at apply — the apply-time US gate is computed by the assembled `us` head loading
`us_classifier_full`.)

### AC7.2 — CCA and relevance calibration sidecars confirmed present (or fit if missing)
**Automated (logic):** `tests/test_preflight_checks.py` (Phase 1 Task 1) —
`calibration_presence_verdict`: relevance-missing → WARN (Phase 4 fixes it); a CCA sidecar missing →
FAIL; all-present → PASS.

**Operational (execution):** The Phase 1 Task 2 gate-script run stats the real sidecars via
`calibration_path_for_weights` and reports the verdict (currently a relevance WARN until AC3.1 fits
it).

### AC7.3 — DoCA freshness propagated, or edit explicitly accepted as incidental
**Automated (logic):** `tests/test_preflight_checks.py` (Phase 1 Task 1) — `doca_freshness_verdict`:
`doca_csv > rds` (match stale) or `rds > positives` (positives stale) → WARN; monotone
`doca_csv ≤ rds ≤ positives` → PASS; a None mtime → WARN with detail.

**Operational (execution):** The Phase 1 Task 2 gate-script run stats the real DoCA-chain mtimes
(`doca.csv`, `cca_matches_good.rds`, `cca_doca_positives.parquet`) and reports; operator either
confirms propagation or records the edit as incidental.

### AC7.4 — `ldc_9507` cache's `us_logit` confirmed computed on dateline-stripped LDC text
**Automated (logic):** `tests/test_preflight_checks.py` (Phase 1 Task 1) — `ldc_channel_verdict`:
`lead_column != "stripped_text"` → FAIL; `lead_column == "stripped_text"` → PASS. (Plus
`ldc_gold_coverage_verdict` returns PASS with the coverage fraction in `detail`.)

**Operational (execution):** The Phase 1 Task 2 gate-script run reads the real `ldc_9507` provenance
and reports (currently a known FAIL, remediated by the Phase 6 `stripped_text` re-embed); the script
exits nonzero if any verdict is FAIL.

---

## Notes on the automated/operational split

- **No tests are invented for purely operational steps.** Embed re-runs, real retrains, real
  calibration fits, full-corpus apply, and hand-coding are verified by documented commands +
  observable conditions, not pytest.
- **No pure logic is mislabeled operational.** Every deterministic helper — fused-US-gate,
  leakage guard, ICA-label derivation, boundary stratification, schema rejection, combiners,
  `select_combiner`, `pick_us_threshold`, `fusion.json` round-trip, assembly/reload,
  `gold_first_us_gate`, apply output schema, and all four pre-flight verdicts — has a named unit
  test.
- **Shared-cache dependency is the dividing line.** A criterion is operational exactly when its
  truth depends on the out-of-repo embedding cache (`cca_doca/embed_cache/*`), trained head weights
  on real data, real Platt fits, real DoCA/gold tables, or human judgment. Where such a criterion
  also has extractable pure logic, that logic is unit-tested separately (AC1.1, AC2.1, AC2.2, AC3.2,
  AC4.1–AC4.3, AC6.1, AC6.3, AC7.1–AC7.4 all appear in both columns).
