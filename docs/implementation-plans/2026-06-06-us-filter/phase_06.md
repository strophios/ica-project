# US/not-US Pre-Filter — Phase 6 Implementation Plan

**Goal:** The three validation instruments + the evidence-driven escalation decision, delivered as runnable tooling. The free 1987–1995 audit runs now; the slice-dependent instruments (pre-1986 transfer eval, DoCA recall, escalation/fine-tuning) are built and unit-tested here but **operator-gated** on the human-coded gold set, which does not exist yet.

**Architecture:** R owns the API↔LDC fuzzy join (reusing the existing `adist`-based matcher); Python owns the audit metrics, the gold-set schema + coding-template generator, the model-application/eval harnesses, and the escalation machinery. The hand-labeled slice is designed as the **seed of a durable full-model gold set** (US + reserved CCA/immig/ICA columns), but Phase 6's evaluation logic is US-only.

**Tech Stack:** R (`adist`, `arrow`), Python (polars, keras, difflib), pytest.

**Scope:** Phase 6 of 8.

**Codebase verified:** 2026-06-09 (fuzzy-join scripts, `eval_cca_classifier.py` Pattern-2 apply, `LayerLRModel` multiplier API, DoCA-match identification; confirmed the hand-labeled slice does NOT exist yet).

---

## Acceptance Criteria Coverage

Implements/tests **us-filter.AC6** and the transfer portion of **us-filter.AC3**:

### us-filter.AC6: Validation instruments produce their reports
- **AC6.1 Success:** the free heuristic audit reports the R heuristic's error rate against dateline labels on joinable 1987–1995 articles.
- **AC6.2 Success:** `stripped_text` ≈ API lead is verified on matched pairs (a similarity figure is reported).
- **AC6.3 Success:** the pre-1986 slice evaluation reports both-class metrics and the dateline-vs-event-location proxy gap.
- **AC6.4 Success:** the DoCA-matched recall diagnostic is produced, with its topic-skew caveat noted.
- **AC6.5 Caveat:** the audit report records the biased-by-joinability caveat.

### us-filter.AC3 (transfer portion)
- **AC3.3 Success:** pre-1986 hand-labeled performance is computed, compared to the in-distribution baseline, and the escalation decision is recorded.

**Operator-gated note:** AC6.3, AC6.4, AC3.3 require the human-coded gold set (and a trained model). This phase delivers their harnesses with synthetic-data unit tests; their *real* execution and the recorded escalation decision are operator steps performed when the coded slice lands.

---

## Verified facts

- Fuzzy join: base R `adist()` (Levenshtein); normalize `str_to_lower(str_remove_all(x, "[^A-z0-9 ]"))`; `dist_cutoff=5`; `00_proc_and_matching_prep.R:456-491` (`fuzzy_string_comp`), match pipeline `:331-452`. Existing join is LDC↔DoCA on date+page+section; API↔LDC uses headline + `pub_date` only (API has no page/section).
- `us_assign()` heuristic in `/Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R` (desk/section + keyword based) — the heuristic being audited.
- DoCA-matched rows: `doca_id` non-null in the LDC parquet (`cca_matches_good.rds`). No recall diagnostic exists.
- `LayerLRModel`: `train_step` scales grads by `get_multiplier(var)` per group; `_default_group_fn` = `variable.path.split("/")[0]`. Custom `group_fn` checking `roberta_layer_N` in `variable.path` enables per-layer unfreezing. `build_endpoint_model(freeze_encoder=False, group_fn=..., layer_multipliers=...)`.
- Pattern-2 apply: `eval_cca_classifier.py` — fresh head → `build_inference_model` → `load_weights(skip_mismatch=False)` → finite predict → attach scores to polars df.
- Hand-labeled slice: **NOT FOUND** (human WIP).

---

## Gold-set schema (durable, seeds the full-model validation set)

One row per article. US filter uses `id`/`alt_corpus_id`/`us_event`/`event_location`/strata; reserved columns tolerated null.

| Column | Type | Purpose |
|---|---|---|
| `id` | str | primary article id (in `corpus`); stored as str (LDC Int64 cast to str) |
| `corpus` | enum `api`/`ldc` | provenance of `id` |
| `alt_corpus_id` | str, nullable | same article's id in the OTHER corpus; NA if only in one |
| `year`, `news_desk`, `section_name` | — | strata / context |
| `headline`, `lead_paragraph` | str | text the coder reads |
| `us_event` | bool, nullable | US filter ground truth (event-location coded) |
| `event_location` | str, nullable | proxy-gap measure |
| `cca_event`, `immig_relevant`, `ica_event` | bool, nullable | reserved — full-model gold |
| `sample_stratum` | str | `doca_matched`/`random_pre1986`/`ambiguous` |
| `coder`, `coded_date`, `confidence`, `notes` | — | coding metadata |

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) — free audit, runs now -->

<!-- START_TASK_1 -->
### Task 1 (R): `r/audit/api_ldc_join.R` — API↔LDC join + heuristic

**Files:** Create `r/audit/api_ldc_join.R` (`# pattern: Imperative Shell`).

Join the API corpus (1987–1995 subset) to the LDC labeled parquet on normalized headline + `pub_date`, using the `adist` matcher (transcribe/reuse `fuzzy_string_comp` + normalization from `00_proc_and_matching_prep.R:456-491`, `dist_cutoff=5`). Run `us_assign()` on the LDC side (`source("/Users/.../nyt_location_checking.R")`). Emit `<US_FILTER_DIR>/audit/api_ldc_matched.parquet` with: `api_id, ldc_id, api_lead (lead_paragraph), ldc_stripped_text, ldc_us_label, ldc_heuristic_us`.

**`id` dtype convention (pinned):** API `id` is `character` (str); LDC `id` is Int64. Emit **both** `api_id` and `ldc_id` as **strings** (`as.character()` the LDC id) so all downstream joins (free-audit metrics, gold-set `alt_corpus_id`, DoCA-recall) use a single str id convention and never silently mismatch int vs str.

Verification (operational): script produces the parquet; print matched-pair count and the joinability rate.
Commit: `feat(us-filter): API<->LDC audit join + heuristic application (R)`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2 (Python): `src/validation/free_audit.py`

**Verifies:** us-filter.AC6.1, AC6.2, AC6.5.

**Files:** Create `src/validation/free_audit.py`; Create `tests/test_free_audit.py`.

Pure metrics (Functional Core) + a thin shell that reads the matched parquet:

```python
import difflib
def _norm(s):  # lowercase, alphanumeric+space only
    import re
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower())

def heuristic_error_rate(heuristic_us, dateline_us):
    """Disagreement rate of the R us_assign heuristic vs dateline labels
    (over rows where both are non-null)."""
    pairs = [(h, d) for h, d in zip(heuristic_us, dateline_us) if h is not None and d is not None]
    if not pairs: return float("nan")
    return sum(h != d for h, d in pairs) / len(pairs)

def lead_similarity(stripped_texts, api_leads):
    """Mean normalized similarity (difflib ratio) of LDC stripped lead vs API lead."""
    sims = [difflib.SequenceMatcher(None, _norm(s), _norm(a)).ratio()
            for s, a in zip(stripped_texts, api_leads)]
    return sum(sims) / len(sims) if sims else float("nan")
```
The shell reads `api_ldc_matched.parquet`, computes both, and prints them with the **biased-by-joinability caveat** (error rates conditional on joinability; AC6.5).

**Testing** (`tests/test_free_audit.py`): synthetic matched pairs — known disagreement → exact error rate; identical-vs-divergent leads → expected similarity ordering; caveat string present in the report.
Commit: `feat(us-filter): free heuristic audit metrics`
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-5) — gold set: schema, template (now), eval (operator-gated) -->

<!-- START_TASK_3 -->
### Task 3: `src/validation/schema.py` — gold-set schema + validator

**Files:** Create `src/validation/schema.py` (`# pattern: Functional Core`); Create `tests/test_validation_schema.py`; Modify `src/config.py` (add `VALIDATION_DIR = PROJECT_ROOT / "validation"`).

Define the column set + dtypes (above) as constants and `validate_gold_set(df) -> None` that raises `ValueError` enumerating any missing required columns / wrong dtypes. Required: `id, corpus, year, headline, lead_paragraph, sample_stratum`. Label columns (`us_event`, `event_location`, reserved) may be present-but-null. `alt_corpus_id` optional-nullable.

**Testing:** valid frame passes; missing `id` / bad `corpus` enum raises with enumeration; null label columns tolerated.
Commit: `feat(us-filter): gold-set validation schema`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: `src/validation/build_coding_template.py` — candidate generator

**Files:** Create `src/validation/build_coding_template.py` (`# pattern: Imperative Shell`); Create `tests/test_build_coding_template.py`.

Sample the API pre-1986 corpus (`API_CORPUS_DIR`, years < 1987) stratified by era-bucket × `news_desk` (configurable n per cell); optional `--doca-matched <path>` merges a DoCA-matched positive set as `sample_stratum="doca_matched"` (tolerated absent). Write `VALIDATION_DIR/coding_template.parquet` conforming to the schema: context columns filled, `corpus="api"`, `alt_corpus_id=None`, label columns null/empty, `sample_stratum` set.

**Testing** (on a synthetic API parquet): stratified sampling hits the requested cells; output validates against `schema.validate_gold_set`; label columns are present-and-null; deterministic under a fixed seed.
**Verification (operational):** `uv run python -m src.validation.build_coding_template` writes a template over the real API corpus.
Commit: `feat(us-filter): hand-labeling coding-template generator`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: `src/validation/slice_eval.py` — transfer eval + proxy gap (operator-gated)

**Verifies:** us-filter.AC6.3, supports AC3.3.

**Files:** Create `src/validation/slice_eval.py`; Create `tests/test_slice_eval.py`.

- Reusable apply helper `apply_us_model(texts) -> probs` following `eval_cca_classifier.py` Pattern-2: load `UsRunConfig` sidecar, fresh `us` head, `build_inference_model`, `load_weights(skip_mismatch=False)`, finite predict, then `PlattCalibrator` (loaded from `.calibration.json`) → calibrated `us_score`.
- `evaluate_slice(gold_df, threshold) -> {precision, recall, f1, n_pos, n_neg}` vs hand `us_event` (AC6.3 both-class).
- `proxy_gap(gold_df) -> {dateline_event_agreement, n}` = agreement between the dateline label (from the row's LDC `alt_corpus_id`, where present) and hand `us_event`/`event_location` (AC6.3).

**Testing** (synthetic coded rows + a fake-backbone model, per `test_assembly.py`): metrics computed correctly from known probs/labels; proxy-gap agreement on constructed dateline-vs-event mismatches. **Real run operator-gated** on the coded gold set + trained model.
Commit: `feat(us-filter): pre-1986 slice eval + proxy-gap (harness)`
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 6-7) — DoCA recall + escalation -->

<!-- START_TASK_6 -->
### Task 6: `src/validation/doca_recall.py` — DoCA-matched recall (operator-gated)

**Verifies:** us-filter.AC6.4.

**Files:** Create `src/validation/doca_recall.py`; Create `tests/test_doca_recall.py`.

`doca_recall(scored_df, threshold) -> {recall, n}` = fraction of DoCA-matched articles (`doca_id` non-null) the calibrated filter scores US (`us_score ≥ threshold`). The report prints the **topic-skew caveat** (DoCA = US protest events; not representative; AC6.4).

**Producing `scored_df` (the `doca_id` source — contract note):** `doca_id` is an **LDC-only** column; the Phase 7 API scores (`id, us_score, us`) do not carry it. The diagnostic's input is assembled one of two ways, both operator-gated:
1. **Score the DoCA-matched LDC rows directly** — filter the LDC labeled parquet to `doca_id` non-null, run `apply_us_model` on those rows (LDC `headline + "</s>" + stripped_text`), attach `doca_id` + `us_score`. This is the simplest path and what the harness assumes by default (a `--ldc` mode).
2. **Join via the gold set** — for pre-1986 (no LDC), use the gold-set `alt_corpus_id` (API id ↔ LDC id) to attach LDC `doca_id` to API-scored rows; only available once the pre-1986 DoCA↔API match exists.
The harness signature takes a `scored_df` that already carries `doca_id` + `us_score`; the producer (one of the above) is the operator's choice and is documented in the module docstring. **`id` dtype:** cast LDC `doca_id`/`id` (Int64) to str at the join boundary to match the API str `id` convention (pinned in Task 1).

**Testing:** synthetic scored DoCA rows (with `doca_id` + `us_score`) → exact recall at a threshold; caveat string present. Real run operator-gated.
Commit: `feat(us-filter): DoCA-matched recall diagnostic`
<!-- END_TASK_6 -->

<!-- START_TASK_7 -->
### Task 7: Escalation knobs + decision helper

**Verifies:** us-filter.AC3.3.

**Files:** Modify `src/us_config.py` (add fine-tuning fields); Modify `src/run_us_classification.py` (unfreeze path); Create `src/validation/escalation.py`; Modify `tests/test_us_config.py`; Create `tests/test_escalation.py`.

- `UsRunConfig`: add `freeze_encoder: bool = True`, `unfreeze_top_n: int = 0`, `layer_multipliers: dict | None = None` (default = frozen probe; extend `to_json`/`from_json` round-trip + `__post_init__` validation: `unfreeze_top_n >= 0`).
- `src/validation/escalation.py`:
  ```python
  def top_n_group_fn(n_top, n_layers=12):
      tops = {f"roberta_layer_{n_layers-1-i}" for i in range(n_top)}
      def group_of(var):
          p = var.path
          if any(t in p for t in tops): return "encoder_top"
          if "roberta" in p: return "encoder_frozen"
          return "head"
      return group_of

  def escalation_decision(baseline_f1, transfer_f1, margin=0.1):
      gap = baseline_f1 - transfer_f1
      escalate = gap > margin
      return {"escalate": escalate,
              "rationale": f"transfer F1 {transfer_f1:.3f} vs baseline {baseline_f1:.3f} "
                           f"(gap {gap:.3f} {'>' if escalate else '<='} margin {margin})"}
  ```
- `run_us_classification`: when `unfreeze_top_n > 0`, build with `freeze_encoder=False`, `group_fn=top_n_group_fn(unfreeze_top_n)`, `layer_multipliers={"head":1.0,"encoder_top":<m>,"encoder_frozen":0.0}`.

**Testing:** `UsRunConfig` round-trips with the new fields; `top_n_group_fn(2)` groups `roberta_layer_11`/`_10` as `encoder_top`, `_0` as `encoder_frozen`, head as `head`; `escalation_decision` flips on the margin. The actual escalation decision is recorded by the operator when the transfer eval runs.
Commit: `feat(us-filter): fine-tuning escalation knobs + decision helper`
<!-- END_TASK_7 -->

<!-- END_SUBCOMPONENT_C -->

---

## Phase 6 Done When

- The free audit runs and reports the R-heuristic error rate + `stripped_text`≈API-lead similarity with the joinability caveat (AC6.1/6.2/6.5).
- The gold-set schema validator + coding-template generator exist and pass tests; a template is materializable over the real API corpus.
- The slice-eval, DoCA-recall, and escalation harnesses are built and unit-tested on synthetic data; `UsRunConfig` carries the unfreeze knobs and `run_us_classification` supports the unfreeze path.
- Their real execution + the recorded escalation decision are operator steps (gated on the coded gold set + trained model).

Covers **us-filter.AC6** and **AC3.3** (tooling; decisions recorded at run time).
