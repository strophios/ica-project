# Tier 5 Implementation Plan — Phase 7: Local Stress Test (Level 1 + Level 2)

> **HUMAN-OPERATED RUNBOOK.** Phases 7–8 are operational empirical validation,
> not subagent-executable TDD. A person runs training on real cached CCA data,
> reads the diagnostic outputs, judges the pass/fail criteria, and writes the
> run-notes. Tasks 1–2 are small code deliverables (a `main()` refactor + a
> short-run wrapper); Tasks 3–5 are run + assess + document.

**Goal:** Run the level-1 (mechanical) and level-2 (numerical, local-portion) stress tests on real cached CCA data with the full diagnostic stack enabled and the corrected prior (π=0.02, already canonical in `DEFAULT_CCA_CONFIG`). Produce `docs/notes/tier5-stress-test-notes.md`.

**Scope:** Phase 7 of 8. Depends on Phases 1–6 (full diagnostic surface assembled + smoke-validated).

**Operational facts (codebase-investigator, 2026-05-17):**
- Launch from project root: `python -m src.run_cca_classification`. **No argv / no `__main__` guard / no functions** — the script runs top-to-bottom on import (Task 1 fixes this).
- `DEFAULT_CCA_CONFIG`: `FLPULossConfig(prior=0.02, ...)` (cca_config.py:817), `epochs=7` (line 820). **The corrected-prior retrain is just running the script — no config edit needed.**
- `steps_per_epoch = math.floor(18300 / (BATCH_SIZE/10))` ≈ **714** at `BATCH_SIZE=256`; `validation_steps = math.floor(1017 / (BATCH_SIZE/2))` ≈ **7**. Full run ≈ 7×714 ≈ 5000 steps; data note: "Train 18300 pos / 1026418 unl; Val 1017 pos / 57024 unl."
- `cca_set/` cache: if `config.CCA_SET_DIR` is absent the script **builds it on first run (~5–10 min)** then loads from disk thereafter (six `{train,val,test}_{pos,unl}.tf`).
- Outputs: checkpoint `config.CCA_CLASSIFIER_DIR / f"{_run_stamp}_checkpoint.weights.h5"`; sidecar via `config_path_for_weights`; CSV `config.CCA_LOGS_DIR / _run_stamp / "metrics.csv"` (added Phase 6 Task 3); TensorBoard `config.CCA_LOGS_DIR / _run_stamp`. `_run_stamp = strftime("%Y%m%d_%H%M%S")`.
- Local `DTYPE_POLICY = "float32"` (cluster is `mixed_float16`; that is Phase 8).
- Diagnostic columns wired (Phase 6): per-step trackers are **train-only** (no `val_*`; `test_step` does not dispatch diagnostics): `grad_norm/cca/max`, `grad_norm/cca/mean`, `grad_overflow_rate`, `cca/positive_risk/mean`, `cca/negative_risk/mean`, `cca/correction_triggered/mean`, `cca/positive_fraction`. Distribution metrics ride `head.metric_objs` → **train + `val_`**: `cca_pred_dist/mean|std|frac_above_0.5` and `val_cca_pred_dist/*`. Under `freeze_encoder=True` there is **no** encoder grad-norm tracker at all (only `cca`) — "encoder grad zero" means "absent by construction," documented in Phase 2.
- No `docs/notes/tier5-stress-test-notes.md` yet. House style: Markdown prose with **Decision / Rationale / Evidence** headings; mirror `docs/notes/process-patterns.md` tone.

**Level 1 (mechanical) pass criteria (design DoD):** end-to-end on real `cca_set/` with diagnostics on; a short run completes with no crash / no shape error / no NaN in final loss; save → load → predict round-trip on a held-out batch; all existing tests still pass.

**Level 2 (numerical, local portion) pass criteria (design DoD):** loss decreases monotonically (final-epoch `cca/positive_risk/mean` < initial by ≥ ~1 order of magnitude); head grad norms non-zero and bounded (`grad_norm/cca/max` finite, no explosion across the run); no NaN/Inf in any tracked scalar; `grad_overflow_rate == 0` (float32 local); FLPU components in expected ranges (`cca/positive_risk/mean` decreasing; `cca/correction_triggered/mean` not pinned at 1.0).

---

<!-- START_TASK_1 -->
### Task 1: Refactor run_cca_classification.py to a `main()` entrypoint

**Files:** Modify `src/run_cca_classification.py`. Code task (enables Tasks 2–5 + the Phase 8 cluster entrypoint).

**Step 1: Wrap the script body in a function**

Read `src/run_cca_classification.py`. Wrap its current top-to-bottom body in:

```python
def main(run_config=None, max_steps=None):
    import dataclasses
    from src import cca_config
    if run_config is None:
        run_config = cca_config.DEFAULT_CCA_CONFIG
    # ... existing body, now indented one level, using `run_config` ...
    # At the steps computation, honor max_steps:
    #   steps_per_epoch = math.floor(18300 / (BATCH_SIZE / 10))
    #   if max_steps is not None:
    #       steps_per_epoch = min(steps_per_epoch, max_steps)
    # ... rest unchanged ...


if __name__ == "__main__":
    main()
```

Preserve every existing line verbatim inside `main` (only: indent; replace the existing `run_config = ...DEFAULT_CCA_CONFIG...` construction with the param-defaulted one; add the `max_steps` cap at the `steps_per_epoch` computation). Do **not** change model/diagnostics/callback wiring (that is Phase 6).

**Step 2: Verify operationally**

Run: `python -c "import ast; ast.parse(open('src/run_cca_classification.py').read()); print('parse-ok')"` → `parse-ok`
Run: `python -c "from src.run_cca_classification import main; print(callable(main))"` → `True` (importing must NOT start training — proves the `__main__` guard works).
Run: `pytest -q` → unchanged total, zero failures.

**Step 3: Commit**

```bash
git add src/run_cca_classification.py
git commit -m "tier5 phase 7: refactor run_cca_classification to importable main(run_config, max_steps)"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add scripts/tier5_short_run.py wrapper

**Files:** Create `scripts/tier5_short_run.py`.

**Step 1: Create the wrapper**

```python
# pattern: Imperative Shell
"""Tier 5 short stress-test run: full diagnostic stack on real cca_set/,
one epoch capped at a few hundred steps. Reproducible level-1 mechanical
check. Run from project root: python scripts/tier5_short_run.py"""

import dataclasses

from src import cca_config
from src.run_cca_classification import main

if __name__ == "__main__":
    short_cfg = dataclasses.replace(cca_config.DEFAULT_CCA_CONFIG, epochs=1)
    main(run_config=short_cfg, max_steps=200)
```

**Step 2: Verify operationally**

Run: `python -c "import ast; ast.parse(open('scripts/tier5_short_run.py').read()); print('parse-ok')"` → `parse-ok`
(Full execution is Task 3 — needs real `cca_set/`.)

**Step 3: Commit**

```bash
git add scripts/tier5_short_run.py
git commit -m "tier5 phase 7: scripts/tier5_short_run.py reproducible short-run wrapper"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Level-1 mechanical run (short, real data) — HUMAN-OPERATED

**Prerequisite:** project root, `source .venv/bin/activate`, local env (`ICA_ENV` unset or `local`; confirm `python -c "from src import config; print(config.IS_CLUSTER, config.DTYPE_POLICY)"` → `False float32`). First run builds `cca_set/` (~5–10 min) if absent.

**Run:**
```
python scripts/tier5_short_run.py
```

**Level-1 pass checklist (record PASS/FAIL + evidence in the notes doc, Task 5):**
- [ ] Process completes, exit 0, no traceback / no shape error.
- [ ] Final-step loss printed by Keras is finite (no `nan`/`inf`).
- [ ] `config.CCA_LOGS_DIR/<stamp>/metrics.csv` exists and contains the wired diagnostic columns (grep header for `grad_overflow_rate`, `grad_norm/cca/`, `cca/positive_risk/mean`, `cca/positive_fraction`, `cca_pred_dist/mean`).
- [ ] A checkpoint `.weights.h5` + `.config.json` sidecar were written.
- [ ] Save→load→predict on a held-out batch succeeds (reuse `python scripts/smoke_test_integrated_stack.py` for the round-trip mechanic, or load the just-written checkpoint via `eval_cca_classifier.py`'s Pattern-2 path and predict one batch — record which).
- [ ] `pytest -q` still green (no code changed since Phase 6, but confirm the working tree).

**If any item fails:** triage before proceeding. Capture the failure verbatim in the notes doc; do not proceed to Task 4 until level-1 is green. Do not "fix" by weakening criteria.
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Level-2 numerical run (full local) — HUMAN-OPERATED

**Run (full 7-epoch local run, corrected prior already in DEFAULT_CCA_CONFIG):**
```
python -m src.run_cca_classification
```

**Level-2 pass checklist (local portion) — read `metrics.csv` across epochs:**
- [ ] **Loss decreasing:** `cca/positive_risk/mean` final epoch < initial epoch by ≥ ~1 order of magnitude (record both values + the per-epoch series).
- [ ] **Head grad norms bounded:** `grad_norm/cca/max` non-zero and finite every epoch; no explosion (record max-over-run; flag if it grows unboundedly).
- [ ] **Encoder grad "zero":** confirm there is **no** `grad_norm/<backbone>/*` column at all (frozen-encoder → tracker absent by construction; this is the expected form of "encoder grad zero").
- [ ] **No NaN/Inf anywhere:** every numeric cell in `metrics.csv` is finite (`grad_overflow_rate == 0.0` for all epochs under float32).
- [ ] **FLPU components sane:** `cca/positive_risk/mean` trends down; `cca/negative_risk/mean` recorded; `cca/correction_triggered/mean` **not pinned at 1.0** (a persistent ~1.0 indicates the prior is materially off — record the value; this is a research-relevant signal even if level-2 "passes" mechanically).
- [ ] **Prediction distribution not collapsed:** `cca_pred_dist/std` not ≈ 0 with `cca_pred_dist/mean` ≈ prior (that pattern = model predicting the prior everywhere). Record `mean/std/frac_above_0.5` and their `val_` counterparts per epoch.
- [ ] All existing tests still green.

**Record the full per-epoch diagnostic series** (copy the `metrics.csv` epoch rows into the notes doc). Note any unexpected reading even if criteria pass.
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Create docs/notes/tier5-stress-test-notes.md — HUMAN-OPERATED

**Files:** Create `docs/notes/tier5-stress-test-notes.md`. Template:

```markdown
# Tier 5 Stress-Test Notes

## Phase 7 — Local (level 1 + level 2)

**Environment:** macOS, float32, tensorflow-metal. Commit: <sha>. Date: <date>.
Config: DEFAULT_CCA_CONFIG (prior=0.02, epochs=7), diagnostics all-enabled.

### Level 1 (mechanical)
- Short run command + outcome.
- Pass checklist: each item PASS/FAIL + evidence (file paths, log excerpts).

### Level 2 (numerical, local)
- Full run command + outcome.
- Per-epoch diagnostic table (paste metrics.csv epoch rows).
- Each pass criterion: value(s) + PASS/FAIL.
- **Decision:** level-1 + level-2(local) PASS / FAIL.
- **Rationale:** ...
- **Unexpected readings / research-relevant observations:** (e.g.,
  correction_triggered rate, prediction-distribution shape — these feed the
  downstream level-3 π=0.03-vs-0.02 comparison workstream, which is OUT of
  Tier 5 scope but seeded here).

## Phase 8 — Cluster (level 2 acceptance bar)
*(filled in Phase 8)*
```

**Done when:** notes doc committed with level-1 + level-2(local) decisions and the per-epoch diagnostic series recorded.

```bash
git add docs/notes/tier5-stress-test-notes.md
git commit -m "tier5 phase 7: local stress-test notes (level 1 + level 2 local)"
```

**Phase 7 Done-when (design DoD):** both local runs complete; level 1 and level 2 (local-portion) pass criteria met and documented in `tier5-stress-test-notes.md`. If level-2 reveals a genuine numerical problem (loss not decreasing, grad explosion, NaN, correction pinned at 1.0), STOP and escalate — do not proceed to cluster time on a broken stack.
<!-- END_TASK_5 -->
