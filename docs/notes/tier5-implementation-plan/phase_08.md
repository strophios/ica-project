# Tier 5 Implementation Plan — Phase 8: Cluster Stress Test (Level-2 Acceptance Bar)

> **HUMAN-OPERATED RUNBOOK.** Requires HPC cluster ("Explorer") access. A
> person submits the jobs, reads `grad_overflow_rate` and the other
> mixed-precision-sensitive diagnostics, calibrates the overflow threshold on
> the short run, judges the full-run acceptance bar, and writes the notes.
> Task 1 is a small committed artifact (SLURM template); Tasks 2–4 are
> submit + assess + document.

**Goal:** Validate `mixed_float16` behavior and any cluster-specific issues via one short cluster run (calibrating the `grad_overflow_rate` threshold + triage), then a full-length cluster run that is the **level-2 acceptance bar**. Update `docs/notes/tier5-stress-test-notes.md` with the cluster summary — input to the post-Tier-5 level-3 (π=0.03-vs-0.02) research workstream.

**Scope:** Phase 8 of 8 (final). Depends on Phase 7 (local runs validated the stack before committing cluster time).

**Operational facts (codebase-investigator, 2026-05-17):**
- Cluster vs. local: `src/config.py` sets `IS_CLUSTER` from `ICA_ENV` (`cluster`/`local`) else the `/projects/ahd` marker. On cluster: `DTYPE_POLICY="mixed_float16"`, `PROJECT_ROOT=/projects/ahd` (so `CCA_SET_DIR`, `CCA_CLASSIFIER_DIR`, `CCA_LOGS_DIR`, checkpoint paths are all under `/projects/ahd`).
- Entry point: `python -m src.run_cca_classification` (Phase 7 Task 1 added `main()` + `__main__` guard) for the full run; `python scripts/tier5_short_run.py` for the short run (reuses `main(epochs=1, max_steps=200)`).
- **No SLURM/sbatch convention exists in the repo and the cluster workflow is undocumented** — partition, account, time limit, and module-load sequence are operator knowledge. Task 1 commits a parameterized template with `<PLACEHOLDER>` fields for the operator to fill once.
- `mixed_float16` consequences (design "Environment behavior"): `GradientFiniteTracker` (`grad_overflow_rate`) becomes the **active** level-2 diagnostic (it is ~0 by construction under local float32). Per-group grad norms are loss-scale-multiplied by `LossScaleOptimizer` — *trends within the cluster run are valid; absolute magnitudes are NOT cross-comparable to the local float32 run*. Loss components: float16 forward, float32 metric accumulation.
- `cca_set/` under `/projects/ahd/cca_set` must exist on the cluster (or first run builds it there — ~5–10 min; budget for it in the short-run job time).

**Level-2 acceptance criteria (cluster, design DoD):** short cluster run completes (cluster paths, CUDA, `mixed_float16` policy applied), `grad_overflow_rate` and other mixed-precision-sensitive diagnostics plausible, any cluster-specific issues triaged; **full** cluster run completes with `grad_overflow_rate < threshold` (low single-digit percent, threshold refined on the short run) and stable training dynamics under mixed precision over the full run (loss decreasing, no NaN/Inf, FLPU components in range, distribution not collapsed).

---

<!-- START_TASK_1 -->
### Task 1: Commit scripts/tier5_cluster.sbatch template

**Files:** Create `scripts/tier5_cluster.sbatch`.

**Step 1: Create the parameterized template**

```bash
#!/bin/bash
#SBATCH --job-name=tier5-cca
#SBATCH --partition=<PLACEHOLDER_PARTITION>      # e.g. gpu
#SBATCH --account=<PLACEHOLDER_ACCOUNT>          # Explorer allocation
#SBATCH --gres=gpu:1
#SBATCH --time=<PLACEHOLDER_TIME>                # e.g. 02:00:00 (short: 00:30:00)
#SBATCH --mem=<PLACEHOLDER_MEM>                  # e.g. 64G
#SBATCH --output=/projects/ahd/cca_logs/slurm-%j.out

set -euo pipefail

# --- Cluster module / env setup (OPERATOR-SUPPLIED) -----------------------
# module load <PLACEHOLDER_CUDA_MODULE>
# module load <PLACEHOLDER_PYTHON_MODULE>
# ------------------------------------------------------------------------

cd /projects/ahd/<PLACEHOLDER_REPO_PATH>          # repo root on the cluster
source .venv/bin/activate
export ICA_ENV=cluster                            # force cluster config

# Mode is chosen by the submit command:
#   sbatch scripts/tier5_cluster.sbatch short      -> short calibration run
#   sbatch scripts/tier5_cluster.sbatch full       -> full acceptance run
MODE="${1:-full}"
if [ "$MODE" = "short" ]; then
    python scripts/tier5_short_run.py
else
    python -m src.run_cca_classification
fi
```

> Operator note: every `<PLACEHOLDER_*>` and the commented `module load`
> lines must be filled with Explorer-specific values before submission.
> `ICA_ENV=cluster` is set explicitly (do not rely solely on the
> `/projects/ahd` marker). The template is committed as a starting artifact;
> filling placeholders is a run-time operator action, not a code edit to be
> committed back unless the values are stable and non-sensitive.

**Step 2: Verify**

Run: `bash -n scripts/tier5_cluster.sbatch && echo syntax-ok` → `syntax-ok` (bash parse only; not executed locally).

**Step 3: Commit**

```bash
git add scripts/tier5_cluster.sbatch
git commit -m "tier5 phase 8: parameterized SLURM template for cluster stress test"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Short cluster run — mixed_float16 validation + threshold calibration — HUMAN-OPERATED

**Prerequisite:** placeholders filled in `scripts/tier5_cluster.sbatch`; repo + `.venv` present under `/projects/ahd/<repo>`; confirm on a login/compute node that `ICA_ENV=cluster python -c "from src import config; print(config.IS_CLUSTER, config.DTYPE_POLICY)"` → `True mixed_float16`.

**Submit:**
```
sbatch scripts/tier5_cluster.sbatch short
```

**Short-run checklist (record in notes, Task 4):**
- [ ] Job completes, exit 0; CUDA visible; `config.DTYPE_POLICY == "mixed_float16"` applied (confirm from a startup log line or add one).
- [ ] No crash / no shape error / no NaN in final loss; cluster paths resolved under `/projects/ahd`.
- [ ] `metrics.csv` written under `/projects/ahd/cca_logs/<stamp>/`; diagnostic columns present.
- [ ] **Calibrate the overflow threshold:** read `grad_overflow_rate` across the short run. Under healthy `mixed_float16` + `LossScaleOptimizer` this is occasional (loss-scale backoff steps) but small. **Set the level-2 threshold = a low single-digit percent above the observed steady-state** (record the observed rate and the chosen threshold + justification).
- [ ] Per-group grad norms (`grad_norm/cca/*`) finite and trending sensibly (recall: loss-scale-multiplied; do NOT compare absolute values to the Phase-7 local run — trend only).
- [ ] Triage any cluster-specific issue (dtype mismatch, OOM, NaN under fp16, loss-scale thrash). Additional short runs allowed until clean. Record each.

**If the short run cannot be made clean:** STOP and escalate with the failure captured. Do not submit the full run on a broken mixed-precision stack.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Full cluster run — the level-2 acceptance bar — HUMAN-OPERATED

**Submit (full 7-epoch run, corrected prior canonical in DEFAULT_CCA_CONFIG):**
```
sbatch scripts/tier5_cluster.sbatch full
```

**Level-2 acceptance checklist (cluster) — read `metrics.csv` across all epochs:**
- [ ] Run completes, exit 0, full 7 epochs.
- [ ] **`grad_overflow_rate < threshold`** (the threshold calibrated in Task 2) for the run / per epoch as specified — this is the headline mixed-precision acceptance criterion.
- [ ] No NaN/Inf in any tracked scalar across the full run.
- [ ] Loss decreasing: `cca/positive_risk/mean` final ≪ initial (≥ ~1 order of magnitude), monotone-ish.
- [ ] Head grad norms bounded across the run (`grad_norm/cca/max` trend stable, no explosion — trend only, fp16-scaled).
- [ ] FLPU components in range: `cca/correction_triggered/mean` not pinned at 1.0; `cca/positive_risk/mean` decreasing.
- [ ] Prediction distribution not collapsed: `cca_pred_dist/std` not ≈0 with `mean`≈prior; record `mean/std/frac_above_0.5` and `val_` counterparts per epoch.
- [ ] Checkpoint + sidecar written (this is the π=0.02 candidate model the level-3 workstream will evaluate).

Record the full per-epoch diagnostic series.
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Update tier5-stress-test-notes.md with cluster summary + level-3 handoff — HUMAN-OPERATED

**Files:** Append to `docs/notes/tier5-stress-test-notes.md` (the "## Phase 8 — Cluster" section stubbed in Phase 7).

Fill:
```markdown
## Phase 8 — Cluster (level-2 acceptance bar)

**Environment:** Explorer, mixed_float16, CUDA. Commit: <sha>. Date: <date>.

### Short run (threshold calibration + triage)
- Submit cmd + job id + outcome.
- Observed grad_overflow_rate (steady state) and the CHOSEN level-2
  threshold + justification.
- Cluster-specific issues found + how triaged (list each short run).

### Full run (acceptance)
- Submit cmd + job id + outcome.
- Per-epoch diagnostic table (paste metrics.csv).
- Each acceptance criterion: value + PASS/FAIL (lead with grad_overflow_rate
  vs threshold).
- **Decision:** Tier 5 level-2 acceptance bar PASS / FAIL.
- **Rationale:** ...

### Level-3 handoff
- Path to the π=0.02 candidate checkpoint + sidecar.
- Diagnostic observations relevant to the downstream π=0.03-vs-0.02
  comparison (correction_triggered rate, prediction-distribution shape,
  loss-component trajectories) — this is the input the separate level-3
  research workstream begins from. (Level-3 itself is OUT of Tier 5 scope.)
```

**Done when:** notes doc updated and committed with the calibrated threshold, the full-run acceptance decision, and the level-3 handoff (candidate model path + diagnostic summary).

```bash
git add docs/notes/tier5-stress-test-notes.md
git commit -m "tier5 phase 8: cluster stress-test notes + level-3 handoff"
```

**Phase 8 Done-when (design DoD):** cluster acceptance run completes; full level-2 pass criteria met (headline: `grad_overflow_rate < calibrated threshold`, stable dynamics under mixed precision); diagnostic outputs documented for the level-3 handoff. **This closes Tier 5.**
<!-- END_TASK_4 -->
