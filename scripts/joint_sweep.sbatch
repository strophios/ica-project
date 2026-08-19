#!/bin/bash
#SBATCH --job-name=joint-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# 8h, matching rel_depth_sweep.sbatch's reasoning: no --gres type constraint
# means the scheduler may hand us a V100 (3-5x slower than the H100s recent
# runs used) -- a slow node + a full 7-epoch TWO-HEAD run + CPU-forced
# scoring needs the margin.
#SBATCH --mem=64G
#SBATCH --array=0-11
#SBATCH --output=/projects/ahd/cca_logs/slurm-%A_%a.out

# Stage-4 joint CCA+rel fine-tune sweep
# (docs/design-plans/2026-08-18-stage4-joint-finetune.md "Shape"): one shared
# DAPT encoder, top-N unfreeze, TWO ClassificationHeads (cca, rel) trained
# together via a single scalarized loss lambda*L_rel + (1-lambda)*L_cca, over
# the 12-point grid
#   N in {1,2}  x  lambda in {0.25, 0.5, 0.75}  x  seeds {200, 201}
#
# --no-hard-freeze always (unlike rel_depth_sweep.sbatch's toggle): the
# stage-3 deploy rule is multiplier-freeze-train + graft-at-deploy, so this
# sweep trains under the same regime the eventual graft will read from.
# --prior is NOT a sweep knob here -- both heads' priors (0.02) are fixed in
# src.run_joint_text._default_joint_text_config, not a CLI override.
#
# Each job trains its point, then scores it CPU-forced with
# scripts/eval_joint_sweep_point.py -> a compact JSON next to the weights.
# Selection happens over the JSONs (rsync only *.eval.json home); only the
# winning artifact's backbone is extracted/synced (mirrors rel_depth_sweep's
# selection discipline).
#
# Submit:              sbatch scripts/joint_sweep.sbatch
# One point (e.g. 5):  sbatch --array=5 scripts/joint_sweep.sbatch
#
# Index decode: N = idx/6 + 1 (idx 0-5 -> N=1, idx 6-11 -> N=2); lambda =
# (0.25, 0.5, 0.75)[(idx%6)/2] (idx%6 in {0,1}->0.25, {2,3}->0.5, {4,5}->0.75);
# seed = 200 + idx%2.

set -euo pipefail
shopt -s nullglob

echo "joint_sweep: task=${SLURM_ARRAY_TASK_ID:-unset} host=$(hostname) $(date)"  # never die log-silent

module load use.own uv

cd /projects/ahd/ica-project
export ICA_ENV=cluster

IDX="${SLURM_ARRAY_TASK_ID}"
N=$(( IDX / 6 + 1 ))
SEED=$(( 200 + IDX % 2 ))

LAMS=(0.25 0.5 0.75)
LAM_TAGS=(025 050 075)
LAM_IDX=$(( (IDX % 6) / 2 ))
LAM="${LAMS[$LAM_IDX]}"
LAM_TAG="${LAM_TAGS[$LAM_IDX]}"

SWEEP_DIR=/projects/ahd/relevance/joint_sweep
mkdir -p "$SWEEP_DIR"

TAG="joint_N${N}_lam${LAM_TAG}_s${SEED}"
WEIGHTS="$SWEEP_DIR/${TAG}.weights.h5"
EVAL_JSON="$SWEEP_DIR/${TAG}.eval.json"

# Score-only resume (2026-08-19): a task that trained but timed out during the
# CPU-forced scoring (weights + config sidecar saved, no eval JSON) must not
# retrain on resubmit. Known hazard on FULL cold-cache grids: the trainer's
# shared tf.data snapshot (relevance/joint_set/*.tf) is written by whichever
# task gets there first — 12 simultaneous cold starts can race a reader
# against a mid-write (job 9340533_0 died exactly so). For a cold cache,
# warm it first: `sbatch --array=0 ...`, wait, then submit the rest.
CONFIG_JSON="${WEIGHTS%.weights.h5}.config.json"
if [[ ! -f "$EVAL_JSON" && -f "$WEIGHTS" && -f "$CONFIG_JSON" ]]; then
    echo "$TAG: weights exist, eval missing -> score-only resume"
    uv run python -m scripts.eval_joint_sweep_point \
        --weights "$WEIGHTS" \
        --out "$EVAL_JSON"
    echo "$TAG: done (score-only) $(date)"
    exit 0
fi

if [[ -f "$EVAL_JSON" ]]; then
    echo "$TAG: eval JSON already present, skipping (delete it to re-run)"
    exit 0
fi

echo "$TAG: training (N=$N lambda=$LAM seed=$SEED, --no-hard-freeze)"
uv run python -m src.run_joint_text \
    --lam "$LAM" \
    --unfreeze-top-n "$N" \
    --seed "$SEED" \
    --no-hard-freeze \
    --epochs 7 \
    --weights-out "$WEIGHTS"

echo "$TAG: scoring (CPU-forced)"
uv run python -m scripts.eval_joint_sweep_point \
    --weights "$WEIGHTS" \
    --out "$EVAL_JSON"

echo "$TAG: done $(date)"
