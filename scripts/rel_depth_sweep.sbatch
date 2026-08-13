#!/bin/bash
#SBATCH --job-name=rel-depth-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# 8h, not 4: no --gres type constraint means the scheduler may hand us a V100
# (3-5x slower than the H100s recent ~30-min runs used) — a slow node + a full
# 7-epoch run + CPU-forced scoring needs the margin. Known trade: longer
# limits queue slightly worse.
#SBATCH --mem=64G
#SBATCH --array=0-11
#SBATCH --output=/projects/ahd/cca_logs/slurm-%A_%a.out

# Stage-3 rel depth sweep (branched-encoder ladder,
# docs/notes/branched-encoder-strategy.md): text-mode rel training with top-N
# unfreeze, HARD freezing below the branch, over the 12-point grid
#   N in {1,2,3}  x  {flat, graded-decay 0.385 = 1/2.6}  x  seeds {200, 201}
# (2 seeds per point: single-run noise on the gold eval is ~+/-0.015 vs-ICA —
# treat smaller deltas as ties; stage-2 record in the strategy note).
#
# Each job trains its point, then scores it CPU-forced with
# scripts/eval_rel_sweep_point.py -> a compact JSON next to the weights.
# Selection happens over the JSONs (rsync only *.eval.json home); only the
# winning artifact's backbone is extracted/synced.
#
# Submit:              sbatch scripts/rel_depth_sweep.sbatch
# One point (e.g. 5):  sbatch --array=5 scripts/rel_depth_sweep.sbatch
#
# Index decode: seed = idx%2 ? 201 : 200; decay = (idx/2)%2 ? graded : flat;
# N = idx/4 + 1. So 0..3 -> N=1, 4..7 -> N=2, 8..11 -> N=3.

set -euo pipefail
shopt -s nullglob

echo "rel_depth_sweep: task=${SLURM_ARRAY_TASK_ID:-unset} host=$(hostname) $(date)"  # never die log-silent

module load use.own uv

cd /projects/ahd/ica-project
export ICA_ENV=cluster

IDX="${SLURM_ARRAY_TASK_ID}"
N=$(( IDX / 4 + 1 ))
SEED=$(( IDX % 2 == 0 ? 200 : 201 ))
if (( (IDX / 2) % 2 == 0 )); then
    DECAY_ARGS=(); DECAY_TAG="flat"
else
    # graded_multipliers' `decay` is a per-layer FACTOR in (0,1] (layer k below
    # the top gets base*decay^k), NOT the ULMFiT divisor convention — 1/2.6.
    DECAY_ARGS=(--graded-decay 0.385); DECAY_TAG="graded"
fi

SWEEP_DIR=/projects/ahd/relevance/sweep
mkdir -p "$SWEEP_DIR"
# Hard-freeze A/B toggle (2026-08-13, the -0.04 vs-July investigation):
#   sbatch --array=0,1 --export=ALL,SWEEP_NO_HARD_FREEZE=1 scripts/rel_depth_sweep.sbatch
# runs the same points with --no-hard-freeze under a "_nhf" tag.
if [[ -n "${SWEEP_NO_HARD_FREEZE:-}" ]]; then
    FREEZE_ARGS=(--no-hard-freeze); FREEZE_TAG="_nhf"
else
    FREEZE_ARGS=(--hard-freeze); FREEZE_TAG=""
fi
TAG="rel_text_N${N}_${DECAY_TAG}_s${SEED}${FREEZE_TAG}"
WEIGHTS="$SWEEP_DIR/${TAG}.weights.h5"
EVAL_JSON="$SWEEP_DIR/${TAG}.eval.json"

if [[ -f "$EVAL_JSON" ]]; then
    echo "$TAG: eval JSON already present, skipping (delete it to re-run)"
    exit 0
fi

echo "$TAG: training (N=$N ${DECAY_TAG} seed=$SEED freeze=${FREEZE_ARGS[0]})"
uv run python -m src.run_relevance_text \
    --unfreeze-top-n "$N" \
    ${DECAY_ARGS[@]+"${DECAY_ARGS[@]}"} \
    --seed "$SEED" \
    "${FREEZE_ARGS[@]}" \
    --prior 0.02 \
    --eta 0.0 \
    --epochs 7 \
    --weights-out "$WEIGHTS"

echo "$TAG: scoring (CPU-forced)"
uv run python -m scripts.eval_rel_sweep_point \
    --weights "$WEIGHTS" \
    --out "$EVAL_JSON"

echo "$TAG: done $(date)"
