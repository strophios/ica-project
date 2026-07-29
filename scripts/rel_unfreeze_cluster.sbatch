#!/bin/bash
#SBATCH --job-name=rel-unfreeze
#SBATCH --partition=gpu     # e.g. gpu-short for the test, gpu for the full
#SBATCH --gres=gpu:h200         # gpu:1:h200 (preferably) or a100 for the full run
#SBATCH --time=00:45:00           # e.g. 06:00:00 (short: 00:30:00)
#SBATCH --mem=64G                 # e.g. 64G
#SBATCH --output=/projects/ahd/relevance/logs/slurm-%j.out

set -euo pipefail

# --- Time/mem budgeting (read before filling in the placeholders above) ---
# Full-encoder backprop happens regardless of unfreeze_top_n: the
# escalation branch's 0.0 multiplier on "encoder_frozen" ZEROES THE UPDATE,
# not the forward/backward COMPUTATION -- every RoBERTa layer still gets a
# gradient computed each step (LayerLRModel scales gradients post-hoc; it
# does not set backbone.trainable=False when unfreeze_top_n>0). Budget compute
# time as if training the full encoder, not just the top N layers.
#
# Population: ~266k rows/epoch (the relevance_train population, minus the
# ICA-eval holdout -- see src/build_relevance_text_table.py's row-count
# report for the exact current count). At batch=256, seq_length=128, that's
# ~1,040 steps/epoch for the Ratio-Batch train_pos=0.1 stream (steps_per_epoch
# = floor(n_pos_train / (batch * train_pos)); n_pos_train is the candidate-
# positive count, a small fraction of the 266k population, so steps/epoch is
# driven by positive count, not total rows -- check the build table's printed
# "positives US-restricted" count to size this run's steps_per_epoch before
# picking --time). mixed_float16 (cluster DTYPE_POLICY) speeds the backward
# pass on Tensor Cores but does not change which layers get gradients.
# --------------------------------------------------------------------------

# --- Cluster module / env setup (OPERATOR-SUPPLIED) -----------------------
# NO CUDA module: tensorflow[and-cuda] bundles the CUDA runtime as pip
# wheels (site-packages/nvidia/*); only the node's NVIDIA driver is needed.
# Loading a CUDA module can SHADOW the bundled libs via LD_LIBRARY_PATH and
# cause version-mismatch errors. If GPUs come back empty, diagnose with
# `nvidia-smi` inside the job (driver/allocation), not a module load.
module load use.own uv
# ------------------------------------------------------------------------

cd "/projects/ahd/ica-project"        # repo root on the cluster
export ICA_ENV=cluster                            # force cluster config
# No venv activation -- project convention is `uv run` (see CLAUDE.md).
# `uv sync --frozen` must have been run ONCE on the login node beforehand;
# --frozen below means jobs never attempt resolution (compute nodes may
# lack internet).

# Prerequisite -- the training table lives OUTSIDE the repo, at the cluster
# DATA root (config.py on-cluster: PROJECT_ROOT == /projects/ahd == data
# root; the repo checkout under it is code-only, same data-outside-repo
# convention as local). Upload the locally-built artifact once:
#   rsync -avz <local>/00_explorer/relevance/relevance_text_table.parquet \
#       <cluster>:/projects/ahd/relevance/
# (Rebuilding on-cluster via `uv run python -m src.build_relevance_text_table`
# is possible but needs api_corpus + the relevance candidates/negatives +
# holdout parquets uploaded first -- the 55MB table upload is the short path.)
# This job assumes /projects/ahd/relevance/relevance_text_table.parquet exists.

# --- TensorBoard (OPERATOR-ADJUSTABLE) -------------------------------------
# Points at a job-id-scoped dir under the cluster relevance artifacts area
# (config.RELEVANCE_DIR == /projects/ahd/relevance on-cluster) so concurrent
# jobs don't collide. Event files are append-only and safe to rsync mid-run:
#   rsync -avz <cluster>:/projects/ahd/relevance/tb_logs/<job_id>/ ./tb_local/
#   uvx tensorboard --logdir ./tb_local
# Or view directly on the cluster: ssh -L 6006:localhost:6006 <login-node>,
# then on the login node: uvx tensorboard --logdir <TB_LOG_DIR> --port 6006.
TB_LOG_DIR="/projects/ahd/relevance/tb_logs/${SLURM_JOB_ID:-manual}"
# ----------------------------------------------------------------------------

# Mode is chosen by the submit command:
#   sbatch scripts/rel_unfreeze_cluster.sbatch short        -> short calibration run
#   sbatch scripts/rel_unfreeze_cluster.sbatch full 1       -> flat N=1 run (baseline)
#   sbatch scripts/rel_unfreeze_cluster.sbatch full 3 0.5   -> graded top-3 run
#                                                              (0.1/0.05/0.025 per-layer LRs)
UNFREEZE_TOP_N="${2:-1}"
GRADED_DECAY="${3:-}"     # empty = flat encoder_top scheme
MODE="${1:-full}"
EXTRA_ARGS=("${@:4}")     # passed through to the trainer, e.g. --eta 0.5 --peak-lr 1e-4
if [ "$MODE" = "short" ]; then
    uv run --frozen python -m src.run_relevance_text --unfreeze-top-n "$UNFREEZE_TOP_N" --epochs 1 --max-steps 200 \
        ${GRADED_DECAY:+--graded-decay "$GRADED_DECAY"} --tensorboard-dir "$TB_LOG_DIR" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    uv run --frozen python -m src.run_relevance_text --unfreeze-top-n "$UNFREEZE_TOP_N" --epochs 7 \
        ${GRADED_DECAY:+--graded-decay "$GRADED_DECAY"} --tensorboard-dir "$TB_LOG_DIR" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
fi

# Preserve this job's artifact pair under a job-id-suffixed name: successive
# runs OVERWRITE relevance_text.*, so without this each experiment clobbers
# the previous candidate. The suffixed copies keep every run comparable via
# scripts/eval_rel_text_artifact.py --weights <path>.
for ext in weights.h5 config.json; do
    src="/projects/ahd/relevance/relevance_text.${ext}"
    if [ -f "$src" ]; then
        cp "$src" "/projects/ahd/relevance/relevance_text.job${SLURM_JOB_ID:-manual}.${ext}"
    fi
done
