#!/bin/bash
#SBATCH --job-name=rel-unfreeze
#SBATCH --partition=<PLACEHOLDER_PARTITION>      # e.g. gpu
#SBATCH --account=<PLACEHOLDER_ACCOUNT>          # Explorer allocation
#SBATCH --gres=gpu:1
#SBATCH --time=<PLACEHOLDER_TIME>                # e.g. 06:00:00 (short: 00:30:00)
#SBATCH --mem=<PLACEHOLDER_MEM>                  # e.g. 64G
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
# module load <PLACEHOLDER_CUDA_MODULE>
# module load <PLACEHOLDER_PYTHON_MODULE>
# ------------------------------------------------------------------------

cd "/projects/ahd/<PLACEHOLDER_REPO_PATH>"        # repo root on the cluster
source .venv/bin/activate
export ICA_ENV=cluster                            # force cluster config

# Prerequisite (run once, not part of this job -- cheap, CPU-only):
#   uv run python -m src.build_relevance_text_table
# This job assumes relevance/relevance_text_table.parquet already exists.

# Mode is chosen by the submit command:
#   sbatch scripts/rel_unfreeze_cluster.sbatch short   -> short calibration run
#   sbatch scripts/rel_unfreeze_cluster.sbatch full    -> full rel-first run
UNFREEZE_TOP_N="${2:-1}"
MODE="${1:-full}"
if [ "$MODE" = "short" ]; then
    python -m src.run_relevance_text --unfreeze-top-n "$UNFREEZE_TOP_N" --epochs 1 --max-steps 200
else
    python -m src.run_relevance_text --unfreeze-top-n "$UNFREEZE_TOP_N" --epochs 7
fi
