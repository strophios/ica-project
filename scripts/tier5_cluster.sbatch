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

cd "/projects/ahd/<PLACEHOLDER_REPO_PATH>"        # repo root on the cluster
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
