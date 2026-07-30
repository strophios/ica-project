#!/bin/bash
#SBATCH --job-name=re-embed
#SBATCH --partition=gpu     # e.g. gpu-short for the test, gpu for the full
#SBATCH --gres=gpu:h200         # gpu:h200 (preferably) or a100 for the full run
#SBATCH --time=04:00:00           # e.g. 06:00:00 (short: 00:30:00)
#SBATCH --mem=64G                 # e.g. 64G
#SBATCH --output=/projects/ahd/relevance/logs/slurm-%j.out

set -euo pipefail

# --- Cluster module / env setup (OPERATOR-SUPPLIED) -----------------------
module load use.own uv
# ------------------------------------------------------------------------

cd "/projects/ahd/ica-project"        # repo root on the cluster
export ICA_ENV=cluster                            # force cluster config

TB=/projects/ahd/relevance/tuned_backbone.job8823087.weights.h5

# 1. relevance_train_tuned — exact original 266,018 API rows, raw channel
uv run python -m src.embed_corpus --include-ids /projects/ahd/relevance/relevance_train_ids.parquet \
    --corpus api_corpus --out-suffix relevance_train_tuned --stamp 20260729 --backbone-weights "$TB"

# 2. train250k_tuned — exact original 250,251 API rows (do NOT replay --sample-n:
#    the corpus grew to 2025, the stratified sample would differ)
uv run python -m src.embed_corpus --include-ids /projects/ahd/relevance/train250k_ids.parquet \
    --corpus api_corpus --out-suffix train250k_tuned --stamp 20260729 --backbone-weights "$TB"

# 3. us_train_ldc_tuned — population is stable (all labeled LDC rows), replay is safe
uv run python -m src.embed_corpus --full --source-pattern us_filter/ldc_labeled.parquet \
    --lead-column stripped_text --label-column us_label --no-year \
    --out-suffix us_train_ldc_tuned --stamp 20260729 --backbone-weights "$TB"
