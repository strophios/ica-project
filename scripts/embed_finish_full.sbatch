#!/bin/bash
#SBATCH --job-name=embed-full-6095
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=/projects/ahd/cca_logs/slurm-%j.out

# Build the `full` API cache (1960-1995, FRESH — no append) and apply the
# deployed IcaModel over it. Runbook:
# docs/notes/ica-apply-results-and-cluster-runbook.md ("2026-07-31 forward apply"
# + the 2026-08-04 addenda).
#
# Why fresh instead of appending to the June part-1 (1960-1975) cache: part 1
# was produced LOCALLY and never synced; at the measured cluster rate
# (~2.7M rows / 20 min on h200) a full rebuild is faster than uploading
# ~5.7GB of shards — and it retires the part-1 metadata contamination (the
# local MPS bug's distorted us_logit column) from the production cache.
# Raw channel, NO coalesce, NO dedupe (all dup ids live in 1996-2025).
#
# Stages (argument, default `all` = embed + apply):
#   sbatch scripts/embed_finish_full.sbatch smoke   # 200-row smoke, throwaway suffix
#   sbatch scripts/embed_finish_full.sbatch embed   # 1960-1995 fresh build
#   sbatch scripts/embed_finish_full.sbatch apply   # apply_ica over the cache
#   sbatch scripts/embed_finish_full.sbatch all
#
# RESUME RULE: a completed embed writes provenance.000.json. If the embed
# dies mid-run the provenance file is ABSENT: clear the cache dir entirely
# (rm /projects/ahd/cca_doca/embed_cache/full/shard_* ) and resubmit.
# Expected runtime: ~3.7M rows — ~30 min on an h200, ~1h on an a100.

set -euo pipefail
shopt -s nullglob
# Counting convention: pure-bash glob arrays, NEVER `ls ... | wc -l` — under
# pipefail an unmatched glob makes ls exit 2 with stderr suppressed, killing
# the script silently (the 2026-08-04 empty-log job deaths).
count_glob() { local -a m=("$@"); echo "${#m[@]}"; }

echo "embed_finish_full: stage=${1:-all} host=$(hostname) $(date)"  # never die log-silent

module load use.own uv

cd "/projects/ahd/ica-project"
export ICA_ENV=cluster
# fp32: the existing 8 part-1 shards are locally-produced float32 — an fp16
# append would seam the cache numerically at the 1975/76 era boundary.
export ICA_DTYPE_POLICY=float32

STAGE="${1:-all}"
CACHE=/projects/ahd/cca_doca/embed_cache/full
STAMP=20260804
# Pin to the 66 top-level parquets (the default glob is recursive and the
# cluster api_corpus/ has extra subdirectory parquets — 2026-08-04 finding).
SOURCE_ARGS=(--source-pattern 'api_corpus/*.parquet')

preflight_embed() {
    # Bookend checks only (no total-count assert — the corpus legitimately
    # grows; the year filter is the population pin). NO coalesce for
    # 1960-1995 (pre-1996 stays on the trained raw channel).
    [ -f /projects/ahd/api_corpus/1960.parquet ] || { echo "FATAL: api_corpus/1960.parquet missing (sync the corpus)"; exit 1; }
    [ -f /projects/ahd/api_corpus/1995.parquet ] || { echo "FATAL: api_corpus/1995.parquet missing (sync the corpus)"; exit 1; }
    # Fresh-build guard: the cluster cache dir must be empty (see RESUME RULE).
    n_shards=$(count_glob "$CACHE"/shard_*_cls.npy)
    [ "$n_shards" -eq 0 ] || { echo "FATAL: $CACHE already has $n_shards shards — fresh build requires an empty cache (see RESUME RULE)"; exit 1; }
}

if [ "$STAGE" = "smoke" ]; then
    uv run python -m src.embed_corpus --full --years 1976-1976 --limit 200 \
        "${SOURCE_ARGS[@]}" --out-suffix smoke_full_7695 --stamp "$STAMP"
    echo "SMOKE OK — inspect stats above, then: rm -r /projects/ahd/cca_doca/embed_cache/smoke_full_7695"
    exit 0
fi

if [ "$STAGE" = "embed" ] || [ "$STAGE" = "all" ]; then
    preflight_embed
    uv run python -m src.embed_corpus --full --years 1960-1995 \
        "${SOURCE_ARGS[@]}" --out-suffix full --stamp "$STAMP"
    [ -f "$CACHE/provenance.000.json" ] || { echo "FATAL: embed finished without provenance.000.json"; exit 1; }
fi

if [ "$STAGE" = "apply" ] || [ "$STAGE" = "all" ]; then
    # Deployed frozen-stack IcaModel; ML US gate (API has no dateline labels).
    # Writes cca_doca/ica_candidates/api_1960_1995.parquet (now truly 1960-1995)
    # + per-year us_filter/api_us_scores/ and cca_doca/api_cca_scores/.
    uv run python -m src.apply_ica --corpus api
fi
