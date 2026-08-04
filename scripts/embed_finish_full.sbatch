#!/bin/bash
#SBATCH --job-name=embed-full-7695
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=/projects/ahd/cca_logs/slurm-%j.out

# Finish the `full` API cache (1976-1995) and re-apply the deployed IcaModel
# over the complete 1960-1995 range. Runbook:
# docs/notes/ica-apply-results-and-cluster-runbook.md ("2026-07-31 forward apply").
#
# Stages (argument, default `all` = embed + apply):
#   sbatch scripts/embed_finish_full.sbatch smoke   # 200-row smoke, throwaway suffix
#   sbatch scripts/embed_finish_full.sbatch embed   # 1976-1995 append into `full`
#   sbatch scripts/embed_finish_full.sbatch apply   # apply_ica over the completed cache
#   sbatch scripts/embed_finish_full.sbatch all
#
# RESUME RULE: `--append` continues shard NUMBERING, not row-level progress.
# If the embed stage dies mid-part, its provenance.008.json will be ABSENT;
# delete the partial shards (shard_008_* and up) before resubmitting:
#   rm /projects/ahd/cca_doca/embed_cache/full/shard_0{08..99}_* 2>/dev/null
# Expected runtime: ~1.9M rows — ~2.5h on an h200 (July tuned re-embed rate),
# roughly double on an a100; the 8h limit covers either. gres is gpu:1 (any
# GPU — a100 queues faster); pin via CLI if wanted:
# `sbatch --gres=gpu:h200 scripts/embed_finish_full.sbatch all`.

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
STAMP=20260731
# Pin to the 66 top-level parquets (the default glob is recursive and the
# cluster api_corpus/ has extra subdirectory parquets — 2026-08-04 finding).
SOURCE_ARGS=(--source-pattern 'api_corpus/*.parquet')

preflight_embed() {
    # Bookend checks only (no total-count assert — the corpus legitimately
    # grows; the year filter is the population pin). NO coalesce for
    # 1976-1995 (pre-1996 stays on the trained raw channel).
    [ -f /projects/ahd/api_corpus/1976.parquet ] || { echo "FATAL: api_corpus/1976.parquet missing (sync the corpus)"; exit 1; }
    [ -f /projects/ahd/api_corpus/1995.parquet ] || { echo "FATAL: api_corpus/1995.parquet missing (sync the corpus)"; exit 1; }
    # Idempotence guard: exactly the 8 part-1 shards, not already appended.
    n_shards=$(count_glob "$CACHE"/shard_*_cls.npy)
    [ "$n_shards" -eq 8 ] || { echo "FATAL: expected 8 existing shards in $CACHE, found $n_shards (already appended, or partial state — see RESUME RULE)"; exit 1; }
    [ ! -f "$CACHE/provenance.008.json" ] || { echo "FATAL: provenance.008.json exists — 1976-1995 already appended"; exit 1; }
}

if [ "$STAGE" = "smoke" ]; then
    uv run python -m src.embed_corpus --full --years 1976-1976 --limit 200 \
        "${SOURCE_ARGS[@]}" --out-suffix smoke_full_7695 --stamp "$STAMP"
    echo "SMOKE OK — inspect stats above, then: rm -r /projects/ahd/cca_doca/embed_cache/smoke_full_7695"
    exit 0
fi

if [ "$STAGE" = "embed" ] || [ "$STAGE" = "all" ]; then
    preflight_embed
    uv run python -m src.embed_corpus --full --years 1976-1995 --append \
        "${SOURCE_ARGS[@]}" --out-suffix full --stamp "$STAMP"
    [ -f "$CACHE/provenance.008.json" ] || { echo "FATAL: append finished without provenance.008.json"; exit 1; }
fi

if [ "$STAGE" = "apply" ] || [ "$STAGE" = "all" ]; then
    # Deployed frozen-stack IcaModel; ML US gate (API has no dateline labels).
    # Writes cca_doca/ica_candidates/api_1960_1995.parquet (now truly 1960-1995)
    # + per-year us_filter/api_us_scores/ and cca_doca/api_cca_scores/.
    uv run python -m src.apply_ica --corpus api
fi
