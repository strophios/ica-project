#!/bin/bash
#SBATCH --job-name=embed-9625
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=/projects/ahd/cca_logs/slurm-%j.out

# Embed the 1996-2025 API corpus into a NEW `api_9625` cache and apply the
# deployed IcaModel -> ranked forward candidates. Runbook:
# docs/notes/ica-apply-results-and-cluster-runbook.md ("2026-07-31 forward apply").
#
# Channel: headline + "</s>" + coalesce(lead_paragraph, abstrct)
# (--lead-fallback-column abstrct). 2025 has NO lead_paragraph; 1996-2024 the
# fallback only fills small gaps (2005 ~15% both-missing is the known bad
# patch). --dedupe-ids resolves the 911 pull-overlap duplicate ids and drops
# the 13 empty-id junk rows; dedupe runs BEFORE the year filter, so part1 and
# part2 agree globally on the canonical copy of cross-year duplicates.
#
# Stages (argument, default `all` = part1 + part2 + apply):
#   sbatch scripts/embed_apply_9625.sbatch smoke   # 200-row 2025 smoke, throwaway suffix
#   sbatch scripts/embed_apply_9625.sbatch part1   # 1996-2010 (~1.4M rows)
#   sbatch scripts/embed_apply_9625.sbatch part2   # 2011-2025 append (~1.3M rows)
#   sbatch scripts/embed_apply_9625.sbatch apply
#   sbatch scripts/embed_apply_9625.sbatch all
#
# RESUME RULE: `--append` continues shard NUMBERING, not row-level progress.
# A completed part writes provenance.<offset>.json (part1 -> provenance.000.json;
# part2 -> provenance.<n_part1_shards>.json). If a part dies mid-run its
# provenance file is ABSENT: delete that part's partial shards (index >= the
# part's offset) and resubmit THAT part only.
# Expected runtime: each part ~1.3-1.4M rows — ~2h on an h200 (July tuned
# re-embed rate: ~1.15M rows inside a 4h window), roughly double on an a100;
# the 8h limit covers either. gres is gpu:1 (any GPU — a100 queues faster);
# pin the fast card at submit time if you want it: CLI overrides the
# directive, e.g. `sbatch --gres=gpu:h200 scripts/embed_apply_9625.sbatch all`.

set -euo pipefail

module load use.own uv

cd "/projects/ahd/ica-project"
export ICA_ENV=cluster

STAGE="${1:-all}"
CACHE=/projects/ahd/cca_doca/embed_cache/api_9625
STAMP=20260731
CHANNEL_ARGS=(--lead-fallback-column abstrct --dedupe-ids)

preflight_corpus() {
    # The 1996-2025 parquets were pulled on the operator's machine — the
    # cluster copy must be synced first (66 files incl. 2025.parquet).
    [ -f /projects/ahd/api_corpus/2025.parquet ] || { echo "FATAL: api_corpus/2025.parquet missing — sync the expanded corpus up first"; exit 1; }
    n_files=$(ls /projects/ahd/api_corpus/*.parquet | wc -l)
    [ "$n_files" -eq 66 ] || { echo "FATAL: expected 66 api_corpus parquets, found $n_files"; exit 1; }
}

if [ "$STAGE" = "smoke" ]; then
    preflight_corpus
    uv run python -m src.embed_corpus --full --years 2025-2025 --limit 200 \
        "${CHANNEL_ARGS[@]}" --out-suffix smoke_9625 --stamp "$STAMP"
    echo "SMOKE OK — check cls_std/us_logit stats above and provenance records"
    echo "lead_fallback_column=abstrct, then: rm -r /projects/ahd/cca_doca/embed_cache/smoke_9625"
    exit 0
fi

if [ "$STAGE" = "part1" ] || [ "$STAGE" = "all" ]; then
    preflight_corpus
    n_shards=$(ls "$CACHE"/shard_*_cls.npy 2>/dev/null | wc -l)
    [ "$n_shards" -eq 0 ] || { echo "FATAL: $CACHE already has $n_shards shards — part1 must start clean (see RESUME RULE)"; exit 1; }
    uv run python -m src.embed_corpus --full --years 1996-2010 \
        "${CHANNEL_ARGS[@]}" --out-suffix api_9625 --stamp "$STAMP"
    [ -f "$CACHE/provenance.000.json" ] || { echo "FATAL: part1 finished without provenance.000.json"; exit 1; }
fi

if [ "$STAGE" = "part2" ] || [ "$STAGE" = "all" ]; then
    preflight_corpus
    [ -f "$CACHE/provenance.000.json" ] || { echo "FATAL: part1 provenance missing — run part1 first"; exit 1; }
    n_prov=$(ls "$CACHE"/provenance.*.json | wc -l)
    [ "$n_prov" -eq 1 ] || { echo "FATAL: expected exactly 1 provenance file before part2, found $n_prov"; exit 1; }
    uv run python -m src.embed_corpus --full --years 2011-2025 --append \
        "${CHANNEL_ARGS[@]}" --out-suffix api_9625 --stamp "$STAMP"
    n_prov=$(ls "$CACHE"/provenance.*.json | wc -l)
    [ "$n_prov" -eq 2 ] || { echo "FATAL: part2 finished without its provenance file"; exit 1; }
fi

if [ "$STAGE" = "apply" ] || [ "$STAGE" = "all" ]; then
    n_prov=$(ls "$CACHE"/provenance.*.json 2>/dev/null | wc -l)
    [ "$n_prov" -eq 2 ] || { echo "FATAL: cache incomplete ($n_prov/2 provenance files) — finish part1+part2 first"; exit 1; }
    # Deployed frozen-stack IcaModel; ML US gate (API has no dateline labels).
    # 2025 rows ride the abstract register — era-slice any eval (roadmap A1 item 3).
    uv run python -m src.apply_ica --corpus api --cache-suffix api_9625 \
        --out-name api_1996_2025.parquet
fi
