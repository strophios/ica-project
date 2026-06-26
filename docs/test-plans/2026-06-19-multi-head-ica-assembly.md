# Human test plan — Multi-Head ICA Assembly

*Generated 2026-06-26 after the 6-phase assembly. Coverage validation: PASS
(18/18 automated-test-obligated ACs covered; 7 operational-only ACs documented).
Companion to `docs/implementation-plans/2026-06-19-multi-head-ica-assembly/test-requirements.md`.*

Most of this plan is the **cluster-deferred** work and the human-judgment checks
that can't be unit-tested. The automated suite (942 passing) covers the decision
logic; this covers the real-data execution. Items already done this session are
marked ✅.

## Prerequisites

- Project root, all commands `uv run …`. `uv run pytest -q` green (942).
- Out-of-repo data present: `cca_doca/embed_cache/*`, `us_filter/ldc_labeled.parquet`,
  `api_corpus/`, `ldc_corpus`, `US_FILTER_FULL_WEIGHTS`, `validation/ica_holdout_ids.parquet`.
- GPU/cluster (Explorer) for the two embed re-runs.

## Phase A — pre-flight gate (run first)

| Step | Action | Expected |
|------|--------|----------|
| A1 | `uv run python -m scripts.preflight_assembly` | 6 verdicts; after Phase 4/6 remediation us_weights/calibration/doca_freshness/ldc_channel/ldc_gold_coverage all PASS (ldc_channel becomes PASS only once the stripped LDC cache is the apply source). Exit 0. |

## Phase B — cluster-deferred embed re-runs (the compute gap)

Full commands in `docs/notes/ica-apply-results-and-cluster-runbook.md`.

| Step | Action | Expected |
|------|--------|----------|
| B1 | API 1976–1995 append embed (`embed_corpus --years 1976-1995 --append --out-suffix full`); smoke with `--limit` first | `full` cache spans 1960–1995; provenance records raw API channel. |
| B2 | `scripts.build_stripped_ldc_source` (cheap; can run locally) | source parquet `id, stripped_text, us_label`. |
| B3 | Stripped LDC embed (`embed_corpus --lead-column stripped_text --out-suffix ldc_9607_stripped`); smoke first | distinct cache; provenance `lead_column=stripped_text`. Optional — only refines the US-head fallback on the 44% non-gold LDC rows. |

## Phase C — full-corpus apply

| Step | Action | Expected |
|------|--------|----------|
| C1 | `uv run python -m src.apply_ica --corpus api` over the completed `full` cache | per-year US+CCA score dirs + `ica_candidates/api_1960_1995.parquet`; schema `id,year,us_score,cca_score,rel_score,ica_score,gated`; scores ∈ [0,1]; gated-out → ica_score 0. ✅ done for 1960–1975. |
| C2 | `uv run python -m src.apply_ica --corpus ldc` (raw `ldc_9507`, or `--cache-suffix ldc_9607_stripped` after B3) | `ica_candidates/ldc_1996_2007.parquet`; logs gold-vs-fallback split (≈56% gold). ✅ done on `ldc_9507`. |

## End-to-end — gold-first gating on real LDC (AC6.3)

From the LDC apply, sample rows where `us_label` is present and disagrees with the
ML US head; confirm the final `gated` follows the gold label (True-gold keeps even
if ML rejects; False-gold blocks even if ML passes); null-`us_label` → ML fallback.
✅ observed this session (e.g. "Hispanic March Draws Crowd to Capital", ML us=0.34,
gated in by gold).

## Human-judgment checks (no oracle)

| Criterion | Step |
|-----------|------|
| AC2.1 hand-coding | ✅ done — `validation/ica_coding_template_coded.csv`, 214 ICA positives; `validate_gold_set` passes. |
| AC3.3 composed-Platt budget | ✅ done — EPV=105 supported the 2-param Platt; ECE 0.140→0.047, recorded in `ica_fusion_metrics.json`. |
| AC6.1/6.2 face validity | ✅ spot-checked this session — LDC top candidates are clean US immigrant collective action (asylum/2006 marches/hunger strikes); API top candidates are diaspora ICA (Soviet Jewry/Cuban exile marches). Re-review the full-corpus candidates after the cluster embed; watch API foreign false positives (the documented US-head diaspora ceiling — see `us-head-retrain-plan.md`). |

## Known limitations (documented, not bugs)

- **US-head diaspora recall ceiling** — the US ML head under-scores diaspora
  collective action (US-soil protest about foreign topics). Gold-first contains it
  for labeled LDC rows; the API corpus (no gold labels) shows some foreign false
  positives at the lenient τ_us=0.02. Root fix: the US-head retrain
  (`docs/notes/us-head-retrain-plan.md`).
- **API 1976–1995 not yet scored** — needs the cluster embed (Phase B1).
