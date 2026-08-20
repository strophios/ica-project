# Branched productionization — operational runbook

*Created 2026-08-20. The mechanical sequence from the completed branched
build (`c4f2f1b`; contract in
`docs/design-plans/2026-08-18-stage4-joint-finetune.md`) to deployed
artifacts. **HOLD STATUS (2026-08-20): steps 5–6 (corpus-scale embeds +
applies) are ON HOLD pending the post-meeting strategic review** — a team
meeting surfaced reprioritization and potential construction errors; do not
spend corpus-scale compute until that review lands. Steps 1–4 are the cheap
validation slice (~1h total) and are safe to run anytime: they validate
machinery, not labels.*

**For the strategic-review session** — triage the meeting's output by blast
radius before touching the roadmap:
1. *Eval-side errors* (gold codings, the 1,131-row set, holdout) —
   invalidate measurements only; artifacts survive; re-draw/re-code + re-run
   evals.
2. *Label-side errors* (DoCA match, rel candidates pool, US label
   semantics, reliable negatives) — invalidate the affected head's
   training; caches + architecture survive; features-mode retrains are
   minutes once labels are fixed.
3. *Corpus/data errors* (ids, eras, text channel) — potentially invalidate
   caches; the most expensive class; exactly why steps 5–6 hold.
4. *Composition/definition mismatches* (us/cca/rel vs the team's
   CA+IMM+CLAIM) — a design conversation, not a bug; the branched
   architecture's new-head extensibility (strategy note) is the relevant
   machinery.
Known soft spots most likely implicated: the rel positives pool
(descriptor provenance), the eval set's non-random draw (no random gold
sample exists), diaspora/US-gate semantics, definitional alignment (§A2).

Execution rules throughout: local runs CPU-forced; cluster embeds
`export ICA_DTYPE_POLICY=float32`; every new artifact gets the CPU-vs-GPU
rank-consistency acceptance check (`metal-execution-findings.md` deployment
rules); nothing overwrites a production path (the guards enforce this —
work in non-default paths until the explicit swap step).

## Step 1 — smoke + fusion-population branched embed (cluster, ~5–10 min)

```
uv run python -m src.embed_corpus \
    --include-ids <the relevance_train population ids parquet> \
    --branch rel_branch=/projects/ahd/relevance/tuned_backbone.job8823087.weights.h5:1 \
    --stamp <today> --out-suffix relevance_train_branched
```

(Reproduce the `relevance_train` population exactly — check that cache's
provenance for its original selection flags; the goal is the same 266,018
rows with TWO cls arrays.) Verify in the log/provenance: the graft
verification block (nonzero only at `transformer_layer_11` vs base, 0.0
there vs donor), per-shard `cls_std` sane for BOTH variants, fp32 policy.
Rsync the cache back (both arrays + metas + provenance).

**Acceptance (local, minutes):** cosine self-check of the branched variant
vs the existing `relevance_train_tuned` cache on shared ids — stage 1 says
graft ≈ full-tuned; mean cosine should be ≥ ~0.999 with the known benign
tail. The base `cls` array must match `relevance_train`'s to ~exactness.

## Step 2 — rel calibration on the branch variant (local, minutes)

Gap: `calibrate_relevance.py` has no `--variant` knob. Two options:
- **A (default, defensible):** reuse the existing p02 calibration
  (`relevance/scratch_diag/relevance_tuned_p02_cpu.calibration.json`, fit
  on the full-tuned cache) — stage 1 proved graft≡tuned within 4e-4 on
  rank metrics, and Platt A/B on near-identical logit distributions shift
  negligibly. Record the approximation in the swap notes.
- **B (strict):** add a `--variant` passthrough to `calibrate_relevance.py`
  (small change, mirrors `load_cache(variant=)`), refit on
  `relevance_train_branched`/`rel_branch`, CPU-forced.

## Step 3 — fusion refit on branched features (local, minutes)

```
PYTHONPATH=. uv run python -m src.fit_fusion \
    --cache-suffix relevance_train_branched \
    --rel-weights ../relevance/scratch_diag/relevance_tuned_p02_cpu.weights.h5 \
    --rel-feature-variant rel_branch \
    --out ../cca_doca/branched_fusion
```

CCA/US weights stay production defaults. The sidecar records
`head_feature_sources={"us":"base","cca":"base","rel":"rel_branch"}`.
**This produces the first correct-math composed number** — compare against
the production fusion's composed metrics AND the branched-baseline proxy
0.8064 (`cca_doca/experiments/branched_baseline_proxy.json`; expect the
full fusion machinery ≥ the plain-product proxy).

## Step 4 — gold re-eval + swap decision inputs (local, minutes)

`reload_and_score_ica(us, cca, rel=p02, fusion=branched, features={"base":
..., "rel_branch": ...}, head_feature_sources=...)` over the gold eval rows
(features from the step-1 cache via id-join). Report composed ROC/PR vs
`ica_event`, diaspora slice, per-head own-terms, vs the production
composed baseline (0.797 metal-era; this re-measures it too — run the
production stack through the same harness for the like-for-like pair).
CPU-vs-GPU rank consistency on both stacks. **Swap decision** = operator
sign-off on the comparison + the calibration-approximation note (step 2A).

## Step 5 — corpus-scale branched embeds (cluster) — ON HOLD

- `full` (1960–1995, ~3.7M rows): the existing `embed_finish_full.sbatch`
  pattern + `--branch ...` (~2× the measured ~20–30 min/backbone on h200).
- 1996–2025 forward (the long-deferred item 3 embed): + `--dedupe-ids`
  + `--lead-fallback-column abstrct` per the recorded coalesce policy;
  era-slice any 2025+ eval.

## Step 6 — applies + swap (cluster + local) — ON HOLD

`apply_ica --rel-variant rel_branch` per corpus (candidates to non-legacy
`--out-name`s first); DoCA-recall + coded-events recall re-report; then the
production swap: promote the p02 rel artifact + branched fusion to the
production paths, update `IcaModel` defaults/`config` constants, reconcile
CLAUDE.md/roadmap/project-state, notify the team with the corrected
composed numbers.

## Artifact inventory the swap will promote

- `relevance/scratch_diag/relevance_tuned_p02_cpu.{weights.h5,config.json,calibration.json}`
  → production rel paths (rename deliberately; sidecars travel together).
- `../cca_doca/branched_fusion/ica_fusion.fusion.json` → production fusion.
- `relevance/tuned_backbone.job8823087.weights.h5` — the rel-branch donor
  (already permanent).
- CCA/US artifacts unchanged.
