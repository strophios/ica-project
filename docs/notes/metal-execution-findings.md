# Metal-execution findings — execution-bound artifacts and corrected tuned-head numbers

*Created 2026-08-12, during the branched-encoder ladder stage 1
(`branched-encoder-strategy.md`). Record of the investigation that found the
production tuned-rel artifact to be tensorflow-metal-execution-bound, the
corrected correct-math numbers for the whole tuned-features head family, and
the deployment rules that follow. Extends the tensorflow-metal predict-bug
record started 2026-08-03/04 (`compare_scoring_paths.py`, commits `c911620`,
`9a9a9a4`) — the ±0.01 characterization there is true for healthy heads and
dangerously incomplete in general.*

## The finding

`relevance/relevance_tuned.weights.h5` (the July mixed-stack rel head, trained
locally on tensorflow-metal 2026-07-29 23:27) scores the 1,131-row gold eval at
**vs-ICA ROC 0.853 under metal-GPU execution and 0.386 under correct CPU/numpy
math** — same bytes, same features, same labels. The archived
`eval_heads_own_terms_tuned.json` numbers (0.836/0.853, 2026-07-29 23:40 local)
reproduce today to four decimals when the GPU is enabled; a pure-numpy forward
pass from the raw h5 bytes reproduces the CPU collapse exactly. Nothing
changed since July; the two measurements were computed by **two different
functions of the same weights**.

Mechanism: the head was *trained* on metal. The optimizer minimizes the loss
of whatever forward function is actually computed, so training fit weights to
metal's distorted math. The artifact is **execution-bound**: valid only on the
execution path it was trained against. On CUDA (cluster) or CPU it is garbage
— rank-inverted, not noisy.

Amplifier: the tuned-cache features drive these heads to extreme logit scales
(|logit| up to ~100). At that scale, float32 accumulation-order differences
between execution paths produce logit-value shifts of ~30 absolute. For
cleanly-trained heads the shift is roughly rank-preserving (ROC moves ≤0.003);
for the metal-trained artifact it was rank-destroying. Note the production
heads (logits O(1)) shift only ~±0.01 — which is why the early-August
characterization looked reassuring.

## Triangulation (all 2026-08-11/12, session record)

1. Graft-test control leg (production rel head, DAPT features, CPU) reproduces
   its recorded numbers → harness sound.
2. Cache features validated two ways (fresh local re-embed vs cluster cache,
   per-row cosine ≈ 1.0) → features sound.
3. Production eval harness (`eval_heads_own_terms.main`) with identical args:
   CPU 0.291/0.385 vs GPU 0.836/0.853 (archived: 0.8358758/0.8526220).
4. Pure-numpy forward from raw h5 bytes == CPU result exactly → the CPU
   number is what the bytes encode.
5. File birth/mtime 2026-07-29 23:27:41, archived eval 23:40:29, no
   modifications since; no relevant code changes since 07-30 (`git log` on the
   scoring path).

## Corrected numbers (correct math, CPU-forced, scratch retrains 2026-08-12)

Fresh probes retrained on the *same tuned caches* with the GPU hidden
(`../{relevance,cca_doca,us_filter}/scratch_diag/`; gold eval = the 1,131-row
ICA set, rank metrics, no calibration):

| head | production (frozen DAPT) | July metal-measured (tuned) | **correct math (tuned)** |
|---|---|---|---|
| rel own-terms / vs-ICA | 0.829 / 0.783 | 0.836 / 0.853 | **0.834–0.836 / 0.852–0.855** |
| rel diaspora @0.30 / @0.10 | 0.397 / 0.176 | 0.662 / 0.250 | **0.662 / 0.250** |
| CCA own-terms | 0.928 (control-verified) | 0.739 | **0.795** |
| US features test F1 | 0.97 | 0.938 | **0.951** |

- **The rel gain is real** — a cleanly-trained π=0.02 probe reproduces (and
  slightly exceeds) the recorded numbers under correct math. A π=0.05 CPU
  probe is also healthy (0.849 vs-ICA): the July artifact's collapse was
  metal-training-specific, **not** a prior artifact.
- **The negative-transfer verdict survives qualitatively, magnitudes
  corrected**: CCA −0.13 (not −0.19), US mild (−0.02 F1; the token-mode gold
  own-terms re-measurement is blocked by the separate US-head full-model load
  issue — `diag_us_head_load.py` thread). No decision in
  `encoder-unfreeze-strategy.md` or `branched-encoder-strategy.md` flips.
- **The mixed-stack composed number (0.797→0.820) is decomposed-verified but
  composed-pending**: it was computed with the metal-bound head + metal
  execution; the composed re-measurement happens at the fusion refit
  (productionization, roadmap §A1).

## Deployment rules (adopt now)

1. **Features-mode heads train and evaluate CPU-forced locally** (they train
   in minutes; metal buys nothing worth an execution-bound artifact), or on
   the cluster (CUDA). The scratch wrappers from this session show the shape;
   promote into the trainers as a default-on `--hide-gpu` or an env guard when
   next touched.
2. **Artifact acceptance check**: score every newly trained head once per
   execution path (CPU vs local GPU); rank divergence beyond ~0.01 ROC ⇒
   reject the artifact as execution-bound. Cheap, catches this whole class.
3. **Calibration does not transfer across execution paths** for large-logit
   heads: CPU-vs-GPU logit *values* differ by ~30 absolute even for healthy
   tuned-cache heads. Fit Platt calibration on the execution path used at
   apply time, and record that path in the calibration sidecar when next
   touched.
4. The July metal-bound artifacts (`relevance_tuned.*`, `cca_doca_tuned.*`,
   `us_classifier_full_tuned.*`, all 2026-07-29 23:2x) must not be deployed or
   re-scored as if portable. Superseding artifacts come from the
   productionization step; until then `scratch_diag/` holds the correct-math
   probes (config sidecars only, no calibration yet).

## Queued: the deeper metal investigation (operator-requested 2026-08-11)

Fixing or precisely characterizing this upstream is both a private and public
good. Scope when picked up: minimal reproduction (a single Dense matmul at
~100-logit scale, CPU vs metal, accumulation-order sensitivity vs a genuine
kernel bug); whether *training* under metal diverges from CPU training on the
same data/seed (it did here, catastrophically — but isolate optimizer-level
vs forward-level causes); tensorflow-metal / keras versions pinned in
`uv.lock` (unchanged since May); file upstream with the reproduction.
Related but distinct: the US-head full-model load failure
(`position_embedding expected 1 variables, received 0`) under the current
keras_hub preset cache — the `diag_us_head_load.py` thread, which also blocks
token-mode US gold evals locally.

## Incidental findings (this session)

- **Location-signals duplicate id**: `load_location_signals` returns one
  duplicated id (`nyt://article/2f64313c-24a8-591b-97e8-3a0328577e1c`,
  api_corpus 991-dup family), which fans out the training-table join; the
  post-07-30 id-uniqueness assert now refuses it (`create_relevance_data`).
  Fix belongs at the boundary: `unique(subset="id", keep="first",
  maintain_order=True)` inside `load_location_signals`, with a logged count.
  The scratch retrains monkeypatched exactly this.
- The archived `eval_rel_text.json` locally is the *collapsed* job8822390
  record; the winning job8823087 text-mode eval JSON was overwritten on the
  cluster (`eval_rel_text_artifact.py` has no `--out`). The correct-math
  reference numbers now live in `cca_doca/experiments/graft_test_v2.json`.

## Artifacts of record

- `cca_doca/experiments/graft_test.json` — v1 run (graft mechanics + the
  collapse discovery; its reference/graft metric legs are metal-artifact
  scores, superseded).
- `cca_doca/experiments/graft_test_v2.json` — stage-1 result of record
  (correct-math reference; graft == full-tuned within 4e-4).
- `../{relevance,cca_doca,us_filter}/scratch_diag/` — CPU-trained probes.
- `scripts/graft_test.py` — the reusable stage-1 harness.
