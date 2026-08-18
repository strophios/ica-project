# Stage 4 — joint CCA+rel fine-tune: design

*Created 2026-08-18. Implementation design for the branched-encoder ladder's
stage 4 (`docs/notes/branched-encoder-strategy.md` execution record, revised
2026-08-18 after operator review: depth swept N ∈ {1,2}, selection on the
composed proxy). This doc specifies the build; the strategy note holds the
why and the decision rules.*

## Shape

One text-mode training run per sweep cell: a shared DAPT encoder with top-N
unfreeze (flat multipliers, multiplier freezing — deploy-side exactness comes
from grafting, per the stage-3 deploy rule) and TWO `ClassificationHead`s
(`cca`, `rel`), each with its own FLPU loss (own prior π=0.02, η=0), weighted
λ·L_rel + (1−λ)·L_cca. Grid: N ∈ {1,2} × λ ∈ {0.25, 0.5, 0.75} × seeds
{200, 201} = 12 cluster jobs, each self-scoring into a compact JSON.

## Components

### 1. `HeadConfig.loss_weight` + `ClassificationHead(loss_weight=...)`

The λ mechanism. `HeadConfig` gains `loss_weight: float = 1.0` (validated
finite and > 0; sidecar back-compat default 1.0 = the historical behavior).
`ClassificationHead` gains a keyword-only `loss_weight=1.0` applied at loss
registration: `add_loss(loss_weight * loss)` (and the scaled value is what
`expose_loss_components` reports as the head's loss share; the components
dict stays unscaled — it diagnoses the FLPU internals, not the mixing).
Byte-identical at 1.0. This is the minimal answer to the pinned multi-head
loss-composition question (`pinned-questions.md` §1 "Multi-head lookahead"):
plain scalarization, per Kurin/Xin 2022 — no GradNorm/PCGrad without a
diagnosed gradient pathology (the per-group grad-norm trackers are the
instrument).

### 2. `src/build_joint_text_table.py` → `relevance/joint_text_table.parquet`

Union-population text table. The two heads' populations are overlapping but
distinct samples (CCA: `train250k` meta; rel: `relevance_train` meta), so:

- Rows: union of the two cache metas by `id` (dedupe; both are API-corpus id
  space). Carry `us_logit` from either (prefer the rel cache's on conflict —
  it is the fused-gate-calibrated channel its threshold was tuned on).
- `cca_label`: id ∈ DoCA positives (`config.CCA_DOCA_POSITIVES`), kept
  regardless of `us` (DoCA events are US by construction).
- `rel_label` inputs: id ∈ `relevance/candidates.parquet`, then
  US-restricted exactly as `build_relevance_text_table` does (positives must
  also pass the fused gate); `reliable_neg` from
  `relevance/reliable_negatives.parquet` (inert at η=0, carried for parity).
- Fused US gate applied once to the union (ML gate ∧ not-clearly-foreign,
  `apply_fused_us_gate`), with the location-signals join **deduped by id**
  (the known one-dup fan-out; fix belongs in `load_location_signals`).
- Clean-ICA holdout dropped at build time + re-verified at train time
  (`assert_holdout_excluded` — this table is the contract's fifth consumer).
- Text attached from `api_corpus` by id (reuse
  `build_relevance_text_table.read_api_text` / `assemble_headline_with_lead`).
- **Prior note**: the union background is larger than either head's own, so
  the per-head effective π shifts modestly. π is an operating-point knob,
  not a quality knob (the 2026-06 sweep), and selection is rank-based —
  record, don't re-estimate.

### 3. `src/run_joint_text.py`

Modeled on `run_relevance_text.py` (importable `main`, same escalation/
hard-freeze/seed knobs, same artifact guard). Differences:

- Two heads from a two-head `RunConfig` (`heads=(cca_head, rel_head)`,
  λ via `loss_weight`: rel=λ, cca=1−λ). `build_endpoint_model` already
  supports multi-head (Pattern A); preprocessor emits both target keys
  (`label_keys={"cca_targets": "cca_label", "rel_targets": "rel_label"}`).
- Ratio-Batch, three streams over the joint table: cca-positives 0.1,
  rel-positives 0.1, unlabeled 0.8 (reliable-neg stream omitted at η=0;
  `--neg-weight` reserved). Streams overlap where a row is positive for
  both — deliberate oversampling, same family as Ratio-Batch itself. Every
  batch carries both labels; both losses see every row (PU semantics: not
  head-X-positive ⇒ head-X-unlabeled).
- CLI adds `--lam` (rel-side λ, required) and `--unfreeze-top-n` as before.

### 4. `scripts/eval_joint_sweep_point.py`

Per-cell scorer (CPU-forced end to end — calibration is NOT rank-invariant
for the composed product, so both the Platt-fit inputs and the gold scoring
stay on one execution path):

1. Score the artifact's two heads text-mode over (a) the table's val split
   at natural balance (~minutes on CPU) and (b) the 1,131-row gold eval.
2. Fit Platt per head on the val logits (val labels: own-terms per head).
3. Composed proxy = calibrated `p_cca · p_rel` on gold → **ROC vs
   `ica_event` (primary), diaspora recall @0.30/@0.10 (secondary)**;
   per-head own-terms ROC as guardrails; rel-solo vs-ICA as diagnostic.
4. JSON out with sweep params (N, λ, seed), all metrics, calibration A/B.

### 5. `scripts/joint_sweep.sbatch`

12-job array (idx → N = idx/6+1; λ = {0.25,0.5,0.75}[(idx%6)/2]; seed =
200 + idx%2), same conventions as `rel_depth_sweep.sbatch` (8h, skip-if-JSON,
pipefail-safe, train + self-score).

### 6. Branched baseline (local, once)

The bar the joint winner must clear: composed proxy of the **branched
stack** — production CCA on DAPT CLS × July-rel probe on tuned CLS, each
Platt-calibrated on its own natural-balance val stream (the tuned probe
needs a calibration fit — `calibrate_relevance.py` tuned knobs), product,
ROC vs `ica_event` + diaspora on gold. Features-mode, minutes.

## Decision rule (pre-registered, from the strategy note)

Joint wins ⇒ single-encoder swap, only if, at some cell: composed proxy ≥
branched baseline within noise AND CCA own-terms holds (~0.93 reference,
guardrail ≥ 0.91) AND rel own-terms holds (~0.83) AND (winner only) US
features-retrained on the joint cache survives (own-terms near 0.925; the
rel-first passenger outcome 0.830 is a fail). Anything less ⇒ branched is
the production architecture; joint retires with a clean negative result.
Tie handling: text-mode per-draw noise is ~±0.01 (stage-3 replicates) —
prefer branched on ties (simpler apply, no US risk).

## Build findings (2026-08-18, components 1/2 + fit_fusion landed)

- **`train250k` ⊆ `relevance_train`** (cca-only = 0 in the union) — the joint
  population IS the rel population (264,887 rows post-holdout). The
  us_logit scale-mismatch concern for cca-only rows is moot in practice.
- Joint-table positives: rel 6,754 (matches the stage-3 table); **CCA 15,269
  vs production's ~13.7k** — the rel population contains more DoCA ids than
  `train250k`, so joint-CCA trains on a slightly larger positive set than
  production CCA. Population note for the guardrail comparison (own-terms is
  measured on gold either way), not a bug.
- **Second physical api_corpus duplicate instance** (the known 991-family):
  one id duplicated across `1901.parquet`/`1968.parquet` fanned out
  `read_api_text`'s cross-file join; the joint builder carries a defensive
  dedupe + test. `build_relevance_text_table.py` has no such guard — if the
  rel text table is ever REBUILT against the current expanded corpus it needs
  the same fix (roadmap small-gaps).
- ~34% of joint-table rows are lead-empty — the known lead-free eras
  (1960–63, 65–69, 1980), same property the stage-3 table trained through.

## Branched baseline (computed 2026-08-18, CPU-forced;
`cca_doca/experiments/branched_baseline_proxy.json`)

Production CCA (DAPT CLS, its production calibration) × the CPU-trained p02
rel probe (tuned CLS, fresh calibration sidecar fit on
`relevance_train_tuned` natural balance — ECE 0.328→0.007): **composed proxy
ROC vs `ica_event` = 0.8064**, diaspora@0.10/@0.30 = 0.265/0.456; guardrails
CCA own 0.9285 / rel own 0.8341; rel-solo vs-ICA 0.8524. **This is the bar
the joint winner's composed proxy must clear.** Two notes: (1) on this proxy
the calibrated product *subtracts* from tuned-rel-solo (0.806 < 0.852) —
production CCA is a weak solo ICA ranker; complementarity is exactly the
axis stage 4 tests. (2) The proxy ≠ the production composed score (no US
gate, no product-vs-LR fusion selection, no composed Platt — those are
ROC-irrelevant or deferred to the winner's full fusion refit), so the
recorded mixed-stack 0.820 is not comparable; 0.8064 on the identical proxy
is.

## Build order

1. `loss_weight` knob (isolated; heads.py + cca_config.py + tests).
2. Joint table builder (isolated; + tests).
3. Trainer + scorer + sbatch (depends on 1+2).
4. Branched baseline computation (local; depends on `fit_fusion`
   parameterization only for the *final* production composition, not for
   the proxy — runs as soon as the tuned probe has a calibration sidecar).
