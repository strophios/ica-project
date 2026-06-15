# Phase 2: DEDPUL prior re-estimation

**Goal:** Re-estimate the positive class prior π for the new DoCA/API/US-restricted population, on
cached embeddings.

**Codebase verified:** 2026-06-15 (prior-estimation modules to be re-read at execution: read
`src/prior_estimation/lu_classifier.py`, `run_prior_estimate.py`, `dedpul_em.py`, `dedpul_utils.py`
before writing tasks).

---

## Acceptance Criteria Coverage
### cca-doca.AC2: Prior
- **cca-doca.AC2.1:** DEDPUL produces a finite π ∈ (0,1) for the new population, recorded with provenance.
- **cca-doca.AC2.2:** π is within a defensible band of the naive labeled positive rate (flagged, not
  silently accepted, if not).

---

## Approach (lean — refine at execution)
- **L/U classifier on cached embeddings.** Train a features-mode head (reuse Phase 0's
  `build_feature_endpoint_model` + `dataset_from_embeddings`) to separate labeled (DoCA positives)
  from unlabeled (US-restricted sample), mirroring `src/prior_estimation/lu_classifier.py`'s task but
  on cached vectors. This is fast (tiny head on 768-d inputs).
- **Convert + DEDPUL.** Feed L/U scores through the existing DEDPUL path (`run_prior_estimate.py` /
  `dedpul_em.py`), honoring the probability-of-unlabeled convention documented in the top-level
  CLAUDE.md (`sigmoid` + `1 - p`). Reuse `dedpul_utils`/`dedpul_em` unchanged.
- **Record π** to a small notes/JSON artifact with provenance (cache suffix, L/U run, kde settings),
  and compute the naive labeled rate `n_pos / (n_pos + n_unl_US)` for the AC2.2 sanity band.

## Vigilance
- Sanity-check π against the naive labeled rate and against the old CCA prior (0.02, different
  population). A π wildly off the labeled rate is a stop-and-investigate (separability artifact, KDE
  bandwidth, L/U leakage). Check `kde_mode`/bandwidth robustness as `run_prior_estimate.py` already supports.

## Tasks (to detail at execution)
1. Features-mode L/U training over cached embeddings → scores. **Verifies:** supports AC2.1.
2. DEDPUL EM over converted scores → π; record artifact + naive-rate comparison. **Verifies:** AC2.1, AC2.2.

## Phase 2 done when
- A recorded π ∈ (0,1) with provenance; naive-rate sanity check documented; `uv run pytest` green for
  any touched modules; `ruff` clean.
