# Phase 5 (stretch): Full-corpus discovered events

**Goal:** Apply the US → CCA pipeline over the full 1960–1995 corpus to surface a sample of
*discovered* CCA events for the demo.

**Gated on:** the overnight full-corpus embedding cache (Phase 0, Task 3) completing.

**Codebase verified:** 2026-06-15.

---

## Acceptance Criteria Coverage
### cca-doca.AC5: Stretch
- **cca-doca.AC5.1:** A per-year scored output over the full corpus exists with US-gated CCA scores;
  a curated discovered-events sample is produced.

---

## Approach (lean — refine at execution)
- On the full cache: `us = us_logit >= threshold`; over US-passing rows, apply the CCA head
  (`apply_cca_model`, Phase 4) → `cca_score`. Write per-year parquet `(id, year, us_score/us_logit,
  cca_score)` under a `config.CCA_DOCA_SCORES_DIR` (mirror `apply_us_filter.py`'s per-year output shape).
- Curate a **discovered-events** sample: high-`cca_score`, US-passing articles **not** in the DoCA
  positive set (i.e., events the classifier surfaces that DoCA didn't code / are outside its window) —
  the compelling demo artifact. Join headlines/ledes back from `api_corpus` for readability.

## Vigilance
- Discovered events should look like plausible collective action on inspection; a high false-positive
  rate among top scores is a signal to revisit threshold/π. Note the DoCA topic-skew caveat when
  framing recall claims.

## Tasks (to detail at execution)
1. Full-cache US→CCA scoring → per-year parquet. **Verifies:** AC5.1.
2. Curate + render a discovered-events sample (top CCA, US, non-DoCA) with headlines/ledes.

## Phase 5 done when
- Per-year scored output exists; a curated discovered-events sample is rendered for the meeting.
