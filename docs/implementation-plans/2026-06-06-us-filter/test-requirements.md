# US/not-US Pre-Filter — Test Requirements

This document maps every acceptance criterion in the US/not-US pre-filter design (`docs/design-plans/2026-06-06-us-filter.md`, criteria `us-filter.AC1.1` through `us-filter.AC7.3`) to its verification mechanism: either an **automated test** (with type, file path, and what it asserts) or **human/operator verification** (with justification and approach).

The mapping is rationalized against the decisions actually made in the eight implementation phases, not the design's idealized statements. The important rationalizations:

- **Phase 1 R resolver logic is tested with `testthat`**, not pytest, run via `Rscript r/tests/run_tests.R`. These are automated but live in a separate runner from the Python suite.
- **Several ACs are operator-gated** because they depend on artifacts that do not exist yet (the human-coded event-location gold set; the DoCA-matched pre-1986 set) or on full/cluster training runs. For these, the phases ship a **synthetic-data unit-test harness** (automatable now) plus a **real-data execution** that is an operator step. Both halves are recorded below.
- **Test types** used: `unit` (pytest, isolated function/class), `integration` (pytest, assembled stack / fake-backbone model / parquet round-trip), `R-testthat` (Rscript), `operational` (a script/build that must run to completion and emit a sane result — verified by running it, not by an assertion suite), and `human` (review / dialogic authorship / judgment that cannot be encoded).

---

## AC1 — Datelines and desk signals produce correct, confident labels

All AC1 resolver/policy logic is automated in the Phase 1 `testthat` suite. The label-build over the real corpus is operational (it produces the parquet that downstream phases consume).

| AC | Automated/Human | Test type | Test file / approach | What it verifies |
|---|---|---|---|---|
| AC1.1 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` | `VANCOUVER, Wash.` resolves `is_us=TRUE` (state qualifier → US). |
| AC1.2 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` | `LISBON, Portugal` resolves `is_us=FALSE` (country qualifier → not-US). |
| AC1.3 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` | Bare `CHICAGO —` → TRUE; bare `LONDON —` → FALSE (AP-30 / AP-46 standalone lists). |
| AC1.4 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` | `PARIS, Texas` → TRUE (state qualifier wins before bare lookup); bare `PARIS` → FALSE (AP-foreign). |
| AC1.5 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` | `PARIS, July 30 —` → FALSE (date field dropped → bare PARIS); `WASHINGTON, July 30 —` → TRUE. |
| AC1.6 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` | `VANCOUVER, Wash., June 1 —` → TRUE (state field detected despite trailing date). |
| AC1.7 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` (or `test-label-policy.R`) | No dateline + Foreign desk → FALSE / `label_source="heuristic"`; National/Metro desk → TRUE / `"heuristic"` (`classify_label` + `desk_section_signal`). |
| AC1.8 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` (or `test-label-policy.R`) | Dateline-US vs desk-Foreign → `us_label=NA`, `label_source="conflict"`. |
| AC1.9 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` (or `test-label-policy.R`) | Unresolved bare city with no confident desk signal → `us_label=NA`. |
| AC1.10 | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` | Collision-trap invariant: bare tokens present in the long lists (`PORTUGAL` in `countries.csv`, `BAYONNE` in `us-area-code-cities`) but absent from AP-30/AP-46 resolve `NA` — proving bare tokens never hit the long lists. |
| AC1 (label build) | Operator/Operational | operational | `Rscript r/dateline/build_labels.R` | Emits `<US_FILTER_DIR>/ldc_labeled.parquet` over LDC 1987–2007 with a sane `label_source` breakdown (dateline-dominant; conflict/null minorities); Python re-reads it (`pl.read_parquet`). Verified by running the build, not an assertion suite — operates on the real ~1.4M-row corpus. |

---

## AC2 — The dateline cannot leak into model input

The detector and its consumer-side relationship are unit-tested in Python; the train-entry runtime assertion is exercised operationally in the Phase 4 short-run. AC2.3's R-side strip-correctness is co-covered in the Phase 1 testthat suite.

| AC | Automated/Human | Test type | Test file / approach | What it verifies |
|---|---|---|---|---|
| AC2.1 | Automated | integration | `tests/test_dateline_guard.py` | On a fixture parquet with correctly-stripped `stripped_text`, `data_from_parquet(..., lead_column="stripped_text")` assembles the input column and `assert_no_dateline_residue(result["stripped_text"])` does not raise. |
| AC2.2 (pytest) | Automated | unit | `tests/test_dateline_guard.py` | A seeded leak row (dateline left in `stripped_text`) makes `assert_no_dateline_residue` raise `ValueError` and `has_dateline_prefix` return `True`. |
| AC2.2 (runtime) | Operator/Operational | operational | `scripts/us_short_run.py` (train-entry `assert_no_dateline_residue` over `stripped_text`) | The guard fires as a runtime assertion at train entry; verified when the short-run executes end-to-end on real labeled data. |
| AC2.3 (Python half) | Automated | unit | `tests/test_dateline_guard.py` | On datelined fixture rows, `has_dateline_prefix(raw_text)` is `True` while `has_dateline_prefix(stripped_text)` is `False`, and `raw_text` ends with `stripped_text` modulo whitespace (removed prefix == dateline span). |
| AC2.3 (R half) | Automated | R-testthat | `r/tests/testthat/test-resolve-dateline.R` | `strip_dateline` removes exactly the dateline span: `WASHINGTON, July 30 — …` → post-delimiter remainder; re-extraction on stripped text finds no dateline; credit-line prefix is also consumed. |

---

## AC3 — Model trains as supervised PN and its transfer is measured

In-distribution config/metrics/split/guard are unit-tested. AC3.1's F1-beats-baseline quality bar requires a **full training run** (operator). AC3.3 (transfer + escalation decision) is harnessed with synthetic data but its real execution needs the hand-labeled gold set (operator).

| AC | Automated/Human | Test type | Test file / approach | What it verifies |
|---|---|---|---|---|
| AC3.1 (path) | Automated/Operational | operational | `scripts/us_short_run.py` (capped 1-epoch / 200-step) | The training path runs end-to-end: model trains, P/R/F1 + majority-class baseline line prints, diagnostics fire. Verifies the path, not the quality bar. |
| AC3.1 (F1 > baseline) | Operator | operational | Full uncapped `src.run_us_classification.main()` (operator-invoked, likely cluster) | The quality bar — in-distribution test F1 exceeds the majority-class baseline — needs a full training run on the full confidently-labeled set. The script computes and reports both numbers so the bar is checkable; the bar itself cannot be asserted in CI (depends on real training to convergence). |
| AC3.2 | Automated | integration | `tests/test_run_us_classification.py` | The assembled US head's metric set includes the prediction-distribution metrics by exact head-prefixed names (`us_pred_dist/mean`, `us_pred_dist/std`, `us_pred_dist/frac_above_0.5`), so they populate for train and val. |
| AC3.3 (harness) | Automated | integration | `tests/test_slice_eval.py`, `tests/test_escalation.py` | `evaluate_slice` computes both-class P/R/F1 from known synthetic probs/labels; `escalation_decision` flips correctly on the margin; `top_n_group_fn` groups encoder layers correctly. |
| AC3.3 (decision) | Operator | operational | `src/validation/slice_eval.py` + `src/validation/escalation.py` over the coded gold set + trained model | Real pre-1986 transfer performance computed, compared to the in-distribution baseline, and the escalation decision recorded. Gated on the human-coded event-location gold set (does not exist yet) and a trained model. |
| AC3.4 | Automated | unit | `tests/test_us_config.py`, `tests/test_run_us_classification.py` | No FLPU/prior/nnPU in the path: walking `dataclasses.fields` finds no `prior` field anywhere in the `UsRunConfig` tree, and the assembled head's `loss_fn` is `BinaryCrossentropy(from_logits=True)`. |
| AC3.5 | Automated | unit | `tests/test_us_data_splits.py`, `tests/test_us_config.py` | Same seed yields identical split membership (by `id` sets) across two calls of `create_us_filter_data`; config sidecar round-trips (including a populated `resolved`) to an equal config. |

---

## AC4 — Scores are calibrated and the calibrator is reusable

Fully automatable on synthetic `(logits, labels)` arrays — no trained model needed at build time. AC4.2's application *to the real pre-1986 slice* is operator-gated (covered under AC6.3); here it is exercised on synthetic/held-out arrays.

| AC | Automated/Human | Test type | Test file / approach | What it verifies |
|---|---|---|---|---|
| AC4.1 | Automated | unit (+ hypothesis) | `tests/test_calibration_platt.py` | `PlattCalibrator.fit` on a natural-balance synthetic set; `transform` outputs all ∈ [0,1] and are monotonic non-decreasing in the logit. |
| AC4.2 | Automated | unit | `tests/test_calibration_report.py` | `calibration_report` returns `{ece, brier, reliability}` with correct structure; on a perfectly-calibrated synthetic set ECE ≈ 0 and Brier is low. |
| AC4.3 | Automated | unit | `tests/test_calibration_sidecar.py` | `save_calibration` → `.calibration.json` → `load_calibration` reproduces an fp-identical `transform` over a logit grid; payload contains `{method, A, B, fit_population, n}`. |
| AC4.4 | Automated | unit | `tests/test_calibration_platt.py` | `fit` requires keyword-only `fit_population`, which is stored on the calibrator; the rebalanced-batch-disallowed rule is documented in the docstring. |
| AC4.5 | Automated | unit | `tests/test_calibration_report.py` | On a constructed miscalibrated set, post-Platt ECE ≤ raw `sigmoid(logits)` ECE on a held-out eval split. |

---

## AC5 — The artifact and output columns are produced and reusable

The apply path and reload reproducibility are unit-tested with a fake-backbone model + stub calibrator. The full real-corpus apply is an operator step.

| AC | Automated/Human | Test type | Test file / approach | What it verifies |
|---|---|---|---|---|
| AC5.1 (path) | Automated | integration | `tests/test_apply_us_filter.py` | On a synthetic API parquet, the apply produces `id, us_score, us`; `us_score ∈ [0,1]`; `us == (us_score >= threshold)`; row count and `id` preserved. |
| AC5.1 (real corpus) | Operator/Operational | operational | `uv run python -m src.apply_us_filter` over the real API corpus | `us_score` + `us` materialized as `us_filter/api_us_scores/{year}.parquet` over 1960–1995. Needs the settled, calibrated model; run by the operator. |
| AC5.2 | Automated | integration | `tests/test_artifact_reload.py` | Save the artifact triple (`*.weights.h5` + `*.config.json` + `*.calibration.json`) to `tmp_path`; `reload_and_score` reproduces pre-save calibrated scores within fp tolerance. |
| AC5.3 | Automated | unit | `tests/test_artifact_reload.py` | The documented default `us` threshold is 0.5, and the recorded recipe references `doca_recall` (the CCA-consumer recall-targeted threshold recipe). |
| AC5.4 | Automated | integration | `tests/test_apply_us_filter.py` | Dateline-less API rows assemble via `data_from_parquet(..., lead_column="lead_paragraph")` — the same concatenation as training, with no stripping/guard — and score consistently. |

---

## AC6 — Validation instruments produce their reports

The "free" 1987–1995 audit is fully automatable on synthetic matched pairs and runs now. The slice-dependent instruments (AC6.3 proxy gap, AC6.4 DoCA recall) ship with synthetic-data unit-test harnesses but their **real reports** need the human-coded gold set / DoCA-matched data (operator).

| AC | Automated/Human | Test type | Test file / approach | What it verifies |
|---|---|---|---|---|
| AC6.1 (metric) | Automated | unit | `tests/test_free_audit.py` | `heuristic_error_rate` returns the exact disagreement rate of the R `us_assign` heuristic vs dateline labels on synthetic matched pairs (over rows where both are non-null). |
| AC6.1 (real audit) | Operator/Operational | operational | `Rscript r/audit/api_ldc_join.R` → `src/validation/free_audit.py` over real 1987–1995 joinable articles | The audit join produces `api_ldc_matched.parquet` and the heuristic error rate is reported on real data. Runs now (no gold set needed) but is an operational build over the real corpora. |
| AC6.2 | Automated | unit | `tests/test_free_audit.py` | `lead_similarity` (difflib ratio over normalized text) returns the expected ordering for identical-vs-divergent leads — the `stripped_text ≈ API lead` similarity figure. (Real figure emitted by the operational audit above.) |
| AC6.3 (harness) | Automated | integration | `tests/test_slice_eval.py` | `evaluate_slice` computes both-class metrics and `proxy_gap` computes dateline-vs-event-location agreement on constructed synthetic mismatches. |
| AC6.3 (real report) | Operator | operational | `src/validation/slice_eval.py` over the coded pre-1986 gold set | Both-class metrics and the real dateline-vs-event-location proxy gap. Gated on the human-coded event-location gold set (does not exist yet). |
| AC6.4 (harness) | Automated | unit | `tests/test_doca_recall.py` | `doca_recall` returns exact recall at a threshold on synthetic scored DoCA rows; the topic-skew caveat string is present in the report. |
| AC6.4 (real report) | Operator | operational | `src/validation/doca_recall.py` over the DoCA-matched scored set | The real DoCA-matched recall diagnostic. Gated on the DoCA-matched data + a calibrated model (`doca_id`-bearing `scored_df` assembled per the operator-chosen producer path). |
| AC6.5 | Automated | unit | `tests/test_free_audit.py` | The free-audit report string includes the biased-by-joinability caveat (error rates conditional on joinability). |

---

## AC7 — Cross-cutting: scope, reproducibility, documentation

Scope is enforced structurally (review-verifiable). The calibration notes are a dialogic, human-authored doc. The shakedown is operational locally and operator-run on the cluster.

| AC | Automated/Human | Test type | Test file / approach | What it verifies |
|---|---|---|---|---|
| AC7.1 | Human | human (review) | Plan/code review against the design's "Out of scope" | No task productionizes the conservative R heuristic or performs pre-1960/post-1995 application+validation. A negative-scope criterion: verified by reviewing that nothing implements the excluded work, not by an executable test. |
| AC7.2 | Human | human (dialogic authorship + review) | `docs/notes/calibration-notes.md`, authored live with the user | The notes exist, were developed dialogically (explicitly NOT auto-generated), cover Platt's `A·logit+B`, what ECE measures, and the load-bearing imbalance/distribution interaction, and are user-reviewed. Cannot be automated — authorship and review are the deliverable. |
| AC7.3 (local) | Automated/Operational | operational | `uv run python scripts/us_short_run.py` (local float32) | The local short-run shakedown completes end-to-end before any full run: paths/env work, metrics + diagnostics fire for train and val. |
| AC7.3 (cluster) | Operator/Operational | operational | Operator-run capped `main()` on Explorer under `mixed_float16` | The cluster shakedown completes: `mixed_float16` / `LossScaleOptimizer` behave, the overflow proxy is active, paths/hardware work. Requires cluster access — operator-run. |

---

## Operator-gated / human-verification summary

Every criterion below is **not automatable in CI**. For each, the reason and the verification approach:

| AC | Why not CI-automatable | Verification approach |
|---|---|---|
| **AC1 (label build)** | Operates on the real ~1.4M-row LDC corpus; success is a "sane `label_source` breakdown," a judgment over real data. | Run `Rscript r/dateline/build_labels.R`; inspect the printed breakdown + Python re-read. (Resolver *logic* is fully CI-automated via testthat.) |
| **AC2.2 (runtime assertion)** | The train-entry guard fires only when training is actually invoked on real labeled data. | Observed during the `scripts/us_short_run.py` execution. (The pytest half of AC2.2 is CI-automated.) |
| **AC3.1 (F1 > baseline)** | The quality bar requires a full training run to convergence on the full labeled set; depends on real data + compute. | Operator runs uncapped `main()` (likely cluster); the script reports F1 and the majority baseline for the operator to check. |
| **AC3.3 (escalation decision)** | Requires the human-coded event-location gold set (does not exist yet) and a trained model. | Synthetic harness is CI-automated; operator runs `slice_eval.py` + `escalation.py` on the coded slice and records the decision. |
| **AC5.1 (real corpus apply)** | Materializing scores over the full 1960–1995 API corpus needs the settled, calibrated model. | Operator runs `python -m src.apply_us_filter`. (Apply path is CI-automated on synthetic data.) |
| **AC6.1 / AC6.2 (real audit)** | The audit join + metrics run over the real LDC↔API corpora (operational build); runs now but not in CI. | Operator runs `r/audit/api_ldc_join.R` then `free_audit.py`. (Metrics are CI-automated on synthetic pairs.) |
| **AC6.3 (real proxy gap)** | Requires the human-coded event-location gold set. | Operator runs `slice_eval.py` on the coded slice. (Harness CI-automated.) |
| **AC6.4 (real DoCA recall)** | Requires the DoCA-matched set + a calibrated model to assemble the `doca_id`-bearing scored frame. | Operator runs `doca_recall.py` on the DoCA-matched scored set. (Harness CI-automated.) |
| **AC7.1 (scope)** | A negative-scope assertion — verifying that excluded work was *not* done is a review judgment, not an executable test. | Plan/code review against the design's "Out of scope" section. |
| **AC7.2 (calibration notes)** | A dialogic, human-authored, user-reviewed doc; auto-generation is explicitly prohibited by the design and Phase 8. | Live drafting session with the user + user review of `docs/notes/calibration-notes.md`. |
| **AC7.3 (cluster shakedown)** | Needs Explorer cluster access + `mixed_float16` hardware. | Operator runs the capped `main()` on the cluster. (The local-float32 shakedown is operational and runnable in this environment.) |

**Fully CI-automated (no operator dependency):** AC1.1–AC1.10 (testthat), AC2.1, AC2.2-pytest, AC2.3 (both halves), AC3.2, AC3.4, AC3.5, AC4.1–AC4.5, AC5.2, AC5.3, AC5.4, AC6.1-metric, AC6.2, AC6.5, and the synthetic harnesses for AC3.3/AC6.3/AC6.4. AC3.1-path, AC5.1-path, and AC7.3-local are automated as **operational short-run** checks (run the capped script; assert it completes), distinct from the operator-gated full/cluster runs.

No acceptance criterion is left unmapped: every `us-filter.ACx.y` appears above with either an automated test or a documented human/operator verification.
