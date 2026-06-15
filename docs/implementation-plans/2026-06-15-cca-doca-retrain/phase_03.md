# Phase 3: CCA retrain (features-mode) + spot-check

**Goal:** Train the CCA head on cached embeddings with FLPU/nnPU at the re-estimated π, frozen probe,
full diagnostics, and a config sidecar; run a small π sensitivity check; spot-check training health.

**Codebase verified:** 2026-06-15 (`run_cca_classification.py`, `cca_config.py`, `loss.py`).

---

## Acceptance Criteria Coverage
### cca-doca.AC3: Retrain
- **cca-doca.AC3.1:** Training writes `*.weights.h5` + `*.config.json` sidecar reloadable by structure.
- **cca-doca.AC3.2:** Diagnostics over the run show no distribution collapse (prediction std not ≈0)
  and FLPU `correction_triggered` behavior is sane.

---

## Approach (lean — refine at execution)
- **New script `src/run_cca_doca.py`** mirroring `run_cca_classification.py` but features-mode:
  - Config: `dataclasses.replace(DEFAULT_CCA_CONFIG, ...)` with `loss=FLPULossConfig(prior=<π from Phase 2>)`,
    and a recorded provenance note (cache suffix, π source). Keep `epochs`, ratio-batch, LR schedule,
    optimizer, diagnostics from the default unless a spot-check motivates change.
  - Data: load split frames (Phase 1) + gather feature arrays by `emb_row`; build train/val sets via
    `dataset_from_embeddings` (Phase 0) with ratio-batch weights from `run_config.ratio_batch`.
  - Model: `cca_head = ClassificationHead(hidden_dim, loss_fn=FLPULoss(prior=π, kiryo_clawback=...),
    metrics=make_cca_metrics()+make_distribution_metrics(diag), name="cca",
    expose_loss_components=diag.enable_loss_components)`; `build_feature_endpoint_model({"cca": cca_head},
    hidden_dim, diagnostics=diag)` + `build_feature_inference_model` (Pattern A).
  - Steps: derive `steps_per_epoch` from the *actual* positive count (not the hardcoded 18300);
    `with_resolved(steps_per_epoch)` for the LR schedule. Compile without loss/metrics (head owns them).
    Save weights to a new `config.CCA_DOCA_WEIGHTS` path + sidecar via `config_path_for_weights`.
  - Keep the token-mode `run_cca_classification.py` untouched (retained for future fine-tuning).
- **π sensitivity:** retrain at a few π values (e.g., 0.5π, π, 2π) — cheap on cached vectors — and
  compare validation metrics + prediction distribution. Record.

## Vigilance (the substitute for the code-review gate)
- Watch the diagnostics each run: FLPU `positive_risk`/`negative_risk`/`correction_triggered`,
  `BatchLabelBalanceTracker` positive fraction (should match ratio-batch weight), prediction
  distribution (`std` not ≈0, `frac_above_0.5` not pinned). Distribution collapse ⇒ stop and investigate.
- Dump top-scored positives/unlabeled (analogue of `cca_classifier/pos_top_*.csv`) and eyeball face validity.

## Tasks (to detail at execution)
1. Add `config.CCA_DOCA_WEIGHTS`; write `src/run_cca_doca.py` (features-mode training). **Verifies:** AC3.1.
2. Run training; inspect diagnostics + sidecar reload. **Verifies:** AC3.1, AC3.2.
3. π sensitivity sweep + top-scored dumps; record findings.

## Phase 3 done when
- Weights + sidecar written and structurally reloadable; diagnostics clean (no collapse); sensitivity
  + face-validity recorded; `uv run pytest` green; `ruff` clean.
