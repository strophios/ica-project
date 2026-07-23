# ICA Project: Transformer-Based Classification for Historical Protest Events

A machine learning system for identifying immigrant collective action (ICA) events in 150+ years of *New York Times* coverage, using positive-unlabeled (PU) learning with a RoBERTa backbone.

*For the authoritative current state, see `docs/notes/project-state-and-data-map.md` (data/artifact map + model state) and `docs/notes/roadmap.md` (what's next). Technical contracts live in `CLAUDE.md`.*

## The Research Problem

The goal is to expand protest event datasets (like the [Dynamics of Collective Action](https://web.stanford.edu/group/collectiveaction/cgi-bin/drupal/)) across 150+ years of *New York Times* articles, finding the tiny fraction of articles reporting immigrant collective action among millions total. Three challenges make this harder than standard text classification:

1. **Positive-Unlabeled data**: We have labeled positives (DoCA-matched articles) but zero confirmed negatives. An unlabeled article may still describe a relevant event.
2. **Severe class imbalance**: ICA articles are a vanishingly small minority of the corpus.
3. **Temporal linguistic drift**: The same phenomenon (a protest) is described in different language in 1870 vs. 2020.

## Approach

The system is a **multi-head classifier over a shared frozen domain-adapted RoBERTa encoder**. Headline + lede text is embedded once; three independently-trained, separately-calibrated heads score it:

- **US gate** — is this a US event? (a hard pre-filter, not a ranking signal)
- **CCA** — is this collective/contentious action? (trained on DoCA-matched articles)
- **Relevance** — is this immigrant-relevant?

A row is scored ICA by gating on US, then combining the calibrated CCA and relevance probabilities (the product beat a learned logistic combiner on a held-out comparison), with a final calibration step. One composed `ica_score` in [0,1] comes out the back. This decomposition lets each head learn from its own (larger) labeled dataset, making the sparse ICA labeling problem tractable.

The pipeline behind the heads:

- **Phase 1 — Domain-Adaptive Pre-Training (DAPT).** Further pre-training of RoBERTa on ~1.16M NYT headline + lede pairs to adapt its representations to the NYT domain (Gururangan et al. 2020). *(Done.)*
- **Phase 2 — Class prior estimation.** A labeled-vs-unlabeled classifier feeds the DEDPUL EM algorithm (Ivanov 2020). The estimate settled at **π ≈ 0.02** for collective action (re-estimated on the DoCA-matched population and swept — an operating-point knob, not a quality knob). *(Done.)*
- **Phase 3 — Head training.** Each head trains **features-mode on cached CLS embeddings** (so training takes minutes, not GPU-hours). CCA and relevance use FLPULoss — focal loss (Lin et al. 2020) + non-negative PU learning (Kiryo et al. 2017), parameterized by the Phase 2 prior; the US head is plain supervised BCE. All three are Platt-calibrated, then combined by an empirically-chosen fusion into the composed ICA score. *(Done — assembled, calibrated, and applied to 1960–2007.)*

### Current state

The assembled multi-head model exists and has been applied end-to-end: it produces ranked ICA candidates for the NYT API corpus (1960–1995) and the LDC corpus (1996–2007). On a hand-coded eval set (1,131 articles, 214 ICA positives, boundary-enriched) the composed score separates ICA from non-ICA at **ROC-AUC 0.80**. Applied to ~625k out-of-period 1996–2007 articles, it surfaces a few hundred to a few thousand high-quality ranked candidates with strong face validity — a tractable haystack reduction for human coding. The CCA head, evaluated against DoCA on a leakage-held-out gold set, reaches reweighted precision ≈ 0.82 at DoCA recall ≈ 0.35 at its chosen operating point (vs. 0.19 for a protest-keyword lexicon). The known ceiling is the US head's diaspora-recall gap (dateline labels encode filing location, not event location). Numbers and caveats: `ml_memo/ica_model_state_2026-06.md`.

### Planned extensions

See `docs/notes/roadmap.md` for the live list. The near-term levers:

- **US-head retrain** on event-location labels (DoCA + section signals) to fix the diaspora-recall ceiling.
- **Encoder unfreezing + per-layer learning rates** (the machinery exists) to lift head discrimination.
- **Temporal adaptation**: time tokens (e.g., `<1900>`) folded into DAPT to handle 150+ years of linguistic drift.
- **Virtual Adversarial Training (ALUM)** for improved generalization under label scarcity.
- **Backward expansion** to pre-1960 NYT, validated against the team's hand-coded historical events.

## Repository Structure

The repo holds code; all large data and trained artifacts live outside it (see `docs/notes/project-state-and-data-map.md`).

```
ica_project/
├── src/
│   ├── dapt.py                       # Phase 1: domain-adaptive pre-training
│   ├── embed_corpus.py               # Cache frozen-DAPT CLS embeddings (the one GPU step)
│   ├── run_cca_doca_prior.py         # Phase 2: DEDPUL prior estimate (DoCA population)
│   ├── build_cca_doca_table.py       # Assemble the CCA training table
│   ├── build_relevance_table.py      # Assemble the relevance training table
│   ├── run_cca_doca.py               # Phase 3: CCA head (features-mode FLPU)
│   ├── run_relevance.py              # Phase 3: relevance head (features-mode FLPU)
│   ├── run_us_features.py            # Phase 3: US head (features-mode BCE)
│   ├── calibrate_{cca,relevance,us_filter}.py   # Platt calibration per head
│   ├── fit_fusion.py                 # Fit the CCA×rel fusion
│   ├── assemble_ica.py               # IcaModel: the assembled multi-head classifier
│   ├── apply_ica.py                  # Apply IcaModel → ranked ICA candidates
│   ├── run_cca_classification.py     # Older text-mode CCA path (reference/tests)
│   ├── eval_cca_classifier.py        # Text-mode eval (reference)
│   ├── model_setup/                  # backbone, heads, assembly, LayerLRModel, DAPT setup
│   ├── loss_functions/loss.py        # FLPULoss (focal + non-negative PU learning)
│   ├── prior_estimation/             # DEDPUL EM + L/U classifier
│   ├── fusion/                       # Combiner + sidecar for the ICA score
│   ├── calibration/                  # Platt scaling (functional core + sidecar I/O)
│   ├── validation/                   # Gold-set schema, eval, DoCA recall, artifact checks
│   ├── diagnostics/                  # Training-run observability (trackers, factory)
│   ├── preproc/                      # Preprocessor, dateline guard, US-location fusion
│   └── data_setup/data.py            # Data loading + tf.data pipeline
├── r/                                # R dateline-labeling pipeline + API ingest (see r/CLAUDE.md)
├── scripts/                          # Runnable experiments, short runs, cluster templates
├── tests/                            # pytest suite (942 tests)
├── docs/notes/                       # Project state, roadmap, design + handoff notes
├── ml_memo/                          # Project memo (ml_memo.qmd/.pdf) + model-state snapshot
├── main.py                           # Entry point
├── CLAUDE.md                         # Detailed technical documentation
├── pyproject.toml                    # Dependencies (Python 3.12, uv)
└── README.md                         # This file
```

## Key Design Decisions

- **Models output logits** (no final activation); all losses use `from_logits=True`.
- **Features-mode training**: the frozen encoder is run over each corpus once and its CLS embeddings cached, so every head trains in minutes.
- **Mixed precision** (`mixed_float16`) on the HPC cluster (CUDA); `float32` locally (macOS MPS mixed-precision support is patchy).
- **Re-balanced batches** via weighted sampling to guarantee labeled positive signal in every batch; calibration is fit on natural-balance data so it maps to the real prior.
- **Headline + lede as input** (not full articles) — captures the signal cheaply and enables expansion to the full NYT archive (1851–present) via the free NYT Archive API.
- **The US head is a gate, not a fusion feature** — ICA is not monotone in US-ness.

## Data Sources

Not included in this repository. Training data is constructed from:

- **Dynamics of Collective Action** (DoCA, 1960–1995): ~23.6k coded protest events. DoCA-matched NYT articles are the CCA head's positives.
- **NYT Archive API corpus** (1960–1995, ~3.7M articles) and the **NYT Annotated Corpus** (LDC, 1987–2007, ~1.16M articles with indexer descriptor tags).

See `docs/notes/project-state-and-data-map.md` for the full data lineage and out-of-repo layout, and the ML memo for labeling strategy.

## Setup

Requires Python 3.12. Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Run commands with `uv run <command>` (e.g. `uv run pytest`) from the project root — do not activate the venv; `uv run` resolves and syncs the environment each time.

## Documentation

The `ml_memo/` directory contains the project memo (`ml_memo.qmd`/`.pdf`, ~27 references), written to make the project legible to non-technical collaborators — the research problem, methodological choices, alternatives considered, and implementation plans. `ml_memo/ica_model_state_2026-06.md` is the current model-state snapshot.

## References

- Ji et al. (2023). PU learning + ALUM for text classification
- Kiryo et al. (2017). Non-negative PU learning
- Ivanov (2020). DEDPUL: Density-based prior estimation
- Lin et al. (2020). Focal loss
- Gururangan et al. (2020). Domain-adaptive pre-training
- Hanna (2017). ML for protest event identification
- Chollet (2025). *Deep Learning with Python*
- Voss, Lauterwasser, & Bloemraad (2024). Immigrant collective action dataset (*Socius*)

## Authors

- **Steven Lauterwasser** — sole technical contributor; all architectural and methodological decisions.
- **Kim Voss** (UC Berkeley) and **Irene Bloemraad** (UBC) — principal investigators and domain experts.
