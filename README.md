# ICA Project: Transformer-Based Classification for Historical Protest Events

A machine learning system for identifying immigrant collective action (ICA) events in 150+ years of *New York Times* coverage, using positive-unlabeled (PU) learning with a RoBERTa backbone.

## The Research Problem

The goal is to expand protest event datasets (like the [Dynamics of Collective Action](https://web.stanford.edu/group/collectiveaction/cgi-bin/drupal/)) across 150+ years of *New York Times* articles, finding the tiny fraction of articles reporting immigrant collective action among millions of articles total. Three challenges make this harder than standard text classification:

1. **Positive-Unlabeled data**: We have labeled positives (from existing datasets + NYT indexer tags) but zero confirmed negatives. An unlabeled article may still describe a relevant event.
2. **Severe class imbalance**: ICA articles are a vanishingly small minority of the corpus.
3. **Temporal linguistic drift**: The same phenomenon (a protest) is described in fundamentally different language in 1870 vs. 2020.

## Approach

The system addresses these challenges through a multi-phase pipeline:

### Phase 1: Domain-Adaptive Pre-Training (DAPT)

Further pre-training of RoBERTa on ~1.16M NYT headline + lede pairs to adapt its general English representations to the NYT domain. This closes the distribution gap between RoBERTa's web-text pre-training corpus and the target newspaper text. *(Implemented.)*

### Phase 2: Class Prior Estimation

Estimating the proportion of true positives in the unlabeled set (α ≈ 0.03 for collective action events). Uses a labeled-vs-unlabeled classifier feeding into the DEDPUL expectation-maximization algorithm (Ivanov 2020). The estimated prior parameterizes the PU learning loss in Phase 3. *(Implemented, under revision.)*

### Phase 3: PU Classification

Binary classification using FLPULoss (Ji et al. 2023), a custom loss function combining focal loss (Lin et al. 2020) with non-negative PU learning (Kiryo et al. 2017), parameterized by the estimated class prior from Phase 2. The classifier uses the DAPT backbone with a classification head (CLS token → dropout → dense → dropout → logits). *(Implemented for single-head CCA classification, under revision.)*

### Planned Extensions

- **Multi-headed architecture**: Separate classification heads for collective action, immigrant involvement, US events, and combined ICA — leveraging larger labeled datasets for each subtask to make the sparse ICA labeling problem tractable.
- **Temporal adaptation**: Time tokens (e.g., `<1900>`) with a temporal masking pre-training task, conditioning word interpretation on temporal context to handle 150+ years of linguistic drift.
- **Virtual Adversarial Training (ALUM)**: Adversarial regularization for improved generalization under label scarcity.
- **Transfer learning**: Pre-training on a related labeled dataset before fine-tuning on the PU task.
- **Per-layer learning rates and encoder unfreezing**.

## Repository Structure

```
ica_project/
├── src/
│   ├── dapt.py                      # Phase 1: Domain-adaptive pre-training script
│   ├── run_prior_estimate.py         # Phase 2: Class prior estimation script
│   ├── run_cca_classification.py     # Phase 3: CCA classifier training script
│   ├── eval_cca_classifier.py        # Evaluation script
│   ├── model_setup/
│   │   ├── dapt_setup.py             # DAPT model initialization
│   │   └── classification_setup.py   # Classification model construction
│   ├── loss_functions/
│   │   └── loss.py                   # FLPULoss (focal + non-negative PU learning)
│   ├── prior_estimation/
│   │   ├── lu_classifier.py          # Labeled vs. unlabeled classifier
│   │   ├── dedpul_em.py              # DEDPUL EM algorithm (adapted from Ivanov 2020)
│   │   ├── dedpul_utils.py           # DEDPUL utility functions
│   │   └── ramaswamy2016.py          # Alternative kernel-based prior estimation
│   ├── preproc/
│   │   └── preprocessor.py           # Text preprocessing (MLM masking + classifier modes)
│   ├── data_setup/                   # Data loading and pipeline construction
│   └── utilities/                    # Shared utilities
├── ml_memo/
│   ├── ml_memo.pdf                   # 22-page project documentation (see below)
│   ├── ml_memo.qmd                   # Quarto source
│   └── references.json               # Bibliography (27 references)
├── main.py                           # Entry point
├── CLAUDE.md                         # Detailed technical documentation
├── pyproject.toml                    # Dependencies (Python 3.12, uv)
└── README.md                         # This file
```

## Key Design Decisions

- **Models output logits** (no final activation); all losses use `from_logits=True`.
- **Mixed precision** (`mixed_float16`) enabled for GPU training on both local (tensorflow-metal) and HPC (CUDA) environments.
- **Re-balanced batches** via weighted sampling to guarantee labeled positive signal in every batch despite extreme class imbalance.
- **Headline + lede as input** (not full articles) — captures the relevant signal at a fraction of the compute cost, and enables future expansion to the full NYT archive (1851–present) via the freely available NYT Archive API.
- **Cross-platform portability**: runs on both macOS (tensorflow-metal) and Linux HPC clusters (tensorflow + CUDA).

## Documentation

The `ml_memo/` directory contains a 22-page project documentation memo with 27 references, written to make the project legible to non-technical collaborators. It covers the research problem, methodological choices, alternatives considered, and implementation plans in depth.

## Data Sources

Training data is constructed from the overlap of two sources:

- **Dynamics of Collective Action** (DoCA, 1960–1995): ~23k coded protest events, including ~661 immigrant collective action events.
- **NYT Annotated Corpus** (LDC, 1987–2007): ~1.8M articles with indexer descriptor tags.

Data is not included in this repository. See the ML memo for details on data construction and labeling strategy.

## Setup

Requires Python 3.12. Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
source .venv/bin/activate
```

All scripts should be run from the project root directory.

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
