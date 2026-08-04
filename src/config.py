"""
Project-wide platform-conditional configuration.

This module is the single source of truth for values that depend on
*where* we're running (local Mac vs. cluster "Explorer"):

  - **Paths**: cluster mounts the data at `/projects/ahd`; local has
    everything under `~/immigration_project/00_ML_data_expansion/00_explorer`.
  - **Dtype policy**: cluster CUDA gets `"mixed_float16"`; local MPS
    (or CPU) gets `"float32"` because mixed precision support on MPS
    has historically been patchy and the speedup motivation evaporates
    when there's no Tensor Core hardware to take advantage of it.

A single bit (`IS_CLUSTER`) is computed once at import time; everything
else flows from it. Adding new platform-conditional values later is a
one-file edit, not a new architectural decision.

**No side effects**: this module exports values; it does not mutate
global Keras state. Scripts that want the dtype policy applied call
`keras.config.set_dtype_policy(config.DTYPE_POLICY)` themselves. This
keeps `import config` from being surprising — tests can override, and
future readers don't have to know that importing has hidden effects.

Example usage::

    import keras
    import src.config as config

    keras.config.set_dtype_policy(config.DTYPE_POLICY)
    weights_path = config.DAPT_BACKBONE_WEIGHTS
    cca_set_dir = config.CCA_SET_DIR
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
# Detection mechanism: file-existence check on `/projects/ahd`, with an
# `ICA_ENV` environment-variable override for the rare cases where the
# automatic detection is wrong (e.g., a laptop that happens to have
# `/projects/ahd` mounted as a network share).
#
# `ICA_ENV=cluster` forces IS_CLUSTER=True; `ICA_ENV=local` forces False.
# Any other value (including "") falls back to the file-existence default
# and emits a warning so a typo doesn't silently change behavior.

_CLUSTER_MARKER = Path("/projects/ahd")
_ENV_OVERRIDE = os.environ.get("ICA_ENV", "").strip().lower()

if _ENV_OVERRIDE == "cluster":
    IS_CLUSTER: bool = True
elif _ENV_OVERRIDE == "local":
    IS_CLUSTER = False
elif _ENV_OVERRIDE == "":
    IS_CLUSTER = _CLUSTER_MARKER.exists()
else:
    warnings.warn(
        f"Unrecognized ICA_ENV value: {_ENV_OVERRIDE!r}. "
        f"Expected 'cluster', 'local', or unset. Falling back to "
        f"file-existence detection on {_CLUSTER_MARKER}.",
        stacklevel=2,
    )
    IS_CLUSTER = _CLUSTER_MARKER.exists()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# All paths are `pathlib.Path` instances. Use `/` to join further
# (e.g., `config.CCA_SET_DIR / "train_pos.tf"`); pass to functions
# that expect strings via `str(path)` or rely on the function's
# implicit `__fspath__` support (most polars / tf.data / os APIs handle
# Path natively).

if IS_CLUSTER:
    PROJECT_ROOT: Path = Path("/projects/ahd")
else:
    PROJECT_ROOT = (
        Path.home() / "immigration_project" / "00_ML_data_expansion" / "00_explorer"
    )

# Data sources
LDC_CORPUS: Path = PROJECT_ROOT / "ldc_corpus"
# NYT Archive API corpus (dateline-less; application target for the US filter)
API_CORPUS_DIR: Path = PROJECT_ROOT / "api_corpus"

# DAPT-phase artifacts
DAPT_TRAINING_SET: Path = PROJECT_ROOT / "dapt_training_set.tf"
DAPT_VALIDATION_SET: Path = PROJECT_ROOT / "dapt_validation_set.tf"
DAPT_BACKBONE_WEIGHTS: Path = PROJECT_ROOT / "dapt_backbone.weights.h5"
DAPT_LM_HEAD_WEIGHTS: Path = PROJECT_ROOT / "lm_head_weights.npy"
DAPT_CURRENT_MODEL: Path = PROJECT_ROOT / "dapt_current_model.keras"
DAPT_LOGS_DIR: Path = PROJECT_ROOT / "dapt_logs"

# Classification-phase artifacts
CCA_SET_DIR: Path = PROJECT_ROOT / "cca_set"
CCA_CLASSIFIER_DIR: Path = PROJECT_ROOT / "cca_classifier"
CCA_CLASSIFIER_MODEL: Path = PROJECT_ROOT / "cca_classifier.keras"
CCA_CLASSIFIER_WEIGHTS: Path = PROJECT_ROOT / "cca_classifier.weights.h5"
CCA_LOGS_DIR: Path = PROJECT_ROOT / "cca_logs"

# Prior-estimation artifacts
LU_CLASSIFIER_MODEL: Path = PROJECT_ROOT / "lu_classifier.keras"
LU_PREDS_DIR: Path = CCA_SET_DIR / "lu"

# US/not-US pre-filter artifacts
US_FILTER_DIR: Path = PROJECT_ROOT / "us_filter"
US_FILTER_LABELED_PARQUET: Path = US_FILTER_DIR / "ldc_labeled.parquet"
US_FILTER_SET_DIR: Path = US_FILTER_DIR / "us_set"
US_FILTER_CLASSIFIER_DIR: Path = US_FILTER_DIR / "classifier"
US_FILTER_CLASSIFIER_WEIGHTS: Path = US_FILTER_DIR / "us_classifier.weights.h5"
US_FILTER_CLASSIFIER_FULL_WEIGHTS: Path = US_FILTER_DIR / "us_classifier_full.weights.h5"
US_FILTER_FULL_WEIGHTS: Path = US_FILTER_CLASSIFIER_FULL_WEIGHTS  # Phase 3 alias for harmonized retrains
US_FILTER_LOGS_DIR: Path = US_FILTER_DIR / "logs"
US_FILTER_SCORES_DIR: Path = US_FILTER_DIR / "api_us_scores"
# API<->LDC cross-match table (from api_ldc_join.R) -- the id-space translation
# table for turning API-format (`nyt://...`) ids into their LDC-integer twins.
US_FILTER_API_LDC_MATCHED: Path = US_FILTER_DIR / "audit" / "api_ldc_matched.parquet"
# US-head retrain (v1, stripped channel): the embed source for the 345 LDC-format
# DoCA positives, and the assembled P/N/U training table.
US_POS_LDC345_SOURCE: Path = US_FILTER_DIR / "us_pos_ldc345_source.parquet"
US_PNU_TABLE: Path = US_FILTER_DIR / "us_pnu_table.parquet"
# The nnPNU retrain candidate itself (src/run_us_pnu.py). Distinct weights file
# from the production `us_classifier_full` head -- this is the validate-before-
# swap candidate, never consumed by apply_ica.py / assemble_ica.py.
US_PNU_WEIGHTS: Path = US_FILTER_DIR / "us_pnu.weights.h5"

# Validation (gold-set) artifacts
VALIDATION_DIR: Path = PROJECT_ROOT / "validation"
ICA_HOLDOUT_IDS: Path = VALIDATION_DIR / "ica_holdout_ids.parquet"
ICA_HOLDOUT_IDS_LDC: Path = VALIDATION_DIR / "ica_holdout_ids_ldc.parquet"

# CCA/DoCA retrain artifacts (2026-06-15 retrain on the NYT API corpus with
# DoCA-confirmed positives; see docs/notes/cca-doca-retrain-design.md).
#
# `DOCA_CCA_MATCHES` is an EXTERNAL, non-checked-in R artifact (the DoCA->NYT
# fuzzy match, keyed by `article_id` in `nyt://article/...` form). Locally it
# lives beside the LDC corpus source tree, one level above PROJECT_ROOT
# (`.../00_ML_data_expansion/LDC2008T19/data`). The cluster path is a best
# guess until the maintenance window ends — verify it then.
if IS_CLUSTER:
    DOCA_CCA_MATCHES: Path = PROJECT_ROOT / "LDC2008T19" / "data" / "cca_matches_good.rds"
else:
    DOCA_CCA_MATCHES = PROJECT_ROOT.parent / "LDC2008T19" / "data" / "cca_matches_good.rds"

# Derived data products (gitignored, like the us_filter family).
CCA_DOCA_DIR: Path = PROJECT_ROOT / "cca_doca"
CCA_DOCA_POSITIVES: Path = CCA_DOCA_DIR / "cca_doca_positives.parquet"
CCA_EMBED_CACHE_DIR: Path = CCA_DOCA_DIR / "embed_cache"
CCA_DOCA_TABLE: Path = CCA_DOCA_DIR / "cca_doca_table.parquet"
CCA_DOCA_WEIGHTS: Path = CCA_DOCA_DIR / "cca_doca.weights.h5"
CCA_DOCA_SCORES_DIR: Path = CCA_DOCA_DIR / "api_cca_scores"
ICA_CANDIDATES_DIR: Path = CCA_DOCA_DIR / "ica_candidates"

# Relevance head artifacts (Phase 3 training output)
RELEVANCE_DOCA_WEIGHTS: Path = PROJECT_ROOT / "relevance" / "relevance.weights.h5"
# Text-mode / encoder-unfreeze rel-first path (docs/notes/encoder-unfreeze-strategy.md).
RELEVANCE_TEXT_TABLE: Path = PROJECT_ROOT / "relevance" / "relevance_text_table.parquet"
RELEVANCE_SET_DIR: Path = PROJECT_ROOT / "relevance" / "rel_set"
RELEVANCE_TEXT_WEIGHTS: Path = PROJECT_ROOT / "relevance" / "relevance_text.weights.h5"


# ---------------------------------------------------------------------------
# Compute / precision
# ---------------------------------------------------------------------------
# Cluster has CUDA + Tensor Cores; local Mac has MPS or CPU. Mixed
# precision pays off on the former and is unreliable / pointless on
# the latter. Callers apply this explicitly via
# `keras.config.set_dtype_policy(DTYPE_POLICY)`.
#
# `ICA_DTYPE_POLICY` overrides the platform rule. Motivating case
# (2026-08-04): cluster embed jobs producing CLS caches that must be
# precision-uniform with the locally-produced fp32 production caches
# (`full` part-1, `ldc_9507`) — an fp16 append would put a numeric seam
# exactly on the 1975/76 era boundary of an era-comparison cache.
# Invalid values fail loudly at import; no silent fallback.

_DTYPE_OVERRIDE = os.environ.get("ICA_DTYPE_POLICY")
if _DTYPE_OVERRIDE is not None and _DTYPE_OVERRIDE not in ("float32", "mixed_float16"):
    raise ValueError(
        f"ICA_DTYPE_POLICY must be 'float32' or 'mixed_float16'; "
        f"got {_DTYPE_OVERRIDE!r}"
    )
DTYPE_POLICY: str = (
    _DTYPE_OVERRIDE
    if _DTYPE_OVERRIDE is not None
    else ("mixed_float16" if IS_CLUSTER else "float32")
)
