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
US_FILTER_LOGS_DIR: Path = US_FILTER_DIR / "logs"
US_FILTER_SCORES_DIR: Path = US_FILTER_DIR / "api_us_scores"

# Validation (gold-set) artifacts
VALIDATION_DIR: Path = PROJECT_ROOT / "validation"


# ---------------------------------------------------------------------------
# Compute / precision
# ---------------------------------------------------------------------------
# Cluster has CUDA + Tensor Cores; local Mac has MPS or CPU. Mixed
# precision pays off on the former and is unreliable / pointless on
# the latter. Callers apply this explicitly via
# `keras.config.set_dtype_policy(DTYPE_POLICY)`.

DTYPE_POLICY: str = "mixed_float16" if IS_CLUSTER else "float32"
