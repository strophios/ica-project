# pattern: Functional Core
"""Relevance-head features-mode inference helper (Phase 2, Task 2).

Mirrors apply_cca_model for the relevance head: loads the RunConfig sidecar,
constructs a fresh features-mode inference model, loads weights by structure,
and predicts logits over cached embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import src.config as config
import src.cca_config as cca_config
from src.model_setup.heads import ClassificationHead
from src.model_setup.assembly import build_feature_inference_model


def apply_relevance_model(
    features: np.ndarray,
    weights_path: Path | str = None,
) -> np.ndarray:
    """Score cached feature vectors with the trained relevance head (Pattern 2 reload).

    Loads the RunConfig sidecar, builds a fresh features-mode inference model,
    loads weights by structure, and returns raw logits (shape (n,)). The head
    outputs logits; threshold at 0.0 for prob 0.5, or sigmoid for [0,1] scores.

    The relevance head reuses the CCA config structure (RunConfig, HeadConfig, etc.),
    so we use cca_config.RunConfig and cca_config.config_path_for_weights to load
    the sidecar metadata.

    Args:
        features: numpy array of shape (n, hidden_dim), typically CLS vectors
        weights_path: path to .h5 weights file. Defaults to RELEVANCE_DOCA_WEIGHTS
                      if not provided. The .config.json sidecar is expected to exist
                      at the same path stem.

    Returns:
        numpy array of shape (n,) containing raw logits.

    Raises:
        ValueError: if logits contain non-finite values.
        FileNotFoundError: if the weights or config sidecar is not found.
    """
    if weights_path is None:
        # Assume there's a RELEVANCE_DOCA_WEIGHTS in config, similar to CCA_DOCA_WEIGHTS
        if hasattr(config, "RELEVANCE_DOCA_WEIGHTS"):
            weights_path = config.RELEVANCE_DOCA_WEIGHTS
        else:
            raise ValueError(
                "weights_path must be provided or config.RELEVANCE_DOCA_WEIGHTS "
                "must be defined"
            )

    weights_path = Path(weights_path)
    run_config = cca_config.RunConfig.from_json(
        cca_config.config_path_for_weights(weights_path)
    )
    head_cfg = run_config.heads[0]
    head = ClassificationHead(hidden_dim=head_cfg.hidden_dim, name=head_cfg.name)
    model = build_feature_inference_model(
        {head_cfg.name: head}, hidden_dim=head_cfg.hidden_dim
    )
    model.load_weights(str(weights_path), skip_mismatch=False)

    logits = model.predict({"features": features}, batch_size=512, verbose=0)
    if isinstance(logits, dict):
        logits = logits[head_cfg.name]
    logits = np.asarray(logits).reshape(-1)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite relevance logits produced by model")
    return logits
