"""
Backbone-loading utilities for the classification stack.

Split out from the legacy `classifier_from_dapt_checkpoint` (which
was doing too many things in one function — backbone loading, head
construction, and assembly all at once). Tier 2 Piece 4b.

Currently exposes a single function: `load_dapt_backbone`. The
legacy "load from full saved `.keras` model and pluck `model.layers[2]`"
path is intentionally not included here — it was only used in
scratch (`test_module.py`) and relied on positional layer-index
access that breaks if the DAPT model's layer order ever changes.
If a future use case needs full-model loading back, it's a small
addition.
"""

from __future__ import annotations

import keras_hub


def load_dapt_backbone(weights_path):
    """
    Load a DAPT-finetuned RoBERTa backbone from saved weights.

    Constructs a fresh `roberta_base_en` backbone (no preprocessor
    attached, no preset weights loaded), then loads the DAPT-trained
    weights from the given path. The backbone's `.trainable`
    attribute is left at its default (True) — callers that want to
    freeze it (e.g., during head-only training) should set it
    themselves, or pass `freeze_encoder=True` to
    `assembly.build_endpoint_model`.

    Args:
        weights_path: path to a `.weights.h5` file containing the
            backbone weights (typically `config.DAPT_BACKBONE_WEIGHTS`).
            Accepts `pathlib.Path` or `str`.

    Returns:
        A `keras_hub.models.Backbone` instance, weights loaded.
    """

    backbone = keras_hub.models.Backbone.from_preset(
        "roberta_base_en", preprocessor=None, load_weights=False
    )
    backbone.load_weights(str(weights_path))

    return backbone
