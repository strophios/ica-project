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

from pathlib import Path

import keras_hub

import src.config as config


def resolve_backbone_path(recorded, canonical=None) -> Path:
    """Resolve a possibly machine-foreign recorded backbone path.

    Sidecar configs record the ABSOLUTE path of the backbone that trained a
    head — correct on the writing machine, foreign after a sync to the other
    platform (2026-08-03 cluster failure: a Mac path inside
    `us_classifier.config.json`). Resolution rule:

      1. an existing recorded path wins (single-machine case, explicit
         non-default backbones — no behavior change);
      2. a missing path whose FILENAME matches the canonical platform DAPT
         backbone (`config.DAPT_BACKBONE_WEIGHTS` unless injected) resolves
         to the canonical path — same artifact identity, platform-correct
         locator, logged loudly;
      3. anything else raises: never silently substitute a different
         artifact (the July backbone-clobber bug is the cautionary tale).

    Args:
        recorded: the path as recorded (str or Path, e.g. from a sidecar).
        canonical: override for the platform-canonical DAPT backbone
            (tests inject a tmp path; production uses the config default).
    """
    p = Path(recorded)
    if p.exists():
        return p
    canonical = Path(canonical) if canonical is not None else config.DAPT_BACKBONE_WEIGHTS
    if p.name == canonical.name and canonical.exists():
        print(f"backbone path {p} absent on this platform; "
              f"resolved to canonical {canonical}")  # LOG
        return canonical
    raise FileNotFoundError(
        f"backbone weights not found: {p} (platform canonical {canonical} "
        f"{'exists but has a different filename' if canonical.exists() else 'is also absent'}). "
        f"For a non-default backbone, pass its platform-local path explicitly "
        f"(e.g. --backbone-weights)."
    )


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

    # Resolve machine-foreign sidecar paths before the h5 open (a no-op for
    # any path that exists — see resolve_backbone_path).
    weights_path = resolve_backbone_path(weights_path)
    backbone = keras_hub.models.Backbone.from_preset(
        "roberta_base_en", preprocessor=None, load_weights=False
    )
    # `skip_mismatch=False`: pin the load-strict discipline (Tier 3
    # Piece 2). Keras 3's `.weights.h5` save format keys variables
    # by layer-class + positional index, so a *rename* of an
    # internal layer wouldn't be caught here, but a *shape*
    # mismatch (e.g., a future keras_hub roberta_base_en variant
    # with different hidden_dim) would silently load partial
    # weights without this. Explicit > implicit, even when
    # explicit matches the framework default.
    backbone.load_weights(str(weights_path), skip_mismatch=False)

    return backbone
