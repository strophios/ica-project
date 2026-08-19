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
import numpy as np

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


def layer_diff_summary(
    paths: list[str], weights_a: list[np.ndarray], weights_b: list[np.ndarray]
) -> dict[str, float]:
    """Pure: max |delta| per top-level backbone group between two weight lists.

    `paths` / `weights_a` / `weights_b` must be parallel (same order, same
    variable identity) -- e.g. `[w.path for w in backbone.weights]` and the
    matching `.numpy()` arrays for two backbones built from the same
    `keras_hub.models.Backbone.from_preset(...)` call, so structure/order is
    guaranteed identical between the two. Groups by the first `/`-delimited
    path segment (`"embeddings"`, `"embeddings_layer_norm"`,
    `"transformer_layer_0"` .. `"transformer_layer_11"` for `roberta_base_en`).

    Lives here (rather than `src.extract_tuned_backbone`, its original home)
    so `build_grafted_backbone` below can use it without a circular import
    (`extract_tuned_backbone` imports `load_dapt_backbone` from this module);
    `extract_tuned_backbone.layer_diff_summary` re-exports this function for
    back-compat.
    """
    if not (len(paths) == len(weights_a) == len(weights_b)):
        raise ValueError(
            f"paths/weights_a/weights_b must be equal length; got "
            f"{len(paths)}/{len(weights_a)}/{len(weights_b)}"
        )
    if not paths:
        raise ValueError("paths must be non-empty")
    out: dict[str, float] = {}
    for path, a, b in zip(paths, weights_a, weights_b):
        group = path.split("/")[0]
        delta = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        out[group] = max(out.get(group, 0.0), delta)
    return out


def build_grafted_backbone(
    base_weights_path, donor_weights_path, layer_groups: set[str]
) -> tuple[keras_hub.models.Backbone, dict]:
    """Build a backbone that is `base_weights_path` everywhere EXCEPT
    `layer_groups`, which are overwritten from `donor_weights_path`.

    Promoted from `scripts/graft_test.py:build_graft_backbone` (the stage-1
    branched-encoder graft harness) into reusable infrastructure for the
    stage-4 branched embed model -- see
    `docs/design-plans/2026-08-18-stage4-joint-finetune.md` "Branched
    productionization -- implementation contract".

    Loads `base`, `donor`, and a fresh `graft` copy of the base (all via
    `load_dapt_backbone`, so each is weight-loaded from its own file
    independently -- Pattern A in-process mutation happens only on `graft`).
    Verifies structural identity across all three (same weight ordering),
    then `graft.get_layer(name).set_weights(donor.get_layer(name).get_weights())`
    for each `name` in `layer_groups`.

    Verification (via `layer_diff_summary`) enforces the graft invariant:
    `graft` differs from `base` at EXACTLY the grafted groups (nonzero there,
    zero elsewhere), and matches `donor` EXACTLY (0.0 delta) at the grafted
    groups. Raises `ValueError` with the offending summary on violation --
    this is a hard-fail correctness gate, not a warning.

    Args:
        base_weights_path: `.weights.h5` path for the backbone everywhere
            outside `layer_groups`.
        donor_weights_path: `.weights.h5` path supplying the grafted groups'
            weights.
        layer_groups: sub-layer names to graft (e.g. `{"transformer_layer_11"}`
            -- top-level `backbone.get_layer(name)` names).

    Returns:
        `(grafted_backbone, {"vs_base": ..., "vs_donor": ...})` -- the two
        diff summaries are `layer_diff_summary` per-group max|delta| tables.
    """
    base = load_dapt_backbone(base_weights_path)
    donor = load_dapt_backbone(donor_weights_path)
    graft = load_dapt_backbone(base_weights_path)

    base_paths = [w.path for w in base.weights]
    donor_paths = [w.path for w in donor.weights]
    graft_paths = [w.path for w in graft.weights]
    if not (base_paths == donor_paths == graft_paths):
        raise ValueError(
            "backbone weight ordering diverged across base/donor/graft "
            "instances -- base, donor, and graft must all be built from "
            "the same backbone architecture"
        )

    for group in layer_groups:
        graft.get_layer(group).set_weights(donor.get_layer(group).get_weights())

    base_arrays = [np.asarray(w.numpy()) for w in base.weights]
    donor_arrays = [np.asarray(w.numpy()) for w in donor.weights]
    graft_arrays = [np.asarray(w.numpy()) for w in graft.weights]

    diff_vs_base = layer_diff_summary(graft_paths, graft_arrays, base_arrays)
    diff_vs_donor = layer_diff_summary(graft_paths, graft_arrays, donor_arrays)

    nonzero_vs_base = {g for g, d in diff_vs_base.items() if d != 0.0}
    if nonzero_vs_base != set(layer_groups):
        raise ValueError(
            f"graft vs base diff mismatch: nonzero groups {sorted(nonzero_vs_base)} "
            f"!= expected grafted groups {sorted(layer_groups)}; "
            f"summary={diff_vs_base}"
        )

    nonzero_vs_donor_at_grafted = {
        g for g in layer_groups if diff_vs_donor.get(g, 0.0) != 0.0
    }
    if nonzero_vs_donor_at_grafted:
        raise ValueError(
            f"graft vs donor diff nonzero at grafted group(s) "
            f"{sorted(nonzero_vs_donor_at_grafted)}; expected exactly 0.0; "
            f"summary={diff_vs_donor}"
        )

    return graft, {"vs_base": diff_vs_base, "vs_donor": diff_vs_donor}
