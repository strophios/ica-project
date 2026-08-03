"""Tests for `resolve_backbone_path` (src/model_setup/backbone.py).

Sidecar configs record the ABSOLUTE backbone path of the writing machine
(`UsRunConfig.backbone_weights_path` materializes `config.DAPT_BACKBONE_WEIGHTS`
at write time). Synced to the other platform, that path is foreign — the
2026-08-03 cluster smoke failed inside `_build_embed_model` trying to open a
Mac path from `us_classifier.config.json`. Resolution rule:

  1. an EXISTING recorded path wins (single-machine case, non-default
     backbones — no behavior change);
  2. a missing path whose FILENAME matches the canonical platform DAPT
     backbone resolves to the canonical path (identity preserved, locator
     swapped, loud log);
  3. anything else raises FileNotFoundError with an actionable message —
     never silently substitute a different artifact (backbone-clobber
     lesson: silent substitution voided a whole re-embed round in July).
"""

from __future__ import annotations

import pytest

from src.model_setup.backbone import resolve_backbone_path


@pytest.fixture
def canonical(tmp_path):
    """A stand-in platform-canonical DAPT backbone file that exists."""
    p = tmp_path / "platform" / "dapt_backbone.weights.h5"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"weights")
    return p


def test_existing_recorded_path_wins(tmp_path, canonical):
    """Rule 1: a recorded path that exists is returned unchanged — even when
    a canonical with the same name exists elsewhere."""
    recorded = tmp_path / "local" / "dapt_backbone.weights.h5"
    recorded.parent.mkdir(parents=True)
    recorded.write_bytes(b"weights")
    assert resolve_backbone_path(recorded, canonical=canonical) == recorded


def test_missing_path_with_canonical_name_resolves(tmp_path, canonical):
    """Rule 2: the cluster-smoke case — a foreign absolute path with the
    canonical filename resolves to the platform canonical."""
    foreign = tmp_path / "other_machine" / "dapt_backbone.weights.h5"  # never created
    assert resolve_backbone_path(foreign, canonical=canonical) == canonical


def test_missing_path_with_other_name_raises(tmp_path, canonical):
    """Rule 3: a missing non-canonical backbone (e.g. a tuned checkpoint)
    must fail loudly, pointing at the explicit-path escape hatch."""
    foreign = tmp_path / "other_machine" / "tuned_backbone.job123.weights.h5"
    with pytest.raises(FileNotFoundError, match="backbone-weights"):
        resolve_backbone_path(foreign, canonical=canonical)


def test_missing_path_and_missing_canonical_raises(tmp_path):
    """Rule 3 edge: same filename but the canonical does not exist either —
    resolution must not return a nonexistent path."""
    foreign = tmp_path / "other_machine" / "dapt_backbone.weights.h5"
    ghost_canonical = tmp_path / "platform" / "dapt_backbone.weights.h5"  # never created
    with pytest.raises(FileNotFoundError):
        resolve_backbone_path(foreign, canonical=ghost_canonical)


def test_accepts_string_input(tmp_path, canonical):
    """Sidecar JSON stores the path as a string; both str and Path work."""
    foreign = str(tmp_path / "other_machine" / "dapt_backbone.weights.h5")
    assert resolve_backbone_path(foreign, canonical=canonical) == canonical
