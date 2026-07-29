# pattern: Imperative Shell
# Reason: monkeypatches module-level I/O entry points to test control flow
# without touching real caches/weights.
"""Tests for the tuned-cache retrain knobs on the three features-mode trainers
(`run_us_features.py`, `run_cca_doca.py`, `run_relevance.py`).

Caches/weights for the tuned artifacts don't exist yet (produced on-cluster),
so these tests never do real training. Each trainer's `main()` is exercised
only up to its first post-guard I/O call, which is monkeypatched to raise a
distinguishable marker exception. This proves two things per script:

  - the production-overwrite guard fires (raises ValueError, marker never
    reached) when a non-default cache suffix pairs with the production
    weights path (the default, or an explicit path equal to it)
  - the guard does NOT fire (marker IS reached) for the default suffix+weights
    combo, and for a non-default suffix paired with a distinct weights path --
    i.e. default behavior is unchanged and the escape hatch works
"""

from __future__ import annotations


import pytest

import src.run_us_features as run_us_features
import src.run_cca_doca as run_cca_doca
import src.run_relevance as run_relevance


class _MarkerReached(Exception):
    """Raised by a monkeypatched I/O call to prove control flow got past the guard."""


def _raise_marker(*args, **kwargs):
    raise _MarkerReached("reached first post-guard I/O call")


# ---------------------------------------------------------------------------
# run_us_features.py
# ---------------------------------------------------------------------------


class TestRunUsFeaturesGuard:
    def test_default_suffix_default_weights_passes_guard(self, monkeypatch):
        """Default invocation (no --suffix, no --out) must reach load_cache,
        i.e. the guard does not block the everyday production run."""
        monkeypatch.setattr(run_us_features, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            run_us_features.main()

    def test_tuned_suffix_default_weights_raises(self, monkeypatch):
        monkeypatch.setattr(run_us_features, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            run_us_features.main(suffix="us_train_ldc_tuned")

    def test_tuned_suffix_explicit_production_weights_raises(self, monkeypatch):
        """Explicitly passing the production path (not just relying on the
        default) must still be caught."""
        monkeypatch.setattr(run_us_features, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            run_us_features.main(
                suffix="us_train_ldc_tuned",
                weights_path=run_us_features.config.US_FILTER_FULL_WEIGHTS,
            )

    def test_tuned_suffix_with_distinct_weights_passes_guard(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_us_features, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            run_us_features.main(
                suffix="us_train_ldc_tuned",
                weights_path=tmp_path / "us_classifier_full_tuned.weights.h5",
            )

    def test_default_suffix_constant_matches_documented_production_cache(self):
        assert run_us_features.DEFAULT_SUFFIX == "us_train_ldc"


class TestRunUsFeaturesBackboneWeights:
    """Bookkeeping knob: --backbone-weights overrides the sidecar's
    backbone_weights_path (used by token-mode eval consumers), default unchanged."""

    def test_backbone_weights_none_leaves_default_backbone_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_us_features, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            run_us_features.main(
                suffix="us_train_ldc_tuned",
                weights_path=tmp_path / "tuned.weights.h5",
                backbone_weights=None,
            )
        # DEFAULT_US_CONFIG itself must be untouched (dataclasses.replace
        # returns a new object; the module-level default is immutable).
        assert run_us_features.us_config.DEFAULT_US_CONFIG.backbone_weights_path == str(
            run_us_features.config.DAPT_BACKBONE_WEIGHTS
        )

    def test_backbone_weights_override_reaches_replace_without_error(self, monkeypatch, tmp_path):
        """Passing --backbone-weights must not itself raise before load_cache
        (i.e. the dataclasses.replace call succeeds)."""
        monkeypatch.setattr(run_us_features, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            run_us_features.main(
                suffix="us_train_ldc_tuned",
                weights_path=tmp_path / "tuned.weights.h5",
                backbone_weights=tmp_path / "tuned_backbone.weights.h5",
            )


# ---------------------------------------------------------------------------
# run_cca_doca.py
# ---------------------------------------------------------------------------


class TestRunCcaDocaGuard:
    def test_default_suffix_default_weights_passes_guard(self, monkeypatch):
        monkeypatch.setattr(run_cca_doca, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            run_cca_doca.main(prior=0.02)

    def test_tuned_suffix_default_weights_raises(self, monkeypatch):
        monkeypatch.setattr(run_cca_doca, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            run_cca_doca.main(prior=0.02, suffix="train250k_tuned")

    def test_tuned_suffix_explicit_production_weights_raises(self, monkeypatch):
        monkeypatch.setattr(run_cca_doca, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            run_cca_doca.main(
                prior=0.02,
                suffix="train250k_tuned",
                weights_path=run_cca_doca.config.CCA_DOCA_WEIGHTS,
            )

    def test_tuned_suffix_with_distinct_weights_passes_guard(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_cca_doca, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            run_cca_doca.main(
                prior=0.02,
                suffix="train250k_tuned",
                weights_path=tmp_path / "cca_doca_tuned.weights.h5",
            )

    def test_default_suffix_constant_matches_documented_production_cache(self):
        assert run_cca_doca.DEFAULT_SUFFIX == "train250k"


# ---------------------------------------------------------------------------
# run_relevance.py
# ---------------------------------------------------------------------------


class TestRunRelevanceGuard:
    def test_default_suffix_default_weights_passes_guard(self, monkeypatch):
        monkeypatch.setattr(run_relevance, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            run_relevance.main(prior=0.02)

    def test_tuned_suffix_default_weights_raises(self, monkeypatch):
        monkeypatch.setattr(run_relevance, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            run_relevance.main(prior=0.02, suffix="relevance_train_tuned")

    def test_tuned_suffix_explicit_production_weights_raises(self, monkeypatch):
        monkeypatch.setattr(run_relevance, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            run_relevance.main(
                prior=0.02,
                suffix="relevance_train_tuned",
                weights_path=run_relevance.DEFAULT_WEIGHTS,
            )

    def test_tuned_suffix_with_distinct_weights_passes_guard(self, monkeypatch, tmp_path):
        monkeypatch.setattr(run_relevance, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            run_relevance.main(
                prior=0.02,
                suffix="relevance_train_tuned",
                weights_path=tmp_path / "relevance_tuned.weights.h5",
            )

    def test_default_suffix_constant_matches_documented_production_cache(self):
        assert run_relevance.DEFAULT_SUFFIX == "relevance_train"

    def test_weights_path_is_coerced_to_path_for_string_input(self, monkeypatch, tmp_path):
        """weights_path accepts a plain string (as argparse.--out supplies) and
        still compares correctly against the guard."""
        monkeypatch.setattr(run_relevance, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            run_relevance.main(
                prior=0.02,
                suffix="relevance_train_tuned",
                weights_path=str(run_relevance.DEFAULT_WEIGHTS),
            )
