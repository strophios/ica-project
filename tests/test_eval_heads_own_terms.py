# pattern: Imperative Shell
# Reason: monkeypatches module-level I/O entry points to test control flow
# without touching the real eval CSV or embed cache.
"""Tests for the tuned-artifact knobs added to `scripts/eval_heads_own_terms.py`.

`main()` now takes cca_weights/rel_weights/us_weights/cache_suffix/out_path,
each independently defaulting to the production artifact. These tests verify
default-argument resolution (Path coercion + production fallback) and that a
tuned run's `out_path` doesn't fall back to the production JSON, all without
touching the real eval CSV -- `pl.read_csv` (the first call in `main()`) is
monkeypatched to a marker exception, so reaching it proves argument resolution
completed successfully.
"""

from __future__ import annotations


import pytest

import scripts.eval_heads_own_terms as eval_heads_own_terms
import src.config as config


class _MarkerReached(Exception):
    """Raised in place of pl.read_csv to prove argument resolution completed."""


def _raise_marker(*args, **kwargs):
    raise _MarkerReached("reached pl.read_csv")


class TestDefaultArgumentResolution:
    def test_all_defaults_resolve_to_production_paths_before_read_csv(self, monkeypatch):
        monkeypatch.setattr(eval_heads_own_terms.pl, "read_csv", _raise_marker)
        with pytest.raises(_MarkerReached):
            eval_heads_own_terms.main()

    def test_explicit_weights_and_cache_suffix_resolve_before_read_csv(self, monkeypatch, tmp_path):
        monkeypatch.setattr(eval_heads_own_terms.pl, "read_csv", _raise_marker)
        with pytest.raises(_MarkerReached):
            eval_heads_own_terms.main(
                cca_weights=tmp_path / "cca_doca_tuned.weights.h5",
                rel_weights=tmp_path / "relevance_tuned.weights.h5",
                us_weights=tmp_path / "us_classifier_full_tuned.weights.h5",
                cache_suffix="relevance_train_tuned",
                out_path=tmp_path / "eval_heads_own_terms_tuned.json",
            )

    def test_default_cache_suffix_constant(self):
        assert eval_heads_own_terms.DEFAULT_CACHE_SUFFIX == "relevance_train"

    def test_default_out_path_is_production_path(self):
        assert eval_heads_own_terms.DEFAULT_OUT == (
            config.CCA_DOCA_DIR / "experiments" / "eval_heads_own_terms.json"
        )

    def test_string_weights_paths_are_coerced_to_path(self, monkeypatch, tmp_path):
        """argparse hands main() plain strings; they must not blow up downstream
        Path-typed consumers (calibration_path_for_weights, apply_*_model)."""
        captured = {}

        def _capture_and_raise(*args, **kwargs):
            captured["called"] = True
            raise _MarkerReached()

        monkeypatch.setattr(eval_heads_own_terms.pl, "read_csv", _capture_and_raise)
        with pytest.raises(_MarkerReached):
            eval_heads_own_terms.main(
                cca_weights=str(tmp_path / "cca_tuned.weights.h5"),
                rel_weights=str(tmp_path / "rel_tuned.weights.h5"),
                us_weights=str(tmp_path / "us_tuned.weights.h5"),
                out_path=str(tmp_path / "out.json"),
            )
        assert captured["called"]
