# pattern: Imperative Shell
# Reason: exercises a Functional Core function; no I/O in the tests themselves,
# but grouped as a shell test module per project test-layout convention.
"""Tests for `src.artifact_guard.check_no_production_overwrite`.

Covers the guard's decision table (suffix x weights-path combinations), path-type
equivalence (str vs Path), and the error message contents an operator needs to
self-correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.artifact_guard import check_no_production_overwrite


PROD_SUFFIX = "train250k"
PROD_WEIGHTS = Path("/data/cca_doca/cca_doca.weights.h5")
TUNED_SUFFIX = "train250k_tuned"
TUNED_WEIGHTS = Path("/data/cca_doca/cca_doca_tuned.weights.h5")


class TestNoRaiseCases:
    """Combinations the guard must let through."""

    def test_default_suffix_default_weights_is_fine(self):
        """The everyday production run: nothing to guard against."""
        check_no_production_overwrite(
            cache_suffix=PROD_SUFFIX,
            production_cache_suffix=PROD_SUFFIX,
            weights_path=PROD_WEIGHTS,
            production_weights_path=PROD_WEIGHTS,
            artifact_label="CCA",
        )

    def test_default_suffix_with_distinct_weights_is_fine(self):
        """Production cache, but writing an experiment weights path -- fine
        (e.g. a hyperparameter sweep that never touches the tuned cache)."""
        check_no_production_overwrite(
            cache_suffix=PROD_SUFFIX,
            production_cache_suffix=PROD_SUFFIX,
            weights_path=TUNED_WEIGHTS,
            production_weights_path=PROD_WEIGHTS,
            artifact_label="CCA",
        )

    def test_tuned_suffix_with_distinct_weights_is_fine(self):
        """The intended tuned-retrain shape: non-default cache, non-production weights."""
        check_no_production_overwrite(
            cache_suffix=TUNED_SUFFIX,
            production_cache_suffix=PROD_SUFFIX,
            weights_path=TUNED_WEIGHTS,
            production_weights_path=PROD_WEIGHTS,
            artifact_label="CCA",
        )


class TestRaiseCases:
    """Combinations the guard must refuse."""

    def test_tuned_suffix_with_production_weights_raises(self):
        with pytest.raises(ValueError, match="production"):
            check_no_production_overwrite(
                cache_suffix=TUNED_SUFFIX,
                production_cache_suffix=PROD_SUFFIX,
                weights_path=PROD_WEIGHTS,
                production_weights_path=PROD_WEIGHTS,
                artifact_label="CCA",
            )

    def test_any_non_default_suffix_with_production_weights_raises(self):
        """Not just '_tuned' -- ANY suffix != the production default triggers it."""
        with pytest.raises(ValueError):
            check_no_production_overwrite(
                cache_suffix="some_other_experiment_cache",
                production_cache_suffix=PROD_SUFFIX,
                weights_path=PROD_WEIGHTS,
                production_weights_path=PROD_WEIGHTS,
                artifact_label="US filter",
            )

    def test_error_message_names_the_artifact_and_paths(self):
        """The message must be actionable: name the artifact, the suffix, and
        both paths, so an operator can fix the invocation without re-reading
        the source."""
        with pytest.raises(ValueError) as exc_info:
            check_no_production_overwrite(
                cache_suffix=TUNED_SUFFIX,
                production_cache_suffix=PROD_SUFFIX,
                weights_path=PROD_WEIGHTS,
                production_weights_path=PROD_WEIGHTS,
                artifact_label="relevance",
            )
        message = str(exc_info.value)
        assert "relevance" in message
        assert TUNED_SUFFIX in message
        assert str(PROD_WEIGHTS) in message


class TestPathTypeEquivalence:
    """str and Path inputs must compare equal where the underlying path is equal."""

    def test_str_weights_path_matches_path_production(self):
        with pytest.raises(ValueError):
            check_no_production_overwrite(
                cache_suffix=TUNED_SUFFIX,
                production_cache_suffix=PROD_SUFFIX,
                weights_path=str(PROD_WEIGHTS),  # str, not Path
                production_weights_path=PROD_WEIGHTS,
                artifact_label="CCA",
            )

    def test_str_production_weights_path_matches_path_weights(self):
        with pytest.raises(ValueError):
            check_no_production_overwrite(
                cache_suffix=TUNED_SUFFIX,
                production_cache_suffix=PROD_SUFFIX,
                weights_path=PROD_WEIGHTS,
                production_weights_path=str(PROD_WEIGHTS),  # str, not Path
                artifact_label="CCA",
            )

    def test_both_str_still_raises(self):
        with pytest.raises(ValueError):
            check_no_production_overwrite(
                cache_suffix=TUNED_SUFFIX,
                production_cache_suffix=PROD_SUFFIX,
                weights_path=str(PROD_WEIGHTS),
                production_weights_path=str(PROD_WEIGHTS),
                artifact_label="CCA",
            )
