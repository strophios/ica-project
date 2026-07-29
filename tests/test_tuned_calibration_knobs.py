# pattern: Imperative Shell
# Reason: monkeypatches module-level I/O entry points to test control flow
# without touching real caches/weights/sidecars.
"""Tests for the tuned-cache retrain knobs on the three calibrators
(`calibrate_us_filter.py`, `calibrate_cca.py`, `calibrate_relevance.py`).

Same strategy as `test_tuned_cache_knobs.py`: each `main()` is exercised only
up to its first post-guard I/O call (monkeypatched to a marker exception), to
prove the guard fires/doesn't-fire on the right suffix x weights-path
combinations without needing real caches or trained artifacts. Also verifies
`calibration_path_for_weights` derives the sidecar path next to whatever
weights path is given (not hardcoded to the production stem).
"""

from __future__ import annotations


import pytest

import src.calibrate_us_filter as calibrate_us_filter
import src.calibrate_cca as calibrate_cca
import src.calibrate_relevance as calibrate_relevance
from src.calibration.sidecar import calibration_path_for_weights


class _MarkerReached(Exception):
    """Raised by a monkeypatched I/O call to prove control flow got past the guard."""


def _raise_marker(*args, **kwargs):
    raise _MarkerReached("reached first post-guard I/O call")


# ---------------------------------------------------------------------------
# calibrate_us_filter.py
# ---------------------------------------------------------------------------


class TestCalibrateUsFilterGuard:
    def test_default_suffix_default_weights_passes_guard(self, monkeypatch):
        monkeypatch.setattr(
            calibrate_us_filter.us_config.UsRunConfig, "from_json",
            staticmethod(_raise_marker),
        )
        with pytest.raises(_MarkerReached):
            calibrate_us_filter.main()

    def test_tuned_suffix_default_weights_raises(self, monkeypatch):
        monkeypatch.setattr(
            calibrate_us_filter.us_config.UsRunConfig, "from_json",
            staticmethod(_raise_marker),
        )
        with pytest.raises(ValueError, match="production"):
            calibrate_us_filter.main(suffix="us_train_ldc_tuned")

    def test_tuned_suffix_with_distinct_weights_passes_guard(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            calibrate_us_filter.us_config.UsRunConfig, "from_json",
            staticmethod(_raise_marker),
        )
        with pytest.raises(_MarkerReached):
            calibrate_us_filter.main(
                suffix="us_train_ldc_tuned",
                weights_path=tmp_path / "us_classifier_full_tuned.weights.h5",
            )

    def test_default_suffix_constant_matches_trainer(self):
        import src.run_us_features as run_us_features

        assert calibrate_us_filter.DEFAULT_SUFFIX == run_us_features.DEFAULT_SUFFIX


# ---------------------------------------------------------------------------
# calibrate_cca.py
# ---------------------------------------------------------------------------


class TestCalibrateCcaGuard:
    def test_default_suffix_default_weights_passes_guard(self, monkeypatch):
        monkeypatch.setattr(calibrate_cca, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            calibrate_cca.main()

    def test_tuned_suffix_default_weights_raises(self, monkeypatch):
        monkeypatch.setattr(calibrate_cca, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            calibrate_cca.main(suffix="train250k_tuned")

    def test_tuned_suffix_with_distinct_weights_passes_guard(self, monkeypatch, tmp_path):
        monkeypatch.setattr(calibrate_cca, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            calibrate_cca.main(
                suffix="train250k_tuned",
                weights_path=tmp_path / "cca_doca_tuned.weights.h5",
            )

    def test_default_suffix_constant_matches_trainer(self):
        import src.run_cca_doca as run_cca_doca

        assert calibrate_cca.DEFAULT_SUFFIX == run_cca_doca.DEFAULT_SUFFIX


# ---------------------------------------------------------------------------
# calibrate_relevance.py
# ---------------------------------------------------------------------------


class TestCalibrateRelevanceGuard:
    def test_default_suffix_default_weights_passes_guard(self, monkeypatch):
        monkeypatch.setattr(calibrate_relevance, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            calibrate_relevance.main()

    def test_tuned_suffix_default_weights_raises(self, monkeypatch):
        monkeypatch.setattr(calibrate_relevance, "load_cache", _raise_marker)
        with pytest.raises(ValueError, match="production"):
            calibrate_relevance.main(suffix="relevance_train_tuned")

    def test_tuned_suffix_with_distinct_weights_passes_guard(self, monkeypatch, tmp_path):
        monkeypatch.setattr(calibrate_relevance, "load_cache", _raise_marker)
        with pytest.raises(_MarkerReached):
            calibrate_relevance.main(
                suffix="relevance_train_tuned",
                weights_path=tmp_path / "relevance_tuned.weights.h5",
            )

    def test_default_suffix_constant_matches_trainer(self):
        import src.run_relevance as run_relevance

        assert calibrate_relevance.DEFAULT_SUFFIX == run_relevance.DEFAULT_SUFFIX


# ---------------------------------------------------------------------------
# calibration_path_for_weights: verify it derives next to ANY weights path,
# tuned or production (Task 3's explicit "verify" ask).
# ---------------------------------------------------------------------------


class TestCalibrationSidecarDerivesNextToTunedWeights:
    def test_tuned_weights_get_tuned_sidecar(self, tmp_path):
        tuned_weights = tmp_path / "cca_doca_tuned.weights.h5"
        sidecar = calibration_path_for_weights(tuned_weights)
        assert sidecar == tmp_path / "cca_doca_tuned.calibration.json"
        assert sidecar != calibration_path_for_weights(tmp_path / "cca_doca.weights.h5")
