# pattern: Functional Core

"""Unit and integration tests for src/calibration/sidecar.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.calibration.calibrator import PlattCalibrator
from src.calibration.sidecar import (
    calibration_path_for_weights,
    load_calibration,
    save_calibration,
)


class TestCalibrationPathForWeights:
    """Test calibration_path_for_weights function."""

    def test_weights_h5_to_calibration_json(self):
        """*.weights.h5 → *.calibration.json."""
        p = calibration_path_for_weights("model.weights.h5")
        assert p == Path("model.calibration.json")

    def test_weights_h5_with_directory(self):
        """Preserves directory path."""
        p = calibration_path_for_weights("/tmp/model.weights.h5")
        assert p == Path("/tmp/model.calibration.json")

    def test_weights_h5_with_stem(self):
        """Handles complex stems."""
        p = calibration_path_for_weights("dir/model_v2_best.weights.h5")
        assert p == Path("dir/model_v2_best.calibration.json")

    def test_non_h5_fallback(self):
        """Non-.weights.h5 files get .calibration.json appended."""
        p = calibration_path_for_weights("model.h5")
        assert p == Path("model.h5.calibration.json")

    def test_pathlib_input(self):
        """Accepts Path objects."""
        p = calibration_path_for_weights(Path("model.weights.h5"))
        assert isinstance(p, Path)
        assert p == Path("model.calibration.json")

    def test_string_input(self):
        """Accepts strings."""
        p = calibration_path_for_weights("model.weights.h5")
        assert isinstance(p, Path)


class TestSaveLoadRoundTrip:
    """Test save_calibration / load_calibration round-trip."""

    def test_save_creates_json_file(self):
        """save_calibration creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal = PlattCalibrator(A=1.5, B=-0.3, fit_population="test", n=100)
            path = Path(tmpdir) / "calibrator.calibration.json"
            save_calibration(cal, path)
            assert path.exists()
            assert path.suffix == ".json"

    def test_saved_json_contains_fields(self):
        """Saved JSON contains method, A, B, fit_population, n."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal = PlattCalibrator(A=1.5, B=-0.3, fit_population="ldc_val", n=100)
            path = Path(tmpdir) / "cal.calibration.json"
            save_calibration(cal, path)
            payload = json.loads(path.read_text())
            assert "method" in payload
            assert "A" in payload
            assert "B" in payload
            assert "fit_population" in payload
            assert "n" in payload

    def test_saved_json_values(self):
        """Saved JSON values match input calibrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal = PlattCalibrator(A=2.0, B=0.5, fit_population="test_pop", n=500)
            path = Path(tmpdir) / "cal.json"
            save_calibration(cal, path)
            payload = json.loads(path.read_text())
            assert payload["A"] == 2.0
            assert payload["B"] == 0.5
            assert payload["fit_population"] == "test_pop"
            assert payload["n"] == 500
            assert payload["method"] == "platt"

    def test_load_from_json(self):
        """load_calibration reconstructs calibrator from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_orig = PlattCalibrator(A=1.5, B=-0.3, fit_population="ldc", n=250)
            path = Path(tmpdir) / "cal.json"
            save_calibration(cal_orig, path)
            cal_loaded = load_calibration(path)
            assert cal_loaded.A == cal_orig.A
            assert cal_loaded.B == cal_orig.B
            assert cal_loaded.fit_population == cal_orig.fit_population
            assert cal_loaded.n == cal_orig.n
            assert cal_loaded.method == "platt"

    def test_roundtrip_transform_identical(self):
        """Loaded calibrator produces identical transform output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Fit on synthetic data
            logits_fit = np.array([-1.0, 0.0, 1.0, 2.0])
            labels_fit = np.array([0, 0, 1, 1])
            cal_orig = PlattCalibrator.fit(logits_fit, labels_fit, fit_population="test")

            # Save and load
            path = Path(tmpdir) / "cal.json"
            save_calibration(cal_orig, path)
            cal_loaded = load_calibration(path)

            # Compare transforms on a grid
            test_logits = np.linspace(-8, 8, 100)
            probs_orig = cal_orig.transform(test_logits)
            probs_loaded = cal_loaded.transform(test_logits)
            np.testing.assert_array_equal(probs_orig, probs_loaded)

    def test_roundtrip_with_path_string(self):
        """Round-trip works with string paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal = PlattCalibrator(A=1.0, B=0.0, fit_population="pop", n=100)
            path_str = str(Path(tmpdir) / "cal.json")
            save_calibration(cal, path_str)
            cal_loaded = load_calibration(path_str)
            assert cal_loaded.A == 1.0


class TestLoadCalibrationValidation:
    """Test load_calibration error handling."""

    def test_unsupported_method_raises(self):
        """Unsupported method value raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal.json"
            payload = {
                "method": "isotonic",  # Not supported
                "A": 1.0,
                "B": 0.0,
                "fit_population": "test",
                "n": 100,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="unsupported"):
                load_calibration(path)

    def test_missing_method_defaults_to_platt(self):
        """Missing method field defaults to 'platt'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal.json"
            payload = {
                # No "method" field
                "A": 1.0,
                "B": 0.0,
                "fit_population": "test",
                "n": 100,
            }
            path.write_text(json.dumps(payload))
            cal = load_calibration(path)
            assert cal.method == "platt"

    def test_missing_required_field_raises(self):
        """Missing required field (e.g., A) raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal.json"
            payload = {
                "method": "platt",
                # Missing "A"
                "B": 0.0,
                "fit_population": "test",
                "n": 100,
            }
            path.write_text(json.dumps(payload))
            with pytest.raises(KeyError):
                load_calibration(path)

    def test_invalid_json_raises(self):
        """Invalid JSON raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cal.json"
            path.write_text("not valid json {")
            with pytest.raises(Exception):  # json.JSONDecodeError
                load_calibration(path)


class TestAC43SaveLoadRoundTrip:
    """AC4.3: calibration params persist to .calibration.json and reload to identical transform."""

    def test_ac43_complete_roundtrip(self):
        """Fit → save → load → transform on grid produces identical results."""
        # Fit on synthetic data
        np.random.seed(42)
        n_pos, n_neg = 30, 70
        pos_logits = np.random.normal(1.0, 1.0, n_pos)
        neg_logits = np.random.normal(-1.0, 1.0, n_neg)
        fit_logits = np.concatenate([pos_logits, neg_logits])
        fit_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

        cal_orig = PlattCalibrator.fit(fit_logits, fit_labels, fit_population="synthetic_natural")

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.weights.h5.calibration.json"
            save_calibration(cal_orig, path)

            # Load and verify structure
            cal_loaded = load_calibration(path)
            assert cal_loaded.A == cal_orig.A
            assert cal_loaded.B == cal_orig.B
            assert cal_loaded.fit_population == "synthetic_natural"
            assert cal_loaded.n == len(fit_labels)

            # Verify transforms are identical on a logit grid
            test_logits = np.linspace(-8, 8, 100)
            probs_orig = cal_orig.transform(test_logits)
            probs_loaded = cal_loaded.transform(test_logits)
            np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-15)

    def test_calibration_path_for_weights_integration(self):
        """calibration_path_for_weights integrates correctly with save/load."""
        np.random.seed(42)
        logits = np.array([-1.0, 0.0, 1.0])
        labels = np.array([0, 0, 1])
        cal = PlattCalibrator.fit(logits, labels, fit_population="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = Path(tmpdir) / "model.weights.h5"
            cal_path = calibration_path_for_weights(weights_path)
            assert cal_path == Path(tmpdir) / "model.calibration.json"

            save_calibration(cal, cal_path)
            cal_loaded = load_calibration(cal_path)
            assert cal_loaded.A == cal.A
            assert cal_loaded.B == cal.B
