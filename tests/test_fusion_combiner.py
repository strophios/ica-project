# pattern: Functional Core

"""Unit and property-based tests for src/fusion/combiner.py."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.fusion.combiner import (
    FusionConfig,
    apply_logistic_combiner,
    combine_and,
    fit_logistic_combiner,
)


class TestCombineAnd:
    """Tests for combine_and function (calibrated-AND combiner)."""

    def test_and_basic_product(self):
        """AND combiner computes elementwise product."""
        p_cca = np.array([0.8, 0.6])
        p_rel = np.array([0.5, 0.4])
        result = combine_and(p_cca, p_rel)
        expected = np.array([0.4, 0.24])
        np.testing.assert_allclose(result, expected)

    def test_and_range_boundaries(self):
        """AND product of probabilities stays in [0, 1]."""
        p_cca = np.array([0.0, 0.5, 1.0, 0.99])
        p_rel = np.array([0.0, 0.5, 1.0, 0.5])
        result = combine_and(p_cca, p_rel)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_and_identity_property(self):
        """AND(p, 1.0) == p (one argument is identity)."""
        p = np.array([0.1, 0.5, 0.9])
        result = combine_and(p, np.ones(3))
        np.testing.assert_allclose(result, p)

    def test_and_absorbing_element(self):
        """AND(p, 0.0) == 0.0 (zero is absorbing)."""
        p = np.array([0.1, 0.5, 0.9])
        result = combine_and(p, np.zeros(3))
        np.testing.assert_allclose(result, np.zeros(3))

    def test_and_commutative(self):
        """AND(a, b) == AND(b, a)."""
        p_cca = np.array([0.8, 0.6, 0.3])
        p_rel = np.array([0.5, 0.4, 0.7])
        result_1 = combine_and(p_cca, p_rel)
        result_2 = combine_and(p_rel, p_cca)
        np.testing.assert_allclose(result_1, result_2)

    @given(
        p1=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=100,
        ),
        p2=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=100,
        ),
    )
    @settings(max_examples=100)
    def test_and_monotonicity_property(self, p1, p2):
        """AND is monotone: if p_cca increases, result increases."""
        if len(p1) != len(p2):
            return  # Skip if lengths don't match

        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)

        result_base = combine_and(p1, p2)

        # Increase each element of p1 slightly and verify result increases or stays same
        for i in range(len(p1)):
            p1_inc = p1.copy()
            p1_inc[i] = min(1.0, p1_inc[i] + 0.01)
            result_inc = combine_and(p1_inc, p2)
            # Product should increase or stay the same
            assert np.all(result_inc >= result_base - 1e-10)

    def test_and_list_input(self):
        """AND accepts Python lists."""
        p_cca = [0.8, 0.6]
        p_rel = [0.5, 0.4]
        result = combine_and(p_cca, p_rel)
        expected = np.array([0.4, 0.24])
        np.testing.assert_allclose(result, expected)

    def test_and_returns_1d_array(self):
        """AND returns 1-D array even if inputs are column vectors."""
        p_cca = np.array([[0.8], [0.6]])
        p_rel = np.array([[0.5], [0.4]])
        result = combine_and(p_cca, p_rel)
        assert result.ndim == 1
        assert len(result) == 2

    def test_and_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        p_cca = np.array([0.8, 0.6])
        p_rel = np.array([0.5])
        with pytest.raises(ValueError, match="length"):
            combine_and(p_cca, p_rel)


class TestFitLogisticCombiner:
    """Tests for fit_logistic_combiner function."""

    def test_fit_basic_2features(self):
        """Fit on 2-feature (CCA + rel) data."""
        scores = np.array([[-1.0, -0.5], [0.0, 0.0], [1.0, 1.5]])
        labels = np.array([0, 0, 1])
        model = fit_logistic_combiner(scores, labels)
        assert model.coef_.shape == (1, 2)
        assert isinstance(model.intercept_[0], (float, np.floating))

    def test_fit_basic_3features(self):
        """Fit on 3-feature (CCA + rel + US) data."""
        scores = np.array([
            [-1.0, -0.5, 0.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.5, -0.3],
        ])
        labels = np.array([0, 0, 1])
        model = fit_logistic_combiner(scores, labels)
        assert model.coef_.shape == (1, 3)

    def test_fit_deterministic(self):
        """Same data + fixed random_state yields identical coefs."""
        scores = np.array([[-1.0, -0.5], [0.0, 0.0], [1.0, 1.5], [2.0, 1.0]])
        labels = np.array([0, 0, 1, 1])
        model1 = fit_logistic_combiner(scores, labels, random_state=42)
        model2 = fit_logistic_combiner(scores, labels, random_state=42)
        # Convert sparse matrices to dense for comparison
        coef1 = model1.coef_ if isinstance(model1.coef_, np.ndarray) else model1.coef_.toarray()
        coef2 = model2.coef_ if isinstance(model2.coef_, np.ndarray) else model2.coef_.toarray()
        np.testing.assert_array_equal(coef1, coef2)
        intercept1 = model1.intercept_ if isinstance(model1.intercept_, np.ndarray) else model1.intercept_.toarray()
        intercept2 = model2.intercept_ if isinstance(model2.intercept_, np.ndarray) else model2.intercept_.toarray()
        np.testing.assert_array_equal(intercept1, intercept2)

    def test_fit_different_random_states_may_differ(self):
        """Different random_state values may yield different results (no guarantee, but allowed)."""
        scores = np.array([[-1.0, -0.5], [0.0, 0.0], [1.0, 1.5]])
        labels = np.array([0, 0, 1])
        # Just verify both states work; don't assert inequality (LR may converge to same solution)
        model1 = fit_logistic_combiner(scores, labels, random_state=42)
        model2 = fit_logistic_combiner(scores, labels, random_state=999)
        assert model1.coef_.shape == model2.coef_.shape

    def test_fit_rejects_1_feature(self):
        """Fewer than 2 features raises ValueError."""
        scores = np.array([[-1.0], [0.0], [1.0]])
        labels = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="2 or 3 columns"):
            fit_logistic_combiner(scores, labels)

    def test_fit_rejects_4_features(self):
        """More than 3 features raises ValueError."""
        scores = np.array([
            [-1.0, -0.5, 0.5, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.5, -0.3, 0.2],
        ])
        labels = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="2 or 3 columns"):
            fit_logistic_combiner(scores, labels)

    def test_fit_rejects_1d_array(self):
        """1-D scores array raises ValueError."""
        scores = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="2-D"):
            fit_logistic_combiner(scores, labels)

    def test_fit_length_mismatch_raises(self):
        """Mismatched scores/labels length raises ValueError."""
        scores = np.array([[-1.0, -0.5], [0.0, 0.0]])
        labels = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="length mismatch"):
            fit_logistic_combiner(scores, labels)

    def test_fit_accepts_list_input(self):
        """scores and labels can be Python lists."""
        scores = [[-1.0, -0.5], [0.0, 0.0], [1.0, 1.5]]
        labels = [0, 0, 1]
        model = fit_logistic_combiner(scores, labels)
        assert model.coef_.shape == (1, 2)

    def test_fit_coefficient_count_2(self):
        """Fitted model has exactly 2 coefficients for 2-feature input."""
        scores = np.array([[-1.0, -0.5], [0.0, 0.0], [1.0, 1.5]])
        labels = np.array([0, 0, 1])
        model = fit_logistic_combiner(scores, labels)
        assert model.coef_.shape[1] == 2

    def test_fit_coefficient_count_3(self):
        """Fitted model has exactly 3 coefficients for 3-feature input."""
        scores = np.array([
            [-1.0, -0.5, 0.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.5, -0.3],
        ])
        labels = np.array([0, 0, 1])
        model = fit_logistic_combiner(scores, labels)
        assert model.coef_.shape[1] == 3


class TestApplyLogisticCombiner:
    """Tests for apply_logistic_combiner function."""

    def test_apply_with_model_object(self):
        """Apply combiner using fitted LogisticRegression object."""
        scores = np.array([[-1.0, -0.5], [0.0, 0.0], [1.0, 1.5]])
        labels = np.array([0, 0, 1])
        model = fit_logistic_combiner(scores, labels)
        test_scores = np.array([[0.5, 0.5], [-0.5, -0.5]])
        probs = apply_logistic_combiner(model, test_scores)
        assert probs.shape == (2,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_apply_with_coef_tuple(self):
        """Apply combiner using (coef, intercept) tuple."""
        coef = np.array([0.5, 0.3])
        intercept = -0.2
        test_scores = np.array([[0.0, 0.0], [1.0, 1.0]])
        probs = apply_logistic_combiner((coef, intercept), test_scores)
        assert probs.shape == (2,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_apply_probs_in_range(self):
        """Applied probabilities always in [0, 1]."""
        scores = np.array([
            [-10.0, -10.0],
            [0.0, 0.0],
            [10.0, 10.0],
        ])
        labels = np.array([0, 0, 1])
        model = fit_logistic_combiner(scores, labels)
        test_scores = np.array([
            [-100.0, -100.0],
            [100.0, 100.0],
        ])
        probs = apply_logistic_combiner(model, test_scores)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_apply_with_list_scores(self):
        """Apply accepts list scores."""
        coef = np.array([0.5, 0.3])
        intercept = -0.2
        test_scores = [[0.0, 0.0], [1.0, 1.0]]
        probs = apply_logistic_combiner((coef, intercept), test_scores)
        assert probs.shape == (2,)

    def test_apply_coef_shape_mismatch_raises(self):
        """Coef shape mismatch with scores raises ValueError."""
        coef = np.array([0.5, 0.3])
        intercept = -0.2
        test_scores = np.array([[0.0, 0.0, 0.0]])  # 3 features, but coef has 2
        with pytest.raises(ValueError, match="shape mismatch"):
            apply_logistic_combiner((coef, intercept), test_scores)


class TestFusionConfig:
    """Tests for FusionConfig dataclass validation."""

    def test_config_product_valid(self):
        """Valid product config constructs without error."""
        cfg = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
        )
        assert cfg.combine == "product"
        assert cfg.coefs is None

    def test_config_logreg_valid_3coefs_no_us(self):
        """Valid logreg config with 3 coefs (2 slopes + intercept, no US)."""
        cfg = FusionConfig(
            gate_threshold=0.5,
            combine="logreg",
            coefs=[0.5, 0.3, -0.1],  # slope_cca, slope_rel, intercept
            score_space="prob",
            includes_us=False,
        )
        assert cfg.combine == "logreg"
        assert cfg.coefs is not None and len(cfg.coefs) == 3

    def test_config_logreg_valid_4coefs_with_us(self):
        """Valid logreg config with 4 coefs (3 slopes + intercept, with US)."""
        cfg = FusionConfig(
            gate_threshold=0.5,
            combine="logreg",
            coefs=[0.5, 0.3, 0.2, -0.1],  # slope_cca, slope_rel, slope_us, intercept
            score_space="prob",
            includes_us=True,
        )
        assert cfg.combine == "logreg"
        assert cfg.coefs is not None and len(cfg.coefs) == 4

    def test_config_rejects_unknown_combine(self):
        """Unknown combine value raises ValueError."""
        with pytest.raises(ValueError, match="product.*logreg"):
            FusionConfig(
                gate_threshold=0.5,
                combine=cast(Any, "invalid"),  # type: ignore[arg-type]  # intentionally invalid
                coefs=None,
                score_space="prob",
                includes_us=False,
            )

    def test_config_rejects_logreg_without_coefs(self):
        """logreg without coefs raises ValueError."""
        with pytest.raises(ValueError, match="coefs.*required"):
            FusionConfig(
                gate_threshold=0.5,
                combine="logreg",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )

    def test_config_rejects_product_with_coefs(self):
        """product with coefs raises ValueError."""
        with pytest.raises(ValueError, match="coefs.*None"):
            FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=[0.5, 0.3],
                score_space="prob",
                includes_us=False,
            )

    def test_config_rejects_wrong_coef_count_3vs4(self):
        """Wrong coef count (3 instead of 4) raises ValueError when includes_us=True."""
        with pytest.raises(ValueError, match="4 elements"):
            FusionConfig(
                gate_threshold=0.5,
                combine="logreg",
                coefs=[0.5, 0.3, 0.2],  # Only 3, but includes_us=True requires 4
                score_space="prob",
                includes_us=True,
            )

    def test_config_rejects_wrong_coef_count_4vs3(self):
        """Wrong coef count (4 instead of 3) raises ValueError when includes_us=False."""
        with pytest.raises(ValueError, match="3 elements"):
            FusionConfig(
                gate_threshold=0.5,
                combine="logreg",
                coefs=[0.5, 0.3, 0.2, -0.1],  # 4, but includes_us=False requires 3
                score_space="prob",
                includes_us=False,
            )

    def test_config_rejects_invalid_threshold_below_zero(self):
        """gate_threshold < 0 raises ValueError."""
        with pytest.raises(ValueError, match="gate_threshold.*0.*1"):
            FusionConfig(
                gate_threshold=-0.1,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )

    def test_config_rejects_invalid_threshold_above_one(self):
        """gate_threshold > 1 raises ValueError."""
        with pytest.raises(ValueError, match="gate_threshold.*0.*1"):
            FusionConfig(
                gate_threshold=1.1,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
            )

    def test_config_accepts_threshold_boundaries(self):
        """gate_threshold of exactly 0.0 and 1.0 are valid."""
        cfg_0 = FusionConfig(
            gate_threshold=0.0,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
        )
        cfg_1 = FusionConfig(
            gate_threshold=1.0,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
        )
        assert cfg_0.gate_threshold == 0.0
        assert cfg_1.gate_threshold == 1.0

    def test_config_rejects_unknown_score_space(self):
        """Unknown score_space raises ValueError."""
        with pytest.raises(ValueError, match="prob.*logit"):
            FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space=cast(Any, "unknown"),  # type: ignore[arg-type]  # intentionally invalid
                includes_us=False,
            )

    def test_config_is_frozen(self):
        """FusionConfig is frozen (immutable)."""
        cfg = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
        )
        # FusionConfig should be frozen per dataclass definition
        assert dataclasses.is_dataclass(cfg)
        # Verify it's actually frozen by checking __dataclass_fields__
        assert cfg.__dataclass_params__.frozen is True

    def test_config_composed_platt_optional(self):
        """composed_platt defaults to None; can be provided as a 2-element list."""
        # Without composed_platt
        cfg_no_platt = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
        )
        assert cfg_no_platt.composed_platt is None

        # With valid composed_platt
        cfg_with_platt = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
            composed_platt=[0.8, -0.5],
        )
        assert cfg_with_platt.composed_platt == [0.8, -0.5]

    def test_config_rejects_invalid_composed_platt_length(self):
        """composed_platt must have exactly 2 elements if present."""
        with pytest.raises(ValueError, match="composed_platt.*2-element"):
            FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
                composed_platt=[0.8],  # Only 1 element
            )

        with pytest.raises(ValueError, match="composed_platt.*2-element"):
            FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
                composed_platt=[0.8, -0.5, 0.1],  # 3 elements
            )

    def test_config_head_calibrators_optional(self):
        """head_calibrators defaults to None; can be provided as a dict."""
        # Without head_calibrators
        cfg_no_cals = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
        )
        assert cfg_no_cals.head_calibrators is None

        # With valid head_calibrators
        cals = {"cca": "cca_stem", "rel": "rel_stem", "us": "us_stem"}
        cfg_with_cals = FusionConfig(
            gate_threshold=0.5,
            combine="product",
            coefs=None,
            score_space="prob",
            includes_us=False,
            head_calibrators=cals,
        )
        assert cfg_with_cals.head_calibrators == cals

    def test_config_rejects_invalid_head_calibrators_type(self):
        """head_calibrators must be a dict if present."""
        with pytest.raises(ValueError, match="head_calibrators.*dict"):
            FusionConfig(
                gate_threshold=0.5,
                combine="product",
                coefs=None,
                score_space="prob",
                includes_us=False,
                head_calibrators=cast(Any, "not_a_dict"),  # type: ignore[arg-type]  # intentionally invalid
            )
