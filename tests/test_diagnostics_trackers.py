# pattern: Functional Core

"""Unit + property-based tests for src/diagnostics/trackers.py."""

from __future__ import annotations

import keras
import numpy as np  # noqa: F401  # used by BatchLabelBalanceTracker property tests (Phase 1 Task 6)
import pytest
import tensorflow as tf
from hypothesis import given, settings
from hypothesis import strategies as st

from src.diagnostics.trackers import PerGroupGradNormTracker
from src.model_setup.assembly import _default_group_fn


def _group_fn_by_first_path_segment(var):
    """Test helper for unit tests: split var.name on / and take the first
    segment as the group name.

    Bare tf.Variable fixtures expose the group via .name (e.g., "head/w1:0");
    real Keras layer variables expose it via .path (e.g., "cca/kernel").
    See test_composes_with_real_default_group_fn for production grouping.
    """
    # var.name is like "head/w1:0", so split on / and take the first part
    return var.name.split("/", 1)[0]


def _make_two_var_setup(group_a_norm, group_b_norm):
    """Two trainable vars + matching gradients with known norms.

    variables[0] is in group 'a', variables[1] is in group 'b'.
    """
    var_a = tf.Variable(tf.zeros([4]), name="a/w")
    var_b = tf.Variable(tf.zeros([4]), name="b/w")
    grad_a = tf.constant([group_a_norm, 0.0, 0.0, 0.0])
    grad_b = tf.constant([group_b_norm, 0.0, 0.0, 0.0])
    return [var_a, var_b], [grad_a, grad_b]


class TestPerGroupGradNormConstruction:
    def test_name_pattern_mean(self):
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        assert t.name == "grad_norm/head/mean"

    def test_name_pattern_max(self):
        t = PerGroupGradNormTracker(group_name="encoder", aggregation="max")
        assert t.name == "grad_norm/encoder/max"

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="aggregation"):
            PerGroupGradNormTracker(group_name="head", aggregation="median")

    def test_empty_group_name_raises(self):
        with pytest.raises(ValueError, match="group_name"):
            PerGroupGradNormTracker(group_name="", aggregation="mean")

    def test_result_default_zero(self):
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        assert float(t.result()) == 0.0


class TestPerGroupGradNormUpdateMean:
    def test_single_group_mean_of_norms(self):
        var_h1 = tf.Variable(tf.zeros([4]), name="head/w1")
        var_h2 = tf.Variable(tf.zeros([4]), name="head/w2")
        grad_h1 = tf.constant([3.0, 0.0, 0.0, 0.0])  # norm 3.0
        grad_h2 = tf.constant([0.0, 4.0, 0.0, 0.0])  # norm 4.0
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state(
            [grad_h1, grad_h2], [var_h1, var_h2], _group_fn_by_first_path_segment
        )
        assert float(t.result()) == pytest.approx(3.5, rel=1e-5)

    def test_filters_other_groups(self):
        variables, gradients = _make_two_var_setup(group_a_norm=5.0, group_b_norm=100.0)
        t = PerGroupGradNormTracker(group_name="a", aggregation="mean")
        t.update_state(gradients, variables, _group_fn_by_first_path_segment)
        assert float(t.result()) == pytest.approx(5.0, rel=1e-5)

    def test_empty_group_reports_zero(self):
        variables, gradients = _make_two_var_setup(5.0, 7.0)
        t = PerGroupGradNormTracker(group_name="nonexistent", aggregation="mean")
        t.update_state(gradients, variables, _group_fn_by_first_path_segment)
        assert float(t.result()) == 0.0

    def test_none_gradient_skipped(self):
        var_a = tf.Variable(tf.zeros([4]), name="head/a")
        var_b = tf.Variable(tf.zeros([4]), name="head/b")
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state(
            [None, tf.constant([3.0, 0.0, 0.0, 0.0])],
            [var_a, var_b],
            _group_fn_by_first_path_segment,
        )
        assert float(t.result()) == pytest.approx(3.0, rel=1e-5)


class TestPerGroupGradNormUpdateMax:
    def test_max_of_norms_within_step(self):
        var_h1 = tf.Variable(tf.zeros([4]), name="head/w1")
        var_h2 = tf.Variable(tf.zeros([4]), name="head/w2")
        grad_h1 = tf.constant([3.0, 0.0, 0.0, 0.0])
        grad_h2 = tf.constant([0.0, 4.0, 0.0, 0.0])
        t = PerGroupGradNormTracker(group_name="head", aggregation="max")
        t.update_state(
            [grad_h1, grad_h2], [var_h1, var_h2], _group_fn_by_first_path_segment
        )
        assert float(t.result()) == pytest.approx(4.0, rel=1e-5)

    def test_max_running_across_steps(self):
        var = tf.Variable(tf.zeros([2]), name="head/w")
        t = PerGroupGradNormTracker(group_name="head", aggregation="max")
        t.update_state([tf.constant([3.0, 0.0])], [var], _group_fn_by_first_path_segment)
        t.update_state([tf.constant([1.0, 0.0])], [var], _group_fn_by_first_path_segment)
        t.update_state([tf.constant([5.0, 0.0])], [var], _group_fn_by_first_path_segment)
        assert float(t.result()) == pytest.approx(5.0, rel=1e-5)

    def test_reset_state_clears_max(self):
        var = tf.Variable(tf.zeros([2]), name="head/w")
        t = PerGroupGradNormTracker(group_name="head", aggregation="max")
        t.update_state([tf.constant([5.0, 0.0])], [var], _group_fn_by_first_path_segment)
        t.reset_state()
        assert float(t.result()) == 0.0


class TestPerGroupGradNormSparse:
    def test_indexed_slices_norm(self):
        var = tf.Variable(tf.zeros([10, 4]), name="head/embedding")
        values = tf.constant([[3.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
        sparse_grad = tf.IndexedSlices(
            values=values, indices=tf.constant([1, 7]), dense_shape=[10, 4]
        )
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state([sparse_grad], [var], _group_fn_by_first_path_segment)
        # Norm of the values tensor = sqrt(3^2 + 4^2) = 5.0
        assert float(t.result()) == pytest.approx(5.0, rel=1e-5)

    def test_mixed_dense_and_sparse_in_same_group(self):
        var_dense = tf.Variable(tf.zeros([4]), name="head/dense")
        var_emb = tf.Variable(tf.zeros([10, 4]), name="head/embedding")
        grad_dense = tf.constant([3.0, 0.0, 0.0, 0.0])  # norm 3
        grad_sparse = tf.IndexedSlices(
            values=tf.constant([[0.0, 4.0, 0.0, 0.0]]),
            indices=tf.constant([0]),
            dense_shape=[10, 4],
        )  # norm 4
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state(
            [grad_dense, grad_sparse],
            [var_dense, var_emb],
            _group_fn_by_first_path_segment,
        )
        assert float(t.result()) == pytest.approx(3.5, rel=1e-5)

    def test_indexed_slices_duplicate_indices_dense_equivalent(self):
        """IndexedSlices with duplicate indices: norm reflects summed dense-equiv."""
        var = tf.Variable(tf.zeros([10, 4]), name="head/embedding")
        # Two rows both targeting index 0; they sum: [3, 0, 0, 0] + [4, 0, 0, 0] = [7, 0, 0, 0]
        # Dense-equivalent norm is 7.0, NOT 5.0 (the norm of separate values).
        sparse_grad = tf.IndexedSlices(
            values=tf.constant([[3.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]),
            indices=tf.constant([0, 0]),
            dense_shape=[10, 4],
        )
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state([sparse_grad], [var], _group_fn_by_first_path_segment)
        assert float(t.result()) == pytest.approx(7.0, rel=1e-5)


# Realistic positive gradient norms. max_value=1e2 keeps float32 rounding
# within meaningful tolerance (rel=1e-4, abs=1e-5) across accumulation.
norm_lists = st.lists(
    st.floats(min_value=0.0, max_value=1e2, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=10,
)


def _build_grads_with_norms(norms):
    """Trainable vars + 1-D gradients all in group 'head' with the given norms."""
    variables = [
        tf.Variable(tf.zeros([1]), name=f"head/w{i}") for i in range(len(norms))
    ]
    gradients = [tf.constant([n]) for n in norms]
    return variables, gradients


class TestPerGroupGradNormProperties:
    @given(norm_lists)
    @settings(max_examples=50, deadline=None)
    def test_mean_equals_sum_div_count(self, norms):
        variables, gradients = _build_grads_with_norms(norms)
        t = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        t.update_state(gradients, variables, _group_fn_by_first_path_segment)
        expected = sum(norms) / len(norms)
        assert float(t.result()) == pytest.approx(expected, rel=1e-4, abs=1e-5)

    @given(norm_lists)
    @settings(max_examples=50, deadline=None)
    def test_permutation_invariance_within_group(self, norms):
        va, ga = _build_grads_with_norms(norms)
        vb, gb = _build_grads_with_norms(list(reversed(norms)))
        ta = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        tb = PerGroupGradNormTracker(group_name="head", aggregation="mean")
        ta.update_state(ga, va, _group_fn_by_first_path_segment)
        tb.update_state(gb, vb, _group_fn_by_first_path_segment)
        assert float(ta.result()) == pytest.approx(float(tb.result()), rel=1e-5)

    @given(st.lists(norm_lists, min_size=1, max_size=5))
    @settings(max_examples=30, deadline=None)
    def test_max_monotone_non_decreasing(self, norm_sequences):
        var = tf.Variable(tf.zeros([1]), name="head/w0")
        t = PerGroupGradNormTracker(group_name="head", aggregation="max")
        prev = 0.0
        for norms in norm_sequences:
            t.update_state(
                [tf.constant([norms[0]])], [var], _group_fn_by_first_path_segment
            )
            current = float(t.result())
            assert current >= prev
            prev = current


class TestPerGroupGradNormProductionGroupFn:
    """Test tracker composition with real Keras layer grouping contract."""

    def test_composes_with_real_default_group_fn(self):
        """Production grouping: real Keras layers expose group via .path."""
        # Build two real Keras Dense layers with distinct group names.
        layer_cca = keras.layers.Dense(8, name="cca")
        layer_other = keras.layers.Dense(8, name="other")

        # Build them on sample inputs so they have trainable variables.
        layer_cca.build((None, 3))
        layer_other.build((None, 3))

        # Collect trainable variables. Real Keras variables have .path like "cca/kernel".
        variables = layer_cca.trainable_variables + layer_other.trainable_variables
        assert len(variables) == 4  # 2 layers × (kernel + bias)

        # Verify that .path attribute exists and groups are correct.
        cca_vars = [v for v in variables if _default_group_fn(v) == "cca"]
        other_vars = [v for v in variables if _default_group_fn(v) == "other"]
        assert len(cca_vars) == 2
        assert len(other_vars) == 2

        # Create matching gradients (all in "cca" group have norm 3.0).
        grads = [tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
                 if _default_group_fn(v) == "cca" else tf.ones_like(v)
                 for v in variables]

        # Track the "cca" group only.
        tracker = PerGroupGradNormTracker(group_name="cca", aggregation="mean")
        tracker.update_state(grads, variables, _default_group_fn)

        # Tracker should report a positive norm for the tracked group.
        result = float(tracker.result())
        assert result > 0.0, "cca group should have non-zero norm"

        # Track a non-existent group; should report zero.
        tracker_missing = PerGroupGradNormTracker(
            group_name="nonexistent", aggregation="mean"
        )
        tracker_missing.update_state(grads, variables, _default_group_fn)
        assert float(tracker_missing.result()) == 0.0


# Task 4: GradientFiniteTracker
from src.diagnostics.trackers import (
    GradientFiniteTracker,
    LossComponentTracker,
    BatchLabelBalanceTracker,
)


class TestGradientFiniteTracker:
    def test_name(self):
        assert GradientFiniteTracker().name == "grad_overflow_rate"

    def test_all_finite_rate_zero(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([1.0, 2.0]), tf.constant([3.0])], None, None)
        assert float(t.result()) == 0.0

    def test_nan_increments_rate(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([float("nan"), 1.0])], None, None)
        assert float(t.result()) == pytest.approx(1.0)

    def test_inf_increments_rate(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([float("inf"), 1.0])], None, None)
        assert float(t.result()) == pytest.approx(1.0)

    def test_mixed_steps_average_correctly(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([1.0])], None, None)             # finite
        t.update_state([tf.constant([float("nan")])], None, None)    # overflow
        t.update_state([tf.constant([1.0])], None, None)             # finite
        t.update_state([tf.constant([1.0])], None, None)             # finite
        assert float(t.result()) == pytest.approx(0.25, rel=1e-5)

    def test_ignores_none_gradients(self):
        t = GradientFiniteTracker()
        t.update_state([None, tf.constant([1.0])], None, None)
        assert float(t.result()) == 0.0

    def test_handles_indexed_slices(self):
        t = GradientFiniteTracker()
        slices = tf.IndexedSlices(
            values=tf.constant([[float("nan")]]),
            indices=tf.constant([0]),
            dense_shape=[3, 1],
        )
        t.update_state([slices], None, None)
        assert float(t.result()) == pytest.approx(1.0)

    def test_reset_state(self):
        t = GradientFiniteTracker()
        t.update_state([tf.constant([float("nan")])], None, None)
        t.reset_state()
        assert float(t.result()) == 0.0


class TestGradientFiniteProperties:
    @given(st.lists(st.booleans(), min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_rate_in_zero_one(self, overflow_per_step):
        t = GradientFiniteTracker()
        for is_overflow in overflow_per_step:
            grad = tf.constant([float("nan")] if is_overflow else [1.0])
            t.update_state([grad], None, None)
        assert 0.0 <= float(t.result()) <= 1.0

    @given(st.lists(st.booleans(), min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_rate_matches_fraction(self, overflow_per_step):
        t = GradientFiniteTracker()
        for is_overflow in overflow_per_step:
            grad = tf.constant([float("nan")] if is_overflow else [1.0])
            t.update_state([grad], None, None)
        expected = sum(overflow_per_step) / len(overflow_per_step)
        assert float(t.result()) == pytest.approx(expected, rel=1e-5, abs=1e-6)
