# pattern: Functional Core

"""Unit + property-based tests for src/diagnostics/trackers.py."""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from src.diagnostics.trackers import PerGroupGradNormTracker


def _group_fn_by_first_path_segment(var):
    """Mirror of assembly._default_group_fn: split var.name on / and take the
    first segment as the group name."""
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
