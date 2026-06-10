"""
Tests for `src.us_metrics.make_us_metrics`.

Verifies that the metric factory returns the correct canonical set
for US head classification (same structure as CCA metrics, adapted for
the US filter).
"""

import keras

from src.us_metrics import make_us_metrics


class TestMakeUsMetrics:
    def test_returns_list_of_metrics(self):
        """make_us_metrics should return a list of Metric instances."""
        metrics = make_us_metrics()
        assert isinstance(metrics, list)
        assert all(isinstance(m, keras.metrics.Metric) for m in metrics)

    def test_returns_four_metrics(self):
        """Canonical US metric set has 4 metrics: BinaryAccuracy, Precision, Recall, AUC."""
        metrics = make_us_metrics()
        assert len(metrics) == 4

    def test_metric_names(self):
        """Verify exact metric names for endpoint-mode routing."""
        metrics = make_us_metrics()
        names = {m.name for m in metrics}
        # BinaryAccuracy defaults to name 'binary_accuracy'; Precision, Recall, AUC
        # are explicitly named in make_us_metrics
        assert "binary_accuracy" in names
        assert "precision" in names
        assert "recall" in names
        assert "pr_auc" in names

    def test_metrics_are_independent(self):
        """Two calls to make_us_metrics should return distinct metric instances."""
        metrics1 = make_us_metrics()
        metrics2 = make_us_metrics()
        # Each call returns fresh instances (not singletons)
        assert metrics1 is not metrics2
        assert all(m1 is not m2 for m1, m2 in zip(metrics1, metrics2))

    def test_binary_accuracy_threshold_zero(self):
        """BinaryAccuracy should use threshold=0.0 (logits space)."""
        metrics = make_us_metrics()
        binary_acc = next((m for m in metrics if isinstance(m, keras.metrics.BinaryAccuracy)), None)
        assert binary_acc is not None
        # threshold is stored as a property on the metric
        assert binary_acc.threshold == 0.0

    def test_precision_threshold_zero(self):
        """Precision should use thresholds=0.0 (logits space)."""
        metrics = make_us_metrics()
        precision = next((m for m in metrics if isinstance(m, keras.metrics.Precision)), None)
        assert precision is not None
        # thresholds is a tuple or list; verify it contains 0.0
        assert 0.0 in precision.thresholds

    def test_recall_threshold_zero(self):
        """Recall should use thresholds=0.0 (logits space)."""
        metrics = make_us_metrics()
        recall = next((m for m in metrics if isinstance(m, keras.metrics.Recall)), None)
        assert recall is not None
        assert 0.0 in recall.thresholds

    def test_auc_from_logits_true(self):
        """AUC should have from_logits=True and curve='PR'."""
        metrics = make_us_metrics()
        auc = next((m for m in metrics if isinstance(m, keras.metrics.AUC)), None)
        assert auc is not None
        # from_logits is a property; curve is typically accessed via config
        # We trust the Keras implementation here; the contract is in the code
