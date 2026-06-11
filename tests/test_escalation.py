# pattern: Functional Core (tests)
"""
Tests for `src.validation.escalation` (fine-tuning decision helpers).

Tests cover:
- top_n_group_fn: grouping of encoder layers by position (top N vs frozen)
- escalation_decision: decision logic and margin-based flipping
"""

import pytest

from src.validation.escalation import top_n_group_fn, escalation_decision


# Helper: a fake variable object with a path attribute
class FakeVariable:
    def __init__(self, path):
        self.path = path


class TestTopNGroupFn:
    """Test top_n_group_fn(n_top, n_layers=12) grouping function."""

    def test_top_2_layers_12_layer_model(self):
        """n_top=2 of 12 layers: layers 11, 10 are encoder_top.
        Layer 0 is encoder_frozen. Non-roberta vars are 'head'."""
        group_fn = top_n_group_fn(n_top=2, n_layers=12)

        # Layer 11 and 10 should be encoder_top
        assert group_fn(FakeVariable("roberta_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_layer_10/kernel")) == "encoder_top"

        # Layer 9 and lower should be encoder_frozen
        assert group_fn(FakeVariable("roberta_layer_9/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("roberta_layer_0/kernel")) == "encoder_frozen"

        # Non-roberta should be 'head'
        assert group_fn(FakeVariable("us/dense/kernel")) == "head"
        assert group_fn(FakeVariable("my_head/var")) == "head"

    def test_top_1_layer(self):
        """n_top=1: only layer 11 is encoder_top."""
        group_fn = top_n_group_fn(n_top=1, n_layers=12)

        assert group_fn(FakeVariable("roberta_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_layer_10/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("head_var")) == "head"

    def test_top_0_layers(self):
        """n_top=0: no encoder_top, all roberta layers are frozen."""
        group_fn = top_n_group_fn(n_top=0, n_layers=12)

        assert group_fn(FakeVariable("roberta_layer_11/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("roberta_layer_0/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("head_var")) == "head"

    def test_top_all_layers(self):
        """n_top=12 (all layers): all roberta layers are encoder_top."""
        group_fn = top_n_group_fn(n_top=12, n_layers=12)

        assert group_fn(FakeVariable("roberta_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_layer_0/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("head_var")) == "head"

    def test_different_n_layers_parameter(self):
        """n_top=2 with n_layers=24: layers 23, 22 are encoder_top."""
        group_fn = top_n_group_fn(n_top=2, n_layers=24)

        assert group_fn(FakeVariable("roberta_layer_23/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_layer_22/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_layer_21/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("roberta_layer_0/kernel")) == "encoder_frozen"

    def test_various_path_patterns(self):
        """Roberta layer matching should work with various path structures."""
        group_fn = top_n_group_fn(n_top=2, n_layers=12)

        # Various path formats that contain roberta_layer_11
        assert group_fn(FakeVariable("roberta_layer_11")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_backbone/roberta_layer_11/w")) == "encoder_top"

        # Non-layer roberta should be encoder_frozen
        assert group_fn(FakeVariable("roberta_token_embedding")) == "encoder_frozen"


class TestEscalationDecision:
    """Test escalation_decision(baseline_f1, transfer_f1, margin=0.1)."""

    def test_escalate_when_gap_exceeds_margin(self):
        """baseline=0.8, transfer=0.6, margin=0.1:
        gap=0.2 > 0.1 -> escalate=True."""
        result = escalation_decision(baseline_f1=0.8, transfer_f1=0.6, margin=0.1)
        assert result["escalate"] is True
        assert "0.6" in result["rationale"] or "0.60" in result["rationale"]
        assert "0.8" in result["rationale"] or "0.80" in result["rationale"]
        assert "gap" in result["rationale"].lower()

    def test_no_escalate_when_gap_within_margin(self):
        """baseline=0.8, transfer=0.75, margin=0.1:
        gap=0.05 <= 0.1 -> escalate=False."""
        result = escalation_decision(baseline_f1=0.8, transfer_f1=0.75, margin=0.1)
        assert result["escalate"] is False
        assert "rationale" in result

    def test_escalate_at_exact_margin_boundary(self):
        """Test margin boundary: gap slightly less than margin should not escalate.
        Use well-separated values to avoid float precision issues."""
        result = escalation_decision(baseline_f1=0.8, transfer_f1=0.715, margin=0.1)
        # gap = 0.085, which is less than 0.1, should NOT escalate
        assert result["escalate"] is False

    def test_escalate_strictly_exceeds_margin(self):
        """baseline=0.75, transfer=0.69, margin=0.05:
        gap=0.06 > 0.05 -> escalate=True."""
        result = escalation_decision(baseline_f1=0.75, transfer_f1=0.69, margin=0.05)
        assert result["escalate"] is True

    def test_default_margin_0_1(self):
        """Default margin is 0.1."""
        result1 = escalation_decision(0.8, 0.69)
        result2 = escalation_decision(0.8, 0.69, margin=0.1)
        assert result1["escalate"] == result2["escalate"]

    def test_zero_margin(self):
        """margin=0: any gap > 0 triggers escalation."""
        result = escalation_decision(0.8, 0.79, margin=0.0)
        assert result["escalate"] is True

    def test_transfer_better_than_baseline(self):
        """transfer > baseline: gap is negative, no escalation."""
        result = escalation_decision(0.7, 0.8, margin=0.1)
        # gap = 0.7 - 0.8 = -0.1, which is not > 0.1
        assert result["escalate"] is False

    def test_equal_f1_scores(self):
        """transfer == baseline: gap=0, no escalation."""
        result = escalation_decision(0.75, 0.75, margin=0.1)
        assert result["escalate"] is False

    def test_rationale_contains_metrics(self):
        """Rationale should contain F1 scores and margin."""
        result = escalation_decision(0.8, 0.6, margin=0.1)
        rationale = result["rationale"].lower()
        # Should mention the F1 values
        assert ("0.6" in result["rationale"] or "0.60" in result["rationale"])
        assert ("0.8" in result["rationale"] or "0.80" in result["rationale"])
        # Should mention gap and margin
        assert "gap" in rationale
        assert "margin" in rationale

    def test_rationale_inequality_operator(self):
        """Rationale should show > or <= based on escalation decision."""
        result_escalate = escalation_decision(0.8, 0.6, margin=0.1)
        result_no_escalate = escalation_decision(0.8, 0.75, margin=0.1)

        # escalate=True should have '>'
        assert ">" in result_escalate["rationale"]
        # escalate=False should have '<='
        assert "<=" in result_no_escalate["rationale"]
