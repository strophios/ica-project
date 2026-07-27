# pattern: Functional Core (tests)
"""
Tests for `src.validation.escalation` (fine-tuning decision helpers).

Tests cover:
- top_n_group_fn: grouping of encoder layers by position (top N vs frozen)
- escalation_decision: decision logic and margin-based flipping
"""

from src.validation.escalation import (
    top_n_group_fn,
    escalation_decision,
    escalation_build_kwargs,
    DEFAULT_ESCALATION_MULTIPLIERS,
)


# Helper: a fake variable object with a path attribute
class FakeVariable:
    def __init__(self, path):
        self.path = path


class TestTopNGroupFn:
    """Test top_n_group_fn(n_top, n_layers=12) grouping function.

    Layer-path naming ("transformer_layer_N", "embeddings/...") matches the
    REAL keras_hub RobertaBackbone (verified via `bb.trainable_variables`
    against the loaded DAPT backbone, 2026-07-27) -- see top_n_group_fn's
    NAMING NOTE docstring for how the prior "roberta_layer_N" default was
    found to be wrong (never matched any real variable path) via the first
    real encoder-unfreeze smoke run.
    """

    def test_top_2_layers_12_layer_model(self):
        """n_top=2 of 12 layers: layers 11, 10 are encoder_top.
        Layer 0 is encoder_frozen. Non-encoder vars are 'head'."""
        group_fn = top_n_group_fn(n_top=2, n_layers=12)

        # Layer 11 and 10 should be encoder_top
        assert group_fn(FakeVariable("transformer_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_10/kernel")) == "encoder_top"

        # Layer 9 and lower should be encoder_frozen
        assert group_fn(FakeVariable("transformer_layer_9/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("transformer_layer_0/kernel")) == "encoder_frozen"

        # Non-encoder should be 'head'
        assert group_fn(FakeVariable("us/dense/kernel")) == "head"
        assert group_fn(FakeVariable("my_head/var")) == "head"

    def test_top_1_layer(self):
        """n_top=1: only layer 11 is encoder_top."""
        group_fn = top_n_group_fn(n_top=1, n_layers=12)

        assert group_fn(FakeVariable("transformer_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_10/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("head_var")) == "head"

    def test_top_0_layers(self):
        """n_top=0: no encoder_top, all transformer layers are frozen."""
        group_fn = top_n_group_fn(n_top=0, n_layers=12)

        assert group_fn(FakeVariable("transformer_layer_11/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("transformer_layer_0/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("head_var")) == "head"

    def test_top_all_layers(self):
        """n_top=12 (all layers): all transformer layers are encoder_top."""
        group_fn = top_n_group_fn(n_top=12, n_layers=12)

        assert group_fn(FakeVariable("transformer_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_0/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("head_var")) == "head"

    def test_different_n_layers_parameter(self):
        """n_top=2 with n_layers=24: layers 23, 22 are encoder_top."""
        group_fn = top_n_group_fn(n_top=2, n_layers=24)

        assert group_fn(FakeVariable("transformer_layer_23/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_22/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_21/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("transformer_layer_0/kernel")) == "encoder_frozen"

    def test_various_path_patterns(self):
        """Transformer-layer matching should work with various path structures."""
        group_fn = top_n_group_fn(n_top=2, n_layers=12)

        # Various path formats that contain transformer_layer_11
        assert group_fn(FakeVariable("transformer_layer_11")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_backbone/transformer_layer_11/w")) == "encoder_top"

    def test_embeddings_are_encoder_frozen(self):
        """Non-layer backbone params (token/position embeddings + their layer
        norm) are always encoder_frozen -- the real keras_hub RobertaBackbone
        top-level component names, confirmed against the loaded DAPT backbone."""
        group_fn = top_n_group_fn(n_top=2, n_layers=12)

        assert group_fn(FakeVariable("embeddings/token_embedding/embeddings")) == "encoder_frozen"
        assert group_fn(FakeVariable("embeddings/position_embedding/embeddings")) == "encoder_frozen"
        assert group_fn(FakeVariable("embeddings_layer_norm/gamma")) == "encoder_frozen"
        assert group_fn(FakeVariable("embeddings_layer_norm/beta")) == "encoder_frozen"

    def test_custom_layer_prefix(self):
        """layer_prefix is a forward-compat override, not hardcoded."""
        group_fn = top_n_group_fn(n_top=1, n_layers=12, layer_prefix="roberta_layer")

        assert group_fn(FakeVariable("roberta_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("roberta_layer_10/kernel")) == "encoder_frozen"
        # And the new default prefix no longer matches under this override.
        assert group_fn(FakeVariable("transformer_layer_11/kernel")) == "head"

    def test_regression_old_roberta_layer_naming_no_longer_matches_default(self):
        """Regression guard for the 2026-07-27 fix: the OLD default naming
        assumption ("roberta_layer_N") must NOT silently match anymore --
        that silent non-match (falling through to "head") was exactly the
        bug the first real smoke run caught (grad_norm/encoder_top/* was
        absent from history.history, not zero -- absent, because no real
        variable path ever matched)."""
        group_fn = top_n_group_fn(n_top=2, n_layers=12)

        assert group_fn(FakeVariable("roberta_layer_11/kernel")) == "head"
        assert group_fn(FakeVariable("roberta_layer_0/kernel")) == "head"


class TestEscalationBuildKwargs:
    """Test escalation_build_kwargs(unfreeze_top_n, layer_multipliers, n_layers).

    Mirrors the semantics of the inline branch in
    run_us_classification.py:154-179 — see that module for the reference
    behavior this function was extracted to match.
    """

    def test_frozen_path_returns_only_freeze_encoder(self):
        """unfreeze_top_n=0: frozen-probe path, no group_fn/layer_multipliers keys."""
        kwargs = escalation_build_kwargs(unfreeze_top_n=0)
        assert kwargs == {"freeze_encoder": True}

    def test_unfreeze_path_sets_freeze_encoder_false(self):
        kwargs = escalation_build_kwargs(unfreeze_top_n=2)
        assert kwargs["freeze_encoder"] is False

    def test_unfreeze_path_default_multipliers(self):
        kwargs = escalation_build_kwargs(unfreeze_top_n=2)
        assert kwargs["layer_multipliers"] == DEFAULT_ESCALATION_MULTIPLIERS
        # Returned dict must be a copy, not the module-level default object
        # (a caller mutating it should not corrupt the shared default).
        assert kwargs["layer_multipliers"] is not DEFAULT_ESCALATION_MULTIPLIERS

    def test_unfreeze_path_custom_multipliers_override_default(self):
        custom = {"head": 1.0, "encoder_top": 0.5, "encoder_frozen": 0.0}
        kwargs = escalation_build_kwargs(unfreeze_top_n=2, layer_multipliers=custom)
        assert kwargs["layer_multipliers"] == custom

    def test_unfreeze_path_empty_dict_multipliers_falls_back_to_default(self):
        """An empty (falsy) dict is treated like None -- caller-supplied but
        empty configuration means 'use the sensible defaults', matching the
        `layer_multipliers or {...}` idiom in run_us_classification.py."""
        kwargs = escalation_build_kwargs(unfreeze_top_n=2, layer_multipliers={})
        assert kwargs["layer_multipliers"] == DEFAULT_ESCALATION_MULTIPLIERS

    def test_unfreeze_path_group_fn_matches_top_n_group_fn_semantics(self):
        """The returned group_fn should behave identically to a directly
        constructed top_n_group_fn with the same parameters."""
        kwargs = escalation_build_kwargs(unfreeze_top_n=2, n_layers=12)
        group_fn = kwargs["group_fn"]

        assert group_fn(FakeVariable("transformer_layer_11/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_10/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_9/kernel")) == "encoder_frozen"
        assert group_fn(FakeVariable("rel/dense/kernel")) == "head"

    def test_unfreeze_path_respects_custom_n_layers(self):
        kwargs = escalation_build_kwargs(unfreeze_top_n=1, n_layers=24)
        group_fn = kwargs["group_fn"]
        assert group_fn(FakeVariable("transformer_layer_23/kernel")) == "encoder_top"
        assert group_fn(FakeVariable("transformer_layer_22/kernel")) == "encoder_frozen"

    def test_frozen_path_ignores_layer_multipliers_argument(self):
        """Even if layer_multipliers is passed, the frozen path (unfreeze_top_n=0)
        does not surface it -- mirrors the source branch's else-clause, which
        only ever sets freeze_encoder."""
        kwargs = escalation_build_kwargs(
            unfreeze_top_n=0, layer_multipliers={"head": 1.0}
        )
        assert kwargs == {"freeze_encoder": True}


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
