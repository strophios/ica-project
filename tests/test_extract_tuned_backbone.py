"""Tests for the tuned-backbone extractor's pure helpers.

Covers only the Functional Core (`default_out_path`, `layer_diff_summary`,
`expected_tuned_groups`, `_group_sort_key`, `_verify_expected_groups`). Does
NOT exercise the keras/backbone reconstruction or file I/O -- those are
covered by the real run against the job8823087 artifact (see
`docs/notes/encoder-unfreeze-strategy.md` and the extraction script's own
printed diff summary).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.extract_tuned_backbone import (
    _group_sort_key,
    _verify_expected_groups,
    default_out_path,
    expected_tuned_groups,
    layer_diff_summary,
)


# ---------------------------------------------------------------------------
# default_out_path
# ---------------------------------------------------------------------------
def test_default_out_path_derives_jobtag():
    p = Path("/some/dir/relevance_text.job8823087.weights.h5")
    out = default_out_path(p)
    assert out == Path("/some/dir/tuned_backbone.job8823087.weights.h5")


def test_default_out_path_handles_multi_dot_stem():
    # jobtag is whatever's between the LAST "." before ".weights.h5" and that suffix.
    p = Path("/x/relevance_text.eta0.3.job123.weights.h5")
    out = default_out_path(p)
    assert out == Path("/x/tuned_backbone.job123.weights.h5")


def test_default_out_path_rejects_missing_suffix():
    with pytest.raises(ValueError, match=r"\.weights\.h5"):
        default_out_path(Path("/x/relevance_text.job8823087.h5"))


def test_default_out_path_rejects_missing_jobtag_segment():
    with pytest.raises(ValueError, match="jobtag segment"):
        default_out_path(Path("/x/relevance_text.weights.h5"))


def test_default_out_path_rejects_empty_jobtag():
    with pytest.raises(ValueError, match="empty jobtag"):
        default_out_path(Path("/x/relevance_text..weights.h5"))


# ---------------------------------------------------------------------------
# layer_diff_summary
# ---------------------------------------------------------------------------
def test_layer_diff_summary_groups_by_top_level_segment():
    paths = [
        "embeddings/token_embedding/embeddings",
        "embeddings_layer_norm/gamma",
        "transformer_layer_0/self_attention_layer/query/kernel",
        "transformer_layer_0/self_attention_layer/query/bias",
        "transformer_layer_11/self_attention_layer/query/kernel",
    ]
    a = [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
    ]
    b = [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.array([1.0, 1.005], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ]
    summary = layer_diff_summary(paths, a, b)
    assert summary["embeddings"] == pytest.approx(0.0)
    assert summary["embeddings_layer_norm"] == pytest.approx(0.0)
    # transformer_layer_0 has two variables (kernel identical, bias off by 0.005)
    assert summary["transformer_layer_0"] == pytest.approx(0.005, abs=1e-6)
    assert summary["transformer_layer_11"] == pytest.approx(1.0)


def test_layer_diff_summary_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        layer_diff_summary(["a", "b"], [np.zeros(1)], [np.zeros(1), np.zeros(1)])


def test_layer_diff_summary_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        layer_diff_summary([], [], [])


# ---------------------------------------------------------------------------
# expected_tuned_groups
# ---------------------------------------------------------------------------
def test_expected_tuned_groups_top_n():
    assert expected_tuned_groups(1) == {"transformer_layer_11"}
    assert expected_tuned_groups(3) == {
        "transformer_layer_11", "transformer_layer_10", "transformer_layer_9",
    }
    assert expected_tuned_groups(0) == set()


def test_expected_tuned_groups_rejects_out_of_range():
    with pytest.raises(ValueError, match="unfreeze_top_n"):
        expected_tuned_groups(13)
    with pytest.raises(ValueError, match="unfreeze_top_n"):
        expected_tuned_groups(-1)


# ---------------------------------------------------------------------------
# _group_sort_key
# ---------------------------------------------------------------------------
def test_group_sort_key_orders_embeddings_before_layers_numerically():
    groups = [
        "transformer_layer_11", "transformer_layer_2", "embeddings",
        "transformer_layer_10", "embeddings_layer_norm",
    ]
    ordered = sorted(groups, key=_group_sort_key)
    # embeddings-family first (alpha order among themselves), then layers 2, 10, 11
    # (numeric, not lexicographic -- "10" and "11" must not sort before "2").
    assert ordered == [
        "embeddings", "embeddings_layer_norm",
        "transformer_layer_2", "transformer_layer_10", "transformer_layer_11",
    ]


# ---------------------------------------------------------------------------
# _verify_expected_groups
# ---------------------------------------------------------------------------
def test_verify_expected_groups_passes_when_clearly_separated():
    summary = {
        "embeddings": 2.3e-3, "embeddings_layer_norm": 2.3e-3,
        "transformer_layer_10": 2.7e-3, "transformer_layer_11": 0.119,
    }
    _verify_expected_groups(summary, unfreeze_top_n=1)  # should not raise


def test_verify_expected_groups_raises_if_tuned_group_missing():
    summary = {"embeddings": 0.0, "transformer_layer_10": 0.0}
    with pytest.raises(ValueError, match="not present"):
        _verify_expected_groups(summary, unfreeze_top_n=1)


def test_verify_expected_groups_raises_if_tuned_group_didnt_move():
    summary = {"embeddings": 0.0, "transformer_layer_11": 0.0}
    with pytest.raises(ValueError, match="zero movement"):
        _verify_expected_groups(summary, unfreeze_top_n=1)


def test_verify_expected_groups_raises_if_not_clearly_separated():
    # tuned group moved, but only 2x the frozen noise floor -- below the
    # required separation margin.
    summary = {"embeddings": 0.01, "transformer_layer_11": 0.02}
    with pytest.raises(ValueError, match="not clearly separated"):
        _verify_expected_groups(summary, unfreeze_top_n=1)


def test_verify_expected_groups_ok_with_no_frozen_groups_left():
    # unfreeze_top_n covers every group present -- nothing to compare against.
    summary = {"transformer_layer_11": 0.05}
    _verify_expected_groups(summary, unfreeze_top_n=1)  # should not raise
