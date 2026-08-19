"""Tests for `src.model_setup.backbone`'s graft builder + moved
`layer_diff_summary` (promoted from `scripts/graft_test.py:build_graft_backbone`,
see `docs/design-plans/2026-08-18-stage4-joint-finetune.md` "Branched
productionization -- implementation contract").

Uses tiny synthetic keras models with named layers mirroring the real
`roberta_base_en` backbone's top-level group names ("embeddings",
"embeddings_layer_norm", "transformer_layer_0".."_11") -- same technique as
`tests/test_assembly.py`'s `_make_layered_fake_backbone`. `load_dapt_backbone`
is monkeypatched (module-local, since `build_grafted_backbone` resolves it
from its own module globals) so no real backbone/weights file is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

import keras  # noqa: F401  (initializes backend)

import src.model_setup.backbone as backbone_module
from src.model_setup.backbone import build_grafted_backbone, layer_diff_summary


SEQ_LEN = 4
HIDDEN_DIM = 4
VOCAB = 20
N_LAYERS = 3

BASE_PATH = "base.weights.h5"
DONOR_PATH = "donor.weights.h5"


def _make_backbone(seed: int, n_layers: int = N_LAYERS):
    """Deterministic-under-seed tiny backbone stand-in: sub-layer names
    mirror the real roberta_base_en backbone's top-level groups."""
    keras.utils.set_random_seed(seed)
    token_ids = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="token_ids")
    padding_mask = keras.Input(shape=(SEQ_LEN,), dtype="int32", name="padding_mask")
    x = keras.layers.Embedding(VOCAB, HIDDEN_DIM, name="embeddings")(token_ids)
    x = keras.layers.LayerNormalization(name="embeddings_layer_norm")(x)
    mask_float = keras.ops.cast(padding_mask, "float32")
    mask_expanded = keras.ops.expand_dims(mask_float, axis=-1)
    x = x * mask_expanded
    for i in range(n_layers):
        x = keras.layers.Dense(HIDDEN_DIM, name=f"transformer_layer_{i}")(x)
    return keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask}, outputs=x
    )


def _fake_loader(path):
    path = str(path)
    if path == BASE_PATH:
        return _make_backbone(seed=1)
    if path == DONOR_PATH:
        return _make_backbone(seed=2)
    raise FileNotFoundError(path)


@pytest.fixture(autouse=True)
def _patch_loader(monkeypatch):
    monkeypatch.setattr(backbone_module, "load_dapt_backbone", _fake_loader)


class TestBuildGraftedBackboneHappyPath:
    def test_grafted_group_matches_donor_exactly(self):
        _, diffs = build_grafted_backbone(BASE_PATH, DONOR_PATH, {"transformer_layer_2"})
        assert diffs["vs_donor"]["transformer_layer_2"] == pytest.approx(0.0)

    def test_grafted_group_differs_from_base(self):
        _, diffs = build_grafted_backbone(BASE_PATH, DONOR_PATH, {"transformer_layer_2"})
        assert diffs["vs_base"]["transformer_layer_2"] > 0.0

    def test_non_grafted_groups_unchanged_vs_base(self):
        _, diffs = build_grafted_backbone(BASE_PATH, DONOR_PATH, {"transformer_layer_2"})
        for group in ("embeddings", "embeddings_layer_norm", "transformer_layer_0",
                      "transformer_layer_1"):
            assert diffs["vs_base"][group] == pytest.approx(0.0)

    def test_returned_backbone_layer_weights_equal_donor(self):
        graft, _ = build_grafted_backbone(BASE_PATH, DONOR_PATH, {"transformer_layer_2"})
        donor = _make_backbone(seed=2)
        for w_g, w_d in zip(
            graft.get_layer("transformer_layer_2").get_weights(),
            donor.get_layer("transformer_layer_2").get_weights(),
        ):
            np.testing.assert_array_equal(w_g, w_d)

    def test_multi_group_graft(self):
        groups = {"transformer_layer_0", "transformer_layer_2"}
        _, diffs = build_grafted_backbone(BASE_PATH, DONOR_PATH, groups)
        for g in groups:
            assert diffs["vs_base"][g] > 0.0
            assert diffs["vs_donor"][g] == pytest.approx(0.0)
        assert diffs["vs_base"]["transformer_layer_1"] == pytest.approx(0.0)

    def test_returns_dict_with_vs_base_and_vs_donor_keys(self):
        _, diffs = build_grafted_backbone(BASE_PATH, DONOR_PATH, {"transformer_layer_0"})
        assert set(diffs.keys()) == {"vs_base", "vs_donor"}


class TestBuildGraftedBackboneFailureModes:
    def test_raises_on_weight_ordering_mismatch(self, monkeypatch):
        def _mismatched_loader(path):
            path = str(path)
            if path == BASE_PATH:
                return _make_backbone(seed=1, n_layers=N_LAYERS)
            if path == "donor_mismatched.weights.h5":
                return _make_backbone(seed=2, n_layers=N_LAYERS + 1)
            raise FileNotFoundError(path)

        monkeypatch.setattr(backbone_module, "load_dapt_backbone", _mismatched_loader)
        with pytest.raises(ValueError, match="ordering diverged"):
            build_grafted_backbone(
                BASE_PATH, "donor_mismatched.weights.h5", {"transformer_layer_0"}
            )

    def test_raises_if_grafted_group_did_not_move_vs_base(self, monkeypatch):
        # Base and donor are bit-identical here -- grafting is a no-op, which
        # violates the "nonzero ONLY at the grafted groups" invariant (it
        # must be nonzero AT the grafted group).
        monkeypatch.setattr(
            backbone_module, "load_dapt_backbone", lambda path: _make_backbone(seed=1)
        )
        with pytest.raises(ValueError, match="vs base"):
            build_grafted_backbone(BASE_PATH, DONOR_PATH, {"transformer_layer_0"})

    def test_raises_if_ungrafted_group_moved_vs_base(self, monkeypatch):
        # Fake out layer_diff_summary to report movement at a NON-grafted
        # group vs base -- the "nonzero ONLY at grafted groups" check must
        # catch this even though the real graft mechanics would never
        # produce it (defense-in-depth on the verification step itself).
        def _fake_diff(paths, a, b):
            return {"transformer_layer_0": 1.0, "transformer_layer_1": 0.5}

        monkeypatch.setattr(backbone_module, "layer_diff_summary", _fake_diff)
        with pytest.raises(ValueError, match="vs base"):
            build_grafted_backbone(BASE_PATH, DONOR_PATH, {"transformer_layer_0"})

    def test_raises_if_graft_still_differs_from_donor_at_grafted_group(self, monkeypatch):
        calls = {"n": 0}

        def _fake_diff(paths, a, b):
            calls["n"] += 1
            if calls["n"] == 1:
                # vs_base: exactly the expected grafted-group movement.
                return {"transformer_layer_0": 1.0}
            # vs_donor: the grafted group should be exactly 0.0 -- violate it.
            return {"transformer_layer_0": 3.0}

        monkeypatch.setattr(backbone_module, "layer_diff_summary", _fake_diff)
        with pytest.raises(ValueError, match="vs donor"):
            build_grafted_backbone(BASE_PATH, DONOR_PATH, {"transformer_layer_0"})


class TestLayerDiffSummaryReExport:
    def test_extract_tuned_backbone_reexports_same_function(self):
        import src.extract_tuned_backbone as extract_tuned_backbone

        assert extract_tuned_backbone.layer_diff_summary is layer_diff_summary
