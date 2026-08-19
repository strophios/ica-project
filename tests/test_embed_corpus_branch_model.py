"""Tests for `src.embed_corpus._build_embed_model`'s branched-embed wiring
(stage-4 branched embed model,
`docs/design-plans/2026-08-18-stage4-joint-finetune.md` "Branched
productionization -- implementation contract"). Separate from
`test_embed_corpus.py` (whose docstring promises no keras forward pass) --
this DOES build tiny synthetic keras models (same technique as
`tests/test_backbone.py` / `tests/test_assembly.py`'s fake-backbone
fixtures) and run a real `.predict()`, but never touches a real backbone or
weights file.

`load_dapt_backbone` is monkeypatched in BOTH `src.embed_corpus` (used to
build the base backbone) and `src.model_setup.backbone` (used inside
`build_grafted_backbone`, resolved from its own module globals) -- see
`tests/test_backbone.py`'s docstring for why both patch targets are needed.

`backbone_weights` (the base-override path) is ALSO raw-`load_weights`-ed a
second time inside `_build_embed_model` (the "CRITICAL ORDER FIX" reapply
step) -- that call bypasses the monkeypatched loader entirely (it's a plain
Keras h5 load on an already-constructed instance), so the base path must be
a REAL `tmp_path`-anchored `.weights.h5` file, not just a lookup key in the
fake loader. The donor path never needs a real file (it's only ever loaded
through the patched `load_dapt_backbone`).
"""

from __future__ import annotations

import numpy as np
import pytest

import keras  # noqa: F401  (initializes backend)

import src.model_setup.backbone as backbone_module
from src.cca_config import LRScheduleConfig, OptimizerConfig
from src.model_setup.assembly import build_inference_model
from src.model_setup.heads import ClassificationHead
from src.us_config import UsHeadConfig, UsRunConfig, config_path_for_weights


SEQ_LEN = 4
HIDDEN_DIM = 4
VOCAB = 20
N_LAYERS = 12  # matches expected_tuned_groups' default n_layers=12


def _make_backbone(seed: int, n_layers: int = N_LAYERS):
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
    model = keras.Model(
        inputs={"token_ids": token_ids, "padding_mask": padding_mask}, outputs=x
    )
    model.hidden_dim = HIDDEN_DIM  # real keras_hub Backbone exposes this
    return model


def _setup(monkeypatch, tmp_path, extra_donors: dict[str, int] | None = None):
    """Patch both `load_dapt_backbone` call sites; write a real us_weights
    h5 + sidecar, and a real base-backbone h5 (needed for the raw reapply
    load -- see module docstring). Returns
    `(embed_corpus, us_weights, base_path, donor_path)`.

    `extra_donors`: optional `{path_str: seed}` for tests wanting more than
    one donor backbone recognized by the fake loader.
    """
    import src.embed_corpus as embed_corpus

    base_path = str(tmp_path / "base_backbone.weights.h5")
    donor_path = str(tmp_path / "donor_backbone.weights.h5")
    donors = {donor_path: 2, **(extra_donors or {})}

    def _fake_loader(path):
        path = str(path)
        if path == base_path:
            return _make_backbone(seed=1)
        if path in donors:
            return _make_backbone(seed=donors[path])
        raise FileNotFoundError(path)

    monkeypatch.setattr(embed_corpus, "load_dapt_backbone", _fake_loader)
    monkeypatch.setattr(backbone_module, "load_dapt_backbone", _fake_loader)

    us_weights = tmp_path / "us_classifier.weights.h5"
    us_cfg = UsRunConfig(
        seq_length=SEQ_LEN, text_key="headline_with_lead", target_dtype="float32",
        head=UsHeadConfig(name="us", source_column="us_label", hidden_dim=HIDDEN_DIM),
        epochs=1, backbone_weights_path="unused_sidecar_backbone.weights.h5",
        lr_schedule=LRScheduleConfig(), optimizer=OptimizerConfig(),
    )
    us_cfg.to_json(config_path_for_weights(us_weights))

    base0 = _make_backbone(seed=1)
    head0 = ClassificationHead(hidden_dim=HIDDEN_DIM, name="us")
    inf0 = build_inference_model(backbone=base0, heads={"us": head0}, seq_length=SEQ_LEN)
    inf0.save_weights(str(us_weights))
    # The raw reapply load (`backbone.load_weights(backbone_path, ...)`)
    # bypasses the monkeypatched loader -- needs a real file on disk.
    base0.save_weights(base_path)

    return embed_corpus, us_weights, base_path, donor_path


def _dummy_inputs(n=2):
    rng = np.random.RandomState(0)
    return {
        "token_ids": rng.randint(0, VOCAB, size=(n, SEQ_LEN)).astype("int32"),
        "padding_mask": np.ones((n, SEQ_LEN), dtype="int32"),
    }


class TestBuildEmbedModelNoBranching:
    def test_output_keys_unchanged_without_branch_specs(self, monkeypatch, tmp_path):
        embed_corpus, us_weights, base_path, _donor_path = _setup(monkeypatch, tmp_path)
        model, _us_cfg, _backbone_path, branch_prov = embed_corpus._build_embed_model(
            us_weights, backbone_weights=base_path,
        )
        # `model.output_names` reflects each output TENSOR's own keras name
        # (auto-generated "get_item_N" for the unnamed CLS-slice op) -- NOT
        # the dict keys passed to `outputs=`. What main() actually relies on
        # is the dict structure `model.predict()` returns, checked here.
        preds = model.predict(_dummy_inputs(), verbose=0)
        assert set(preds.keys()) == {"cls", "us"}
        assert branch_prov == {}


class TestBuildEmbedModelWithBranching:
    def test_adds_cls_variant_output(self, monkeypatch, tmp_path):
        embed_corpus, us_weights, base_path, donor_path = _setup(monkeypatch, tmp_path)
        model, _us_cfg, _backbone_path, branch_prov = embed_corpus._build_embed_model(
            us_weights, backbone_weights=base_path,
            branch_specs={"rel_branch": (donor_path, 1)},
        )
        preds = model.predict(_dummy_inputs(), verbose=0)
        assert set(preds.keys()) == {"cls", "us", "cls.rel_branch"}

    def test_branch_provenance_shape(self, monkeypatch, tmp_path):
        embed_corpus, us_weights, base_path, donor_path = _setup(monkeypatch, tmp_path)
        _model, _us_cfg, _backbone_path, branch_prov = embed_corpus._build_embed_model(
            us_weights, backbone_weights=base_path,
            branch_specs={"rel_branch": (donor_path, 1)},
        )
        assert set(branch_prov.keys()) == {"rel_branch"}
        entry = branch_prov["rel_branch"]
        assert entry["groups"] == ["transformer_layer_11"]
        assert entry["unfreeze_top_n"] == 1
        assert entry["donor"]["path"] == donor_path
        assert set(entry["graft_verification"].keys()) == {"vs_base", "vs_donor"}

    def test_multiple_branches_each_get_their_own_output(self, monkeypatch, tmp_path):
        donor2_path = str(tmp_path / "donor2_backbone.weights.h5")
        embed_corpus, us_weights, base_path, donor_path = _setup(
            monkeypatch, tmp_path, extra_donors={donor2_path: 3}
        )
        model, _us_cfg, _backbone_path, branch_prov = embed_corpus._build_embed_model(
            us_weights, backbone_weights=base_path,
            branch_specs={
                "rel_branch": (donor_path, 1),
                "cca_branch": (donor2_path, 2),
            },
        )
        preds = model.predict(_dummy_inputs(), verbose=0)
        assert set(preds.keys()) == {"cls", "us", "cls.rel_branch", "cls.cca_branch"}
        assert branch_prov["cca_branch"]["groups"] == sorted(
            ["transformer_layer_11", "transformer_layer_10"]
        )

    def test_branch_cls_differs_from_base_cls_end_to_end(self, monkeypatch, tmp_path):
        embed_corpus, us_weights, base_path, donor_path = _setup(monkeypatch, tmp_path)
        model, _us_cfg, _backbone_path, _branch_prov = embed_corpus._build_embed_model(
            us_weights, backbone_weights=base_path,
            branch_specs={"rel_branch": (donor_path, 1)},
        )
        preds = model.predict(_dummy_inputs(), verbose=0)
        # The grafted top layer differs from the base's -> different CLS.
        assert not np.allclose(preds["cls"], preds["cls.rel_branch"])
        # The US head reads the BASE cls regardless of branching (deployed
        # fusion config keeps "us" on "base" -- see the design plan).
        assert preds["us"].shape == (2, 1)

    def test_raises_on_unresolvable_top_n(self, monkeypatch, tmp_path):
        """No explicit top_n and no RunConfig sidecar for the donor -- must
        fail loudly (`_resolve_branch_groups`'s contract), not silently pick
        a default depth."""
        embed_corpus, us_weights, base_path, donor_path = _setup(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="top_n"):
            embed_corpus._build_embed_model(
                us_weights, backbone_weights=base_path,
                branch_specs={"rel_branch": (donor_path, None)},
            )
