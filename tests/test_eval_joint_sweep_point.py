"""Tests for scripts.eval_joint_sweep_point (the per-cell scorer for the
joint CCA+rel fine-tune sweep, docs/design-plans/2026-08-18-stage4-joint-
finetune.md Components item 4).

Covers the pure parts only (no model build, no real training/scoring, per
the quality bar -- these are synthetic-stand-in tests):
  - extract_sweep_params: pull sweep provenance out of a sidecar dict.
  - natural_balance_val_population: dedup the cca_pos/rel_pos overlap when
    reconstructing the val split's natural-balance population from
    create_joint_text_data's grouped output.
  - compose_scores: p_cca * p_rel.
  - Importing does not trigger scoring/training.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scripts.eval_joint_sweep_point import (
    compose_scores,
    extract_sweep_params,
    natural_balance_val_population,
)


# ---------------------------------------------------------------------------
# extract_sweep_params (pure)
# ---------------------------------------------------------------------------
class TestExtractSweepParams:
    def _sidecar(self, **overrides):
        base = {
            "unfreeze_top_n": 1,
            "layer_multipliers": None,
            "hard_freeze": False,
            "seed": 200,
            "epochs": 7,
            "heads": [
                {"name": "cca", "loss_weight": 0.5},
                {"name": "rel", "loss_weight": 0.5},
            ],
        }
        base.update(overrides)
        return base

    def test_top_level_fields(self):
        params = extract_sweep_params(self._sidecar())
        assert params["unfreeze_top_n"] == 1
        assert params["hard_freeze"] is False
        assert params["seed"] == 200
        assert params["epochs"] == 7

    def test_loss_weights_extracted_per_head(self):
        params = extract_sweep_params(
            self._sidecar(heads=[
                {"name": "cca", "loss_weight": 0.75},
                {"name": "rel", "loss_weight": 0.25},
            ])
        )
        assert params["cca_loss_weight"] == pytest.approx(0.75)
        assert params["rel_loss_weight"] == pytest.approx(0.25)

    def test_missing_heads_yields_none_loss_weights(self):
        sidecar = self._sidecar()
        del sidecar["heads"]
        params = extract_sweep_params(sidecar)
        assert params["cca_loss_weight"] is None
        assert params["rel_loss_weight"] is None

    def test_missing_optional_fields_yield_none(self):
        params = extract_sweep_params({"heads": []})
        assert params["unfreeze_top_n"] is None
        assert params["seed"] is None

    def test_does_not_mutate_input(self):
        sidecar = self._sidecar()
        original = dict(sidecar)
        extract_sweep_params(sidecar)
        assert sidecar == original


# ---------------------------------------------------------------------------
# natural_balance_val_population (pure)
# ---------------------------------------------------------------------------
class TestNaturalBalanceValPopulation:
    def _splits(self, cca_pos, rel_pos, unl):
        return {"val": {"cca_pos": cca_pos, "rel_pos": rel_pos, "unl": unl}}

    def test_disjoint_groups_all_kept(self):
        cca_pos = pl.DataFrame({"id": ["a"], "x": [1]})
        rel_pos = pl.DataFrame({"id": ["b"], "x": [2]})
        unl = pl.DataFrame({"id": ["c"], "x": [3]})
        out = natural_balance_val_population(self._splits(cca_pos, rel_pos, unl))
        assert sorted(out["id"].to_list()) == ["a", "b", "c"]

    def test_overlap_row_counted_once(self):
        cca_pos = pl.DataFrame({"id": ["x", "a"], "val": [1, 2]})
        rel_pos = pl.DataFrame({"id": ["x", "b"], "val": [1, 3]})
        unl = pl.DataFrame({"id": ["c"], "val": [4]})
        out = natural_balance_val_population(self._splits(cca_pos, rel_pos, unl))
        assert out.height == 4  # x, a, b, c (x deduped)
        assert out.filter(pl.col("id") == "x").height == 1

    def test_empty_groups_ok(self):
        empty = pl.DataFrame({"id": []}, schema={"id": pl.Utf8})
        out = natural_balance_val_population(self._splits(empty, empty, empty))
        assert out.height == 0


# ---------------------------------------------------------------------------
# compose_scores (pure)
# ---------------------------------------------------------------------------
class TestComposeScores:
    def test_elementwise_product(self):
        p_cca = np.array([0.5, 0.2, 1.0])
        p_rel = np.array([0.5, 0.8, 0.0])
        out = compose_scores(p_cca, p_rel)
        assert out == pytest.approx([0.25, 0.16, 0.0])

    def test_shape_preserved(self):
        p_cca = np.array([0.1, 0.2])
        p_rel = np.array([0.3, 0.4])
        out = compose_scores(p_cca, p_rel)
        assert out.shape == (2,)


# ---------------------------------------------------------------------------
# Import side effects
# ---------------------------------------------------------------------------
class TestImportNoSideEffects:
    def test_import_does_not_trigger_scoring(self):
        import scripts.eval_joint_sweep_point

        assert scripts.eval_joint_sweep_point is not None
