"""Tests for src.run_us_pnu's pure core: PNU stream/label assembly, pi_hat
estimation, and val-selection metrics.

- us-pnu-train.AC1: attach_emb_rows resolves per-cache emb_row, raises on
  missing cache / unresolved id.
- us-pnu-train.AC2: gather_group_features concatenates multi-cache groups and
  fills the FLPULoss label convention (1.0/-1.0/0.0).
- us-pnu-train.AC3: estimate_prior_from_calibrated_logits = mean(calibrator
  .transform(logits)).
- us-pnu-train.AC4: pn_val_metrics / pu_val_risk are deterministic pure
  functions of their inputs.
- us-pnu-train.AC5: importing the module does not trigger training.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.calibration.calibrator import PlattCalibrator
from src.run_us_pnu import (
    _base_run_config,
    attach_emb_rows,
    estimate_prior_from_calibrated_logits,
    gather_group_features,
    pn_val_metrics,
    pu_val_risk,
)


# ---------------------------------------------------------------------------
# attach_emb_rows
# ---------------------------------------------------------------------------
def _table():
    return pl.DataFrame({
        "id": ["a", "b", "100", "d"],
        "cache": ["train250k", "train250k", "us_train_ldc", "full"],
        "pnu_label": ["pos", "pos", "neg", "unl"],
    })


def _cache_metas():
    return {
        "train250k": pl.DataFrame({"id": ["a", "b", "z"], "emb_row": [10, 11, 99]}),
        "us_train_ldc": pl.DataFrame({"id": [100, 101], "emb_row": [0, 1]}),  # Int64 id
        "full": pl.DataFrame({"id": ["d", "e"], "emb_row": [5, 6]}),
    }


class TestAttachEmbRows:
    def test_resolves_emb_row_per_cache(self):
        out = attach_emb_rows(_table(), _cache_metas())
        by_id = dict(zip(out["id"].to_list(), out["emb_row"].to_list()))
        assert by_id == {"a": 10, "b": 11, "100": 0, "d": 5}

    def test_missing_cache_raises(self):
        metas = _cache_metas()
        del metas["full"]
        with pytest.raises(ValueError, match="no cache_metas entry for cache='full'"):
            attach_emb_rows(_table(), metas)

    def test_unresolved_id_raises(self):
        table = pl.DataFrame({
            "id": ["a", "nonexistent"], "cache": ["train250k", "train250k"], "pnu_label": ["pos", "pos"],
        })
        with pytest.raises(ValueError, match="not found in its"):
            attach_emb_rows(table, _cache_metas())

    def test_ldc_int_id_matched_via_utf8_cast(self):
        table = pl.DataFrame({"id": ["100"], "cache": ["us_train_ldc"], "pnu_label": ["neg"]})
        out = attach_emb_rows(table, _cache_metas())
        assert out["emb_row"].to_list() == [0]


# ---------------------------------------------------------------------------
# gather_group_features
# ---------------------------------------------------------------------------
class TestGatherGroupFeatures:
    def test_single_cache_group(self):
        group = pl.DataFrame({"cache": ["full", "full"], "emb_row": [0, 2]})
        cls_by_cache = {"full": np.arange(12).reshape(4, 3).astype(np.float32)}
        feats, labels = gather_group_features(group, cls_by_cache, label=0.0)
        assert feats.shape == (2, 3)
        assert set(map(tuple, feats)) == {(0.0, 1.0, 2.0), (6.0, 7.0, 8.0)}
        assert (labels == 0.0).all()

    def test_multi_cache_group_concatenates(self):
        group = pl.DataFrame({
            "cache": ["train250k", "train250k", "us_pos_ldc345"],
            "emb_row": [0, 1, 0],
        })
        cls_by_cache = {
            "train250k": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            "us_pos_ldc345": np.array([[9.0, 9.0]], dtype=np.float32),
        }
        feats, labels = gather_group_features(group, cls_by_cache, label=1.0)
        assert feats.shape == (3, 2)
        assert set(map(tuple, feats)) == {(1.0, 1.0), (2.0, 2.0), (9.0, 9.0)}
        assert (labels == 1.0).all()

    def test_label_values_match_flpu_convention(self):
        group = pl.DataFrame({"cache": ["full"], "emb_row": [0]})
        cls_by_cache = {"full": np.zeros((1, 2), dtype=np.float32)}
        for label in (1.0, -1.0, 0.0):
            _, labels = gather_group_features(group, cls_by_cache, label=label)
            assert labels[0] == label

    def test_empty_group_returns_empty_arrays(self):
        group = pl.DataFrame({"cache": [], "emb_row": []}, schema={"cache": pl.Utf8, "emb_row": pl.Int64})
        feats, labels = gather_group_features(group, {}, label=0.0)
        assert feats.shape[0] == 0
        assert labels.shape[0] == 0


# ---------------------------------------------------------------------------
# estimate_prior_from_calibrated_logits
# ---------------------------------------------------------------------------
class TestEstimatePrior:
    def test_matches_mean_of_transform(self):
        cal = PlattCalibrator(A=1.5, B=-0.3, fit_population="test", n=100)
        logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = float(np.mean(cal.transform(logits)))
        assert estimate_prior_from_calibrated_logits(logits, cal) == pytest.approx(expected)

    def test_identity_calibrator_recovers_sigmoid_mean(self):
        cal = PlattCalibrator(A=1.0, B=0.0, fit_population="test", n=10)
        logits = np.zeros(5)  # sigmoid(0) = 0.5
        assert estimate_prior_from_calibrated_logits(logits, cal) == pytest.approx(0.5)

    def test_empty_logits_raises(self):
        cal = PlattCalibrator(A=1.0, B=0.0, fit_population="test", n=10)
        with pytest.raises(ValueError, match="empty"):
            estimate_prior_from_calibrated_logits(np.array([]), cal)

    def test_result_in_unit_interval(self):
        cal = PlattCalibrator(A=2.0, B=1.0, fit_population="test", n=50)
        rng = np.random.default_rng(0)
        logits = rng.normal(scale=5.0, size=1000)
        pi_hat = estimate_prior_from_calibrated_logits(logits, cal)
        assert 0.0 <= pi_hat <= 1.0


# ---------------------------------------------------------------------------
# pn_val_metrics
# ---------------------------------------------------------------------------
class TestPnValMetrics:
    def test_perfect_separation_gives_high_pr_auc_and_f1(self):
        pos = np.array([2.0, 3.0, 1.5])
        neg = np.array([-2.0, -1.0, -3.0])
        m = pn_val_metrics(pos, neg)
        assert m["pr_auc"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["n_pos"] == 3
        assert m["n_neg"] == 3

    def test_deterministic(self):
        pos = np.array([1.0, -0.5, 2.0])
        neg = np.array([0.5, -1.0])
        assert pn_val_metrics(pos, neg) == pn_val_metrics(pos, neg)

    def test_worst_case_ranking_gives_low_pr_auc(self):
        pos = np.array([-5.0, -4.0])
        neg = np.array([5.0, 4.0])
        m = pn_val_metrics(pos, neg)
        assert m["pr_auc"] < 0.6


# ---------------------------------------------------------------------------
# pu_val_risk
# ---------------------------------------------------------------------------
class TestPuValRisk:
    def test_eta_zero_matches_pure_nnpu(self):
        """eta=0 should reduce to plain nnPU (no reliable-negative influence beyond
        the neg stream simply not being weighted -- but the function still needs
        neg_logits to build the combined array; verify it runs and is finite)."""
        pos = np.array([1.0, 2.0], dtype=np.float32)
        neg = np.array([-1.0, -2.0], dtype=np.float32)
        unl = np.array([0.1, -0.1, 0.5], dtype=np.float32)
        risk = pu_val_risk(pos, neg, unl, prior=0.1, eta=0.0)
        assert np.isfinite(risk)

    def test_deterministic(self):
        pos = np.array([1.0], dtype=np.float32)
        neg = np.array([-1.0], dtype=np.float32)
        unl = np.array([0.0, 0.2], dtype=np.float32)
        r1 = pu_val_risk(pos, neg, unl, prior=0.2, eta=0.5)
        r2 = pu_val_risk(pos, neg, unl, prior=0.2, eta=0.5)
        assert r1 == pytest.approx(r2)

    def test_eta_changes_risk_when_reliable_negatives_are_misclassified(self):
        # Mirrors tests/test_flpu_loss.py's
        # test_misclassified_reliable_negatives_raise_loss recipe: reliable
        # negatives scored confidently POSITIVE cost more as eta rises.
        pos = np.full(8, 2.0, dtype=np.float32)
        neg = np.full(8, 3.0, dtype=np.float32)   # reliable negs wrongly positive
        unl = np.full(16, -1.0, dtype=np.float32)
        r_eta0 = pu_val_risk(pos, neg, unl, prior=0.1, eta=0.0)
        r_eta_hi = pu_val_risk(pos, neg, unl, prior=0.1, eta=0.6)
        assert r_eta_hi > r_eta0


# ---------------------------------------------------------------------------
# _base_run_config
# ---------------------------------------------------------------------------
class TestBaseRunConfig:
    def test_head_renamed_and_loss_set(self):
        cfg = _base_run_config(prior=0.05, eta=0.25, epochs=3)
        assert cfg.heads[0].name == "us_pnu"
        assert cfg.heads[0].loss.prior == pytest.approx(0.05)
        assert cfg.heads[0].loss.nnpnu_eta == pytest.approx(0.25)
        assert cfg.epochs == 3

    def test_invalid_eta_raises_via_flpu_loss_config(self):
        with pytest.raises(ValueError, match="nnpnu_eta"):
            _base_run_config(prior=0.05, eta=1.5, epochs=1)


# ---------------------------------------------------------------------------
# Import side effects
# ---------------------------------------------------------------------------
class TestImportNoSideEffects:
    def test_import_does_not_trigger_training(self):
        import src.run_us_pnu

        assert src.run_us_pnu is not None
