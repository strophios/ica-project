"""CCA slice eval + score-stratified coding template (Phase 4).

- cca-doca.AC4.1: template is schema-conformant, cca_event null, prefix-stratified.
- cca-doca.AC4.2: evaluate_cca_slice computes P/R/F1 at a threshold, drops nulls.
"""

from __future__ import annotations

import polars as pl

import pytest

import numpy as np

from src.validation.cca_slice_eval import (
    band_ipw_weights,
    evaluate_cca_slice,
    evaluate_cca_slice_weighted,
    recall_at_thresholds,
)
from src.validation.build_cca_coding_template import assign_score_band, build_cca_template
from src.validation.schema import validate_gold_set


def test_evaluate_cca_slice_metrics_and_null_drop():
    gold = pl.DataFrame({
        "cca_event": [True, True, False, False, None],
        "cca_logit": [2.0, -1.0, 1.5, -2.0, 3.0],
    })
    # threshold 0.0: preds = [T, F, T, F, (dropped)]
    # among coded (non-null): tp=1 (row0), fn=1 (row1), fp=1 (row2), tn=1 (row3)
    r = evaluate_cca_slice(gold, threshold=0.0)
    assert r["n_pos"] == 2 and r["n_neg"] == 2  # null row excluded
    assert r["precision"] == 0.5  # tp/(tp+fp) = 1/2
    assert r["recall"] == 0.5     # tp/(tp+fn) = 1/2
    assert abs(r["f1"] - 0.5) < 1e-9


def _scored(n_unl=300, n_pos=50):
    rows = []
    # unlabeled spread across logit bands
    for i in range(n_unl):
        logit = -4.0 + 8.0 * (i / n_unl)  # -4 .. +4
        rows.append({"id": f"u{i}", "year": str(1960 + i % 36),
                     "news_desk": "National Desk", "section_name": "A",
                     "headline": f"hl {i}", "lead_paragraph": f"lead {i}",
                     "cca_label": 0, "cca_logit": logit})
    for i in range(n_pos):  # positives must be excluded from the template
        rows.append({"id": f"p{i}", "year": "1980", "news_desk": "Metro",
                     "section_name": "B", "headline": f"pos {i}",
                     "lead_paragraph": "x", "cca_label": 1, "cca_logit": 5.0})
    return pl.DataFrame(rows)


def test_build_cca_template_schema_and_exclusions():
    scored = _scored()
    tmpl = build_cca_template(scored, alloc={"cca_score_high": 40, "cca_score_mid": 30,
                                             "cca_score_low": 30}, seed=200)
    validate_gold_set(tmpl)  # schema-conformant (also raises inside build, belt+braces)
    # cca_event null for coding (AC4.1)
    assert tmpl["cca_event"].null_count() == tmpl.height
    # only unlabeled sampled -- no positives leaked in
    assert not any(i.startswith("p") for i in tmpl["id"].to_list())
    # all strata are CCA bands
    assert set(tmpl["sample_stratum"].unique()).issubset(
        {"cca_score_high", "cca_score_mid", "cca_score_low"}
    )
    # year cast to Int64
    assert tmpl.schema["year"] == pl.Int64


def test_band_ipw_weights_are_corpus_over_gold():
    w = band_ipw_weights(
        gold_band_counts={"cca_score_high": 148, "cca_score_low": 176},
        corpus_band_counts={"cca_score_high": 753, "cca_score_low": 217062},
    )
    assert w["cca_score_high"] == pytest.approx(753 / 148)
    assert w["cca_score_low"] == pytest.approx(217062 / 176)


def test_band_ipw_weights_reject_empty_corpus_or_gold():
    with pytest.raises(ValueError):  # band coded but absent from corpus
        band_ipw_weights({"cca_score_mid": 10}, {"cca_score_high": 5})
    with pytest.raises(ValueError):  # non-positive gold count
        band_ipw_weights({"cca_score_mid": 0}, {"cca_score_mid": 5})


def test_weighted_equals_raw_under_uniform_sampling():
    # Oracle: equal corpus/gold ratio across bands -> uniform weights -> the
    # reweighted metrics collapse to the raw (unweighted) ones.
    gold = pl.DataFrame({
        "cca_event": [True, False, True, False],
        "cca_logit": [2.0, 1.5, -2.0, -1.5],  # thr 0 -> preds [T, T, F, F]
        "sample_stratum": ["cca_score_high", "cca_score_high",
                           "cca_score_low", "cca_score_low"],
    })
    w = band_ipw_weights({"cca_score_high": 2, "cca_score_low": 2},
                         {"cca_score_high": 20, "cca_score_low": 20})  # both -> 10
    gold = gold.with_columns(
        pl.col("sample_stratum").replace_strict(w).alias("ipw")
    )
    raw = evaluate_cca_slice(gold, threshold=0.0)
    wtd = evaluate_cca_slice_weighted(gold, threshold=0.0)
    assert wtd["precision"] == pytest.approx(raw["precision"])
    assert wtd["recall"] == pytest.approx(raw["recall"])
    assert wtd["f1"] == pytest.approx(raw["f1"])


def test_weighted_recall_corrects_stratification_bias():
    # One positive in a light band (caught) + one in a heavy band (missed).
    # Raw recall = 1/2; corpus recall is dominated by the heavy-band miss.
    gold = pl.DataFrame({
        "cca_event": [True, True],
        "cca_logit": [2.0, -2.0],      # thr 0 -> [caught, missed]
        "sample_stratum": ["cca_score_high", "cca_score_low"],
        "ipw": [1.0, 100.0],
    })
    raw = evaluate_cca_slice(gold, threshold=0.0)
    wtd = evaluate_cca_slice_weighted(gold, threshold=0.0)
    assert raw["recall"] == pytest.approx(0.5)
    assert wtd["recall"] == pytest.approx(1.0 / 101.0)
    assert wtd["support_tp"] == 1 and wtd["support_fn"] == 1


def test_recall_at_thresholds_counts_fraction_above():
    scores = np.array([2.0, -1.0, 0.5])  # all known positives
    out = recall_at_thresholds(scores, (0.0, 1.0))
    assert out[0] == {"threshold": 0.0, "recall": 2 / 3, "n": 3}  # 2 of 3 >= 0
    assert out[1] == {"threshold": 1.0, "recall": 1 / 3, "n": 3}  # 1 of 3 >= 1


def test_recall_at_thresholds_empty_is_zero():
    out = recall_at_thresholds(np.array([]), (0.0,))
    assert out == [{"threshold": 0.0, "recall": 0.0, "n": 0}]


def test_assign_score_band_boundaries():
    df = pl.DataFrame({"cca_logit": [1.0, 0.99, -1.0, -1.01, 0.0]})
    bands = df.select(assign_score_band(pl.col("cca_logit")).alias("b"))["b"].to_list()
    # >= 1.0 high; < -1.0 low; else mid. Note -1.0 is mid (boundary is strict <).
    assert bands == ["cca_score_high", "cca_score_mid", "cca_score_mid",
                     "cca_score_low", "cca_score_mid"]


def test_build_cca_template_prefix_is_stratified():
    scored = _scored()
    tmpl = build_cca_template(scored, alloc={"cca_score_high": 40, "cca_score_mid": 40,
                                             "cca_score_low": 40}, seed=200)
    # the first third should already contain all three bands (interleaved order)
    prefix = tmpl.head(tmpl.height // 3)
    assert prefix["sample_stratum"].n_unique() == 3
