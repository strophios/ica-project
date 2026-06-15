"""CCA slice eval + score-stratified coding template (Phase 4).

- cca-doca.AC4.1: template is schema-conformant, cca_event null, prefix-stratified.
- cca-doca.AC4.2: evaluate_cca_slice computes P/R/F1 at a threshold, drops nulls.
"""

from __future__ import annotations

import polars as pl

from src.validation.cca_slice_eval import evaluate_cca_slice
from src.validation.build_cca_coding_template import build_cca_template
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


def test_build_cca_template_prefix_is_stratified():
    scored = _scored()
    tmpl = build_cca_template(scored, alloc={"cca_score_high": 40, "cca_score_mid": 40,
                                             "cca_score_low": 40}, seed=200)
    # the first third should already contain all three bands (interleaved order)
    prefix = tmpl.head(tmpl.height // 3)
    assert prefix["sample_stratum"].n_unique() == 3
