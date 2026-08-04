"""Tests for the pure helpers in scripts/topline_candidates_eval.py.

The topline eval reports anchor-set recall (DoCA matches, hand-coded ICA
positives) against the ranked candidates files. Pure core: rank lookup and
recall-at-budget; the shell (parquet I/O, joins, CSV dumps) is exercised by
running the script on the real files.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scripts.topline_candidates_eval import anchor_ranks, recall_at_top


def _ranked(ids_scores):
    return pl.DataFrame(
        {"id": [i for i, _ in ids_scores], "ica_score": [s for _, s in ids_scores]}
    ).sort("ica_score", descending=True)


class TestAnchorRanks:
    def test_ranks_are_one_based_positions_in_score_order(self):
        df = _ranked([("a", 0.9), ("b", 0.5), ("c", 0.1)])
        found, ranks = anchor_ranks(df, ["b", "c"])
        assert found == 2
        assert sorted(ranks.tolist()) == [2, 3]

    def test_missing_anchors_counted_not_ranked(self):
        df = _ranked([("a", 0.9), ("b", 0.5)])
        found, ranks = anchor_ranks(df, ["b", "zzz", "yyy"])
        assert found == 1
        assert ranks.tolist() == [2]

    def test_empty_anchor_set(self):
        df = _ranked([("a", 0.9)])
        found, ranks = anchor_ranks(df, [])
        assert found == 0
        assert ranks.size == 0


class TestRecallAtTop:
    def test_recall_counts_ranks_within_budget_over_all_anchors(self):
        """Denominator is ALL anchors (found + missing) — an anchor absent
        from the candidates is a miss, not an exclusion."""
        ranks = np.array([1, 5, 100])
        assert recall_at_top(ranks, n_anchors=4, k=10) == 0.5  # 2 of 4

    def test_budget_boundary_inclusive(self):
        ranks = np.array([10])
        assert recall_at_top(ranks, n_anchors=1, k=10) == 1.0
        assert recall_at_top(ranks, n_anchors=1, k=9) == 0.0

    def test_zero_anchors_is_zero(self):
        assert recall_at_top(np.array([]), n_anchors=0, k=10) == 0.0


class TestRecallAtThreshold:
    def test_fraction_of_all_anchors_scoring_at_or_above_t(self):
        from scripts.memo_pr_tables import recall_at_threshold
        import numpy as np
        scores = np.array([0.9, 0.5, 0.1])   # scores of FOUND anchors
        assert recall_at_threshold(scores, n_anchors=4, t=0.5) == 0.5  # 2 of 4

    def test_boundary_inclusive_and_empty(self):
        from scripts.memo_pr_tables import recall_at_threshold
        import numpy as np
        assert recall_at_threshold(np.array([0.5]), n_anchors=1, t=0.5) == 1.0
        assert recall_at_threshold(np.array([]), n_anchors=0, t=0.5) == 0.0
