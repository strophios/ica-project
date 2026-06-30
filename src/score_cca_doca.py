# pattern: Imperative Shell
"""
Score the embedding cache with the trained CCA model + face-validity dump.

Applies the DoCA-trained CCA head to the cached CLS vectors, joins headlines/desk
from the API corpus, writes a scored-candidate parquet (input to the gold-set
builder), and prints/writes a face-validity view: top-scored overall, top
"discovered" (high-scoring unlabeled = candidate CCA events DoCA didn't match),
and positive "misses" (DoCA positives the model scores low).

Run from project root:
    uv run python -m src.score_cca_doca --suffix train250k --top-n 40
"""

from __future__ import annotations

import argparse

import polars as pl

import src.config as config
from src.embed_corpus import load_cache
from src.build_cca_doca_table import label_and_restrict
from src.validation.cca_slice_eval import apply_cca_model

_DISPLAY = ["cca_logit", "cca_label", "year", "news_desk", "headline"]


def score_cache(suffix: str = "train250k", threshold: float = 0.0) -> pl.DataFrame:
    """Score the cache and join corpus text fields.

    Returns: id, year, news_desk, section_name, headline, lead_paragraph,
    us_logit, cca_label, us, cca_logit (one row per cached article).
    """
    meta, cls = load_cache(config.CCA_EMBED_CACHE_DIR / suffix)
    positives = pl.read_parquet(config.CCA_DOCA_POSITIVES)["id"].to_list()
    table = label_and_restrict(meta, positives, threshold)
    table = table.with_columns(pl.Series("cca_logit", apply_cca_model(cls)))

    api = (
        pl.scan_parquet(f"{config.API_CORPUS_DIR}/**/*.parquet")
        .select(["id", "headline", "lead_paragraph", "news_desk", "section_name"])
        .collect()
    )
    return table.join(api, on="id", how="left")


def _show(title: str, df: pl.DataFrame) -> None:
    print(f"\n=== {title} ===")  # LOG
    for r in df.select(_DISPLAY).iter_rows(named=True):
        tag = "POS" if r["cca_label"] == 1 else "unl"
        hl = (r["headline"] or "")[:90]
        print(f"  [{r['cca_logit']:+6.2f}] {tag} {r['year']} {r['news_desk'] or '?':<12.12} {hl}")  # LOG


def main(suffix: str = "train250k", threshold: float = 0.0, top_n: int = 40) -> None:
    scored = score_cache(suffix, threshold)
    config.CCA_DOCA_DIR.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(config.CCA_DOCA_DIR / "scored_candidates.parquet")

    top = scored.sort("cca_logit", descending=True).head(top_n)
    discovered = (
        scored.filter(pl.col("cca_label") == 0)
        .sort("cca_logit", descending=True).head(top_n)
    )
    misses = (
        scored.filter(pl.col("cca_label") == 1)
        .sort("cca_logit").head(top_n)
    )

    _show("TOP CCA-SCORED (overall)", top)
    _show("TOP 'DISCOVERED' (unlabeled, highest CCA score)", discovered)
    _show("POSITIVE MISSES (DoCA positives, lowest CCA score)", misses)

    for name, df in [("top", top), ("discovered", discovered), ("misses", misses)]:
        out = config.CCA_DOCA_DIR / f"face_validity_{name}.csv"
        df.select(_DISPLAY + ["lead_paragraph", "id"]).write_csv(out)
    print(f"\nWrote scored_candidates.parquet + face_validity_*.csv to "
          f"{config.CCA_DOCA_DIR}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score cache with CCA model + face validity.")
    ap.add_argument("--suffix", default="train250k")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args()
    main(suffix=args.suffix, threshold=args.threshold, top_n=args.top_n)
