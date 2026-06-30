# pattern: Imperative Shell (gazetteer predicates are the pure core)
"""Select reliable NEGATIVES for the relevance head's nnPNU loss.

The pass-1 relevance head over-fires on US-datelined FOREIGN news (the dateline-
based US gate can't catch it). These confidently-foreign, no-US-footprint articles
are fed to the loss as reliable negatives (label -1) so the head learns to reject
them -- which should also sharpen the diaspora boundary by contrast.

Selection (conservative, to avoid mislabeling immigrant-relevant DIASPORA as
negative -- diaspora tags US enclaves + ethnicities, so the guards exclude it):

  REQUIRE  a foreign-COUNTRY glocation as the PRIMARY location (rank 1). A
           positive foreign signal (gazetteer country/foreign-city) is robust;
           "absence of US" alone leaked bare US city names (Syracuse, Bronx).
  EXCLUDE  any US glocation (United States / state / US-state parenthetical /
           AP US city) -- "fully foreign, no US footprint";
           any immigration-content descriptor (the relevance positive lexicon);
           any "-American" ethnic tag (Cuban-Americans -> likely diaspora).
  RESTRICT to cache articles that pass the US gate (us_logit >= threshold) and are
           neither relevance positives nor ICA anchors (asserted zero overlap).

Validated on the 500-row gold set: of the selected gold rows, ~92% are coded
immig=False (the axis that matters -- we are not suppressing relevance). Gazetteers
reused from the US-filter dateline resolver (r/dateline/gazetteers/).

Run from the project root:
    uv run python -m scripts.build_reliable_negatives
Writes <PROJECT_ROOT>/relevance/reliable_negatives.parquet with column `id`.
"""

from __future__ import annotations

import argparse
import glob
import re

import polars as pl

import src.config as config
from scripts.build_relevance_candidates import CONTENT_NORMALIZED
from src.embed_corpus import load_cache_meta


def _load_gazetteers():
    base = _repo_root() / "r" / "dateline" / "gazetteers"
    countries = set(pl.read_csv(base / "countries.csv")["value"].str.to_lowercase().to_list())
    countries.discard("united states")
    us_cities = set(pl.read_csv(base / "ap_us_cities.csv")["city"].str.to_lowercase().to_list())
    fcities = set(pl.read_csv(base / "ap_foreign_cities.csv")["city"].str.to_lowercase().to_list())
    sdf = pl.read_csv(base / "state_long_abbrs.csv")
    states = set(sdf["full"].str.to_lowercase().to_list())
    sabbr = (
        {re.sub(r"[.\s]", "", x.lower()) for x in sdf["long_abbr"].to_list()}
        | {x.lower() for x in sdf["usps_abbr"].to_list()}
        | {"nyc"}
    )
    return countries, us_cities, fcities, states, sabbr


def _repo_root():
    # This file lives at <repo>/scripts/build_reliable_negatives.py
    import pathlib
    return pathlib.Path(__file__).resolve().parents[1]


def _parts(value: str):
    vl = value.lower().strip()
    m = re.search(r"\(([^)]+)\)", vl)
    par = m.group(1).strip() if m else None
    base = re.sub(r"\s*\([^)]*\)", "", vl).strip()
    return vl, par, base


def make_predicates(countries, us_cities, fcities, states, sabbr):
    def is_us_gloc(value: str) -> bool:
        vl, par, base = _parts(value)
        if "united states" in vl or "u.s." in vl:
            return True
        if base in states or base in us_cities:
            return True
        return par is not None and (re.sub(r"[.\s]", "", par) in sabbr or par in states)

    def is_foreign_country(value: str) -> bool:
        _, par, base = _parts(value)
        return base in countries or base in fcities or (
            par is not None and (par in countries or par in fcities)
        )

    return is_us_gloc, is_foreign_country


def main(threshold: float = 0.5) -> None:
    relevance_dir = config.PROJECT_ROOT / "relevance"
    is_us_gloc, is_foreign_country = make_predicates(*_load_gazetteers())

    meta = load_cache_meta(config.CCA_EMBED_CACHE_DIR / "relevance_train")
    pos = set(pl.read_parquet(relevance_dir / "candidates.parquet")["id"].to_list())
    anchors = set(pl.read_parquet(relevance_dir / "ica_anchors.parquet")["article_id"].to_list())
    bg_ids = set(
        meta.filter((pl.col("us_logit") >= threshold) & ~pl.col("id").is_in(list(pos | anchors)))["id"].to_list()
    )
    print(f"US-passing background (non-positive, non-anchor): {len(bg_ids):,}")  # LOG

    parts: list[pl.DataFrame] = []
    for f in sorted(glob.glob(str(config.API_CORPUS_DIR / "*.parquet"))):
        d = pl.read_parquet(f, columns=["id", "keywords"]).filter(
            pl.col("id").is_in(list(bg_ids)) & pl.col("keywords").is_not_null()
        )
        if d.height:
            parts.append(d.explode("keywords").unnest("keywords").select("id", "type", "value", "rank"))
    ex = pl.concat(parts)

    gl = ex.filter(pl.col("type") == "glocations").with_columns(
        pl.col("value").map_elements(is_us_gloc, return_dtype=pl.Boolean).alias("us"),
        pl.col("value").map_elements(is_foreign_country, return_dtype=pl.Boolean).alias("fc"),
    )
    per = gl.group_by("id").agg(
        has_us=pl.col("us").any(),
        fc_rank=pl.when(pl.col("fc")).then(pl.col("rank")).otherwise(None).min(),
    )
    sub = ex.filter(pl.col("type").is_in(["subject", "organizations"])).with_columns(
        pl.col("value").str.to_lowercase().str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().alias("nm")
    )
    content_ids = set(sub.filter(pl.col("nm").is_in(list(CONTENT_NORMALIZED)))["id"].to_list())
    ethnic_ids = set(sub.filter(pl.col("value").str.to_lowercase().str.contains(r"[a-z]+-american"))["id"].to_list())

    rel = per.filter(
        (pl.col("fc_rank") <= 1)
        & ~pl.col("has_us")
        & ~pl.col("id").is_in(list(content_ids))
        & ~pl.col("id").is_in(list(ethnic_ids))
    )
    rel_ids = rel.select("id")
    overlap = set(rel_ids["id"].to_list()) & (pos | anchors)
    assert not overlap, f"reliable negatives overlap positives/anchors: {len(overlap)}"

    out = relevance_dir / "reliable_negatives.parquet"
    rel_ids.write_parquet(out)
    print(f"reliable negatives: {rel_ids.height:,} -> {out}")  # LOG


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Select reliable foreign negatives for the relevance head.")
    ap.add_argument("--threshold", type=float, default=0.5, help="calibrated US prob gate")
    args = ap.parse_args()
    main(threshold=args.threshold)
