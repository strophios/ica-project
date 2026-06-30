# pattern: Imperative Shell (I/O in main; pure lexicon-matching core below)
"""Build the immigrant-relevance POSITIVE candidate set from NYT descriptors.

The immigrant-relevance head's positives come from two complementary sources,
unioned here:

  1. DESCRIPTOR-SELECTED breadth -- API-corpus articles whose NYT indexer
     keywords (the `keywords` list-column: subject / organizations / glocations,
     each {type,value,rank,major}) match a curated immigration-CONTENT lexicon.
     This supplies non-protest coverage the anchors lack.
  2. HAND-VERIFIED anchors -- the 466 ICA-matched articles exported by
     r/doca/export_ica_anchors.R. Clean but protest-confined.

Design decisions (see docs/notes for the full reasoning):

* CURATED INCLUDE-LIST, not stem-matching. Stem-matching ("immigr", "visa",
  "alien") drags in false friends -- `Visa USA Inc`, `Emigrant Savings Bank`,
  `NAZI POLICIES TOWARD JEWS AND FOREIGN NATIONALITIES`, `Inter-American
  Development Bank`. An include-list never matches those in the first place.

* NORMALIZED matching (casefold + collapse non-alphanumerics). The NYT indexer
  vocabulary drifts across the period (ALLCAPS early -> Titlecase late) and
  carries punctuation/unicode-hyphen variants. Normalization folds
  `ASYLUM, POLITICAL` / `Asylum (Political)` and the three spellings of
  `FOREIGN POPULATION AND FOREIGN-DESCENT GROUPS` into one match each.

* FORM descriptors are EXCLUDED by construction. `Demonstrations and Riots`,
  `Hunger Strikes`, `Boycotts`, `Assembly`, `Assaults` rank high in the anchors
  ONLY because the anchors are all protests (a confound). They carry no
  immigration signal (confirmed: ~0 discrimination in the gold contrast), and
  including them as selectors would inject non-immigration protests and re-couple
  the relevance head to the CCA head. They are simply absent from the lexicon.

* GLOCATIONS are NOT selectors. Country tags (Cuba, Haiti, Iran) are
  high-recall / low-precision for relevance; left to the embedding head + anchors.

Run from the project root:
    uv run python -m scripts.build_relevance_candidates
Writes  <PROJECT_ROOT>/relevance/candidates.parquet  with columns
    id, year, headline, lead_paragraph, matched (list[str]), is_anchor (bool).
"""

from __future__ import annotations

import glob
import re

import polars as pl

import src.config as config

# --------------------------------------------------------------------------- #
# Lexicon (the checked-in record of what counts as an immigration-content tag)  #
# --------------------------------------------------------------------------- #

# Content subjects / organizations. Listed in display form; matched normalized,
# so case/punctuation variants in the corpus fold in automatically.
CONTENT_DESCRIPTORS: tuple[str, ...] = (
    "Immigration and Emigration",
    "Refugees and Expatriates",
    "Refugees",
    "Illegal Aliens",
    "Immigration and Refugees",
    "Deportation",
    "Asylum (Political)",
    "Asylum, Political",
    "Asylum, Right of",
    "Foreign Population and Foreign-Descent Groups",
    "Foreign Population",
    "Immigration and Naturalization Service",
    "Immigration and Naturalization Service (US)",
    "Bilingual Education",
    "Migrant Labor",
    "Migrant Workers Children",
    "Visas",
    "Immigration Reform and Control Act of 1986",
    "Curbs on Aliens",
    "Denaturalization and Deportation of Criminals",
    "Exiles",  # kept; EXPATRIATES deliberately dropped (Americans-abroad sense)
)

# Hyphenated ethnic tags ("<nationality>-American(s)"). DISABLED for pass 1
# (INCLUDE_ETHNIC=False): on US-gated face validity this tier was mostly noise --
# the ethnic tag is usually *incidental* to the story (ARAB-AMERICANS on an FBI
# surveillance piece, ASIAN-AMERICANS on the O.J. trial), because ethnicity is a
# poor article-level proxy for "special and specific relevance to immigrants".
# Exclusionary recall it would add is better recovered later by the typed head +
# anchors. Kept here, gated, so the decision is recorded and reversible.
INCLUDE_ETHNIC: bool = False

# Leading tokens that are NOT nationalities (corporate / rhetorical false friends:
# "All-American Engineering", "Un-American Activities") plus the ETHNICITIES
# excluded per the original ICA coding (groups very unlikely to be immigrants).
ETHNIC_DENY: frozenset[str] = frozenset({
    "african", "white", "jewish", "native",          # ICA-coding exclusions
    "all", "un", "anti", "non", "pan", "inter", "latin",  # not-a-nationality
})

# Substrings that mark an "-american" value as a corporate / institutional /
# historical false friend rather than an ethnic-community tag.
ETHNIC_FALSE_FRIENDS: tuple[str, ...] = (
    "bank", "corp", "inc", "games", " war", "council", "assn", "association",
    "press", "development", "economic", "relations", "center", "museum",
    "journal", "inter-", "pan-",
)

_ETHNIC_RE = re.compile(r"([a-z]+)-americans?")


def normalize(value: str) -> str:
    """Casefold and collapse non-alphanumeric runs to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


CONTENT_NORMALIZED: frozenset[str] = frozenset(normalize(d) for d in CONTENT_DESCRIPTORS)


def is_ethnic_match(value_lower: str) -> bool:
    """True for a `<nationality>-American(s)` community tag, excluding denied
    ethnicities and corporate/institutional false friends."""
    m = _ETHNIC_RE.search(value_lower)
    if m is None:
        return False
    if m.group(1) in ETHNIC_DENY:
        return False
    return not any(ff in value_lower for ff in ETHNIC_FALSE_FRIENDS)


# --------------------------------------------------------------------------- #
# Selection core (pure: dataframe in -> matched (id, value) rows out)           #
# --------------------------------------------------------------------------- #

def select_matches(articles: pl.DataFrame) -> pl.DataFrame:
    """Given an articles frame with `id` and the exploded keyword struct already
    flattened to (`id`, `type`, `value`), return matched (`id`, `value`) rows."""
    low = articles.with_columns(pl.col("value").str.to_lowercase().alias("low"))
    low = low.with_columns(
        pl.col("low").str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars().alias("nm")
    )
    content = low.filter(pl.col("nm").is_in(list(CONTENT_NORMALIZED)))
    if not INCLUDE_ETHNIC:
        return content.select("id", "value")
    ethnic = low.filter(
        pl.col("low").map_elements(is_ethnic_match, return_dtype=pl.Boolean)
    )
    return pl.concat([content.select("id", "value"), ethnic.select("id", "value")])


# --------------------------------------------------------------------------- #
# Imperative shell                                                              #
# --------------------------------------------------------------------------- #

def main() -> None:
    relevance_dir = config.PROJECT_ROOT / "relevance"
    relevance_dir.mkdir(parents=True, exist_ok=True)
    anchors_path = relevance_dir / "ica_anchors.parquet"
    out_path = relevance_dir / "candidates.parquet"

    anchors = pl.read_parquet(anchors_path)
    anchor_ids = set(anchors["article_id"].unique().to_list())
    print(f"anchors: {len(anchor_ids)} unique articles")

    parts: list[pl.DataFrame] = []
    for f in sorted(glob.glob(str(config.API_CORPUS_DIR / "*.parquet"))):
        df = pl.read_parquet(
            f, columns=["id", "year", "headline", "lead_paragraph", "keywords"]
        ).filter(pl.col("keywords").is_not_null())
        df = df.with_columns(pl.col("year").cast(pl.Int32, strict=False))
        exploded = (
            df.select("id", "keywords")
            .explode("keywords")
            .unnest("keywords")
            .filter(pl.col("type").is_in(["subject", "organizations"]))
            .select("id", "type", "value")
        )
        matched = select_matches(exploded)
        if matched.height:
            grouped = matched.group_by("id").agg(pl.col("value").unique().alias("matched"))
            grouped = grouped.join(
                df.select("id", "year", "headline", "lead_paragraph"), on="id", how="left"
            )
            parts.append(grouped)

    candidates = pl.concat(parts).unique(subset="id")
    candidates = candidates.with_columns(
        pl.col("id").is_in(list(anchor_ids)).alias("is_anchor")
    )

    # Anchors not caught by any descriptor: add as positive rows (matched = null).
    caught = set(candidates["id"].to_list())
    anchor_only_ids = [a for a in anchor_ids if a not in caught]
    if anchor_only_ids:
        anchor_only = pl.DataFrame({"id": anchor_only_ids}).with_columns(
            matched=pl.lit(None, dtype=pl.List(pl.String)),
            year=pl.lit(None, dtype=pl.Int32),
            headline=pl.lit(None, dtype=pl.String),
            lead_paragraph=pl.lit(None, dtype=pl.String),
            is_anchor=pl.lit(True),
        )
        candidates = pl.concat([candidates, anchor_only], how="diagonal")

    candidates = candidates.unique(subset="id")
    candidates.write_parquet(out_path)

    n_desc = candidates.filter(pl.col("matched").is_not_null()).height
    print(f"descriptor-selected: {n_desc}")
    print(f"anchor-only (no descriptor): {len(anchor_only_ids)}")
    print(f"UNION positives written: {candidates.height} -> {out_path}")


if __name__ == "__main__":
    main()
