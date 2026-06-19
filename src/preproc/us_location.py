# pattern: Functional Core (pure predicates; one cached gazetteer load)
"""US/not-US LOCATION signal from NYT glocation keywords + desk/section.

A Python port of the vendored `r/vendored/us_assign.R` location heuristic, used to
make the US gate smarter than the dateline-based ML filter alone. The ML filter
gates on where an article is *reported from* (dateline); this adds where the
*event* is, read off the indexer `glocations` and the desk/section.

The load-bearing distinction (validated on the gold `us_event` set): an article
is **clearly foreign** when it has a foreign location signal and NO US one
(`any_not_us AND NOT any_us`). Diaspora events carry a US location (Miami,
Brooklyn) so they are NOT clearly foreign -- which is exactly the foreign-news /
diaspora separation the content-based relevance head could not make.

BOUNDARY-INVENTORY PAIR with `r/vendored/us_assign.R`: the desk/section lists and
the US/foreign place logic mirror that heuristic. Any change to the place
gazetteers or the desk/section constants in either file should be reflected in the
other. Gazetteers are the shared CSVs under `r/dateline/gazetteers/`.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_GAZETTEER_DIR = Path(__file__).resolve().parents[2] / "r" / "dateline" / "gazetteers"

# Desk/section US-signal lists (us_assign.R lines 33-50, str_to_lower'd).
SECTION_US: frozenset[str] = frozenset({
    "u.s.", "new york", "new york and region", "washington",
})
SECTION_NON_US: frozenset[str] = frozenset({"world"})
DSK_US: frozenset[str] = frozenset({
    "national desk", "metropolitan desk", "connecticut weekly desk",
    "westchester weekly desk", "long island weekly desk", "new york region",
    "new jersey weekly desk", "the city weekly desk",
})
DSK_NON_US: frozenset[str] = frozenset({"foreign desk"})


@dataclass(frozen=True)
class PlaceSets:
    """Gazetteer place name sets (all lowercased)."""

    countries: frozenset[str]
    us_cities: frozenset[str]
    foreign_cities: frozenset[str]
    states: frozenset[str]
    state_abbrevs: frozenset[str]


@functools.lru_cache(maxsize=1)
def load_place_sets() -> PlaceSets:
    """Load the shared dateline-resolver gazetteers (cached)."""
    def col(name: str, c: str) -> list[str]:
        return pl.read_csv(_GAZETTEER_DIR / name)[c].str.to_lowercase().to_list()

    sdf = pl.read_csv(_GAZETTEER_DIR / "state_long_abbrs.csv")
    abbrevs = {re.sub(r"[.\s]", "", x.lower()) for x in sdf["long_abbr"].to_list()}
    abbrevs |= {x.lower() for x in sdf["usps_abbr"].to_list()}
    abbrevs.add("nyc")
    countries = set(col("countries.csv", "value"))
    countries.discard("united states")
    return PlaceSets(
        countries=frozenset(countries),
        us_cities=frozenset(col("ap_us_cities.csv", "city")),
        foreign_cities=frozenset(col("ap_foreign_cities.csv", "city")),
        states=frozenset(sdf["full"].str.to_lowercase().to_list()),
        state_abbrevs=frozenset(abbrevs),
    )


def _parts(value: str) -> tuple[str, str | None, str]:
    """(lowercased value, parenthetical content, base without parenthetical)."""
    vl = value.lower().strip()
    m = re.search(r"\(([^)]+)\)", vl)
    par = m.group(1).strip() if m else None
    base = re.sub(r"\s*\([^)]*\)", "", vl).strip()
    return vl, par, base


def is_us_place(value: str, places: PlaceSets) -> bool:
    """A glocation that denotes a US place (country='United States', a state, a
    US-state parenthetical like 'Miami (Fla)', or an AP US city)."""
    vl, par, base = _parts(value)
    if "united states" in vl or "u.s." in vl:
        return True
    if base in places.states or base in places.us_cities:
        return True
    return par is not None and (
        re.sub(r"[.\s]", "", par) in places.state_abbrevs or par in places.states
    )


def is_foreign_place(value: str, places: PlaceSets) -> bool:
    """A glocation that denotes a foreign country or AP foreign city."""
    _, par, base = _parts(value)
    return base in places.countries or base in places.foreign_cities or (
        par is not None and (par in places.countries or par in places.foreign_cities)
    )


def location_signals(
    glocations: list[str], news_desk: str | None, section_name: str | None,
    places: PlaceSets | None = None,
) -> tuple[bool, bool]:
    """Return (any_us, any_not_us) fusing glocation place signals with desk/section
    signals, mirroring `us_assign`'s `any_us` / `any_not_us`."""
    places = places or load_place_sets()
    loc_us = any(is_us_place(v, places) for v in glocations)
    loc_not = any(is_foreign_place(v, places) for v in glocations)
    desk = (news_desk or "").lower()
    section = (section_name or "").lower()
    meta_us = section in SECTION_US or desk in DSK_US
    meta_not = section in SECTION_NON_US or desk in DSK_NON_US
    return (loc_us or meta_us), (loc_not or meta_not)


def compute_location_signals(
    articles: pl.DataFrame, places: PlaceSets | None = None
) -> pl.DataFrame:
    """Pure: from an articles frame (`id`, `keywords` list-struct, `news_desk`,
    `section_name`) compute per-article `any_us` / `any_not_us`.

    Vectorized: classify the UNIQUE glocation values once (far fewer than rows),
    aggregate `any` per id, then fuse the desk/section signals. Articles with no
    location keyword fall back to their desk/section signal (both False if neither).
    """
    places = places or load_place_sets()
    glocs = (
        articles.select("id", "keywords")
        .explode("keywords")
        .unnest("keywords")
        .filter(pl.col("type").str.contains("location"))  # matches "glocations"
        .select("id", pl.col("value").alias("loc"))
    )
    uniq = glocs.select("loc").unique().with_columns(
        is_us=pl.col("loc").map_elements(lambda v: is_us_place(v, places), return_dtype=pl.Boolean),
        is_for=pl.col("loc").map_elements(lambda v: is_foreign_place(v, places), return_dtype=pl.Boolean),
    )
    per = (
        glocs.join(uniq, on="loc", how="left")
        .group_by("id")
        .agg(loc_us=pl.col("is_us").any(), loc_not=pl.col("is_for").any())
    )
    out = articles.select("id", "news_desk", "section_name").join(per, on="id", how="left")
    # fill_null("") on desk/section so is_in returns False (not null) on missing
    # metadata; null OR False is null under Kleene logic and would leak through.
    section = pl.col("section_name").str.to_lowercase().fill_null("")
    desk = pl.col("news_desk").str.to_lowercase().fill_null("")
    return out.select(
        "id",
        any_us=(
            pl.col("loc_us").fill_null(False)
            | section.is_in(list(SECTION_US))
            | desk.is_in(list(DSK_US))
        ),
        any_not_us=(
            pl.col("loc_not").fill_null(False)
            | section.is_in(list(SECTION_NON_US))
            | desk.is_in(list(DSK_NON_US))
        ),
    )


def is_clearly_foreign(any_us: bool, any_not_us: bool) -> bool:
    """True when the event is clearly foreign: a foreign location signal and NO US
    one. This is what the fused gate excludes on top of the ML US filter."""
    return any_not_us and not any_us


def passes_us_gate(
    us_logit: float, any_us: bool, any_not_us: bool, threshold: float = 0.5
) -> bool:
    """Fused US gate: the ML filter passes AND the event is not clearly foreign."""
    return us_logit >= threshold and not is_clearly_foreign(any_us, any_not_us)
