# pattern: Mixed (pure detector core + thin cached gazetteer I/O loader)
"""No-residue dateline guard.

`has_dateline_prefix` is a Python port of the *conditional-strip detection* half
of the R extractor `resolve_dateline` (r/dateline/resolve_dateline.R). The two are
a boundary-inventory pair: any change to the R extractor's credit-line / caps-block /
delimiter / conditional-stripping logic MUST be mirrored here, and vice versa.

CONDITIONAL STRIPPING SEMANTICS (critical for AC2.1):
The R extractor strips a caps-block dateline prefix only when it:
  1. Contains a date field (e.g., "WASHINGTON, July 30"), OR
  2. Has a qualifier resolving against US states/countries (e.g., "LISBON, Portugal"), OR
  3. Is a bare city in the AP-30 (US) or AP-46 (foreign) lists (e.g., "NEW YORK")

Emphasis-caps ledes without any of the above (e.g., "PILOBOLUS - that dance troupe")
are deliberately NOT treated as datelines and are NOT stripped. This guard must mirror
that decision to avoid false positives on unstripped emphasis-caps leads.

This guard is the load-bearing leakage check: a dateline left in the model input
would silently inflate apparent performance.
"""

import re
from collections.abc import Iterable
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalization: match R's normalize_token (lowercase + strip non-alpha)
# ---------------------------------------------------------------------------
def _normalize_token(x: str) -> str:
    """Normalize a place/qualifier token to a comparison key.

    Mirrors R's `normalize_token`: lowercase, alpha-only.
    Examples: "Wash." → "wash"; "N.Y." → "ny"; "Los Angeles" → "losangeles".
    """
    if not x:
        return ""
    return re.sub(r"[^a-z]", "", x.lower())


# ---------------------------------------------------------------------------
# Gazetteer loading and caching
# ---------------------------------------------------------------------------
_GAZETTEERS_CACHE = None


def _load_gazetteers_from_csv(gazetteer_dir: Path) -> dict:
    """Load normalized gazetteer sets from CSV files in gazetteer_dir.

    Returns a dict with keys: states, countries, us_cities, foreign_cities.
    Each value is a set of normalized (lowercase, alpha-only) strings.

    CSV files expected:
      - state_long_abbrs.csv (columns: full, long_abbr)
      - countries.csv (column: value)
      - ap_us_cities.csv (column: city)
      - ap_foreign_cities.csv (column: city)
    """
    import csv

    def load_csv_column(filename: str, column: str) -> set[str]:
        """Load a CSV file and return a set of normalized values from a column."""
        filepath = gazetteer_dir / filename
        result = set()
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get(column, "").strip()
                if val:
                    normalized = _normalize_token(val)
                    if normalized:  # Only add non-empty normalized values
                        result.add(normalized)
        return result

    return {
        "states": load_csv_column("state_long_abbrs.csv", "full")
        | load_csv_column("state_long_abbrs.csv", "long_abbr"),
        "countries": load_csv_column("countries.csv", "value"),
        "us_cities": load_csv_column("ap_us_cities.csv", "city"),
        "foreign_cities": load_csv_column("ap_foreign_cities.csv", "city"),
    }


def _get_default_gazetteers() -> dict:
    """Load gazetteers from the in-repo directory (caching on first call)."""
    global _GAZETTEERS_CACHE
    if _GAZETTEERS_CACHE is None:
        # Determine the repo root by walking up from this file
        this_file = Path(__file__).resolve()
        # src/preproc/dateline_guard.py → src/preproc → src → root
        repo_root = this_file.parent.parent.parent
        gazetteer_dir = repo_root / "r" / "dateline" / "gazetteers"
        if not gazetteer_dir.exists():
            raise FileNotFoundError(
                f"Gazetteer directory not found at {gazetteer_dir}. "
                "Expected r/dateline/gazetteers/*.csv files."
            )
        _GAZETTEERS_CACHE = _load_gazetteers_from_csv(gazetteer_dir)
    return _GAZETTEERS_CACHE


# ---------------------------------------------------------------------------
# Regex patterns for date/weekday detection (from R)
# ---------------------------------------------------------------------------
# Matches complete month names (full or AP abbreviations) + day number
_DATE_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|may|jun|jul|aug|sept|oct|nov|dec)\.?\s*[0-9]{1,2}$",
    re.IGNORECASE,
)

# Matches complete weekday names (full or AP abbreviations) with optional period
_WEEKDAY_RE = re.compile(
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tues|wed|thurs|fri|sat|sun)\.?$",
    re.IGNORECASE,
)


def _is_date_field(field: str) -> bool:
    """Does a field look like a date? Matches month + optional day."""
    if not field:
        return False
    return _DATE_RE.match(field.strip()) is not None


def _is_weekday_field(field: str) -> bool:
    """Does a field look like a weekday?"""
    if not field:
        return False
    return _WEEKDAY_RE.match(field.strip()) is not None


# ---------------------------------------------------------------------------
# Dateline block extraction (mirrors R's extract_dateline_block)
# ---------------------------------------------------------------------------
_CREDIT_RE = re.compile(r"^\s*Special to The New York Times\s*", re.IGNORECASE)
# Caps block: [A-Z][A-Z .',\-]*[A-Z.] (strict all-caps)
# Followed by 1-2 optional qualifier fields: ,\s*[A-Za-z. 0-9]{2,20}
# Then delimiter: em-dash, --, or spaced hyphen
_DATELINE_RE = re.compile(
    r"^\s*([A-Z][A-Z.' -]*[A-Z.])"
    r"(?:,\s*([A-Za-z. 0-9]{2,20}))?"
    r"(?:,\s*([A-Za-z. 0-9]{2,20}))?"
    r"\s*(-|--|—)\s"
)


def _extract_dateline_block(text: str) -> dict:
    """Extract the leading dateline block structure.

    Returns a dict with:
      found (bool): True if a dateline-shaped block was found
      block (str or None): The extracted block (city + optional qualifiers)
      match_len (int): Number of characters consumed (for exact stripping)
    """
    if not text:
        return {"found": False, "block": None, "match_len": 0}

    work = text
    offset = 0

    # Strip a leading credit line if present
    m = _CREDIT_RE.match(work)
    if m:
        consumed = len(m.group(0))
        offset += consumed
        work = work[consumed:]

    # Try to match the dateline structure
    m = _DATELINE_RE.match(work)
    if not m:
        return {"found": False, "block": None, "match_len": 0}

    # Extract the main city block (group 1)
    city = m.group(1)
    q1 = m.group(2)
    q2 = m.group(3)

    # Build the block: city + optional qualifiers
    block = city
    if q1 and q1.strip():
        block += ", " + q1.strip()
    if q2 and q2.strip():
        block += ", " + q2.strip()

    # Strip trailing wire tag "(AP)"
    block = re.sub(r"\(AP\)\s*$", "", block).strip()

    match_len = offset + len(m.group(0))

    return {"found": True, "block": block, "match_len": match_len}


# ---------------------------------------------------------------------------
# Field parsing (non-date, non-weekday fields)
# ---------------------------------------------------------------------------
def _parse_dateline_fields(block: str) -> list[str]:
    """Parse a block into non-date, non-weekday fields.

    Splits on commas and filters out date and weekday fields.
    Returns a list of remaining (location/qualifier) fields.
    """
    if not block:
        return []

    fields = [f.strip() for f in block.split(",")]
    fields = [f for f in fields if f]  # Remove empty

    # Filter out date and weekday fields
    non_temporal = [
        f for f in fields if not (_is_date_field(f) or _is_weekday_field(f))
    ]
    return non_temporal


# ---------------------------------------------------------------------------
# Place resolution (US/not-US via gazetteer lookups)
# ---------------------------------------------------------------------------
def _resolve_place(fields: list[str], gazetteers: dict) -> dict:
    """Resolve fields to US/not-US signal.

    Returns dict with is_us (True/False/None) and place (str or None).

    Logic mirrors R:
      - If 2+ fields: check last field (qualifier) against states/countries
      - If 1 field: check against AP-list cities only
    """
    if not fields:
        return {"is_us": None, "place": None}

    city = fields[0]
    if len(fields) >= 2:
        qualifier = fields[-1]
        qn = _normalize_token(qualifier)
        if qn in gazetteers["states"]:
            return {"is_us": True, "place": f"{city}, {qualifier}"}
        if qn in gazetteers["countries"]:
            return {"is_us": False, "place": f"{city}, {qualifier}"}
        # Qualifier present but unrecognized
        return {"is_us": None, "place": f"{city}, {qualifier}"}

    # Bare city: only check AP lists, not countries/area-codes
    cn = _normalize_token(city)
    if cn in gazetteers["us_cities"]:
        return {"is_us": True, "place": city}
    if cn in gazetteers["foreign_cities"]:
        return {"is_us": False, "place": city}

    return {"is_us": None, "place": city}


# ---------------------------------------------------------------------------
# Conditional stripping decision (the critical gate for AC2.1)
# ---------------------------------------------------------------------------
def _should_strip_dateline_block(
    block: str, fields_after_parse: list[str], resolve_result: dict, gazetteers: dict
) -> bool:
    """Decide whether a matched block is a REAL dateline (and should be stripped).

    A block is treated as a dateline if:
      1. It contains a date field, OR
      2. It has a recognized state/country qualifier, OR
      3. It's a bare city in the AP lists (AP-30 or AP-46)

    Otherwise (emphasis-caps ledes, unrecognized qualifiers) → False (not stripped).

    This mirrors R's `should_strip_dateline_block`.
    """
    if not block:
        return False

    # Check if the original block contains a date field
    all_fields = [f.strip() for f in block.split(",")]
    all_fields = [f for f in all_fields if f]
    if any(_is_date_field(f) for f in all_fields):
        return True

    # Check if we resolved to US/not-US (recognized qualifier)
    if resolve_result["is_us"] is not None:
        return True

    # Check if it's a bare AP-list city (1 field after filtering, in AP lists)
    if len(fields_after_parse) == 1:
        cn = _normalize_token(fields_after_parse[0])
        if cn in gazetteers["us_cities"] or cn in gazetteers["foreign_cities"]:
            return True

    # Otherwise: emphasis-caps lede or unrecognized qualifier → not a dateline
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def has_dateline_prefix(text: str) -> bool:
    """True if `text` begins with a dateline prefix that WOULD be stripped.

    This is the conditional-strip version: returns True only if the detected
    dateline block contains a date field, a recognized qualifier, or is a bare
    AP-list city. Emphasis-caps ledes and unrecognized qualifiers → False.

    Used for leakage detection and assertion on model input.
    """
    if not text:
        return False

    gazetteers = _get_default_gazetteers()

    # Extract the block structure
    block_info = _extract_dateline_block(text)
    if not block_info["found"]:
        return False

    block = block_info["block"]
    # Parse out date/weekday fields
    fields = _parse_dateline_fields(block)
    # Resolve to US/not-US
    resolve_result = _resolve_place(fields, gazetteers)
    # Apply conditional logic
    should_strip = _should_strip_dateline_block(block, fields, resolve_result, gazetteers)

    return should_strip


def assert_no_dateline_residue(texts: Iterable[str], *, max_report: int = 10) -> None:
    """Raise ValueError if any text retains a dateline prefix.

    Used as a pytest assertion and as a runtime guard at train entry (Phase 4).
    Only reports offenders up to max_report to avoid overwhelming error output.
    """
    offenders = [(i, t) for i, t in enumerate(texts) if has_dateline_prefix(t)]
    if offenders:
        sample = offenders[:max_report]
        raise ValueError(
            f"Dateline residue detected in {len(offenders)} input(s); "
            f"model input would leak the label. First offenders (index, text): "
            + "; ".join(f"({i}, {t[:60]!r})" for i, t in sample)
        )
