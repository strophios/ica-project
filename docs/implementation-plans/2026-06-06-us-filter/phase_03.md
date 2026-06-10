# US/not-US Pre-Filter — Phase 3 Implementation Plan

**Goal:** Assemble the model input from `stripped_text` and prove the dateline cannot leak into it.

**Architecture:** Parameterize the existing `data_from_parquet` on the lead-column name (no fork) so the same null-handling + `headline + "</s>" + lead` concatenation points at `stripped_text`. Add a Python port of the R dateline *detector* as a no-residue guard, used both as a pytest and (in Phase 4) as a runtime assertion at train entry. The guard is the load-bearing leakage check; it is why Phase 1 retains `raw_text`.

**Tech Stack:** Python, polars, `re`, pytest.

**Scope:** Phase 3 of 8.

**Codebase verified:** 2026-06-09 (direct read of `src/data_setup/data.py` and `tests/test_data_loading.py`).

---

## Acceptance Criteria Coverage

This phase implements and tests **us-filter.AC2**:

### us-filter.AC2: The dateline cannot leak into model input
- **us-filter.AC2.1 Success:** the assembled input column (from `stripped_text`) contains no dateline prefix — the no-residue guard passes on real data.
- **us-filter.AC2.2 Failure:** a seeded leak (dateline left in) makes the guard fail loudly, both as a pytest and as the runtime assertion at train entry.
- **us-filter.AC2.3 Success:** `raw_text` differs from `stripped_text` exactly by the dateline span on datelined rows.

(AC2.3 is co-covered: the R-side strip-correctness half is tested in Phase 1's testthat suite; the consumer-side relationship is tested here in Python.)

---

## Verified facts

- `data_from_parquet(project_root, db_folder="ldc_corpus", addl_columns=None)` at `src/data_setup/data.py:5`. Hardcoded `lead_paragraph` at: select list (line 9), null-fill (line 21), "NA"-replace (lines 28–31), concat zip (line 40). Returns a polars `DataFrame` with an added `headline_with_lead` column.
- `tests/test_data_loading.py` writes a fixture parquet to `tmp_path / db_folder / test_data.parquet` and calls `data_from_parquet`. The fixture helper `_write_parquet` builds `id/headline/lead_paragraph`. 11 tests; all rely on default-arg behavior.
- Phase 1 carries `headline` into `<US_FILTER_DIR>/ldc_labeled.parquet` (columns: `id, headline, us_label, label_source, dateline_place, stripped_text, raw_text`).

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Parameterize `data_from_parquet` on the lead column

**Verifies:** us-filter.AC2.1 (enables `stripped_text` assembly).

**Files:**
- Modify: `src/data_setup/data.py:5-47`
- Modify: `tests/test_data_loading.py` (add lead-column cases)

**Implementation:** add a keyword-only-friendly param `lead_column="lead_paragraph"` and replace every hardcoded `lead_paragraph` / `pl.col.lead_paragraph` reference with `lead_column`. Use `pl.col(lead_column)` (string form) throughout for the dynamic name. Final function:

```python
def data_from_parquet(project_root, db_folder="ldc_corpus", addl_columns=None,
                      lead_column="lead_paragraph"):
    ldc_pq = pl.scan_parquet(
        f"{project_root}/{db_folder}/**/*.parquet", hive_partitioning=True
    )
    cols_to_select = ["id", "headline", lead_column]
    if addl_columns is not None:
        [cols_to_select.append(x) for x in addl_columns]

    ldc_data = ldc_pq.select(pl.col(cols_to_select)).collect()
    ldc_data = ldc_data.with_columns(
        pl.col("headline").fill_null(""),
        pl.col(lead_column).fill_null(""),
    )
    ldc_data = ldc_data.with_columns(
        pl.when(pl.col("headline") == "NA").then(pl.lit("")).otherwise(pl.col("headline")).alias("headline"),
        pl.when(pl.col(lead_column) == "NA").then(pl.lit("")).otherwise(pl.col(lead_column)).alias(lead_column),
    )
    headline_lead = [
        x + "</s>" + y
        for x, y in zip(
            ldc_data.get_column("headline"), ldc_data.get_column(lead_column)
        )
    ]
    ldc_data = ldc_data.with_columns(
        pl.Series(name="headline_with_lead", values=headline_lead)
    )
    print(ldc_data.shape)  # LOG
    return ldc_data
```

Note: keep the `print(...)  # LOG` line (project convention). Default value `"lead_paragraph"` keeps existing behavior byte-identical.

**Testing** (`tests/test_data_loading.py`): tests must verify AC2.1's enabling behavior —
- A new test writes a fixture parquet with columns `id/headline/stripped_text` and asserts `data_from_parquet(tmp_path, db_folder="us_filter", lead_column="stripped_text")` produces `headline_with_lead == headline + "</s>" + stripped_text` (and applies the same null/"NA" handling to `stripped_text`).
- A test asserts the default call still assembles from `lead_paragraph` (regression guard — existing 11 tests already cover this; add one explicit assertion that the default param name is unchanged).
- Extend `_write_parquet` (or add a sibling helper) to accept an arbitrary lead-column name.

**Verification:**
Run: `uv run pytest tests/test_data_loading.py`
Expected: all tests pass (existing 11 + new lead-column cases).

**Commit:** `feat(us-filter): parameterize data_from_parquet lead column`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: No-residue dateline guard `src/preproc/dateline_guard.py`

**Verifies:** us-filter.AC2.1, AC2.2, AC2.3.

**Files:**
- Create: `src/preproc/dateline_guard.py`
- Create: `tests/test_dateline_guard.py`

**Implementation** (`# pattern: Functional Core` for the detector; the assertion is a thin shell):

```python
# pattern: Functional Core
"""No-residue dateline guard.

`has_dateline_prefix` is a Python port of the *detection* half of the R
extractor `extract_dateline_block` (r/dateline/resolve_dateline.R). The two are a
boundary-inventory pair: any change to the R extractor's credit-line / caps-block /
delimiter patterns MUST be mirrored here, and vice versa. This guard is the
load-bearing leakage check -- a dateline left in the model input would silently
inflate apparent performance.
"""
import re
from collections.abc import Iterable

_CREDIT_RE = re.compile(r"^\s*Special to The New York Times\s*", re.IGNORECASE)
# Leading ALL-CAPS place block, then a dateline delimiter (em dash, --, or spaced -).
_DATELINE_RE = re.compile(r"^\s*[A-Z][A-Z .,'\-]*[A-Z.)]\s*(—|--|-)\s")


def has_dateline_prefix(text: str) -> bool:
    """True if `text` begins with a dateline prefix (optionally after a
    'Special to The New York Times' credit line)."""
    if not text:
        return False
    work = _CREDIT_RE.sub("", text, count=1)
    return _DATELINE_RE.match(work) is not None


def assert_no_dateline_residue(texts: Iterable[str], *, max_report: int = 10) -> None:
    """Raise ValueError if any text retains a dateline prefix.

    Used as a pytest assertion and as a runtime guard at train entry (Phase 4).
    """
    offenders = [(i, t) for i, t in enumerate(texts) if has_dateline_prefix(t)]
    if offenders:
        sample = offenders[:max_report]
        raise ValueError(
            f"Dateline residue detected in {len(offenders)} input(s); "
            f"model input would leak the label. First offenders (index, text): "
            + "; ".join(f"({i}, {t[:60]!r})" for i, t in sample)
        )
```

**Guard target — important:** the guard runs over the **`stripped_text` column** (the lead component), NOT the assembled `headline_with_lead`. A residual dateline appears at the *start of the lead*, which in the assembled string `headline + "</s>" + stripped_text` is mid-string (after the headline and separator); `has_dateline_prefix` anchors at string-start, so it must be applied to `stripped_text` directly. The assembled column's dateline-freedom follows from `stripped_text` being clean.

**Testing** (`tests/test_dateline_guard.py`):
- **AC2.1**: write a fixture parquet (`id/headline/raw_text/stripped_text/label_source`) with datelined `raw_text` and correctly-stripped `stripped_text`; call `data_from_parquet(..., db_folder=..., lead_column="stripped_text", addl_columns=["raw_text","label_source"])`; assert `assert_no_dateline_residue(result["stripped_text"])` does not raise.
- **AC2.2**: add one row whose `stripped_text` still contains the dateline; assert `assert_no_dateline_residue(result["stripped_text"])` raises `ValueError`, and `has_dateline_prefix` returns True on that row.
- **AC2.3**: for the datelined fixture rows, assert `raw_text.endswith(stripped_text)` (modulo trailing/leading whitespace) and the removed prefix is the dateline — i.e. `has_dateline_prefix(raw_text)` is True while `has_dateline_prefix(stripped_text)` is False.
- Direct unit cases for `has_dateline_prefix`: `"WASHINGTON, July 30 — text"` → True; `"Special to The New York Times CHICAGO — text"` → True; `"The workers met to discuss terms."` → False; `""` → False.

**Verification:**
Run: `uv run pytest tests/test_dateline_guard.py`
Expected: all tests pass.

**Commit:** `feat(us-filter): no-residue dateline leakage guard + tests`
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

---

## Execution deviation note (2026-06-10, follows from the Phase 1 deviation)

Phase 1's revision (see phase_01.md "Execution deviation note") made text-channel stripping **conditional**: the R extractor strips a caps-block prefix only when it (i) contains a date field, (ii) has a qualifier resolving against states/countries, or (iii) is a bare AP-list city. Emphasis-caps ledes (`"PILOBOLUS - that dance troupe..."` — ~1,400 rows in the real LDC corpus) are deliberately NOT stripped.

Consequence for Task 2: the Python guard must port the **would-strip decision**, not the bare prefix pattern given in the original spec above (which would false-alarm on every unstripped emphasis-caps lede and fail AC2.1 on real data). Concretely, `has_dateline_prefix` (rename or keep; keep the public name) must mirror `r/dateline/resolve_dateline.R`'s current logic: strict all-caps city block + optional short mixed-case comma fields + delimiter, optional credit prefix, then the three-way conditional (date field | recognized state/country qualifier | bare AP-30/AP-foreign city). This requires the gazetteers; load them from the in-repo `r/dateline/gazetteers/*.csv` (single thin I/O helper, default path derived from the repo layout, results passed into the pure detector — keep the Functional Core honest). The boundary-inventory pairing note must name the conditional semantics explicitly. Tests gain cases: `"PILOBOLUS - that dance troupe specializing in mad scrambles"` → False (not residue); `"LISBON, Portugal — Officials said."` → True; `"WASHINGTON, March 2 - A girl from Westmont"` → True (spaced-hyphen real form); `"MEMORY, memory - is there ever enough of it?"` → False.

## Phase 3 Done When

- `data_from_parquet` builds `headline_with_lead` from `stripped_text` when called with `lead_column="stripped_text"`, and the default path is unchanged (existing tests pass).
- `assert_no_dateline_residue` passes on correctly-stripped input and raises loudly on a seeded leak.
- The R↔Python detector sync note is documented in `dateline_guard.py`.

Covers **us-filter.AC2**. (The runtime train-entry call of `assert_no_dateline_residue` over the `stripped_text` column is wired in Phase 4.)
