# US/not-US Pre-Filter — Phase 2 Implementation Plan

**Goal:** Bring the 1960–1995 NYT Archive API corpus into the parquet world with the nested `keywords` list-column intact, so Python can later score it.

**Architecture:** A one-time R conversion script reads each per-year `.rds` and writes a per-year parquet via `arrow::write_parquet` (the only R↔Python bridge that preserves the nested `keywords` list-column). Output is a read-only application-target corpus, separate from the `us_filter/` derived-artifact directory.

**Tech Stack:** R + `arrow`, run via `Rscript`. Python touchpoint: one new path constant in `src/config.py`.

**Scope:** Phase 2 of 8.

**Codebase verified:** 2026-06-09 (codebase-investigator + direct `.rds` schema inspection).

---

## Acceptance Criteria Coverage

This is an **infrastructure** phase. **Verifies: none** (no acceptance criteria). It produces the application-target corpus consumed in Phase 7. Verification is operational (conversion succeeds, schema preserved, row counts match).

---

## Verified facts (from `.rds` inspection)

- Source: `/Users/strophios/immigration_project/00_ML_data_expansion/nyt_archive_by_year/{1960..1995}.rds` — 36 files.
- Each is a `tbl_df`/`data.frame` with 14 columns:
  `year, month, headline, abstrct, lead_paragraph, web_url, keywords (list), pub_date (Date), document_type, news_desk, section_name, uri, id (character), word_count (integer)`.
- `keywords` is a list-column; each element is a `data.frame` with columns `type, value, rank, major`.
- 1960 = 145,134 rows (corpus is large; expect millions of rows total).
- API leads are dateline-less — no stripping on this side.
- `id` is `character` here (LDC `id` is Int64) — relevant to the Phase 6 join, not this phase.

## Environment precondition

Run: `Rscript -e 'cat(requireNamespace("arrow", quietly=TRUE))'` → expect `TRUE`. If FALSE, STOP and surface to the user.

---

<!-- START_TASK_1 -->
### Task 1: Conversion script `r/api_ingest/rds_to_parquet.R`

**Files:**
- Create: `r/api_ingest/rds_to_parquet.R` (`# pattern: Imperative Shell`)

**Implementation:**

```r
# pattern: Imperative Shell
# Convert per-year NYT Archive API .rds files to parquet, preserving the nested
# `keywords` list-column (arrow is the only bridge that does this correctly).

suppressMessages({ library(arrow) })

SRC_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/nyt_archive_by_year"
OUT_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/api_corpus"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

rds_files <- list.files(SRC_DIR, pattern = "\\.rds$", full.names = TRUE)
stopifnot(length(rds_files) == 36)

total_src <- 0L; total_out <- 0L
for (f in sort(rds_files)) {
  year <- sub("\\.rds$", "", basename(f))
  d <- readRDS(f)
  out_path <- file.path(OUT_DIR, paste0(year, ".parquet"))
  write_parquet(d, out_path)
  back <- open_dataset(out_path)$num_rows
  cat(sprintf("%s: src=%d written=%d %s\n", year, nrow(d), back,
              ifelse(nrow(d) == back, "OK", "MISMATCH")))
  stopifnot(nrow(d) == back)
  total_src <- total_src + nrow(d); total_out <- total_out + back
}
cat(sprintf("TOTAL src=%d written=%d\n", total_src, total_out))
stopifnot(total_src == total_out)
```

**Step 1:** Write the script.

**Step 2: Run**

```bash
Rscript r/api_ingest/rds_to_parquet.R
```
Expected: 36 lines each ending `OK`; a final `TOTAL src=N written=N` with equal counts. The script `stopifnot`s on any mismatch.

**Step 3: Commit**

```bash
git add r/api_ingest/rds_to_parquet.R
git commit -m "feat(us-filter): convert NYT API rds corpus (1960-1995) to parquet"
```

**Verifies:** none (infrastructure).
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Config path + nested-schema verification

**Files:**
- Modify: `src/config.py` (add `API_CORPUS_DIR`)

**Step 1: Add config path.** In `src/config.py`, alongside the other data-source/artifact paths, add:

```python
# NYT Archive API corpus (dateline-less; application target for the US filter)
API_CORPUS_DIR: Path = PROJECT_ROOT / "api_corpus"
```

**Step 2: Verify the nested structure survived round-trip (Python side)**

```bash
uv run python -c "
import polars as pl, src.config as c
df = pl.read_parquet(c.API_CORPUS_DIR / '1960.parquet')
print('columns:', df.columns)
print('keywords dtype:', df.schema['keywords'])   # expect List(Struct({type,value,rank,major}))
print('rows:', df.height)                          # expect 145134
assert df.height == 145134, df.height
"
```
Expected: `keywords` dtype is `List(Struct(...))` with fields `type, value, rank, major`; 1960 row count is 145,134.

**Step 3: Confirm all 36 years are readable as one dataset**

```bash
uv run python -c "
import polars as pl, src.config as c, pathlib
files = sorted(c.API_CORPUS_DIR.glob('*.parquet'))
print('year files:', len(files))                   # expect 36
assert len(files) == 36, len(files)
print('first/last:', files[0].stem, files[-1].stem) # expect 1960 / 1995
"
```
Expected: 36 files spanning 1960–1995.

**Step 4: Commit**

```bash
git add src/config.py
git commit -m "feat(us-filter): api_corpus config path + nested-schema verification"
```

**Verifies:** none (infrastructure).
<!-- END_TASK_2 -->

---

## Phase 2 Done When

- All 36 years convert with matching row counts (script `stopifnot`s on mismatch).
- The nested `keywords` column round-trips as `List(Struct({type,value,rank,major}))`, confirmed by reading the parquet schema in Python.
- `src/config.py` exposes `API_CORPUS_DIR`.

Infrastructure phase — operational verification only.
