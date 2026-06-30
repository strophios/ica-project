# US/not-US Pre-Filter — Phase 1 Implementation Plan

**Goal:** Produce a derived labeled parquet over the full LDC corpus (1987–2007) with confident US/not-US labels and dateline-stripped model-input text.

**Architecture:** A new in-repo R module (top-level `r/` tree) owns dateline extraction, place resolution, a desk/section fusion policy, and span stripping. It reads the existing partitioned LDC parquet read-only and writes a *derived* labeled parquet to an out-of-repo `us_filter/` directory. Pure resolver logic (Functional Core) is separated from the I/O build script (Imperative Shell). R → Python contract is the derived parquet only.

**Tech Stack:** R (base + `arrow` + `testthat`), run via `Rscript`. Python touchpoint: two new path constants in `src/config.py`.

**Scope:** Phase 1 of 8.

**Codebase verified:** 2026-06-09 (two codebase-investigator passes + AP-dateline internet research).

---

## Acceptance Criteria Coverage

This phase implements and tests **us-filter.AC1** in full:

### us-filter.AC1: Datelines and desk signals produce correct, confident labels
- **us-filter.AC1.1 Success:** A US state qualifier (`VANCOUVER, Wash.`) → `us_label=true`, `label_source="dateline"`.
- **us-filter.AC1.2 Success:** A country qualifier (`LISBON, Portugal`) → `us_label=false`, `label_source="dateline"`.
- **us-filter.AC1.3 Success:** A bare standalone US city (`CHICAGO —`) → true; a bare standalone foreign city (`LONDON —`) → false.
- **us-filter.AC1.4 Success (collision):** `Paris, Texas` → true (state qualifier wins, before any bare lookup); bare `PARIS` → false (AP-foreign list).
- **us-filter.AC1.5 Edge (date field):** `PARIS, July 30 —` → bare PARIS → false; `WASHINGTON, July 30 —` → true (date field dropped, place resolved).
- **us-filter.AC1.6 Edge (multi-comma):** `VANCOUVER, Wash., June 1 —` → state field detected despite trailing date → true.
- **us-filter.AC1.7 Success (backfill):** no dateline + Foreign desk → false, `label_source="heuristic"`; no dateline + National/Metro desk → true, `"heuristic"`.
- **us-filter.AC1.8 Failure (conflict):** dateline US but desk Foreign (or vice versa) → `us_label=null`, `label_source="conflict"`.
- **us-filter.AC1.9 Edge (unresolved):** bare unlisted city with no confident desk signal → `us_label=null`.
- **us-filter.AC1.10 Guard (collision-trap invariant):** a bare token is never resolved against `us-area-code-cities` or `countries.csv`.

---

## Design refinements adopted (agreed with user, deviate from written design)

1. **Label/train span = full LDC 1987–2007**, not the 1987–1995 DoCA overlap. Dateline labels are era-independent; the overlap restriction was a carryover from DoCA-label thinking that does not apply here. (Affects AC3.1 year wording downstream.)
2. **Output = derived parquet** at `<US_FILTER_DIR>/ldc_labeled.parquet`, not columns mutated onto the shared `ldc_corpus/` dataset.
3. **R module lives in-repo** under `r/`, owning its own copies of the gazetteer CSVs (self-contained, version-controlled).
4. **`us-area-code-cities.csv` is not used by the resolver** (collision-trap invariant AC1.10); it belongs to the keyword heuristic, out of scope here. Not loaded in Phase 1.
5. Desk/section lists are **copied** in-repo from `nyt_location_checking.R:65-82` (with provenance comment), not `source()`-d from the out-of-repo file.

## Environment preconditions (verify before Task 1)

Run: `Rscript --version` → expect R present (the existing sibling R scripts rely on it).
Run: `Rscript -e 'cat(requireNamespace("arrow", quietly=TRUE))'` → expect `TRUE` (arrow already in use).
If `arrow` is FALSE, STOP and surface to the user (the build script needs it).

Out-of-repo absolute paths used (verified to exist):
- LDC dataset: `/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/ldc_corpus/` (partitioned by `publication_year`).
- Gazetteer source CSVs: `/Users/strophios/immigration_project/00_ML_data_expansion/context_data/{countries.csv,state_long_abbrs.csv}`.
- Desk-list provenance: `/Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R` (lines 65–82).

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Bring gazetteers in-repo + author the two curated AP lists

**Files:**
- Create: `r/dateline/gazetteers/countries.csv` (copy of out-of-repo `context_data/countries.csv`; columns `id`,`value`)
- Create: `r/dateline/gazetteers/state_long_abbrs.csv` (copy; columns `full`,`long_abbr`,`usps_abbr`)
- Create: `r/dateline/gazetteers/ap_us_cities.csv` (single column `city`, 30 rows)
- Create: `r/dateline/gazetteers/ap_foreign_cities.csv` (single column `city`, ~49 rows)
- Create: `r/dateline/gazetteers/SOURCES.md` (provenance)

**Step 1: Copy the two existing CSVs into the repo**

```bash
mkdir -p r/dateline/gazetteers
cp /Users/strophios/immigration_project/00_ML_data_expansion/context_data/countries.csv r/dateline/gazetteers/countries.csv
cp /Users/strophios/immigration_project/00_ML_data_expansion/context_data/state_long_abbrs.csv r/dateline/gazetteers/state_long_abbrs.csv
```

**Step 2: Author `ap_us_cities.csv`** (header `city`, one per line):

```
city
Atlanta
Baltimore
Boston
Chicago
Cincinnati
Cleveland
Dallas
Denver
Detroit
Honolulu
Houston
Indianapolis
Las Vegas
Los Angeles
Miami
Milwaukee
Minneapolis
New Orleans
New York
Oklahoma City
Philadelphia
Phoenix
Pittsburgh
St. Louis
Salt Lake City
San Antonio
San Diego
San Francisco
Seattle
Washington
```

**Step 3: Author `ap_foreign_cities.csv`** (header `city`; seed list from AP research — membership may be finalized during execution):

```
city
Amsterdam
Baghdad
Bangkok
Beijing
Beirut
Berlin
Brussels
Cairo
Djibouti
Dublin
Geneva
Gibraltar
Guatemala City
Havana
Helsinki
Hong Kong
Islamabad
Istanbul
Jerusalem
Johannesburg
Kuwait City
London
Luxembourg
Macau
Madrid
Mexico City
Milan
Monaco
Montreal
Moscow
Munich
New Delhi
Panama City
Paris
Prague
Quebec City
Rio de Janeiro
Rome
San Marino
Sao Paulo
Shanghai
Singapore
Stockholm
Sydney
Tokyo
Toronto
Vatican City
Vienna
Zurich
```

**Step 4: Write `SOURCES.md`** documenting: the two large CSVs are copied from `context_data/` (curated, near-static; in-repo copy is the resolver's source of truth); the AP-30 and AP-foreign lists derive from AP Stylebook dateline conventions (NYT follows AP style). Note the collision rule (bare `PARIS`→France; `PARIS, Texas`→US) and the eight never-abbreviated states.

**Step 5: Verify operationally**

```bash
Rscript -e 'stopifnot(nrow(read.csv("r/dateline/gazetteers/ap_us_cities.csv"))==30); cat("ap_us OK\n")'
Rscript -e 'invisible(lapply(list.files("r/dateline/gazetteers", pattern="csv$", full.names=TRUE), function(f){d<-read.csv(f); cat(f, nrow(d), "rows\n")}))'
```
Expected: `ap_us_cities.csv` = 30 rows; all CSVs parse and print row counts.

**Step 6: Commit**

```bash
git add r/dateline/gazetteers
git commit -m "feat(us-filter): in-repo dateline gazetteers + curated AP standalone lists"
```

**Verifies:** none directly (underpins AC1).
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Resolver module `r/dateline/resolve_dateline.R`

**Files:**
- Create: `r/dateline/resolve_dateline.R`

**Implementation** — pure functions (`# pattern: Functional Core`). Gazetteers are loaded once and passed in; only `load_gazetteers` touches disk.

```r
# pattern: Functional Core
# Dateline extraction + structure-first US/not-US place resolution.
# Pure functions; load_gazetteers is the single thin I/O helper.

# Normalize a place/qualifier token to a comparison key: lowercase, alpha-only.
# "Wash." -> "wash"; "N.Y." -> "ny"; "Los Angeles" -> "losangeles".
normalize_token <- function(x) {
  if (is.na(x)) return(NA_character_)
  gsub("[^a-z]", "", tolower(x))
}

# Load normalized gazetteer sets from a directory of CSVs.
# states: from state full names + AP abbreviations ONLY (NOT 2-letter USPS codes,
#   which collide with English words OR/IN/OK) -- satisfies the AC1.10/guard intent.
load_gazetteers <- function(dir) {
  states_raw   <- utils::read.csv(file.path(dir, "state_long_abbrs.csv"),
                                  stringsAsFactors = FALSE)
  countries    <- utils::read.csv(file.path(dir, "countries.csv"),
                                  stringsAsFactors = FALSE)
  us_cities    <- utils::read.csv(file.path(dir, "ap_us_cities.csv"),
                                  stringsAsFactors = FALSE)
  foreign      <- utils::read.csv(file.path(dir, "ap_foreign_cities.csv"),
                                  stringsAsFactors = FALSE)
  norm_set <- function(v) unique(stats::na.omit(vapply(v, normalize_token, character(1))))
  list(
    states         = norm_set(c(states_raw$full, states_raw$long_abbr)),
    countries      = norm_set(countries$value),
    us_cities      = norm_set(us_cities$city),
    foreign_cities = norm_set(foreign$city)
  )
}

# Does a field look like a date field? Matches "July 30", "Jul. 30", "June 1", "Jan 5".
.DATE_RE <- "^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\\.?\\s*[0-9]{0,2}$"
is_date_field <- function(field) {
  grepl(.DATE_RE, trimws(tolower(field)))
}

# Isolate the leading ALL-CAPS place block before the dateline delimiter.
# Handles a leading "Special to The New York Times" credit and trailing "(AP)" wire tag.
# Returns list(found, block, match_len) where match_len is the number of leading
# characters of `lead` consumed by block + delimiter (for exact stripping).
extract_dateline_block <- function(lead) {
  empty <- list(found = FALSE, block = NA_character_, match_len = 0L)
  if (is.na(lead) || !nzchar(lead)) return(empty)

  work <- lead
  offset <- 0L
  # Strip a leading credit line if present (case-insensitive), recording consumed length.
  credit_re <- "^\\s*Special to The New York Times\\s*"
  m <- regexpr(credit_re, work, ignore.case = TRUE)
  if (m == 1L) {
    consumed <- attr(m, "match.length")
    offset <- offset + consumed
    work <- substr(work, consumed + 1L, nchar(work))
  }

  # Dateline = caps block (letters, spaces, commas, periods, parens for (AP)),
  # then a delimiter: em dash, "--", or spaced hyphen.
  # Caps block: uppercase words, allowing commas/periods/spaces and a trailing (AP).
  # NOTE: the em dash is written as the PCRE unicode escape \x{2014}; do NOT use
  # — (PCRE does not support \u — it would match a literal 'u'). Equivalently,
  # embed the literal — character in the string. Validate against real leads in
  # Step 3 to confirm the delimiter actually present (em dash vs --).
  dl_re <- "^\\s*([A-Z][A-Z .,'-]*[A-Z.)])\\s*(\\x{2014}|--|-)\\s"
  m2 <- regexpr(dl_re, work, perl = TRUE)
  if (m2 != 1L) return(empty)

  full_match <- regmatches(work, m2)
  block <- sub(dl_re, "\\1", full_match, perl = TRUE)
  # Remove a trailing wire tag like "(AP)" from the block.
  block <- trimws(sub("\\(AP\\)\\s*$", "", block))
  match_len <- offset + attr(m2, "match.length")
  list(found = TRUE, block = block, match_len = as.integer(match_len))
}

# Split a caps block into non-date fields (city + optional qualifier).
parse_dateline_fields <- function(block) {
  if (is.na(block) || !nzchar(block)) return(character(0))
  fields <- trimws(strsplit(block, ",", fixed = TRUE)[[1]])
  fields <- fields[nzchar(fields)]
  fields[!vapply(fields, is_date_field, logical(1))]
}

# Resolve fields -> list(is_us = TRUE|FALSE|NA, place).
# Structure-first, ordered. Bare tokens consult ONLY AP-30/AP-46 (never countries/area-codes).
resolve_place <- function(fields, gz) {
  if (length(fields) == 0) return(list(is_us = NA, place = NA_character_))
  city <- fields[1]
  if (length(fields) >= 2) {
    qualifier <- fields[length(fields)]
    qn <- normalize_token(qualifier)
    if (!is.na(qn) && qn %in% gz$states)    return(list(is_us = TRUE,  place = paste(city, qualifier, sep = ", ")))
    if (!is.na(qn) && qn %in% gz$countries) return(list(is_us = FALSE, place = paste(city, qualifier, sep = ", ")))
    return(list(is_us = NA, place = paste(city, qualifier, sep = ", ")))  # qualifier present, unrecognized
  }
  # Bare token: only the short curated standalone lists.
  cn <- normalize_token(city)
  if (!is.na(cn) && cn %in% gz$us_cities)      return(list(is_us = TRUE,  place = city))
  if (!is.na(cn) && cn %in% gz$foreign_cities) return(list(is_us = FALSE, place = city))
  list(is_us = NA, place = city)
}

# Remove the matched dateline span (block + delimiter) from the lead -> stripped_text.
strip_dateline <- function(lead, block_info) {
  if (is.na(lead)) return(NA_character_)
  if (!isTRUE(block_info$found) || block_info$match_len <= 0L) return(lead)
  trimws(substr(lead, block_info$match_len + 1L, nchar(lead)))
}

# Convenience: full dateline resolution for one lead -> list(is_us, place, block_info).
resolve_dateline <- function(lead, gz) {
  bi <- extract_dateline_block(lead)
  if (!isTRUE(bi$found)) return(list(is_us = NA, place = NA_character_, block_info = bi))
  fields <- parse_dateline_fields(bi$block)
  rp <- resolve_place(fields, gz)
  list(is_us = rp$is_us, place = rp$place, block_info = bi)
}
```

**Step 1:** Write the file.

**Step 2: Confirm it parses**

```bash
Rscript -e 'source("r/dateline/resolve_dateline.R"); cat("parsed OK\n")'
```
Expected: `parsed OK`.

**Step 3: Validate the extractor against REAL data** (the documented-format gap — the 1980s–90s "Special to The New York Times" credit format is not authoritatively documented):

```bash
Rscript -e '
source("r/dateline/resolve_dateline.R")
library(arrow)
ds <- open_dataset("/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/ldc_corpus/")
samp <- head(as.data.frame(ds), 30)$lead_paragraph
for (s in samp) { bi <- extract_dateline_block(s); cat(ifelse(bi$found, paste0("[", bi$block, "] "), "[no-dl] "), substr(s,1,70), "\n") }
'
```
Expected: datelined leads show an isolated caps block; non-datelined leads show `[no-dl]`. If the credit-line or delimiter pattern does not match real data, adjust the regexes in `extract_dateline_block` and re-run. Note: `—` in the regex is the em dash; confirm real leads use it (vs `--`).

**Step 4: Commit**

```bash
git add r/dateline/resolve_dateline.R
git commit -m "feat(us-filter): dateline extractor + structure-first place resolver"
```

**Verifies:** AC1.1–AC1.6, AC1.10 logic (tests in Task 3).
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Resolver tests `r/tests/testthat/test-resolve-dateline.R`

**Verifies:** us-filter.AC1.1, AC1.2, AC1.3, AC1.4, AC1.5, AC1.6, AC1.10, and the R half of AC2.3 (strip removes exactly the dateline span).

**Files:**
- Create: `r/tests/testthat/test-resolve-dateline.R`
- Create: `r/tests/run_tests.R`

**Step 1: Test runner `r/tests/run_tests.R`**

```r
if (!requireNamespace("testthat", quietly = TRUE))
  install.packages("testthat", repos = "https://cloud.r-project.org")
library(testthat)
source("r/dateline/resolve_dateline.R")
testthat::test_dir("r/tests/testthat", stop_on_failure = TRUE)
```

**Step 2: Tests** — each helper resolves a raw lead end-to-end through the public API.

```r
gz <- load_gazetteers("r/dateline/gazetteers")

resolve_lead <- function(lead) resolve_dateline(lead, gz)$is_us

test_that("AC1.1 US state qualifier -> US", {
  expect_true(resolve_lead("VANCOUVER, Wash. — The city council met."))
})
test_that("AC1.2 country qualifier -> not-US", {
  expect_false(resolve_lead("LISBON, Portugal — Officials said."))
})
test_that("AC1.3 bare standalone cities", {
  expect_true(resolve_lead("CHICAGO — The mayor spoke."))
  expect_false(resolve_lead("LONDON — Parliament voted."))
})
test_that("AC1.4 collision: Paris, Texas vs bare PARIS", {
  expect_true(resolve_lead("PARIS, Texas — A local story."))
  expect_false(resolve_lead("PARIS — The president of France."))
})
test_that("AC1.5 date field dropped", {
  expect_false(resolve_lead("PARIS, July 30 — A summit opened."))
  expect_true(resolve_lead("WASHINGTON, July 30 — Congress acted."))
})
test_that("AC1.6 multi-comma with trailing date", {
  expect_true(resolve_lead("VANCOUVER, Wash., June 1 — Rain fell."))
})
test_that("Geneva/Moscow collisions", {
  expect_true(resolve_lead("Geneva, N.Y. — A town meeting."))
  expect_false(resolve_lead("GENEVA — Talks resumed."))
  expect_true(resolve_lead("Moscow, Idaho — The university."))
  expect_false(resolve_lead("MOSCOW — The Kremlin said."))
})
test_that("AC1.10 bare token never hits long lists", {
  # The structural invariant: bare tokens consult ONLY AP-30/AP-46, never
  # countries.csv or us-area-code-cities. Prove it with bare tokens that ARE
  # present in the long lists but absent from the short standalone lists ->
  # they must resolve NA. A resolver that (wrongly) consulted the long lists
  # would resolve these to US/not-US and fail the test.
  # 'Portugal' is in countries.csv (a country name) but not AP-46:
  expect_true(is.na(resolve_lead("PORTUGAL — A bare country name, not a dateline city.")))
  # 'Bayonne' is in us-area-code-cities (a US city) but not AP-30:
  expect_true(is.na(resolve_lead("BAYONNE — A bare US city not on the AP-30 list.")))
  # And a plain unlisted token:
  expect_true(is.na(resolve_lead("LISBON — An unlisted bare city.")))
})
test_that("AC2.3 (R half) strip removes exactly the dateline span", {
  # raw == removed_span + stripped: stripping yields the post-delimiter remainder,
  # and re-extracting from the stripped text finds no dateline.
  lead <- "WASHINGTON, July 30 — Congress acted on the budget today."
  bi <- extract_dateline_block(lead)
  stripped <- strip_dateline(lead, bi)
  expect_equal(stripped, "Congress acted on the budget today.")
  expect_false(extract_dateline_block(stripped)$found)
  # Credit-line case: the "Special to The New York Times" prefix is consumed too.
  lead2 <- "Special to The New York Times CHICAGO — The mayor spoke."
  bi2 <- extract_dateline_block(lead2)
  expect_equal(strip_dateline(lead2, bi2), "The mayor spoke.")
})
```

**Step 3: Run**

```bash
Rscript r/tests/run_tests.R
```
Expected: all tests pass (testthat reports `[ FAIL 0 | ... ]`).

**Step 4: Commit**

```bash
git add r/tests
git commit -m "test(us-filter): resolver canonical collision-set tests"
```
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->

<!-- START_TASK_4 -->
### Task 4: Desk/section signal + fusion policy

**Verifies:** us-filter.AC1.7, AC1.8, AC1.9.

**Files:**
- Create: `r/dateline/label_policy.R`
- Modify: `r/tests/testthat/test-resolve-dateline.R` (append fusion tests) OR create `r/tests/testthat/test-label-policy.R`
- Modify: `r/tests/run_tests.R` (source `label_policy.R` too)

**Implementation** `r/dateline/label_policy.R` (`# pattern: Functional Core`). The desk/section lists are **copied** from `nyt_location_checking.R:65-82` (provenance comment); transcribe the exact list contents from that file during execution.

```r
# pattern: Functional Core
# Desk/section US-signal + dateline/desk fusion policy.
# Desk lists copied from nyt_location_checking.R:65-82 (provenance: out-of-repo
# /Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R).

# Transcribe the exact vectors from nyt_location_checking.R:65-82.
DSK_US        <- c(<COPY from nyt_location_checking.R dsk_us>)
DSK_NON_US    <- c(<COPY from nyt_location_checking.R dsk_non_us>)
SECTION_US    <- c(<COPY from nyt_location_checking.R section_us>)
SECTION_NON_US<- c(<COPY from nyt_location_checking.R section_non_us>)

# Desk/section US signal: TRUE | FALSE | NA.
desk_section_signal <- function(dsk, print_section) {
  is_us  <- (!is.na(dsk) && dsk %in% DSK_US)         || (!is.na(print_section) && print_section %in% SECTION_US)
  is_non <- (!is.na(dsk) && dsk %in% DSK_NON_US)     || (!is.na(print_section) && print_section %in% SECTION_NON_US)
  if (is_us && !is_non) return(TRUE)
  if (is_non && !is_us) return(FALSE)
  NA  # silent or internally-conflicting desk signal -> no confident desk label
}

# Fuse dateline + desk signals -> list(us_label, label_source).
classify_label <- function(dateline_is_us, desk_is_us) {
  has_dl   <- !is.na(dateline_is_us)
  has_desk <- !is.na(desk_is_us)
  if (has_dl && has_desk && (dateline_is_us != desk_is_us))
    return(list(us_label = NA, label_source = "conflict"))   # AC1.8
  if (has_dl)
    return(list(us_label = dateline_is_us, label_source = "dateline"))  # AC1.1-1.6
  if (has_desk)
    return(list(us_label = desk_is_us, label_source = "heuristic"))     # AC1.7
  list(us_label = NA, label_source = NA_character_)           # AC1.9
}
```

**Step 1:** Open `/Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R`, read lines 65–82, and transcribe the four vectors verbatim into the placeholders. Keep the provenance comment.

**Step 2: Tests** (append to the testthat suite):

```r
test_that("AC1.7 desk backfill when no dateline", {
  # Use a real DSK_NON_US value (e.g. 'Foreign Desk') and a real DSK_US value.
  expect_false(classify_label(NA, desk_section_signal("Foreign Desk", NA))$us_label)
  expect_equal(classify_label(NA, desk_section_signal("Foreign Desk", NA))$label_source, "heuristic")
  expect_true(classify_label(NA, desk_section_signal("Metropolitan Desk", NA))$us_label)
})
test_that("AC1.8 dateline/desk conflict -> null/conflict", {
  res <- classify_label(TRUE, FALSE)
  expect_true(is.na(res$us_label)); expect_equal(res$label_source, "conflict")
})
test_that("AC1.9 unresolved -> null", {
  res <- classify_label(NA, NA)
  expect_true(is.na(res$us_label)); expect_true(is.na(res$label_source))
})
test_that("dateline wins when desk agrees or is silent", {
  expect_true(classify_label(TRUE, NA)$us_label)
  expect_equal(classify_label(TRUE, TRUE)$label_source, "dateline")
})
```
Adjust the desk string literals in the AC1.7 test to actual values present in `DSK_US`/`DSK_NON_US` after transcription.

**Step 3:** Update `r/tests/run_tests.R` to also `source("r/dateline/label_policy.R")`.

**Step 4: Run**

```bash
Rscript r/tests/run_tests.R
```
Expected: all tests pass.

**Step 5: Commit**

```bash
git add r/dateline/label_policy.R r/tests
git commit -m "feat(us-filter): desk/section heuristic + dateline-desk fusion policy"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Build script + config paths — emit the derived labeled parquet

**Files:**
- Create: `r/dateline/build_labels.R` (`# pattern: Imperative Shell`)
- Modify: `src/config.py` (add `US_FILTER_DIR`, `US_FILTER_LABELED_PARQUET`)

**Step 1: Add config paths.** In `src/config.py`, in the artifacts section (after `CCA_*` constants), add:

```python
# US/not-US pre-filter artifacts
US_FILTER_DIR: Path = PROJECT_ROOT / "us_filter"
US_FILTER_LABELED_PARQUET: Path = US_FILTER_DIR / "ldc_labeled.parquet"
```

Verify: `uv run python -c "import src.config as c; print(c.US_FILTER_DIR, c.US_FILTER_LABELED_PARQUET)"` prints the two paths and does not raise.

**Step 2: Build script `r/dateline/build_labels.R`**

```r
# pattern: Imperative Shell
# Read the full LDC partitioned dataset (1987-2007), apply the dateline resolver +
# desk fusion row-wise, and write a derived labeled parquet for the US filter.

suppressMessages({ library(arrow) })
source("r/dateline/resolve_dateline.R")
source("r/dateline/label_policy.R")

LDC_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/ldc_corpus/"
OUT_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter"
OUT_PARQUET <- file.path(OUT_DIR, "ldc_labeled.parquet")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

gz <- load_gazetteers("r/dateline/gazetteers")

df <- as.data.frame(open_dataset(LDC_DIR)) [, c("id", "headline", "lead_paragraph", "dsk", "print_section")]
n <- nrow(df)

us_label <- logical(n); label_source <- character(n)
dateline_place <- character(n); stripped_text <- character(n); raw_text <- df$lead_paragraph

for (i in seq_len(n)) {
  rd   <- resolve_dateline(df$lead_paragraph[i], gz)
  desk <- desk_section_signal(df$dsk[i], df$print_section[i])
  lab  <- classify_label(rd$is_us, desk)
  us_label[i]       <- lab$us_label
  label_source[i]   <- lab$label_source
  dateline_place[i] <- rd$place
  stripped_text[i]  <- strip_dateline(df$lead_paragraph[i], rd$block_info)
}

out <- data.frame(
  id = df$id, headline = df$headline, us_label = us_label, label_source = label_source,
  dateline_place = dateline_place, stripped_text = stripped_text,
  raw_text = raw_text, stringsAsFactors = FALSE
)
write_parquet(out, OUT_PARQUET)

# Operational check: label_source breakdown + extraction coverage.
cat("rows:", n, "\n")
print(table(label_source, useNA = "ifany"))
cat("dateline-extracted coverage:", round(mean(!is.na(dateline_place)) * 100, 1), "%\n")
```

Note: the row-wise loop over ~1.4M rows may be slow; if it is prohibitive during execution, vectorize via `vapply`/`Map` over the columns, but keep the per-row resolver contract identical. (Surface to the user if a vectorization refactor is needed rather than changing the resolver semantics.)

**Step 3: Verify operationally**

```bash
Rscript r/dateline/build_labels.R
```
Expected: writes `ldc_labeled.parquet`; prints a `label_source` breakdown where `dateline` is the dominant confident class and `conflict`/`<NA>` are minorities. Then confirm Python can read it:

```bash
uv run python -c "import polars as pl, src.config as c; df=pl.read_parquet(c.US_FILTER_LABELED_PARQUET); print(df.columns); print(df['label_source'].value_counts())"
```
Expected: columns `id, headline, us_label, label_source, dateline_place, stripped_text, raw_text`; a sane value-count breakdown. (`headline` is carried so Phase 3 can assemble `headline + "</s>" + stripped_text` from this single parquet.)

**Step 4: Commit**

```bash
git add src/config.py r/dateline/build_labels.R
git commit -m "feat(us-filter): build derived LDC labeled parquet (1987-2007)"
```

**Verifies:** none directly (data-path gate for AC1; feeds Phase 3).
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->

---

## Execution deviation note (2026-06-09, agreed with user)

**Empirical finding:** the LDC parquet at `00_explorer/ldc_corpus/` does NOT contain datelines in `lead_paragraph`/`full_text` (~0.02% embedded tail only, e.g. `"WASHINGTON, March 2 - ..."`). The dateline is a separate NITF element (`/nitf/body/body.head/dateline`) that the parquet-feeding CSV pipeline (`LDC2008T19/data/scripts/01_all_to_csv.R`) never extracted. The parallel rds pipeline (`01_all_to_rds.R:35`) DID extract it: `LDC2008T19/data/parsed_to_rds/{1987..2007}.rds` each carry a `dateline` column with 27–40% non-NA coverage (e.g. 2000: 17,939/64,068). Values are clean structured fields: `"PASADENA, Calif., Dec. 31"`, `"ZAGREB, Croatia"`, `"HAMILTON, New Zealand, Saturday, Jan. 1"`.

**Revised approach (replaces text-extraction as the primary label channel):**
1. **Label channel = rds join.** `build_labels.R` reads per-year rds (select `id`, `dateline`, dedup by id), left-joins onto the LDC parquet by `id`, and resolves the dateline **field** via a new pure entry point `resolve_dateline_field(dateline_str, gz)` — same field-splitting / date-dropping / structure-first resolution core (AC1 logic unchanged), plus weekday-token filtering (`"Saturday"`).
2. **Text channel = hygiene only.** The lead-text extractor remains solely to catch the embedded tail and keep `stripped_text` leakage-proof. Its regex must be strict (all-caps city block; optional short mixed-case qualifier/date fields) and stripping is **conditional**: a block is treated as a dateline only if it has a date field, a recognized state/country qualifier, or is a bare AP-list city. Emphasis-caps ledes (`"PILOBOLUS - that dance troupe..."`) must never strip. (The first implementation's loosened regex falsely matched ~33k rows and corrupted their `stripped_text`; this is the corrective.)
3. Output schema, desk fusion policy, and config constants are unchanged.

Phase 3's no-residue guard must mirror the conditional text-channel semantics (its detector is the text channel's Python port; the rds field never reaches model-input text).

## Phase 1 Done When

- The resolver testthat suite passes on the canonical collision set (`Rscript r/tests/run_tests.R` → 0 failures), including field-channel and conditional-strip tests.
- The derived labeled parquet exists over LDC 1987–2007 with a sane `label_source` breakdown (dateline-dominant per the rds join — expect roughly 27–40% dateline-labeled; conflict/null minorities).
- `src/config.py` exposes `US_FILTER_DIR` / `US_FILTER_LABELED_PARQUET`.

Covers **us-filter.AC1**.
