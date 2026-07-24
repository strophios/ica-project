# r/ — Dateline labeling & corpus ingest (R)

*Last updated: 2026-06-10. Domain doc for the R-side pipeline that produces US/not-US labels and the API corpus. Python consumers and the overall ML system live under `src/` (see the root `CLAUDE.md`, "US/not-US Pre-Filter").*

## Purpose

This tree derives the **US/not-US labels** for the pre-filter and converts the NYT Archive API corpus to parquet. It exists in R (not Python) because the source data is in R `.rds` files whose nested structure only `arrow` bridges correctly, and because the dateline/desk resolution logic is ported from prior out-of-repo R scripts.

**Run R code and tests from the project root** (scripts source `r/dateline/*.R` by relative path and read `r/dateline/gazetteers`).

## Layout

- `r/dateline/` — the label pipeline:
  - `resolve_dateline.R` — Functional Core. Dateline extraction + structure-first US/not-US place resolution against gazetteers. Pure functions plus one thin I/O helper (`load_gazetteers`).
  - `label_policy.R` — Functional Core. Desk/section US-signal lists (`DSK_US`, `DSK_NON_US`, `SECTION_US`, `SECTION_NON_US`, transcribed from out-of-repo `nyt_location_checking.R:65-82`) + dateline/desk fusion policy.
  - `build_labels.R` — Imperative Shell. One-shot batch: reads LDC parquet (1987–2007) + per-year rds `dateline`, resolves, fuses, writes the derived labeled parquet.
  - `gazetteers/` — `state_long_abbrs.csv`, `countries.csv`, `ap_us_cities.csv` (AP-30), `ap_foreign_cities.csv` (AP-46), `SOURCES.md` (provenance).
- `r/api_ingest/` — the NYT Archive API pull + corpus conversion:
  - `archive_transform.R` — Functional Core. Story-list → canonical per-year tibble schema (`ARCHIVE_COLS`, 14 columns incl. the historical `abstrct` name; keywords as nested `type/value/rank/major` tibbles), fetch planning (`month_plan`, incl. the rotating-month `--skeleton` mode for the pre-1960 backfill), CLI-value parsing, and the `validate_archive_tibble` schema gate.
  - `pull_archive.R` — Imperative Shell. Resumable Archive API pull: per-month RAW-response checkpoints (`nyt_archive_raw/{year}_{MM}.rds` — transform bugs are fixable without re-pulling), 12s rate-limit sleep, retry/backoff, a per-run request budget (default 480, under the API's 500/day cap), and assembly of complete years only into `nyt_archive_by_year/{year}.rds` (a partial year — skeleton or `--through`-capped — can never leak into the corpus). Key via `NYT_API_KEY` env var, never hardcoded. Supersedes the interactive out-of-repo `nyt_headlines.R` (which pulled 1960–1995).
  - `rds_to_parquet.R` — Imperative Shell. Converts per-year API `.rds` files to `api_corpus/{year}.parquet`, preserving the nested `keywords` list-column. Incremental (skips years whose parquet exists; `--force`, `--years YYYY:YYYY` to override) — the original hard-asserted exactly 36 files (1960–1995); the pull now grows the set.
- `r/audit/api_ldc_join.R` — Imperative Shell. Applies the legacy `us_assign` location logic (from `nyt_location_checking.R:247-344`) for the free-audit cross-check; LDC-side (not API-side — see the provenance comment).
- `r/tests/` — testthat suite. `Rscript r/tests/run_tests.R` from the project root; **stop-on-failure**, currently 161 passing assertions (2026-07-24). `run_tests.R` sources the resolver + policy + `archive_transform.R`, loads gazetteers into the parent env, then `test_dir`s `r/tests/testthat/`.

**Linting:** project-root `.lintr` (DCF format — no comments allowed in the file, so the reasoning lives here): UPPER_SNAKE allowed for module constants (tree convention: `DSK_US`, `SRC_DIR`, `ARCHIVE_COLS`); `commented_code_linter` off (script-header usage examples false-positive); `object_usage_linter` off (this is a `source()`-based tree, so cross-file functions are invisible to per-file lint — the testthat suite is what proves name resolution); line length 120. lintr's config discovery doesn't find the root file from subdirectory paths in a non-package tree — invoke with the option set: `Rscript -e 'options(lintr.linter_file = ".lintr"); print(lintr::lint("<file>"))'` from the project root.

## Key contracts

- **Datelines live in rds metadata, NOT in the LDC parquet text.** The parquet-feeding CSV pipeline never extracted the NITF `dateline` element; the parallel `parsed_to_rds/{year}.rds` did (27–40% coverage). So the **primary label channel is the rds join** via `resolve_dateline_field(dateline_str, gz)` — structure-first field resolution (split on commas, drop date/weekday fields, resolve the leading place + trailing qualifier against the gazetteers). This is the authoritative amendment in `docs/implementation-plans/2026-06-06-us-filter/phase_01.md`.
- **Text channel = hygiene only, and stripping is CONDITIONAL.** The lead-text extractor (`resolve_dateline` / `should_strip_dateline_block`) strips a caps-block prefix only when it (i) has a date field, (ii) has a qualifier resolving against US states/countries, or (iii) is a bare AP-list city. Emphasis-caps ledes (e.g. `"PILOBOLUS - that dance troupe…"`, ~1,400 real rows) are deliberately NOT stripped. A looser earlier regex falsely matched ~33k rows and corrupted their `stripped_text`; the strict conditional form is the corrective.
- **Boundary-inventory pair with Python.** `src/preproc/dateline_guard.py` is the Python port of the conditional-strip *detection* half of `resolve_dateline.R`. Any change to the credit-line / caps-block / delimiter / conditional-strip logic in either file MUST be mirrored in the other. `normalize_token` (lowercase + alpha-only) is shared verbatim across the boundary.
- **Gazetteer scope.** States come from full names + AP long abbreviations ONLY — NOT 2-letter USPS codes, which collide with English words (OR/IN/OK). This is load-bearing for the AC1.10 / no-residue guard intent.

## Outputs (derived data products, gitignored)

- `us_filter/ldc_labeled.parquet` — training source. Carries `us_label` (bool, null = unresolved/conflict), `label_source`, and the leakage-proof `stripped_text`. Written by `build_labels.R`. Consumed by `src/run_us_classification.py` (`data_from_parquet(..., db_folder="us_filter", lead_column="stripped_text")`).
- `api_corpus/` — read-only application target, written by `rds_to_parquet.R`.

If gazetteers are finalized or `build_labels.R` is re-run, the downstream `us_filter/us_set/` tf.data cache must be deleted so it rebuilds (stale cache silently diverges from recomputed steps).
