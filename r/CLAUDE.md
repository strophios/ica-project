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
- `r/api_ingest/rds_to_parquet.R` — Imperative Shell. Converts the 36 per-year API `.rds` files to `api_corpus/{year}.parquet`, preserving the nested `keywords` list-column.
- `r/audit/api_ldc_join.R` — Imperative Shell. Applies the legacy `us_assign` location logic (from `nyt_location_checking.R:247-344`) for the free-audit cross-check; LDC-side (not API-side — see the provenance comment).
- `r/tests/` — testthat suite. `Rscript r/tests/run_tests.R` from the project root; **stop-on-failure**, currently 79 passing assertions. `run_tests.R` sources the resolver + policy, loads gazetteers into the parent env, then `test_dir`s `r/tests/testthat/`.

## Key contracts

- **Datelines live in rds metadata, NOT in the LDC parquet text.** The parquet-feeding CSV pipeline never extracted the NITF `dateline` element; the parallel `parsed_to_rds/{year}.rds` did (27–40% coverage). So the **primary label channel is the rds join** via `resolve_dateline_field(dateline_str, gz)` — structure-first field resolution (split on commas, drop date/weekday fields, resolve the leading place + trailing qualifier against the gazetteers). This is the authoritative amendment in `docs/implementation-plans/2026-06-06-us-filter/phase_01.md`.
- **Text channel = hygiene only, and stripping is CONDITIONAL.** The lead-text extractor (`resolve_dateline` / `should_strip_dateline_block`) strips a caps-block prefix only when it (i) has a date field, (ii) has a qualifier resolving against US states/countries, or (iii) is a bare AP-list city. Emphasis-caps ledes (e.g. `"PILOBOLUS - that dance troupe…"`, ~1,400 real rows) are deliberately NOT stripped. A looser earlier regex falsely matched ~33k rows and corrupted their `stripped_text`; the strict conditional form is the corrective.
- **Boundary-inventory pair with Python.** `src/preproc/dateline_guard.py` is the Python port of the conditional-strip *detection* half of `resolve_dateline.R`. Any change to the credit-line / caps-block / delimiter / conditional-strip logic in either file MUST be mirrored in the other. `normalize_token` (lowercase + alpha-only) is shared verbatim across the boundary.
- **Gazetteer scope.** States come from full names + AP long abbreviations ONLY — NOT 2-letter USPS codes, which collide with English words (OR/IN/OK). This is load-bearing for the AC1.10 / no-residue guard intent.

## Outputs (derived data products, gitignored)

- `us_filter/ldc_labeled.parquet` — training source. Carries `us_label` (bool, null = unresolved/conflict), `label_source`, and the leakage-proof `stripped_text`. Written by `build_labels.R`. Consumed by `src/run_us_classification.py` (`data_from_parquet(..., db_folder="us_filter", lead_column="stripped_text")`).
- `api_corpus/` — read-only application target, written by `rds_to_parquet.R`.

If gazetteers are finalized or `build_labels.R` is re-run, the downstream `us_filter/us_set/` tf.data cache must be deleted so it rebuilds (stale cache silently diverges from recomputed steps).
