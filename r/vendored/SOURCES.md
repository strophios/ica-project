# r/vendored — Provenance and Scope

This directory vendors the `us_assign` heuristic from the out-of-repo analysis script `/Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R` (extracted 2026-06-11).

## Vendored Code

**us_assign.R**: Location-based US/not-US assignment heuristic. Extracted from the original source at lines 65–82 (constants), 102–206 (gazetteer construction), and 247–344 (function definition). See the provenance header in that file for detailed sourcing.

**Why vendored**: The original out-of-repo file is a mixed library/analysis script that errors on wholesale `source()` due to interactive analysis at the end of file. It is a load-bearing un-versioned dependency for the audit in `r/audit/api_ldc_join.R`. Freezing it here is intentional to establish a stable baseline: the audit benchmarks this specific implementation against the dateline resolver (Phase 1 audit commitment). The original may drift post-extraction; this copy is stable and pinned to the extraction date.

## Gazetteers

**us-area-code-cities.csv**: Vendored copy of `/Users/strophios/immigration_project/00_ML_data_expansion/context_data/us-area-code-cities.csv`. Used by the `loc_df` construction in `us_assign.R` for matching US city location keywords.

Load-bearing scoping note (AC1.10 boundary): **This file is intentionally NOT in `r/dateline/gazetteers/`.** The dateline resolver in `r/dateline/resolve_dateline.R` must never match against raw city names (bare `"New York"` would match both the state and NYC; collision-trap invariant AC1.10). This gazetteer is scoped ONLY to the `us_assign` keyword-matching heuristic, which operates in a different context (keywords are explicitly tagged as location descriptors in article metadata). To maintain this invariant, any gazetteer reuse must pass through `us_assign.R` (the intended consumer), not directly imported elsewhere.

**state_long_abbrs.csv, countries.csv**: Reuse existing in-repo copies from `r/dateline/gazetteers/` to avoid duplication. These are shared across the dateline resolver and the `us_assign` heuristic.
