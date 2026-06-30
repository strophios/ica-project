# Dateline Resolver Gazetteers

## Source Files

### `countries.csv`
Copied from `/Users/strophios/immigration_project/00_ML_data_expansion/context_data/countries.csv`.
Curated list of country names and identifiers. **In-repo copy is the resolver's source of truth.**

### `state_long_abbrs.csv`
Copied from `/Users/strophios/immigration_project/00_ML_data_expansion/context_data/state_long_abbrs.csv`.
US state full names and abbreviations (both AP long-form and USPS 2-letter codes).

### `ap_us_cities.csv`
Curated list of 30 major US cities recognized as AP-stylebook standalone dateline cities.
These cities can appear bare (e.g., `CHICAGO —`) without a state qualifier in dateline context.

Cities: Atlanta, Baltimore, Boston, Chicago, Cincinnati, Cleveland, Dallas, Denver, Detroit, Honolulu, Houston, Indianapolis, Las Vegas, Los Angeles, Miami, Milwaukee, Minneapolis, New Orleans, New York, Oklahoma City, Philadelphia, Phoenix, Pittsburgh, St. Louis, Salt Lake City, San Antonio, San Diego, San Francisco, Seattle, Washington.

### `ap_foreign_cities.csv`
Curated list of 49 major foreign cities recognized as AP-stylebook standalone dateline cities.
These cities can appear bare (e.g., `LONDON —`) without a country qualifier in dateline context.

Source: AP Stylebook dateline conventions for major international news hubs.

## Resolver Behavior

**Collision rule (AC1.10 invariant):**
- A bare city name (no qualifier) is resolved **only** against the short AP-30 US list and AP-foreign list.
- Bare tokens **never** consult the long `countries.csv` or area-code lists (if used elsewhere).
- This prevents false positives like resolving a bare `Portugal` (a country name) as a dateline city.

**State/country qualifier precedence:**
- `PARIS, Texas` → state qualifier present → US
- `PARIS, France` → country qualifier present → not-US
- `PARIS` (bare) → AP-foreign list → not-US
- `PARIS, N.Y.` → state qualifier present → US

**Eight never-abbreviated US states:**
The `state_long_abbrs.csv` includes full names and AP long-form abbreviations for all 50 US states. The resolver normalizes tokens to lowercase alpha-only for matching, so both "Wash." and "Washington" resolve identically in the state context.
