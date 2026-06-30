# ---- PROVENANCE HEADER ----
#
# Vendored heuristic for US/not-US location assignment.
#
# Source: /Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R
#         Lines 65-82 (constants), 102-206 (loc_df construction), 247-344 (us_assign function)
# Extracted: 2026-06-11
#
# What was taken:
#   - Constants: section_us, dsk_us, section_non_us, dsk_non_us (lines 65-82)
#   - Location keyword gazetteer construction: loc_df (lines 102-206)
#   - Core function: us_assign(df, return_long=FALSE) (lines 247-344)
#
# Why vendored:
#   - The original out-of-repo file is a mixed library/analysis script whose
#     interactive analysis tail errors on wholesale source() (end-of-file interactivity).
#   - It is an un-versioned, load-bearing dependency for the audit in api_ldc_join.R.
#   - Freezing the heuristic here is intentional: the audit benchmarks this specific
#     implementation against the dateline resolver (Phase 1 audit commitment).
#   - The original file may drift after this extraction date; this copy is stable.
#
# CSV paths updated (only edit to the original):
#   - CSV reads now point to in-repo copies under r/vendored/gazetteers/ and r/dateline/gazetteers/
#     (see the comment on each read_csv call below).
#
# ---- END PROVENANCE HEADER ----

suppressMessages({
  library(tidyverse)
})

# ---- Section/Desk constants (from original lines 65-82) ----
section_us <- c("U.S.", "New York", "New York and Region", "Washington") |>
  str_to_lower()
dsk_us <- c(
  "National Desk",
  "Metropolitan Desk",
  "Connecticut Weekly Desk",
  "Westchester Weekly Desk",
  "Long Island Weekly Desk",
  "New York Region",
  "New Jersey Weekly Desk",
  "The City Weekly Desk"
) |>
  str_to_lower()

section_non_us <- c("World") |>
  str_to_lower()
dsk_non_us <- c("Foreign Desk") |>
  str_to_lower()

# ---- Location Keywords & Gazetteers (from original lines 102-206) ----
# CSV paths have been updated to point to in-repo copies.
# state_long_abbrs.csv and countries.csv reuse the in-repo copies from r/dateline/gazetteers/
# (to avoid duplication; they are scoped to the dateline resolver and this heuristic).
state_abbrs <- read_csv(
  "r/dateline/gazetteers/state_long_abbrs.csv",
  show_col_types = FALSE
)
countries <- read_csv(
  "r/dateline/gazetteers/countries.csv",
  show_col_types = FALSE
)
# us-area-code-cities.csv is vendored here (not in r/dateline/gazetteers/) because it is
# BANNED from the dateline resolver (AC1.10 collision-trap invariant — bare dateline tokens
# must never resolve against it). It is scoped to this audit keyword heuristic only.
us_cities <- read_csv(
  "r/vendored/gazetteers/us-area-code-cities.csv",
  col_names = c("area_code", "city", "state", "country", "lat", "long"),
  show_col_types = FALSE
)

# Search order is going to matter. I want to search such that we don't keep
# searching once we have a match. Rough order is something like:
# - United States, plus abbrs?
# - New York (City/State)
# - continents/areas (Middle East, Europe, etc.)
# - USSR / Union of Soviet Socialist Republics
# - selection of most common foreign countries (China, Great Britain, Canada, Russia, France, India, Cuba, Japan, Vietnam, Israel, Italy, South Africa, Iran, Iraq, Germany...)
# - US states (full names? long abbreviations? normal abbreviations?)
# - US abbreviations (if not already)
# - Other foreign countries
# - US state short abbreviations
# - US cities

loc_df <- tibble(
  value = c(
    "United States",
    "New York City",
    "New York",
    "NYC",
    "USA",
    "Middle East",
    "Europe",
    "Asia",
    "Africa",
    "North America",
    "South America",
    "Antarctica",
    "Australia",
    "USSR",
    "Union of Soviet Socialist Republics",
    "China",
    "Great Britain",
    "Canada",
    "Russia",
    "France",
    "Germany",
    "United Kingdom",
    "India",
    "Japan",
    "Vietnam",
    "Israel",
    "Italy",
    "Spain",
    "South Africa",
    "Iran",
    "Iraq",
    "Yugoslavia",
    "Paris",
    "Moscow",
    "London",
    "Manhattan",
    "Brooklyn",
    "Staten Island",
    "U.S.",
    "U.S.A."
  ),
  is_us = c(rep(TRUE, times = 5), rep(FALSE, times = 30), rep(TRUE, times = 5))
)


loc_df <- rbind(
  loc_df,
  tibble(
    value = c(
      state_abbrs[["full"]][!(state_abbrs[["full"]] %in% loc_df[["value"]])],
      state_abbrs |>
        mutate(long_abbr = str_remove_all(long_abbr, "[^A-Za-z]")) |>
        filter(str_to_lower(long_abbr) != str_to_lower(usps_abbr)) |>
        pull(long_abbr)
    ),
    is_us = TRUE
  )
)

loc_df <- rbind(
  loc_df,
  tibble(
    value = countries[["value"]][
      !(countries[["value"]] %in% loc_df[["value"]])
    ],
    is_us = FALSE
  )
)

loc_df <- rbind(
  loc_df,
  tibble(
    value = c(
      state_abbrs[["usps_abbr"]],
      us_cities[["city"]][!(us_cities[["city"]] %in% loc_df[["value"]])]
    ),
    is_us = TRUE
  )
)

loc_df[["pattern"]] <- paste("\\b", loc_df[["value"]], "\\b", sep = "")
# not sure whether we want str_to_lower for these

# ---- us_assign function (from original lines 247-344, byte-faithful transcription) ----
us_assign <- function(df, return_long = FALSE) {
  #' Determine whether each story in a df of NYT stories is in the US or not
  #' `return_long` sets whether `us_assign()` we return the long form of the
  #' location assigned result (i.e., one row per location keyword) or the
  #' aggregated form (one row per story).

  # First we assign location based on section and desk
  df <- df |>
    mutate(
      meta_is_us = str_to_lower(section_name) %in%
        section_us |
        str_to_lower(news_desk) %in% dsk_us,
      meta_not_us = str_to_lower(section_name) %in%
        section_non_us |
        str_to_lower(news_desk) %in% dsk_non_us
    )

  # the LDC corpus sometimes has multiple section names separated by semicolons
  # so we check for that, address those separately, and update the first pass
  rns <- which(str_detect(df[["section_name"]], "; "))
  if (length(rns) > 0) {
    sec_rns_us <- map(rns, function(rn) {
      sec <- str_to_lower(df[["section_name"]][rn])
      sec <- str_split(sec, "; ")[[1]]
      sec_us <- any(sec %in% section_us)
      sec_not <- any(sec %in% section_non_us)

      tibble(meta_is_us = sec_us, meta_not_us = sec_not)
    }) |>
      bind_rows()

    df[["meta_is_us"]][rns] <- sec_rns_us[["meta_is_us"]] |
      df[["meta_is_us"]][rns]
    df[["meta_not_us"]][rns] <- sec_rns_us[["meta_not_us"]] |
      df[["meta_not_us"]][rns]
  }

  # Then we assign based on keywords
  # First, unnest (location) keywords so that each location
  # assigned to a story gets one row (we'll stick them together
  # afterwards)
  df <- df |>
    mutate(
      location = map(keywords, function(kws) {
        out <- kws[str_detect(kws[["type"]], "location"), ][["value"]]
        if (length(out) == 0) {
          out <- NA_character_
        }
        out
      })
    ) |>
    unnest(cols = location)

  loc_present <- which(!is.na(df[["location"]]))
  assigned <- rep(FALSE, times = length(loc_present))
  is_us <- rep(FALSE, times = length(loc_present))
  not_us <- rep(FALSE, times = length(loc_present))

  for (rn in seq_len(nrow(loc_df))) {
    to_test <- df[["location"]][loc_present[!assigned]]
    # if the pattern we're testing doesn't rely on all caps (i.e., isn't
    # an abbreviation), we title case the values we're testing against
    if (!str_detect(loc_df[["value"]][rn], "^[A-Z]+$")) {
      to_test <- str_to_title(to_test)
    }
    out <- str_detect(to_test, loc_df[["pattern"]][rn])
    if (loc_df[["is_us"]][rn]) {
      is_us[!assigned] <- out
    } else {
      not_us[!assigned] <- out
    }
    assigned[!assigned] <- out
  }

  df[["is_us"]] <- FALSE
  df[["not_us"]] <- FALSE
  df[["is_us"]][loc_present] <- is_us
  df[["not_us"]][loc_present] <- not_us

  df_long <- df
  df <- df |>
    select(-location) |>
    group_by(id) |>
    mutate(is_us = any(is_us), not_us = any(not_us)) |>
    ungroup() |>
    distinct()

  # Finally, we create `any_us` and `any_not_us` columns to aggregate
  # the section/desk assigned locations and keyword based locations
  df <- df |>
    mutate(any_us = is_us | meta_is_us, any_not_us = not_us | meta_not_us)

  if (return_long) {
    df_long
  } else {
    df
  }
}
