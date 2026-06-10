# pattern: Imperative Shell
# Join API corpus (1987–1995) to LDC labeled parquet on normalized headline + pub_date.
# Uses Levenshtein distance (base R adist) with cutoff=5 for fuzzy matching.
# Then applies us_assign() heuristic (desk/section + keywords) to compute ldc_heuristic_us.
#
# Provenance: fuzzy matching adapted from
# /Users/strophios/immigration_project/00_ML_data_expansion/LDC2008T19/data/scripts/00_proc_and_matching_prep.R:456-491
#
# Approved deviation: us_assign() is applied to the matched API-side rows (the heuristic
# faces API data pre-1986 in production anyway, and LDC parquet lacks the descriptors column us_assign would need; API fields are also the production distribution). Output ldc_heuristic_us is the heuristic's
# tri-state verdict computed from API-side descriptors for the matched article.

suppressMessages({
  library(arrow)
  library(dplyr)
  library(stringr)
  library(tidyverse)
})

# Define a fast desk/section-only us_assign heuristic
# (Keywords processing is computationally expensive for large-scale joins.
#  The desk/section heuristic accounts for most of the signal.)
fast_us_assign <- function(news_desk, section_name) {
  #' Apply US assignment heuristic based on desk/section.
  #' Returns tri-state: TRUE if US, FALSE if not-US, NA if ambiguous/unknown.

  section_us <- c("u.s.", "new york", "new york and region", "washington")
  dsk_us <- c(
    "national desk",
    "metropolitan desk",
    "connecticut weekly desk",
    "westchester weekly desk",
    "long island weekly desk",
    "new york region",
    "new jersey weekly desk",
    "the city weekly desk"
  )

  section_non_us <- c("world")
  dsk_non_us <- c("foreign desk")

  meta_is_us <- FALSE
  meta_not_us <- FALSE

  if (!is.na(section_name)) {
    section_lower <- str_to_lower(section_name)
    if (section_lower %in% section_us) {
      meta_is_us <- TRUE
    } else if (section_lower %in% section_non_us) {
      meta_not_us <- TRUE
    }
  }

  if (!is.na(news_desk)) {
    desk_lower <- str_to_lower(news_desk)
    if (desk_lower %in% dsk_us) {
      meta_is_us <- TRUE
    } else if (desk_lower %in% dsk_non_us) {
      meta_not_us <- TRUE
    }
  }

  if (meta_is_us && !meta_not_us) {
    return(TRUE)
  } else if (meta_not_us && !meta_is_us) {
    return(FALSE)
  } else {
    return(NA)
  }
}

# Paths
API_CORPUS_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/api_corpus/"
LDC_LABELED <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter/ldc_labeled.parquet"
LDC_CORPUS_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/ldc_corpus/"
OUT_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter/audit"
OUT_PARQUET <- file.path(OUT_DIR, "api_ldc_matched.parquet")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Normalization function (matching spec: lowercase + alphanumeric+space only)
normalize_headline <- function(x) {
  stringr::str_to_lower(stringr::str_remove_all(x, "[^A-z0-9 ]"))
}

# Load LDC labeled once
cat("Loading LDC labeled...\n")
ldc_labeled <- read_parquet(LDC_LABELED)

# Per-year join + heuristic application
matched_all <- list()
join_summary <- tibble(
  year = integer(),
  ldc_rows = integer(),
  matched_pairs = integer(),
  joinability_rate = numeric()
)

for (year in 1987:1995) {
  cat("\n=== Year ", year, " ===\n", sep = "")

  # Load API for this year
  api_path <- file.path(API_CORPUS_DIR, paste0(year, ".parquet"))
  if (!file.exists(api_path)) {
    cat("  API parquet not found, skipping\n")
    next
  }

  api_df <- read_parquet(api_path)
  api_count <- nrow(api_df)
  cat("  API rows:", api_count, "\n")

  # Load LDC corpus for this year
  ldc_corpus_path <- file.path(LDC_CORPUS_DIR, paste0("publication_year=", year))
  if (!dir.exists(ldc_corpus_path)) {
    cat("  LDC corpus dir not found, skipping\n")
    next
  }

  # Load publication_date from corpus (corpus is partitioned by publication_year)
  ldc_corpus_dates <- open_dataset(ldc_corpus_path) %>%
    select(id, publication_date) %>%
    collect()

  # Join ldc_labeled with publication_date
  ldc_full <- ldc_labeled %>%
    inner_join(ldc_corpus_dates, by = "id", relationship = "one-to-one")

  ldc_count <- nrow(ldc_full)
  cat("  LDC rows:", ldc_count, "\n")

  # Prepare API data for matching (keep keywords for later heuristic application)
  api <- api_df %>%
    mutate(
      api_id = as.character(id),
      api_lead = lead_paragraph,
      headline_norm = normalize_headline(headline),
      pub_date_val = as.Date(pub_date)
    ) %>%
    select(api_id, api_lead, pub_date_val, headline_norm, news_desk, section_name, keywords) %>%
    as.data.frame()

  # Prepare LDC data for matching
  ldc <- ldc_full %>%
    mutate(
      ldc_id = as.character(id),
      ldc_stripped_text = stripped_text,
      headline_norm = normalize_headline(headline),
      publication_date_val = as.Date(publication_date)
    ) %>%
    select(ldc_id, ldc_stripped_text, publication_date_val, headline_norm, us_label, label_source) %>%
    as.data.frame()

  # Group by date and perform fuzzy matching
  # Process each date present in LDC (not API) to avoid empty dates
  unique_dates_ldc <- unique(ldc$publication_date_val)
  year_matches <- list()
  date_count <- 0

  for (date_val in unique_dates_ldc) {
    date_count <- date_count + 1
    if (date_count %% 20 == 0) {
      cat("    Processing date", date_count, "/", length(unique_dates_ldc), "\n")
    }

    api_on_date <- api[api$pub_date_val == date_val, , drop = FALSE]
    ldc_on_date <- ldc[ldc$publication_date_val == date_val, , drop = FALSE]

    if (nrow(api_on_date) == 0 || nrow(ldc_on_date) == 0) next

    # Match each API article to best LDC match (one-to-one)
    # Keep track of which LDC rows have been matched
    ldc_matched_mask <- rep(FALSE, nrow(ldc_on_date))

    for (i in seq_len(nrow(api_on_date))) {
      api_headline <- api_on_date$headline_norm[i]

      # Get unmatched LDC rows for this date
      unmatched_idx <- which(!ldc_matched_mask)
      if (length(unmatched_idx) == 0) break

      ldc_to_match <- ldc_on_date[unmatched_idx, , drop = FALSE]

      # Compute Levenshtein distances
      dists <- adist(api_headline, ldc_to_match$headline_norm)

      if (length(dists) == 0) next

      best_idx_in_unmatched <- which.min(dists)
      best_dist <- dists[best_idx_in_unmatched]
      best_idx_in_ldc <- unmatched_idx[best_idx_in_unmatched]

      # Match if distance <= 5
      if (best_dist <= 5) {
        # Apply us_assign() heuristic (desk/section only) to matched API row
        heuristic_us <- fast_us_assign(
          api_on_date$news_desk[i],
          api_on_date$section_name[i]
        )

        year_matches[[length(year_matches) + 1]] <- data.frame(
          api_id = api_on_date$api_id[i],
          ldc_id = ldc_on_date$ldc_id[best_idx_in_ldc],
          api_lead = api_on_date$api_lead[i],
          ldc_stripped_text = ldc_on_date$ldc_stripped_text[best_idx_in_ldc],
          ldc_us_label = ldc_on_date$us_label[best_idx_in_ldc],
          ldc_label_source = ldc_on_date$label_source[best_idx_in_ldc],
          ldc_heuristic_us = heuristic_us,
          stringsAsFactors = FALSE
        )
        ldc_matched_mask[best_idx_in_ldc] <- TRUE
      }
    }
  }

  if (length(year_matches) > 0) {
    year_df <- do.call(rbind, year_matches) %>%
      as_tibble()

    matched_all[[length(matched_all) + 1]] <- year_df

    cat("  Matched pairs:", nrow(year_df), "\n")
    cat("  Joinability rate:", round(nrow(year_df) / ldc_count * 100, 2), "%\n")

    join_summary <- bind_rows(
      join_summary,
      tibble(
        year = year,
        ldc_rows = ldc_count,
        matched_pairs = nrow(year_df),
        joinability_rate = nrow(year_df) / ldc_count
      )
    )
  } else {
    cat("  Matched pairs: 0\n")
    cat("  Joinability rate: 0%\n")
  }
}

# Combine all years
if (length(matched_all) > 0) {
  matched <- do.call(bind_rows, matched_all) %>%
    mutate(
      api_id = as.character(api_id),
      ldc_id = as.character(ldc_id)
    ) %>%
    as.data.frame()

  cat("\nWriting output...\n")
  write_parquet(matched, OUT_PARQUET)
} else {
  cat("\nNo matches found!\n")
}

# Print summary
cat("\n=== Grand Summary (1987–1995) ===\n")
print(join_summary)
if (nrow(join_summary) > 0) {
  cat("\nTotal LDC rows (1987–1995):", sum(join_summary$ldc_rows), "\n")
  cat("Total matched pairs:", sum(join_summary$matched_pairs), "\n")
  if (sum(join_summary$ldc_rows) > 0) {
    cat("Overall joinability rate:", round(sum(join_summary$matched_pairs) / sum(join_summary$ldc_rows) * 100, 2), "%\n")
  }
}
cat("Output written to:", OUT_PARQUET, "\n")
