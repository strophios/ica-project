# pattern: Imperative Shell
#
# Semantics-preserving completion strategy for us_assign() application.
# Deduplicates unique location strings and applies the pure-function location pattern logic.
#
# Sourced from nyt_location_checking.R lines 247-344 (us_assign function).
# OPTIMIZATION: Load each RDS file once and cache all needed data (locations + metadata).

suppressMessages({
  library(arrow)
  library(dplyr)
  library(stringr)
  library(tidyverse)
})

# ---- Load vendored us_assign heuristic ----
cat("=== Step 0: Load location gazetteer and constants ===\n")
source("r/vendored/us_assign.R")
cat(sprintf("Loaded location gazetteer (%d patterns), constants loaded\n\n", nrow(loc_df)))

# ---- Paths ----
OUT_PARQUET <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter/audit/api_ldc_matched.parquet"
LDC_PARSED_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/LDC2008T19/data/parsed_to_rds/"

# ---- Load matched parquet ----
cat("=== Step 1: Load matched parquet ===\n")
matched <- read_parquet(OUT_PARQUET) %>%
  mutate(ldc_heuristic_us = NA) %>%
  as_tibble()
cat(sprintf("Loaded %d matched pairs\n", nrow(matched)))

# ---- Step 2-4: Load RDS once, extract everything, build deduplicated cache ----
cat("\n=== Step 2-4: Load RDS, extract locations & metadata, deduplicate ===\n")

ldc_ids_set <- sort(unique(as.integer(matched$ldc_id)))
cat(sprintf("Need keywords for %d unique LDC IDs\n", length(ldc_ids_set)))

# Single pass through RDS files: extract locations, metadata, and build location->verdict cache
location_strings <- c()
article_locs_list <- list()  # will be converted to tibble after loop
article_ids_list <- list()
ldc_meta_list <- list()

for (year in 1987:1995) {
  rds_path <- file.path(LDC_PARSED_DIR, paste0(year, ".rds"))
  if (!file.exists(rds_path)) next

  cat(sprintf("  Loading year %d RDS (%.0f MB)...\n", year, file.size(rds_path) / 1e6))
  ldc_year_full <- readRDS(rds_path)

  ids_in_year <- as.integer(ldc_year_full$id)
  rows_to_keep <- which(ids_in_year %in% ldc_ids_set)

  if (length(rows_to_keep) == 0) {
    cat(sprintf("    No matched articles in year %d\n", year))
    next
  }

  cat(sprintf("    Processing %d matched articles...\n", length(rows_to_keep)))

  # Vectorized extraction: extract all locations and metadata at once
  locs_per_article <- map(rows_to_keep, function(idx) {
    descs <- ldc_year_full$descriptors[[idx]]
    if (is.data.frame(descs)) {
      loc_rows <- descs[str_detect(descs$type, "^location$"), , drop = FALSE]
      if (nrow(loc_rows) > 0) {
        return(loc_rows$content)
      }
    }
    return(character(0))
  })

  # Extract all location strings
  for (locs in locs_per_article) {
    location_strings <- c(location_strings, locs)
  }

  # Add to lists (much faster than add_row per row)
  article_ids_list[[length(article_ids_list) + 1]] <- as.character(ids_in_year[rows_to_keep])
  article_locs_list[[length(article_locs_list) + 1]] <- locs_per_article

  # Extract metadata
  ldc_meta_year <- tibble(
    ldc_id = as.character(ids_in_year[rows_to_keep]),
    section_name = ldc_year_full$online_sections[rows_to_keep],
    news_desk = ldc_year_full$dsk[rows_to_keep]
  )
  ldc_meta_list[[length(ldc_meta_list) + 1]] <- ldc_meta_year

  cat(sprintf("    Extracted: %d locations, %d metadata entries\n",
              length(location_strings), length(rows_to_keep)))
}

# Convert lists to tibbles
ldc_meta_full <- bind_rows(ldc_meta_list) %>% distinct(ldc_id, .keep_all = TRUE)
article_ids_vec <- unlist(article_ids_list)
article_locs_vec <- unlist(article_locs_list, recursive = FALSE)
article_locations <- tibble(ldc_id = article_ids_vec, location_list = article_locs_vec)

ldc_meta_full <- ldc_meta_full %>% distinct(ldc_id, .keep_all = TRUE)
unique_locations <- sort(unique(location_strings))

cat(sprintf("\nTotal unique location strings: %d\n", length(unique_locations)))
cat(sprintf("Total articles with location data: %d\n", nrow(article_locations)))

if (length(unique_locations) > 200000) {
  stop(sprintf("ERROR: Unique location count %d exceeds threshold. Aborting.", length(unique_locations)))
}

# ---- Step 5: Create synthetic dataframe and run us_assign ----
cat("\n=== Step 5: Run us_assign on unique locations ===\n")

synthetic_df <- tibble(
  id = seq_along(unique_locations),
  news_desk = NA_character_,
  section_name = NA_character_,
  keywords = map(unique_locations, function(loc_str) {
    data.frame(type = "location", value = loc_str, stringsAsFactors = FALSE)
  })
)

cat(sprintf("Synthetic df: %d unique location rows\n", nrow(synthetic_df)))
cat("Applying us_assign to unique locations...")
start_time <- Sys.time()
result_df <- us_assign(synthetic_df, return_long = FALSE)
elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat(sprintf(" (%.1f seconds)\n", as.numeric(elapsed)))

location_verdicts <- tibble(
  location = unique_locations,
  loc_is_us = result_df$is_us,
  loc_not_us = result_df$not_us
)

cat("Verdict distribution (unique locations):\n")
print(location_verdicts %>%
      mutate(verdict = case_when(
        loc_is_us & !loc_not_us ~ "US",
        loc_not_us & !loc_is_us ~ "non-US",
        TRUE ~ "ambiguous"
      )) %>%
      count(verdict))

# ---- Step 6: Aggregate verdicts per article ----
cat("\n=== Step 6: Aggregate location verdicts to articles ===\n")

article_loc_verdicts <- article_locations %>%
  mutate(
    kw_is_us = map_lgl(location_list, function(locs) {
      if (length(locs) == 0) return(FALSE)
      any(location_verdicts[location_verdicts$location %in% locs, ]$loc_is_us, na.rm = TRUE)
    }),
    kw_not_us = map_lgl(location_list, function(locs) {
      if (length(locs) == 0) return(FALSE)
      any(location_verdicts[location_verdicts$location %in% locs, ]$loc_not_us, na.rm = TRUE)
    })
  ) %>%
  select(ldc_id, kw_is_us, kw_not_us)

cat(sprintf("Aggregated: %d articles\n", nrow(article_loc_verdicts)))

# ---- Step 7: Combine with metadata and create final verdict ----
cat("\n=== Step 7: Combine metadata and create tri-state verdict ===\n")

matched_with_verdicts <- matched %>%
  left_join(ldc_meta_full, by = "ldc_id") %>%
  left_join(article_loc_verdicts, by = "ldc_id") %>%
  mutate(
    # Meta flags from section/desk
    meta_is_us = str_to_lower(section_name) %in% section_us |
                 str_to_lower(news_desk) %in% dsk_us,
    meta_not_us = str_to_lower(section_name) %in% section_non_us |
                  str_to_lower(news_desk) %in% dsk_non_us
  )

# Handle multi-section names
rns <- which(str_detect(matched_with_verdicts[["section_name"]], "; "))
if (length(rns) > 0) {
  cat(sprintf("Processing %d multi-section articles\n", length(rns)))
  sec_rns_us <- map(rns, function(rn) {
    sec <- str_to_lower(matched_with_verdicts[["section_name"]][rn])
    sec <- str_split(sec, "; ")[[1]]
    sec_us <- any(sec %in% section_us)
    sec_not <- any(sec %in% section_non_us)
    tibble(meta_is_us = sec_us, meta_not_us = sec_not)
  }) %>% bind_rows()

  matched_with_verdicts[["meta_is_us"]][rns] <- sec_rns_us[["meta_is_us"]] |
    matched_with_verdicts[["meta_is_us"]][rns]
  matched_with_verdicts[["meta_not_us"]][rns] <- sec_rns_us[["meta_not_us"]] |
    matched_with_verdicts[["meta_not_us"]][rns]
}

# Create tri-state verdict
matched_with_verdicts <- matched_with_verdicts %>%
  mutate(
    any_us = kw_is_us | meta_is_us,
    any_not_us = kw_not_us | meta_not_us,
    ldc_heuristic_us = case_when(
      any_us & !any_not_us ~ TRUE,
      any_not_us & !any_us ~ FALSE,
      TRUE ~ NA
    )
  )

cat("\nTri-state verdict distribution:\n")
print(matched_with_verdicts %>% count(ldc_heuristic_us))

# ---- Step 8: Write output ====
cat("\n=== Step 8: Write output ===\n")

final_output <- matched_with_verdicts %>%
  select(api_id, ldc_id, api_lead, ldc_stripped_text, ldc_us_label, ldc_label_source, ldc_heuristic_us) %>%
  as.data.frame()

cat(sprintf("Final output: %d rows\n", nrow(final_output)))
write_parquet(final_output, OUT_PARQUET)

# Verify
cat("Verifying written parquet...\n")
verified <- read_parquet(OUT_PARQUET)
cat(sprintf("Verified: %d rows, %d columns\n", nrow(verified), ncol(verified)))

# ---- Final Report ----
cat("\n=== FINAL REPORT ===\n")
cat(sprintf("Unique location strings deduced: %d\n", length(unique_locations)))
cat(sprintf("us_assign time: %.1f seconds (3,155 patterns × %d strings)\n", as.numeric(elapsed), length(unique_locations)))
cat(sprintf("Matched pairs in output: %d (unchanged)\n", nrow(final_output)))
cat("\nTri-state verdict distribution:\n")
print(verified %>% as_tibble() %>% count(ldc_heuristic_us))

cat("\nComparison with dateline heuristic:\n")
comparison <- verified %>%
  as_tibble() %>%
  mutate(
    ldc_us_label = as.logical(ldc_us_label),
    disagreement = !is.na(ldc_heuristic_us) & !is.na(ldc_us_label) & (ldc_heuristic_us != ldc_us_label)
  ) %>%
  summarise(
    total_pairs = n(),
    both_non_null = sum(!is.na(ldc_heuristic_us) & !is.na(ldc_us_label)),
    na_heuristic = sum(is.na(ldc_heuristic_us)),
    na_dateline = sum(is.na(ldc_us_label)),
    disagreements = sum(disagreement, na.rm = TRUE),
    disagreement_rate = round(sum(disagreement, na.rm = TRUE) / sum(!is.na(ldc_heuristic_us) & !is.na(ldc_us_label)), 4)
  )
print(comparison)

cat("\n=== COMPLETION ===\n")
cat("Output written to:", OUT_PARQUET, "\n")
