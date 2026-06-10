# pattern: Imperative Shell
# Apply us_assign() heuristic (desk/section + keywords) to pre-matched API/LDC pairs.
#
# NOTE: This script loads api_ldc_matched.parquet (created by the full fuzzy-join loop)
# and applies the real us_assign() heuristic from nyt_location_checking.R to all matched articles.
#
# us_assign heuristic from: /Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R

suppressMessages({
  library(arrow)
  library(dplyr)
  library(stringr)
  library(tidyverse)
})

# Load us_assign heuristic from nyt_location_checking.R
cat("Loading us_assign heuristic from nyt_location_checking.R (partial-eval)...\n")
nyt_loc_file <- "/Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R"
exprs <- parse(nyt_loc_file)
for (i in seq_along(exprs)) {
  tryCatch({
    eval(exprs[[i]], envir = globalenv())
    if (exists("filter_and_us_assign")) {
      cat(sprintf("  Loaded at expression %d/%d\n", i, length(exprs)))
      break
    }
  }, error = function(e) {
    # Suppress errors in the analysis section
  })
}

# Verify all needed components loaded
if (!exists("us_assign")) {
  stop("Failed to load us_assign function from nyt_location_checking.R")
}
if (!exists("loc_df")) {
  stop("Failed to load loc_df gazetteer from nyt_location_checking.R")
}
if (nrow(loc_df) == 0) {
  stop("loc_df gazetteer is empty")
}
cat("  Successfully loaded us_assign and loc_df (", nrow(loc_df), " location patterns)\n", sep = "")

# Paths
API_CORPUS_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/api_corpus/"
OUT_PARQUET <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter/audit/api_ldc_matched.parquet"

# Load already-matched parquet
cat("\nLoading pre-matched parquet...\n")
matched <- read_parquet(OUT_PARQUET) %>%
  mutate(ldc_heuristic_us = NA) %>%  # Reset heuristic column to recompute
  as_tibble()

cat("Loaded", nrow(matched), "matched pairs\n")

# Extract api_ids for batch lookup
api_ids_needed <- matched$api_id

# Load all API years and extract required rows
cat("\nLoading API corpus years and extracting matched articles...\n")
api_all_rows <- list()

for (year in 1987:1995) {
  api_path <- file.path(API_CORPUS_DIR, paste0(year, ".parquet"))
  if (!file.exists(api_path)) next

  api_df <- read_parquet(api_path)

  # Filter to articles in our matched set
  api_year <- api_df %>%
    mutate(api_id = as.character(id)) %>%
    filter(api_id %in% api_ids_needed) %>%
    select(api_id, news_desk, section_name, keywords) %>%
    as_tibble()

  if (nrow(api_year) > 0) {
    cat("  Year ", year, ": found ", nrow(api_year), " articles\n", sep = "")
    api_all_rows[[length(api_all_rows) + 1]] <- api_year
  }
}

# Combine all years' API data
if (length(api_all_rows) == 0) {
  stop("No API articles found for matched set")
}

api_full <- do.call(bind_rows, api_all_rows) %>%
  # Add row ID for correspondence
  mutate(temp_row_id = row_number()) %>%
  as_tibble()

cat("\nCombined API articles:", nrow(api_full), "\n")

# Check keywords structure and coerce if needed
cat("Checking keywords format...\n")
kw_sample <- api_full$keywords[[1]]
if (!is.data.frame(kw_sample)) {
  cat("  Coercing keywords to data.frame format\n")
  api_full <- api_full %>%
    mutate(keywords = map(keywords, function(kws) {
      if (is.data.frame(kws)) kws else as.data.frame(kws)
    }))
}

# Prepare input for us_assign
cat("Applying us_assign heuristic to all ", nrow(api_full), " articles...\n", sep = "")
start_time <- Sys.time()

api_input <- api_full %>%
  mutate(id = row_number()) %>%  # Fresh row numbers for us_assign
  select(id, news_desk, section_name, keywords) %>%
  as.data.frame()

# Apply us_assign
year_heuristic <- us_assign(api_input, return_long = FALSE)

# Extract the tri-state verdict from any_us and any_not_us
cat("Extracting verdicts...\n")
verdict_list <- list()
for (i in seq_len(nrow(year_heuristic))) {
  any_us <- year_heuristic$any_us[i]
  any_not_us <- year_heuristic$any_not_us[i]

  if (isTRUE(any_us) && !isTRUE(any_not_us)) {
    verdict_list[[i]] <- TRUE
  } else if (isTRUE(any_not_us) && !isTRUE(any_us)) {
    verdict_list[[i]] <- FALSE
  } else {
    verdict_list[[i]] <- NA
  }
}
heuristic_verdicts <- as.logical(unlist(verdict_list))

# Create verdict map
verdict_map <- data.frame(
  api_id = api_full$api_id,
  ldc_heuristic_us = heuristic_verdicts,
  stringsAsFactors = FALSE
)

elapsed <- difftime(Sys.time(), start_time, units = "secs")
cat(sprintf("Heuristic application completed in %.1f seconds\n", as.numeric(elapsed)))

# Join verdicts back to matched
cat("\nJoining verdicts to matched articles...\n")
matched_final <- matched %>%
  left_join(verdict_map, by = "api_id") %>%
  select(api_id, ldc_id, api_lead, ldc_stripped_text, ldc_us_label, ldc_label_source, ldc_heuristic_us) %>%
  as.data.frame()

# Write output
cat("Writing output...\n")
write_parquet(matched_final, OUT_PARQUET)

# Print summary
cat("\n=== Summary ===\n")
matched_final_check <- read_parquet(OUT_PARQUET)
cat("Final parquet:", nrow(matched_final_check), "rows\n")
cat("ldc_heuristic_us distribution:\n")
print(table(matched_final_check$ldc_heuristic_us, useNA="always"))
cat("Output written to:", OUT_PARQUET, "\n")
