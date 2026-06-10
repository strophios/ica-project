# pattern: Imperative Shell
# Join API corpus (1987–1995) to LDC labeled parquet on normalized headline + pub_date.
# Uses Levenshtein distance (base R adist) with cutoff=5.
# Provenance: fuzzy matching adapted from
# /Users/strophios/immigration_project/00_ML_data_expansion/LDC2008T19/data/scripts/00_proc_and_matching_prep.R:456-491

suppressMessages({
  library(arrow)
  library(dplyr)
  library(stringr)
})

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

# Load data
cat("Loading LDC labeled...\n")
ldc_labeled <- read_parquet(LDC_LABELED)

cat("Loading LDC corpus...\n")
ldc_corpus <- open_dataset(LDC_CORPUS_DIR) %>%
  select(id, publication_date, dsk, online_sections) %>%
  collect()

cat("Merging LDC...\n")
ldc_full <- ldc_labeled %>%
  left_join(ldc_corpus, by = "id", relationship = "one-to-one")

cat("Loading API (1987-1995)...\n")
api_combined <- NULL
total_api_rows <- 0
for (year in 1987:1995) {
  path <- file.path(API_CORPUS_DIR, paste0(year, ".parquet"))
  if (file.exists(path)) {
    df <- read_parquet(path)
    total_api_rows <- total_api_rows + nrow(df)
    api_combined <- if (is.null(api_combined)) df else bind_rows(api_combined, df)
    cat("  ", year, ": ", nrow(df), "\n", sep = "")
  }
}

# Prepare data
cat("Normalizing...\n")
api <- api_combined %>%
  mutate(
    api_id = as.character(id),
    api_lead = lead_paragraph,
    headline_norm = normalize_headline(headline),
    pub_date = as.Date(pub_date)
  ) %>%
  select(api_id, api_lead, pub_date, headline_norm) %>%
  as.data.frame()

ldc <- ldc_full %>%
  mutate(
    ldc_id = as.character(id),
    ldc_stripped_text = stripped_text,
    headline_norm = normalize_headline(headline),
    publication_date = as.Date(publication_date)
  ) %>%
  select(ldc_id, ldc_stripped_text, publication_date, headline_norm, us_label, dsk, online_sections) %>%
  as.data.frame()

cat("Matching by date and headline (adist, cutoff=5)...\n")

# Group by date for efficiency
ldc_by_date <- split(ldc, ldc$publication_date)
matched_list <- list()
match_count <- 0

# For each date present in API
api_dates <- unique(api$pub_date)
for (date in api_dates) {
  api_on_date <- api[api$pub_date == date, , drop = FALSE]
  ldc_on_date <- ldc_by_date[[as.character(date)]]

  if (is.null(ldc_on_date) || nrow(ldc_on_date) == 0) next

  # Match each API article on this date to best LDC match
  for (i in seq_len(nrow(api_on_date))) {
    api_headline <- api_on_date$headline_norm[i]

    # Compute distances to all LDC articles on this date
    dists <- adist(api_headline, ldc_on_date$headline_norm)

    if (length(dists) == 0) next
    best_idx <- which.min(dists)
    best_dist <- dists[best_idx]

    # Match if distance <= 5
    if (best_dist <= 5) {
      matched_list[[length(matched_list) + 1]] <- data.frame(
        api_id = api_on_date$api_id[i],
        ldc_id = ldc_on_date$ldc_id[best_idx],
        api_lead = api_on_date$api_lead[i],
        ldc_stripped_text = ldc_on_date$ldc_stripped_text[best_idx],
        ldc_us_label = ldc_on_date$us_label[best_idx],
        ldc_dsk = ldc_on_date$dsk[best_idx],
        ldc_online_sections = ldc_on_date$online_sections[best_idx],
        stringsAsFactors = FALSE
      )
      match_count <- match_count + 1
    }
  }

  if (match(date, api_dates) %% 50 == 0) {
    cat("  ", match(date, api_dates), "/", length(api_dates), " dates processed\n", sep = "")
  }
}

matched <- if (length(matched_list) > 0) do.call(rbind, matched_list) else data.frame()

cat("Writing output...\n")
if (nrow(matched) > 0) {
  # Ensure IDs are character
  matched$api_id <- as.character(matched$api_id)
  matched$ldc_id <- as.character(matched$ldc_id)
  write_parquet(matched, OUT_PARQUET)
}

cat("\n=== Join Summary ===\n")
cat("API rows (1987–1995):", nrow(api), "\n")
cat("LDC rows:", nrow(ldc), "\n")
cat("Matched pairs:", nrow(matched), "\n")
if (nrow(api) > 0) {
  cat("Joinability rate:", round(nrow(matched) / nrow(api) * 100, 2), "%\n")
}
cat("Output written to:", OUT_PARQUET, "\n")
