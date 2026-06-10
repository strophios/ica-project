# pattern: Imperative Shell
# Read the full LDC partitioned dataset (1987-2007), join with per-year RDS dateline fields,
# apply structured + text dateline resolution, desk fusion, and write derived labeled parquet.

suppressMessages({ library(arrow) })
source("r/dateline/resolve_dateline.R")
source("r/dateline/label_policy.R")

LDC_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/ldc_corpus/"
RDS_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/LDC2008T19/data/parsed_to_rds/"
OUT_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter"
OUT_PARQUET <- file.path(OUT_DIR, "ldc_labeled.parquet")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

gz <- load_gazetteers("r/dateline/gazetteers")

# Load LDC parquet
cat("Loading LDC parquet...\n")
df <- as.data.frame(open_dataset(LDC_DIR))[, c("id", "headline", "lead_paragraph", "dsk", "print_section")]
n <- nrow(df)

# Load and rbind per-year RDS dateline columns
cat("Loading per-year RDS dateline fields...\n")
rds_years <- seq(1987, 2007)
rds_list <- lapply(rds_years, function(year) {
  rds_path <- file.path(RDS_DIR, paste0(year, ".rds"))
  if (!file.exists(rds_path)) {
    cat("  Warning: missing", rds_path, "\n")
    return(data.frame(id = integer(0), dateline = character(0)))
  }
  d <- readRDS(rds_path)
  data.frame(id = d$id, dateline = d$dateline, stringsAsFactors = FALSE)
})
rds_combined <- do.call(rbind, rds_list)

# Remove duplicates (keep first) and report
rds_dedup <- rds_combined[!duplicated(rds_combined$id), ]
dup_count <- nrow(rds_combined) - nrow(rds_dedup)
if (dup_count > 0) cat("  Deduplicated", dup_count, "duplicate IDs from RDS\n")

# Left-join RDS dateline onto LDC by id
cat("Joining RDS dateline onto LDC...\n")
df <- merge(df, rds_dedup, by = "id", all.x = TRUE)
rds_matched <- sum(!is.na(df$dateline))
cat("  Rows with non-NA RDS dateline:", rds_matched, "/", nrow(df), "\n")

# Initialize output columns
us_label <- logical(n); label_source <- character(n)
dateline_place <- character(n); stripped_text <- character(n); raw_text <- df$lead_paragraph

# Row-wise resolution: use RDS channel (primary) or text channel (fallback)
# NOTE: This is an intentional one-shot batch script (full corpus in memory, 1.16M-row loop).
# It is not a hot path and does not require optimization. The row-wise loop is kept for
# clarity and simplicity over the full dataset load in a single session.
cat("Resolving datelines row-wise...\n")
for (i in seq_len(n)) {
  # Try structured RDS channel first
  if (!is.na(df$dateline[i])) {
    rds_res <- resolve_dateline_field(df$dateline[i], gz)
    dateline_place[i] <- rds_res$place
    text_res <- resolve_dateline(df$lead_paragraph[i], gz)  # for stripping only
    stripped_text[i] <- if (isTRUE(text_res$should_strip)) strip_dateline(df$lead_paragraph[i], text_res$block_info) else df$lead_paragraph[i]
    dateline_signal <- rds_res$is_us
  } else {
    # Fallback to text channel for both signal and stripping
    text_res <- resolve_dateline(df$lead_paragraph[i], gz)
    dateline_place[i] <- text_res$place
    stripped_text[i] <- if (isTRUE(text_res$should_strip)) strip_dateline(df$lead_paragraph[i], text_res$block_info) else df$lead_paragraph[i]
    dateline_signal <- text_res$is_us
  }

  # Desk/section backfill
  desk <- desk_section_signal(df$dsk[i], df$print_section[i])

  # Fuse dateline + desk
  lab <- classify_label(dateline_signal, desk)
  us_label[i] <- lab$us_label
  label_source[i] <- lab$label_source
}

# Construct output dataframe
out <- data.frame(
  id = df$id, headline = df$headline, us_label = us_label, label_source = label_source,
  dateline_place = dateline_place, stripped_text = stripped_text,
  raw_text = raw_text, stringsAsFactors = FALSE
)

# Write parquet
cat("Writing output parquet...\n")
write_parquet(out, OUT_PARQUET)

# Operational checks
cat("\n=== Build Summary ===\n")
cat("Total rows:", nrow(out), "\n")
cat("\nLabel source breakdown:\n")
print(table(label_source, useNA = "ifany"))
cat("\nDateline-extracted coverage:", round(mean(!is.na(dateline_place)) * 100, 1), "%\n")

# Spot check: most rows should have stripped_text == raw_text except for embedded tails
differ_count <- sum(out$stripped_text != out$raw_text, na.rm = TRUE)
cat("Rows where stripped_text != raw_text:", differ_count, "(expect <1% due to embedded tails)\n")

if (differ_count > 0) {
  cat("\nSample rows with different stripped_text:\n")
  diff_idx <- which(out$stripped_text != out$raw_text)[1:min(5, differ_count)]
  for (idx in diff_idx) {
    cat("  id=", out$id[idx], " source=", out$label_source[idx], "\n")
    cat("    raw[:60]:", substr(out$raw_text[idx], 1, 60), "\n")
    cat("    str[:60]:", substr(out$stripped_text[idx], 1, 60), "\n")
  }
}
