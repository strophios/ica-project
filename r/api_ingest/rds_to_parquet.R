# pattern: Imperative Shell
# Convert per-year NYT Archive API .rds files to parquet, preserving the nested
# `keywords` list-column (arrow is the only bridge that does this correctly).
# Incremental: converts only years whose parquet is missing; --force redoes
# all, --years YYYY:YYYY restricts the sweep. (The original one-shot version
# hard-asserted exactly 36 files, 1960-1995; the pull now grows the set.)

suppressMessages({
  library(arrow)
})
source("r/api_ingest/archive_transform.R")

SRC_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/nyt_archive_by_year"
OUT_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/api_corpus"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

args <- commandArgs(trailingOnly = TRUE)
force <- "--force" %in% args
years_filter <- NULL
yi <- which(args == "--years")
if (length(yi) == 1) {
  if (yi == length(args)) stop("missing value for --years")
  years_filter <- parse_years(args[yi + 1])
}

rds_files <- list.files(SRC_DIR, pattern = "^[0-9]{4}\\.rds$", full.names = TRUE)
if (length(rds_files) == 0) stop("no per-year rds files found in ", SRC_DIR)
if (!is.null(years_filter)) {
  keep <- sub("\\.rds$", "", basename(rds_files)) %in% as.character(years_filter)
  rds_files <- rds_files[keep]
}

converted <- 0L
skipped <- 0L
total_src <- 0L
total_out <- 0L
for (f in sort(rds_files)) {
  year <- sub("\\.rds$", "", basename(f))
  out_path <- file.path(OUT_DIR, paste0(year, ".parquet"))
  if (file.exists(out_path) && !force) {
    skipped <- skipped + 1L
    next
  }
  d <- readRDS(f)
  write_parquet(d, out_path)
  back <- open_dataset(out_path)$num_rows
  cat(sprintf(
    "%s: src=%d written=%d %s\n", year, nrow(d), back,
    ifelse(nrow(d) == back, "OK", "MISMATCH")
  ))
  stopifnot(nrow(d) == back)
  total_src <- total_src + nrow(d)
  total_out <- total_out + back
  converted <- converted + 1L
}
stopifnot(total_src == total_out)
cat(sprintf(
  "converted=%d skipped(existing)=%d rows=%d\n", converted, skipped, total_out
))
