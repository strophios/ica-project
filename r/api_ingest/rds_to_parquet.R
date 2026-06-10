# pattern: Imperative Shell
# Convert per-year NYT Archive API .rds files to parquet, preserving the nested
# `keywords` list-column (arrow is the only bridge that does this correctly).

suppressMessages({ library(arrow) })

SRC_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/nyt_archive_by_year"
OUT_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/api_corpus"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

rds_files <- list.files(SRC_DIR, pattern = "\\.rds$", full.names = TRUE)
stopifnot(length(rds_files) == 36)

total_src <- 0L; total_out <- 0L
for (f in sort(rds_files)) {
  year <- sub("\\.rds$", "", basename(f))
  d <- readRDS(f)
  out_path <- file.path(OUT_DIR, paste0(year, ".parquet"))
  write_parquet(d, out_path)
  back <- open_dataset(out_path)$num_rows
  cat(sprintf("%s: src=%d written=%d %s\n", year, nrow(d), back,
              ifelse(nrow(d) == back, "OK", "MISMATCH")))
  stopifnot(nrow(d) == back)
  total_src <- total_src + nrow(d); total_out <- total_out + back
}
cat(sprintf("TOTAL src=%d written=%d\n", total_src, total_out))
stopifnot(total_src == total_out)
