# pattern: Imperative Shell
# Export DoCA-confirmed CCA positives to parquet for the CCA/DoCA retrain.
#
# Reads the DoCA->NYT fuzzy-match artifact (cca_matches_good.rds), keeps only
# successful matches, and writes the UNIQUE set of matched API article ids
# (`article_id` is in nyt://article/... form, identical to the API corpus `id`).
# One article can match multiple DoCA events, so we dedupe to unique ids and
# record the per-article event count. Consumed by the Python embedding extractor
# (--include-ids) and the Phase 1 label join.
#
# Run from the project root: Rscript r/doca/export_cca_positives.R

suppressMessages({
  library(arrow)
  library(dplyr)
})

IN_RDS <- "/Users/strophios/immigration_project/00_ML_data_expansion/LDC2008T19/data/cca_matches_good.rds"
OUT_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/cca_doca"
OUT_PARQUET <- file.path(OUT_DIR, "cca_doca_positives.parquet")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Reading", IN_RDS, "\n")
m <- readRDS(IN_RDS)
cat(sprintf("  total match rows: %d\n", nrow(m)))

# Keep only successful matches (drops ~4k 'failed' rows with NA match_dist).
good <- m %>% filter(match_quality == "succeeded")
cat(sprintf("  succeeded match rows: %d\n", nrow(good)))

# Dedupe to unique article ids; record per-article DoCA event count.
pos <- good %>%
  group_by(id = article_id) %>%
  summarise(n_doca_events = n(), .groups = "drop")
cat(sprintf("  unique positive article ids: %d\n", nrow(pos)))

# Sanity: ids should be in nyt://article/... form (the API corpus join key).
n_nyt <- sum(startsWith(pos$id, "nyt://article/"))
cat(sprintf("  ids in nyt://article/ form: %d / %d\n", n_nyt, nrow(pos)))
if (n_nyt < nrow(pos)) {
  cat("  NOTE: some ids are not API-form (likely LDC-only matches); these\n")
  cat("        will drop out naturally on the inner join to the API corpus.\n")
}

write_parquet(pos, OUT_PARQUET)
cat("Wrote", OUT_PARQUET, "\n")
