# pattern: Imperative Shell
# Export the hand-coded ICA events as immigrant-relevance ANCHOR articles.
#
# These 681 events are the authors' hand-coded "Immigrant Collective Action"
# subset of DoCA -- events of "special and specific relevance to immigrants."
# Joined to NYT articles through the same DoCA->NYT match artifact that feeds
# the CCA positives (cca_matches_good.rds), they become clean, hand-verified
# POSITIVES for the immigrant-relevance head. They are protest-confined (DoCA
# only contains protest events), so they anchor the relevance signal in the
# protest subspace; the descriptor-selected positives supply non-protest breadth.
#
# Output columns (one row per article x event_type4, multi-label by design):
#   article_id      nyt://article/... (the API corpus join key)
#   eventid         DoCA event id
#   event_type4     Documentation | Access | Diasporic | Exclusionary
#   immig_recode    graded immigrant-involvement code (0/1/2/3/4/5/6/9/NA)
#   immigrant_involved  bool carried from the match file (recode 1-5)
#
# Run from the project root: Rscript r/doca/export_ica_anchors.R

suppressMessages({
  library(arrow)
  library(dplyr)
})

ROOT <- "/Users/strophios/immigration_project/00_ML_data_expansion"
IN_RDS <- file.path(ROOT, "LDC2008T19/data/cca_matches_good.rds")
IN_ICA <- file.path(ROOT, "00_explorer/ica_events.csv")
OUT_DIR <- file.path(ROOT, "00_explorer/relevance")
OUT_PARQUET <- file.path(OUT_DIR, "ica_anchors.parquet")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("Reading", IN_RDS, "and", IN_ICA, "\n")
m <- readRDS(IN_RDS)
ica <- read.csv(IN_ICA)
cat(sprintf("  ICA events: %d (unique eventid %d)\n", nrow(ica), length(unique(ica$eventid))))

# Successful matches, in API-joinable nyt:// form only (LDC-only ids drop out).
good <- m %>%
  filter(match_quality == "succeeded", startsWith(article_id, "nyt://article/"))

anchors <- good %>%
  filter(eventid %in% ica$eventid) %>%
  select(eventid, article_id, immigrant_involved) %>%
  inner_join(ica %>% select(eventid, event_type4, immig_recode), by = "eventid") %>%
  distinct()

cat(sprintf("  anchor rows (article x event): %d\n", nrow(anchors)))
cat(sprintf("  unique anchor article_id: %d\n", length(unique(anchors$article_id)))) # ~466
cat("  by event_type4 (article-level):\n")
print(table(distinct(anchors, article_id, event_type4)$event_type4))

write_parquet(anchors, OUT_PARQUET)
cat("Wrote", OUT_PARQUET, "\n")
