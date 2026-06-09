# pattern: Imperative Shell
# Read the full LDC partitioned dataset (1987-2007), apply the dateline resolver +
# desk fusion row-wise, and write a derived labeled parquet for the US filter.

suppressMessages({ library(arrow) })
source("r/dateline/resolve_dateline.R")
source("r/dateline/label_policy.R")

LDC_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/ldc_corpus/"
OUT_DIR <- "/Users/strophios/immigration_project/00_ML_data_expansion/00_explorer/us_filter"
OUT_PARQUET <- file.path(OUT_DIR, "ldc_labeled.parquet")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

gz <- load_gazetteers("r/dateline/gazetteers")

df <- as.data.frame(open_dataset(LDC_DIR)) [, c("id", "headline", "lead_paragraph", "dsk", "print_section")]
n <- nrow(df)

us_label <- logical(n); label_source <- character(n)
dateline_place <- character(n); stripped_text <- character(n); raw_text <- df$lead_paragraph

for (i in seq_len(n)) {
  rd   <- resolve_dateline(df$lead_paragraph[i], gz)
  desk <- desk_section_signal(df$dsk[i], df$print_section[i])
  lab  <- classify_label(rd$is_us, desk)
  us_label[i]       <- lab$us_label
  label_source[i]   <- lab$label_source
  dateline_place[i] <- rd$place
  stripped_text[i]  <- strip_dateline(df$lead_paragraph[i], rd$block_info)
}

out <- data.frame(
  id = df$id, headline = df$headline, us_label = us_label, label_source = label_source,
  dateline_place = dateline_place, stripped_text = stripped_text,
  raw_text = raw_text, stringsAsFactors = FALSE
)
write_parquet(out, OUT_PARQUET)

# Operational check: label_source breakdown + extraction coverage.
cat("rows:", n, "\n")
print(table(label_source, useNA = "ifany"))
cat("dateline-extracted coverage:", round(mean(!is.na(dateline_place)) * 100, 1), "%\n")
