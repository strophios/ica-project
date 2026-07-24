# pattern: Functional Core
# Pure transforms for the NYT Archive API pull: story lists -> the canonical
# per-year tibble schema (matching the existing 1960-1995 nyt_archive_by_year
# files, including the historical `abstrct` column name), plus fetch planning
# and CLI-value parsing. No I/O here; the shell is pull_archive.R.

# The canonical column order/names of a per-year archive tibble. `abstrct`
# (sic) is the historical name in all 36 existing files -- keep it so the
# downstream parquet corpus stays schema-uniform.
ARCHIVE_COLS <- c(
  "year", "month", "headline", "abstrct", "lead_paragraph", "web_url",
  "keywords", "pub_date", "document_type", "news_desk", "section_name",
  "uri", "id", "word_count"
)

ARCHIVE_MIN_YEAR <- 1851L # NYT founding; the Archive API starts here
ARCHIVE_MAX_YEAR <- 2100L

chr_or_na <- function(x) {
  if (is.null(x) || length(x) == 0) {
    return(NA_character_)
  }
  as.character(x[[1]])
}

int_or_na <- function(x) {
  if (is.null(x) || length(x) == 0) {
    return(NA_integer_)
  }
  suppressWarnings(as.integer(x[[1]]))
}

date_or_na <- function(x) {
  raw <- chr_or_na(x)
  if (is.na(raw)) {
    return(as.Date(NA))
  }
  as.Date(sub("T.*", "", raw))
}

# API keyword entries carry (name, value, rank, major); the canonical schema
# renames name -> type. Accepts a list of entries or an already-tabular value;
# empty -> NULL (matching the historical convention in the existing files).
normalize_keywords <- function(kws) {
  if (is.null(kws) || length(kws) == 0) {
    return(NULL)
  }
  out <- if (is.data.frame(kws)) {
    tibble::as_tibble(kws)
  } else {
    dplyr::bind_rows(lapply(kws, tibble::as_tibble))
  }
  if ("name" %in% names(out) && !("type" %in% names(out))) {
    out <- dplyr::rename(out, type = "name")
  }
  for (col in c("type", "value", "major")) {
    if (!(col %in% names(out))) out[[col]] <- NA_character_
  }
  if (!("rank" %in% names(out))) out[["rank"]] <- NA_integer_
  out[, c("type", "value", "rank", "major")]
}

# One API story (a nested list) -> a one-row tibble (without year/month).
# Missing fields become NA rather than erroring: old decades have sparse
# metadata and one gappy story must not kill a 1,300-request pull.
story_to_row <- function(story) {
  tibble::tibble(
    headline = chr_or_na(story[["headline"]][["main"]]),
    abstrct = chr_or_na(story[["abstract"]]),
    lead_paragraph = chr_or_na(story[["lead_paragraph"]]),
    web_url = chr_or_na(story[["web_url"]]),
    keywords = list(normalize_keywords(story[["keywords"]])),
    pub_date = date_or_na(story[["pub_date"]]),
    document_type = chr_or_na(story[["document_type"]]),
    news_desk = chr_or_na(story[["news_desk"]]),
    section_name = chr_or_na(story[["section_name"]]),
    uri = chr_or_na(story[["uri"]]),
    id = chr_or_na(story[["_id"]]),
    word_count = int_or_na(story[["word_count"]])
  )
}

empty_archive_tibble <- function() {
  tibble::tibble(
    year = character(), month = character(), headline = character(),
    abstrct = character(), lead_paragraph = character(),
    web_url = character(), keywords = list(),
    pub_date = as.Date(character()), document_type = character(),
    news_desk = character(), section_name = character(), uri = character(),
    id = character(), word_count = integer()
  )
}

# A month's worth of stories -> canonical tibble. year/month are stored as
# character ("1987", "1".."12", unpadded) to match the existing files.
month_tibble <- function(stories, year, month) {
  rows <- dplyr::bind_rows(lapply(stories, story_to_row))
  if (nrow(rows) == 0) {
    return(empty_archive_tibble())
  }
  rows[["year"]] <- as.character(year)
  rows[["month"]] <- as.character(as.integer(month))
  rows[, ARCHIVE_COLS]
}

# Business-layer schema gate: refuse to hand a malformed tibble to saveRDS.
validate_archive_tibble <- function(d) {
  if (!identical(names(d), ARCHIVE_COLS)) {
    stop(
      "archive tibble schema mismatch: got [",
      paste(names(d), collapse = ", "), "]"
    )
  }
  if (!inherits(d[["pub_date"]], "Date")) stop("pub_date must be Date")
  if (!is.list(d[["keywords"]])) stop("keywords must be a list-column")
  if (!is.integer(d[["word_count"]])) stop("word_count must be integer")
  invisible(d)
}

# "YYYY" or "YYYY:YYYY" -> integer vector of years.
parse_years <- function(v) {
  m <- regmatches(v, regexec("^([0-9]{4})(:([0-9]{4}))?$", v))[[1]]
  if (length(m) == 0) {
    stop("invalid --years value: '", v, "' (expected YYYY or YYYY:YYYY)")
  }
  from <- as.integer(m[2])
  to <- if (nzchar(m[4])) as.integer(m[4]) else from
  if (to < from) stop("--years range reversed: ", v)
  from:to
}

# "YYYY-MM" -> list(year=, month=). Caps the plan (e.g. exclude the current,
# still-incomplete month).
parse_through <- function(through) {
  m <- regmatches(through, regexec("^([0-9]{4})-([0-9]{1,2})$", through))[[1]]
  if (length(m) == 0) {
    stop("invalid --through value: '", through, "' (expected YYYY-MM)")
  }
  out <- list(year = as.integer(m[2]), month = as.integer(m[3]))
  if (out$month < 1 || out$month > 12) {
    stop("invalid --through month: ", out$month)
  }
  out
}

# The fetch plan: one row per (year, month) to request, in chronological
# order. skeleton = one month per year, rotating deterministically through
# the calendar ((year - 1851) %% 12 + 1) so a sampled backfill spreads over
# seasons instead of always hitting January.
month_plan <- function(years, skeleton = FALSE, through = NULL) {
  years <- sort(unique(as.integer(years)))
  if (any(is.na(years)) ||
        any(years < ARCHIVE_MIN_YEAR) || any(years > ARCHIVE_MAX_YEAR)) {
    stop(
      "years out of range [", ARCHIVE_MIN_YEAR, ", ", ARCHIVE_MAX_YEAR, "]"
    )
  }
  plan <- expand.grid(month = 1:12, year = years)[, c("year", "month")]
  plan <- plan[order(plan$year, plan$month), ]
  if (skeleton) {
    plan <- plan[plan$month == ((plan$year - ARCHIVE_MIN_YEAR) %% 12) + 1, ]
  }
  if (!is.null(through)) {
    tp <- parse_through(through)
    keep <- plan$year < tp$year |
      (plan$year == tp$year & plan$month <= tp$month)
    plan <- plan[keep, ]
  }
  tibble::as_tibble(plan)
}

raw_checkpoint_path <- function(raw_dir, year, month) {
  file.path(raw_dir, sprintf("%d_%02d.rds", as.integer(year), as.integer(month)))
}
