# pattern: Functional Core
# Dateline extraction + structure-first US/not-US place resolution.
# Pure functions; load_gazetteers is the single thin I/O helper.

# Normalize a place/qualifier token to a comparison key: lowercase, alpha-only.
# "Wash." -> "wash"; "N.Y." -> "ny"; "Los Angeles" -> "losangeles".
normalize_token <- function(x) {
  if (is.na(x)) return(NA_character_)
  gsub("[^a-z]", "", tolower(x))
}

# Load normalized gazetteer sets from a directory of CSVs.
# states: from state full names + AP abbreviations ONLY (NOT 2-letter USPS codes,
#   which collide with English words OR/IN/OK) -- satisfies the AC1.10/guard intent.
load_gazetteers <- function(dir) {
  states_raw   <- utils::read.csv(file.path(dir, "state_long_abbrs.csv"),
                                  stringsAsFactors = FALSE)
  countries    <- utils::read.csv(file.path(dir, "countries.csv"),
                                  stringsAsFactors = FALSE)
  us_cities    <- utils::read.csv(file.path(dir, "ap_us_cities.csv"),
                                  stringsAsFactors = FALSE)
  foreign      <- utils::read.csv(file.path(dir, "ap_foreign_cities.csv"),
                                  stringsAsFactors = FALSE)
  norm_set <- function(v) unique(stats::na.omit(vapply(v, normalize_token, character(1))))
  list(
    states         = norm_set(c(states_raw$full, states_raw$long_abbr)),
    countries      = norm_set(countries$value),
    us_cities      = norm_set(us_cities$city),
    foreign_cities = norm_set(foreign$city)
  )
}

# Does a field look like a date field? Matches "July 30", "Jul. 30", "June 1", "Jan 5".
.DATE_RE <- "^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\\.?\\s*[0-9]{0,2}$"
is_date_field <- function(field) {
  grepl(.DATE_RE, trimws(tolower(field)))
}

# Isolate the leading ALL-CAPS place block before the dateline delimiter.
# Handles a leading "Special to The New York Times" credit and trailing "(AP)" wire tag.
# Returns list(found, block, match_len) where match_len is the number of leading
# characters of `lead` consumed by block + delimiter (for exact stripping).
extract_dateline_block <- function(lead) {
  empty <- list(found = FALSE, block = NA_character_, match_len = 0L)
  if (is.na(lead) || !nzchar(lead)) return(empty)

  work <- lead
  offset <- 0L
  # Strip a leading credit line if present (case-insensitive), recording consumed length.
  credit_re <- "^\\s*Special to The New York Times\\s*"
  m <- regexpr(credit_re, work, ignore.case = TRUE)
  if (m == 1L) {
    consumed <- attr(m, "match.length")
    offset <- offset + consumed
    work <- substr(work, consumed + 1L, nchar(work))
  }

  # Dateline = block starting with CAPS, containing AT LEAST ONE COMMA (structure: CITY, STATE/COUNTRY),
  # then optional date field(s), then a delimiter: em dash, "--", or spaced hyphen.
  # Pattern: starts with optional space, opening CAPS word(s), comma, then qualifier/date, then dash.
  # Real LDC data uses spaced hyphen " - " (validated against real corpus).
  # Examples: " MEXICO CITY, Jan. 1 - ", " LONDON, Jan. 7 - ", "VANCOUVER, Wash. — "
  # Key constraints:
  #   - Starts with capital letter (rules out most body text)
  #   - Contains exactly one comma in the captured block
  #   - After comma: short content (<50 chars) — likely a qualifier or date
  #   - Total block length: <80 chars (real datelines are 20-40 chars)
  # Note: this accepts some false positives (titles with commas) but false negatives on real datelines is worse.
  dl_re <- "^\\s*([A-Z][^,]*?[,][^,]{0,50})\\s*(-|—)\\s"
  m2 <- regexpr(dl_re, work, perl = TRUE)
  if (m2 != 1L) return(empty)

  full_match <- regmatches(work, m2)
  block <- sub(dl_re, "\\1", full_match, perl = TRUE)
  # Remove a trailing wire tag like "(AP)" from the block.
  block <- trimws(sub("\\(AP\\)\\s*$", "", block))
  match_len <- offset + attr(m2, "match.length")
  list(found = TRUE, block = block, match_len = as.integer(match_len))
}

# Split a caps block into non-date fields (city + optional qualifier).
parse_dateline_fields <- function(block) {
  if (is.na(block) || !nzchar(block)) return(character(0))
  fields <- trimws(strsplit(block, ",", fixed = TRUE)[[1]])
  fields <- fields[nzchar(fields)]
  fields[!vapply(fields, is_date_field, logical(1))]
}

# Resolve fields -> list(is_us = TRUE|FALSE|NA, place).
# Structure-first, ordered. Bare tokens consult ONLY AP-30/AP-46 (never countries/area-codes).
resolve_place <- function(fields, gz) {
  if (length(fields) == 0) return(list(is_us = NA, place = NA_character_))
  city <- fields[1]
  if (length(fields) >= 2) {
    qualifier <- fields[length(fields)]
    qn <- normalize_token(qualifier)
    if (!is.na(qn) && qn %in% gz$states)    return(list(is_us = TRUE,  place = paste(city, qualifier, sep = ", ")))
    if (!is.na(qn) && qn %in% gz$countries) return(list(is_us = FALSE, place = paste(city, qualifier, sep = ", ")))
    return(list(is_us = NA, place = paste(city, qualifier, sep = ", ")))  # qualifier present, unrecognized
  }
  # Bare token: only the short curated standalone lists.
  cn <- normalize_token(city)
  if (!is.na(cn) && cn %in% gz$us_cities)      return(list(is_us = TRUE,  place = city))
  if (!is.na(cn) && cn %in% gz$foreign_cities) return(list(is_us = FALSE, place = city))
  list(is_us = NA, place = city)
}

# Remove the matched dateline span (block + delimiter) from the lead -> stripped_text.
strip_dateline <- function(lead, block_info) {
  if (is.na(lead)) return(NA_character_)
  if (!isTRUE(block_info$found) || block_info$match_len <= 0L) return(lead)
  trimws(substr(lead, block_info$match_len + 1L, nchar(lead)))
}

# Convenience: full dateline resolution for one lead -> list(is_us, place, block_info).
resolve_dateline <- function(lead, gz) {
  bi <- extract_dateline_block(lead)
  if (!isTRUE(bi$found)) return(list(is_us = NA, place = NA_character_, block_info = bi))
  fields <- parse_dateline_fields(bi$block)
  rp <- resolve_place(fields, gz)
  list(is_us = rp$is_us, place = rp$place, block_info = bi)
}
