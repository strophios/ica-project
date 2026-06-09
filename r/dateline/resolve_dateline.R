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

# Does a field look like a weekday? Matches "Monday", "Sat.", "Saturday", etc.
.WEEKDAY_RE <- "^(mon|tue|wed|thu|fri|sat|sun)[a-z.]*$"
is_weekday_field <- function(field) {
  grepl(.WEEKDAY_RE, trimws(tolower(field)))
}

# Isolate the leading ALL-CAPS place block before the dateline delimiter.
# Handles a leading "Special to The New York Times" credit and trailing "(AP)" wire tag.
# Returns list(found, block, match_len) where match_len is the number of leading
# characters of `lead` consumed by block + delimiter (for exact stripping).
# STRICT: the caps block must be genuinely all-caps (city part [A-Z][A-Z.' -]*[A-Z.],
# optionally followed by 1-2 comma-separated mixed-case fields like "Wash.", "Portugal", "Jan. 1").
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

  # Dateline block: all-caps city (uppercase words/periods/apostrophes/spaces/hyphens,
  # ending with uppercase letter or period to avoid matching emphasis-caps ledes),
  # optionally followed by 1-2 comma-separated fields of 2-20 mixed-case chars
  # ([A-Za-z. 0-9], covering "Wash.", "N.Y.", "Portugal", "July 30"),
  # then delimiter (em dash, --, or spaced hyphen).
  # Strategy: strictly enforce all-caps and proper termination on leading field,
  # allow 1-2 optional fields, then dash.
  dl_re <- "^\\s*([A-Z][A-Z.' -]*[A-Z.])(?:,\\s*([A-Za-z. 0-9]{2,20}))?(?:,\\s*([A-Za-z. 0-9]{2,20}))?\\s*(-|--|—)\\s"
  m2 <- regexpr(dl_re, work, perl = TRUE)
  if (m2 != 1L) return(empty)

  full_match <- regmatches(work, m2)
  block <- sub(dl_re, "\\1", full_match, perl = TRUE)
  # If there are 1-2 qualifier/date fields, append them to the block.
  q1 <- sub(dl_re, "\\2", full_match, perl = TRUE)
  q2 <- sub(dl_re, "\\3", full_match, perl = TRUE)
  if (nzchar(q1)) block <- paste0(block, ", ", q1)
  if (nzchar(q2)) block <- paste0(block, ", ", q2)

  # Remove a trailing wire tag like "(AP)" from the block.
  block <- trimws(sub("\\(AP\\)\\s*$", "", block))
  match_len <- offset + attr(m2, "match.length")
  list(found = TRUE, block = block, match_len = as.integer(match_len))
}

# Split a caps block into non-date, non-weekday fields (city + optional qualifier).
parse_dateline_fields <- function(block) {
  if (is.na(block) || !nzchar(block)) return(character(0))
  fields <- trimws(strsplit(block, ",", fixed = TRUE)[[1]])
  fields <- fields[nzchar(fields)]
  is_date <- vapply(fields, is_date_field, logical(1))
  is_weekday <- vapply(fields, is_weekday_field, logical(1))
  fields[!is_date & !is_weekday]
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

# Resolve a structured dateline FIELD (from NITF <dateline> metadata, no delimiter).
# "PASADENA, Calif., Dec. 31" -> list(is_us=TRUE, place="PASADENA, Calif.");
# "ZAGREB, Croatia" -> list(is_us=FALSE, place="ZAGREB, Croatia");
# "HAMILTON, New Zealand, Saturday, Jan. 1" -> list(is_us=NA, place=...);
# NA/empty -> list(is_us=NA, place=NA_character_).
# Strategy: split on commas, drop date + weekday fields, then resolve through resolve_place.
resolve_dateline_field <- function(dateline, gz) {
  if (is.na(dateline) || !nzchar(trimws(dateline))) {
    return(list(is_us = NA, place = NA_character_))
  }

  # Parse comma-separated fields and drop dates/weekdays
  fields <- trimws(strsplit(dateline, ",", fixed = TRUE)[[1]])
  fields <- fields[nzchar(fields)]
  is_date <- vapply(fields, is_date_field, logical(1))
  is_weekday <- vapply(fields, is_weekday_field, logical(1))
  fields <- fields[!is_date & !is_weekday]

  # Resolve through the standard place logic
  resolve_place(fields, gz)
}

# Remove the matched dateline span (block + delimiter) from the lead -> stripped_text.
# Returns NA if lead is NA; otherwise returns stripped text or original lead if no match.
strip_dateline <- function(lead, block_info) {
  if (is.na(lead)) return(NA_character_)
  if (!isTRUE(block_info$found) || block_info$match_len <= 0L) return(lead)
  trimws(substr(lead, block_info$match_len + 1L, nchar(lead)))
}

# Conditional-strip: decide whether a matched block is a REAL dateline or false positive.
# A block is a real dateline if it contains a date field, a recognized state/country qualifier,
# or is a bare AP-list city. Emphasis-caps ledes and bare unrecognized qualifiers -> not stripped.
# Returns list(is_us, place, should_strip) where should_strip indicates whether to strip
# the block from the lead (for stripped_text). If should_strip=FALSE, stripped text = original lead.
should_strip_dateline_block <- function(block, fields_after_parse, resolve_result, gz) {
  if (is.na(block) || !nzchar(block)) return(FALSE)

  # Check if the original block (before field filtering) contains a date field
  all_fields <- trimws(strsplit(block, ",", fixed = TRUE)[[1]])
  has_date <- any(vapply(all_fields, is_date_field, logical(1)))
  if (has_date) return(TRUE)

  # Check if we resolved to a US/not-US (not NA)
  if (!is.na(resolve_result$is_us)) return(TRUE)

  # Check if it's a bare AP-list city (1 field, resolved to US or foreign)
  if (length(fields_after_parse) == 1) {
    cn <- normalize_token(fields_after_parse[1])
    is_ap_us <- !is.na(cn) && cn %in% gz$us_cities
    is_ap_foreign <- !is.na(cn) && cn %in% gz$foreign_cities
    if (is_ap_us || is_ap_foreign) return(TRUE)
  }

  # Otherwise, it's a false positive (emphasis-caps lede or bare unrecognized token)
  FALSE
}

# Convenience: full dateline resolution for one lead -> list(is_us, place, block_info, should_strip).
# The text-channel is CONDITIONAL: blocks are treated as datelines only if they have
# a date field, a recognized qualifier, or are bare AP-list cities.
resolve_dateline <- function(lead, gz) {
  bi <- extract_dateline_block(lead)
  if (!isTRUE(bi$found)) {
    return(list(is_us = NA, place = NA_character_, block_info = bi, should_strip = FALSE))
  }

  fields <- parse_dateline_fields(bi$block)
  rp <- resolve_place(fields, gz)

  # Decide whether to actually treat this as a dateline (and strip it)
  should_strip <- should_strip_dateline_block(bi$block, fields, rp, gz)

  # If we're not stripping, return NA for the signal (text channel hygiene only)
  final_is_us <- if (should_strip) rp$is_us else NA
  final_place <- if (should_strip) rp$place else NA_character_

  list(is_us = final_is_us, place = final_place, block_info = bi, should_strip = should_strip)
}
