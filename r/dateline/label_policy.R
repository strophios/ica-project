# pattern: Functional Core
# Desk/section US-signal + dateline/desk fusion policy.
# Desk lists copied from nyt_location_checking.R:65-82 (provenance: out-of-repo
# /Users/strophios/immigration_project/00_ML_data_expansion/nyt_location_checking.R).

# Transcribed from nyt_location_checking.R:65-82
DSK_US <- c(
  "national desk",
  "metropolitan desk",
  "connecticut weekly desk",
  "westchester weekly desk",
  "long island weekly desk",
  "new york region",
  "new jersey weekly desk",
  "the city weekly desk"
)

DSK_NON_US <- c("foreign desk")

SECTION_US <- c("u.s.", "new york", "new york and region", "washington")

SECTION_NON_US <- c("world")

# Desk/section US signal: TRUE | FALSE | NA.
desk_section_signal <- function(dsk, print_section) {
  # Normalize inputs to lowercase for comparison
  dsk_lower <- if (!is.na(dsk)) tolower(dsk) else NA_character_
  section_lower <- if (!is.na(print_section)) tolower(print_section) else NA_character_

  is_us  <- (!is.na(dsk_lower) && dsk_lower %in% DSK_US)         || (!is.na(section_lower) && section_lower %in% SECTION_US)
  is_non <- (!is.na(dsk_lower) && dsk_lower %in% DSK_NON_US)     || (!is.na(section_lower) && section_lower %in% SECTION_NON_US)
  if (is_us && !is_non) return(TRUE)
  if (is_non && !is_us) return(FALSE)
  NA  # silent or internally-conflicting desk signal -> no confident desk label
}

# Fuse dateline + desk signals -> list(us_label, label_source).
classify_label <- function(dateline_is_us, desk_is_us) {
  has_dl   <- !is.na(dateline_is_us)
  has_desk <- !is.na(desk_is_us)
  if (has_dl && has_desk && (dateline_is_us != desk_is_us))
    return(list(us_label = NA, label_source = "conflict"))   # AC1.8
  if (has_dl)
    return(list(us_label = dateline_is_us, label_source = "dateline"))  # AC1.1-1.6
  if (has_desk)
    return(list(us_label = desk_is_us, label_source = "heuristic"))     # AC1.7
  list(us_label = NA, label_source = NA_character_)           # AC1.9
}
