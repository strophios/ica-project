# Note: gazetteers loaded in run_tests.R before test_dir() is called
# This avoids path issues when testthat changes the working directory

resolve_lead <- function(lead) resolve_dateline(lead, gz)$is_us

# Test the structured field channel
resolve_field <- function(field) resolve_dateline_field(field, gz)$is_us

test_that("AC1.1 US state qualifier -> US", {
  expect_true(resolve_lead("VANCOUVER, Wash. — The city council met."))
})
test_that("AC1.2 country qualifier -> not-US", {
  expect_false(resolve_lead("LISBON, Portugal — Officials said."))
})
test_that("AC1.3 bare standalone cities", {
  expect_true(resolve_lead("CHICAGO — The mayor spoke."))
  expect_false(resolve_lead("LONDON — Parliament voted."))
})
test_that("AC1.4 collision: Paris, Texas vs bare PARIS", {
  expect_true(resolve_lead("PARIS, Texas — A local story."))
  expect_false(resolve_lead("PARIS — The president of France."))
})
test_that("AC1.5 date field dropped", {
  expect_false(resolve_lead("PARIS, July 30 — A summit opened."))
  expect_true(resolve_lead("WASHINGTON, July 30 — Congress acted."))
})
test_that("AC1.6 multi-comma with trailing date", {
  expect_true(resolve_lead("VANCOUVER, Wash., June 1 — Rain fell."))
})
test_that("Geneva/Moscow collisions", {
  # All-caps versions (matching actual NYT dateline format)
  expect_true(resolve_lead("GENEVA, N.Y. — A town meeting."))
  expect_false(resolve_lead("GENEVA — Talks resumed."))
  expect_true(resolve_lead("MOSCOW, Idaho — The university."))
  expect_false(resolve_lead("MOSCOW — The Kremlin said."))
})
test_that("AC1.10 bare token never hits long lists", {
  # The structural invariant: bare tokens consult ONLY AP-30/AP-46, never
  # countries.csv or us-area-code-cities. Prove it with bare tokens that ARE
  # present in the long lists but absent from the short standalone lists ->
  # they must resolve NA. A resolver that (wrongly) consulted the long lists
  # would resolve these to US/not-US and fail the test.
  # 'Portugal' is in countries.csv (a country name) but not AP-46:
  expect_true(is.na(resolve_lead("PORTUGAL — A bare country name, not a dateline city.")))
  # 'Bayonne' is in us-area-code-cities (a US city) but not AP-30:
  expect_true(is.na(resolve_lead("BAYONNE — A bare US city not on the AP-30 list.")))
  # And a plain unlisted token:
  expect_true(is.na(resolve_lead("LISBON — An unlisted bare city.")))
})
test_that("Field channel: PASADENA, Calif., Dec. 31 -> TRUE", {
  # Structured field from RDS: state qualifier present, date field dropped
  expect_true(resolve_field("PASADENA, Calif., Dec. 31"))
})
test_that("Field channel: ZAGREB, Croatia -> FALSE", {
  # Structured field: country qualifier present
  expect_false(resolve_field("ZAGREB, Croatia"))
})
test_that("Field channel: HAMILTON, New Zealand, Saturday, Jan. 1 -> FALSE", {
  # Structured field: weekday + date dropped, qualifier "New Zealand" -> country
  expect_false(resolve_field("HAMILTON, New Zealand, Saturday, Jan. 1"))
})
test_that("Field channel: WASHINGTON, Dec. 31 -> TRUE", {
  # Structured field: bare AP-30 city after date drop
  expect_true(resolve_field("WASHINGTON, Dec. 31"))
})
test_that("Field channel: LONDON, Dec. 31 -> FALSE", {
  # Structured field: bare AP-foreign city after date drop
  expect_false(resolve_field("LONDON, Dec. 31"))
})
test_that("Field channel: NA/empty -> NA", {
  expect_true(is.na(resolve_field(NA)))
  expect_true(is.na(resolve_field("")))
})
test_that("Field channel: BAYONNE (bare unlisted) -> NA", {
  # Bare token not on AP-30 or AP-46
  expect_true(is.na(resolve_field("BAYONNE")))
})

test_that("Conditional-strip: PILOBOLUS - ... not stripped, NA signal", {
  # Emphasis-caps false positive: no date, no recognized qualifier, bare PILOBOLUS not on AP lists
  lead <- "PILOBOLUS - that dance troupe specializing in mad scrambles"
  rd <- resolve_dateline(lead, gz)
  expect_true(is.na(rd$is_us))
  expect_false(rd$should_strip)  # should_strip flag is FALSE
  # Only strip if should_strip is TRUE
  stripped <- if (isTRUE(rd$should_strip)) strip_dateline(lead, rd$block_info) else lead
  expect_equal(stripped, lead)  # unchanged
})
test_that("Conditional-strip: MEMORY, memory - ... not stripped, NA signal", {
  # Emphasis-caps false positive: mixed-case second field should not match strict extractor
  lead <- "MEMORY, memory - is there ever enough of it?"
  rd <- resolve_dateline(lead, gz)
  expect_true(is.na(rd$is_us))
  expect_false(rd$should_strip)  # should_strip flag is FALSE
  # Only strip if should_strip is TRUE
  stripped <- if (isTRUE(rd$should_strip)) strip_dateline(lead, rd$block_info) else lead
  expect_equal(stripped, lead)  # unchanged
})
test_that("Conditional-strip: real spaced-hyphen dateline WASHINGTON, March 2 - ...", {
  # Real dateline from corpus: has date field, so should strip
  lead <- "WASHINGTON, March 2 - A girl from Westmont, Ill., attended the ceremony."
  rd <- resolve_dateline(lead, gz)
  expect_true(rd$is_us)  # US state present after date drop
  stripped <- strip_dateline(lead, rd$block_info)
  expect_equal(stripped, "A girl from Westmont, Ill., attended the ceremony.")
})

test_that("AC2.3 (R half) strip removes exactly the dateline span", {
  # raw == removed_span + stripped: stripping yields the post-delimiter remainder,
  # and re-extracting from the stripped text finds no dateline.
  lead <- "WASHINGTON, July 30 — Congress acted on the budget today."
  bi <- extract_dateline_block(lead)
  stripped <- strip_dateline(lead, bi)
  expect_equal(stripped, "Congress acted on the budget today.")
  expect_false(extract_dateline_block(stripped)$found)
  # Credit-line case: the "Special to The New York Times" prefix is consumed too.
  lead2 <- "Special to The New York Times CHICAGO — The mayor spoke."
  bi2 <- extract_dateline_block(lead2)
  expect_equal(strip_dateline(lead2, bi2), "The mayor spoke.")
})

test_that("AC1.7 desk backfill when no dateline", {
  # Use a real DSK_NON_US value (e.g. 'Foreign Desk') and a real DSK_US value.
  expect_false(classify_label(NA, desk_section_signal("Foreign Desk", NA))$us_label)
  expect_equal(classify_label(NA, desk_section_signal("Foreign Desk", NA))$label_source, "heuristic")
  expect_true(classify_label(NA, desk_section_signal("National Desk", NA))$us_label)
})
test_that("AC1.8 dateline/desk conflict -> null/conflict", {
  res <- classify_label(TRUE, FALSE)
  expect_true(is.na(res$us_label)); expect_equal(res$label_source, "conflict")
})
test_that("AC1.9 unresolved -> null", {
  res <- classify_label(NA, NA)
  expect_true(is.na(res$us_label)); expect_true(is.na(res$label_source))
})
test_that("dateline wins when desk agrees or is silent", {
  expect_true(classify_label(TRUE, NA)$us_label)
  expect_equal(classify_label(TRUE, TRUE)$label_source, "dateline")
})

test_that("REGRESSION: city names with month prefixes not dropped as date fields", {
  # JUNEAU, Alaska: "JUNEAU" should resolve, not be dropped as a false "Jun" date field
  expect_true(resolve_field("JUNEAU, Alaska"))
  # AUGUSTA, Georgia: not dropped as "Aug"
  expect_true(resolve_field("AUGUSTA, Georgia"))
  # MARQUETTE, Michigan: not dropped as "Mar"
  expect_true(resolve_field("MARQUETTE, Michigan"))
})

test_that("REGRESSION: city names with weekday prefixes not dropped as weekday fields", {
  # MONTGOMERY, Ala.: "MONTGOMERY" should resolve, not be dropped as a false "Mon" weekday
  expect_true(resolve_field("MONTGOMERY, Ala."))
  # SUNNYVALE, Calif.: not dropped as "Sun"
  expect_true(resolve_field("SUNNYVALE, Calif."))
  # MONTREAL, Canada: MONTREAL is in ap_foreign_cities; not dropped as "Mon"
  # (with unrecognized "Quebec" it would be NA, but with "Canada" it resolves false)
  expect_false(resolve_field("MONTREAL, Canada"))
  # FRISCO, Texas: "Fri" is only 3 chars, and our regex now requires anchored full/abbrev form
  expect_true(resolve_field("FRISCO, Texas"))
  # SUNDERLAND, Mass.: "Sun" is only 3 chars, anchored
  expect_true(resolve_field("SUNDERLAND, Mass."))
})

test_that("Date field regex: genuine dates accepted, ambiguous bare months rejected", {
  # Genuine date fields (with day numbers) should match
  expect_true(is_date_field("Dec. 31"))
  expect_true(is_date_field("July 30"))
  expect_true(is_date_field("Jan 5"))
  expect_true(is_date_field("May 1"))
  # Bare month names without day number should NOT match
  expect_false(is_date_field("May"))
  expect_false(is_date_field("June"))
  expect_false(is_date_field("December"))
})

test_that("Weekday field regex: genuine weekdays accepted, ambiguous names rejected", {
  # Full weekday names with optional period should match
  expect_true(is_weekday_field("Saturday"))
  expect_true(is_weekday_field("Sat."))
  expect_true(is_weekday_field("Monday"))
  expect_true(is_weekday_field("Mon."))
  # City names starting with weekday prefix should NOT match (now anchored)
  expect_false(is_weekday_field("SUNNYVALE"))
  expect_false(is_weekday_field("MONTGOMERY"))
  expect_false(is_weekday_field("FRISCO"))
  expect_false(is_weekday_field("SUNDERLAND"))
})

test_that("REGRESSION FIX: AP abbreviations Sept., Tues., Thurs. now match", {
  # AP style uses 4-letter "Sept." not 3-letter "Sep."
  expect_true(is_date_field("Sept. 28"))
  expect_true(is_date_field("Sept. 5"))
  # AP style uses "Tues." and "Thurs." not "Tue." and "Thu."
  expect_true(is_weekday_field("Tues."))
  expect_true(is_weekday_field("Thurs."))
  expect_true(is_weekday_field("Tuesday"))
  expect_true(is_weekday_field("Thursday"))
})

test_that("REGRESSION FIX: other month abbreviations still work", {
  # Oct., March, May, etc.
  expect_true(is_date_field("Oct. 9"))
  expect_true(is_date_field("March 2"))
  expect_true(is_date_field("May 1"))
})

test_that("REGRESSION FIX: end-to-end dateline with Sept. resolves correctly", {
  # WASHINGTON, Sept. 28 should resolve to US (DC known on AP list)
  expect_true(resolve_field("WASHINGTON, Sept. 28"))
  # ATLANTA, Sept. 3 should resolve to US (Georgia city on AP list)
  expect_true(resolve_field("ATLANTA, Sept. 3"))
  # PARIS, Sept. 5 should resolve to FALSE (PARIS bare -> foreign, not US)
  expect_false(resolve_field("PARIS, Sept. 5"))
})
