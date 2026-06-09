# Note: gazetteers loaded in run_tests.R before test_dir() is called
# This avoids path issues when testthat changes the working directory

resolve_lead <- function(lead) resolve_dateline(lead, gz)$is_us

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
  expect_true(resolve_lead("Geneva, N.Y. — A town meeting."))
  expect_false(resolve_lead("GENEVA — Talks resumed."))
  expect_true(resolve_lead("Moscow, Idaho — The university."))
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
