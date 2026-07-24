# Tests for r/api_ingest/archive_transform.R (sourced by run_tests.R).

fake_story <- function(headline = "PROTEST AT U.N.") {
  list(
    headline = list(main = headline, kicker = NULL),
    abstract = "Something happened.",
    lead_paragraph = "NEW YORK, May 1 - Lead text.",
    web_url = "https://www.nytimes.com/x",
    keywords = list(
      list(name = "subject", value = "Demonstrations", rank = 1L, major = "N"),
      list(name = "glocations", value = "NEW YORK CITY", rank = 2L, major = "N")
    ),
    pub_date = "1902-05-01T00:00:00+0000",
    document_type = "article",
    news_desk = "None",
    section_name = "Archives",
    uri = "nyt://article/abc",
    `_id` = "nyt://article/abc",
    word_count = 123L
  )
}

test_that("story_to_row extracts a full story", {
  row <- story_to_row(fake_story())
  expect_equal(row$headline, "PROTEST AT U.N.")
  expect_equal(row$abstrct, "Something happened.")
  expect_equal(row$id, "nyt://article/abc")
  expect_equal(row$word_count, 123L)
  expect_s3_class(row$pub_date, "Date")
  expect_equal(row$pub_date, as.Date("1902-05-01"))
  kw <- row$keywords[[1]]
  expect_equal(names(kw), c("type", "value", "rank", "major"))
  expect_equal(kw$type, c("subject", "glocations"))
})

test_that("story_to_row tolerates missing fields", {
  gappy <- list(headline = NULL, keywords = list(), pub_date = NULL)
  row <- story_to_row(gappy)
  expect_true(is.na(row$headline))
  expect_true(is.na(row$abstrct))
  expect_true(is.na(row$pub_date))
  expect_true(is.na(row$word_count))
  expect_null(row$keywords[[1]])
})

test_that("normalize_keywords handles tabular input and empties", {
  expect_null(normalize_keywords(NULL))
  expect_null(normalize_keywords(list()))
  tab <- data.frame(
    name = "subject", value = "Strikes", rank = 1L, major = "N"
  )
  out <- normalize_keywords(tab)
  expect_equal(names(out), c("type", "value", "rank", "major"))
  expect_equal(out$type, "subject")
})

test_that("month_tibble matches the canonical schema", {
  d <- month_tibble(list(fake_story(), fake_story("SECOND")), 1902, 5)
  expect_equal(names(d), ARCHIVE_COLS)
  expect_equal(d$year, c("1902", "1902"))
  expect_equal(d$month, c("5", "5")) # unpadded character, historical format
  template <- empty_archive_tibble()
  expect_equal(
    vapply(d, function(x) class(x)[1], character(1)),
    vapply(template, function(x) class(x)[1], character(1))
  )
  expect_silent(validate_archive_tibble(d))
})

test_that("month_tibble of no stories is an empty canonical tibble", {
  d <- month_tibble(list(), 1902, 5)
  expect_equal(nrow(d), 0)
  expect_equal(names(d), ARCHIVE_COLS)
  expect_silent(validate_archive_tibble(d))
})

test_that("validate_archive_tibble rejects schema drift", {
  d <- empty_archive_tibble()
  d$word_count <- NULL
  expect_error(validate_archive_tibble(d), "schema mismatch")
  d2 <- empty_archive_tibble()
  d2$word_count <- numeric()
  expect_error(validate_archive_tibble(d2), "word_count must be integer")
})

test_that("month_plan covers full years in order", {
  plan <- month_plan(c(1997, 1996))
  expect_equal(nrow(plan), 24)
  expect_equal(plan$year[1], 1996)
  expect_equal(plan$month[1:12], 1:12)
})

test_that("month_plan skeleton picks one rotating month per year", {
  plan <- month_plan(1851:1959, skeleton = TRUE)
  expect_equal(nrow(plan), 109)
  expect_equal(plan$year, 1851:1959)
  expect_true(all(plan$month %in% 1:12))
  expect_equal(plan$month[1], 1) # 1851 -> month 1
  expect_equal(plan$month[13], 1) # 1863 wraps back to month 1
  expect_equal(length(unique(plan$month[1:12])), 12) # all months hit
})

test_that("month_plan --through caps the tail", {
  plan <- month_plan(2025:2026, through = "2026-06")
  expect_equal(nrow(plan), 18)
  expect_equal(max(plan$month[plan$year == 2026]), 6)
})

test_that("month_plan rejects out-of-range years", {
  expect_error(month_plan(1800), "out of range")
})

test_that("parse_years handles single years, ranges, and garbage", {
  expect_equal(parse_years("1996"), 1996L)
  expect_equal(parse_years("1996:1998"), 1996:1998)
  expect_error(parse_years("1998:1996"), "reversed")
  expect_error(parse_years("96:98"), "invalid")
  expect_error(parse_years("1996-1998"), "invalid")
})

test_that("parse_through validates its format", {
  expect_equal(parse_through("2026-06"), list(year = 2026L, month = 6L))
  expect_equal(parse_through("2026-6")$month, 6L)
  expect_error(parse_through("2026-13"), "month")
  expect_error(parse_through("June 2026"), "invalid")
})

test_that("raw_checkpoint_path pads months", {
  expect_equal(
    basename(raw_checkpoint_path("/tmp", 1902, 5)), "1902_05.rds"
  )
})
