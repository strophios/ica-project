context("vendored us_assign heuristic")

# Note: us_assign, loc_df, and constants are loaded in run_tests.R before test_dir() is called.
# This avoids path issues when testthat changes the working directory.

# Verify the module loaded correctly
test_that("vendored us_assign module loads and defines required objects", {
  expect_true(exists("us_assign"))
  expect_true(exists("loc_df"))
  expect_true(exists("section_us"))
  expect_true(exists("dsk_us"))
  expect_true(exists("section_non_us"))
  expect_true(exists("dsk_non_us"))

  # Verify loc_df structure
  expect_equal(nrow(loc_df), 3155)
  expect_true(all(c("value", "is_us", "pattern") %in% colnames(loc_df)))
})

# Test the core function with synthetic data
test_that("us_assign correctly assigns US/not-US based on section/desk", {
  # Create a synthetic dataframe with no location keywords (test section/desk only)
  df <- tibble(
    id = c(1, 2, 3),
    section_name = c("U.S.", "World", "New York"),
    news_desk = c("National Desk", "Foreign Desk", "Metropolitan Desk"),
    keywords = list(
      data.frame(type = "keyword", value = "test", stringsAsFactors = FALSE),
      data.frame(type = "keyword", value = "test", stringsAsFactors = FALSE),
      data.frame(type = "keyword", value = "test", stringsAsFactors = FALSE)
    )
  )

  result <- us_assign(df)

  # US section should assign meta_is_us
  expect_true(result$any_us[1])
  expect_false(result$any_not_us[1])

  # Foreign Desk should assign meta_not_us
  expect_false(result$any_us[2])
  expect_true(result$any_not_us[2])

  # New York + Metro desk should assign US
  expect_true(result$any_us[3])
  expect_false(result$any_not_us[3])
})

test_that("us_assign correctly assigns based on location keywords", {
  # Create dataframe with location keywords
  df <- tibble(
    id = c(1, 2, 3),
    section_name = c(NA, NA, NA),
    news_desk = c(NA, NA, NA),
    keywords = list(
      # New York City (US)
      data.frame(type = "location", value = "New York City", stringsAsFactors = FALSE),
      # Paris (not US)
      data.frame(type = "location", value = "Paris", stringsAsFactors = FALSE),
      # France (not US)
      data.frame(type = "location", value = "France", stringsAsFactors = FALSE)
    )
  )

  result <- us_assign(df)

  # NYC should be US
  expect_true(result$any_us[1])
  expect_false(result$any_not_us[1])

  # Paris should be not-US
  expect_false(result$any_us[2])
  expect_true(result$any_not_us[2])

  # France should be not-US
  expect_false(result$any_us[3])
  expect_true(result$any_not_us[3])
})

test_that("us_assign handles no location signals (ambiguous/unresolved)", {
  # Create dataframe with no location signals
  df <- tibble(
    id = c(1),
    section_name = c(NA),
    news_desk = c(NA),
    keywords = list(
      data.frame(type = "keyword", value = "test", stringsAsFactors = FALSE)
    )
  )

  result <- us_assign(df)

  # Should have no signal
  expect_false(result$any_us[1])
  expect_false(result$any_not_us[1])
})

test_that("us_assign handles multi-section names (semicolon-separated)", {
  # Test the multi-section handling
  df <- tibble(
    id = c(1),
    section_name = c("U.S.; New York"),
    news_desk = c(NA),
    keywords = list(
      data.frame(type = "keyword", value = "test", stringsAsFactors = FALSE)
    )
  )

  result <- us_assign(df)

  # Multi-section with a US section should be US
  expect_true(result$any_us[1])
  expect_false(result$any_not_us[1])
})

test_that("us_assign aggregates multiple location keywords per article", {
  # Create dataframe where one article has both US and non-US locations
  df <- tibble(
    id = c(1, 1),  # Same id, two location keywords
    section_name = c(NA, NA),
    news_desk = c(NA, NA),
    keywords = list(
      data.frame(type = "location", value = "New York City", stringsAsFactors = FALSE),
      data.frame(type = "location", value = "France", stringsAsFactors = FALSE)
    )
  )

  result <- us_assign(df, return_long = FALSE)

  # Note: the function aggregates is_us/not_us by id, but because keywords
  # is still multi-row (2 different keyword values), distinct() keeps both rows.
  # Both rows will have the same is_us/not_us aggregated values.
  expect_equal(nrow(result), 2)

  # Both rows should have the aggregated signals (any_us=TRUE, any_not_us=TRUE)
  # because the id was aggregated to have both US and not-US locations
  expect_true(result$any_us[1])
  expect_true(result$any_not_us[1])
  expect_true(result$any_us[2])
  expect_true(result$any_not_us[2])
})

test_that("us_assign return_long parameter works correctly", {
  # Create dataframe with multiple locations per article
  df <- tibble(
    id = c(1, 1),
    section_name = c(NA, NA),
    news_desk = c(NA, NA),
    keywords = list(
      data.frame(type = "location", value = "New York City", stringsAsFactors = FALSE),
      data.frame(type = "location", value = "Paris", stringsAsFactors = FALSE)
    )
  )

  result_long <- us_assign(df, return_long = TRUE)
  result_short <- us_assign(df, return_long = FALSE)

  # Long form should have 2 rows (one per location keyword after unnesting)
  expect_equal(nrow(result_long), 2)

  # Short form also has 2 rows due to keywords list-column keeping distinct rows,
  # but both rows represent the same id with aggregated is_us/not_us values
  expect_equal(nrow(result_short), 2)

  # All rows in short form should have the aggregated verdict
  expect_true(all(result_short$any_us))
  expect_true(all(result_short$any_not_us))
})
