if (!requireNamespace("testthat", quietly = TRUE))
  install.packages("testthat", repos = "https://cloud.r-project.org")
library(testthat)
# Ensure we're in the project root directory
if (!file.exists("r/dateline/gazetteers")) {
  stop("run_tests.R must be executed from the project root directory")
}
source("r/dateline/resolve_dateline.R")
source("r/dateline/label_policy.R")
# Load gazetteers and make available to tests via parent environment
gz <- load_gazetteers("r/dateline/gazetteers")
# Load vendored us_assign heuristic for tests
source("r/vendored/us_assign.R")
# Archive-pull transforms (pure core of r/api_ingest/pull_archive.R)
source("r/api_ingest/archive_transform.R")
testthat::test_dir("r/tests/testthat", stop_on_failure = TRUE)
