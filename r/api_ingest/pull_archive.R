# pattern: Imperative Shell
# Resumable NYT Archive API pull. Fetches month-by-month, checkpointing each
# RAW API response to disk (so a failure, ban, or budget stop loses nothing
# and transform bugs can be fixed without re-pulling), then assembles any year
# whose 12 months are all present into nyt_archive_by_year/{year}.rds in the
# canonical schema. Partial years (skeleton mode, --through caps) stay as raw
# checkpoints only -- an incomplete year can never leak into the corpus.
#
# Run from the ica_project root. The API key comes from the NYT_API_KEY env
# var (never hardcode it here):
#
#   export NYT_API_KEY=...   # same key as in the old nyt_headlines.R
#
#   # Forward pull, 1996 -> mid-2026 (skips the incomplete current month):
#   Rscript r/api_ingest/pull_archive.R --years 1996:2026 --through 2026-06
#
#   # Backward skeleton (one rotating month per year, 1851-1959):
#   Rscript r/api_ingest/pull_archive.R --years 1851:1959 --skeleton
#
# Resume: re-run the same command; already-fetched months are skipped. The
# per-run request budget (--max-requests, default 480) stays under the API's
# 500/day cap; the script stops cleanly at the budget and tells you what
# remains. Rate limit: 5 requests/min, enforced via --sleep (>= 12s).

suppressMessages({
  library(tibble)
  library(dplyr)
})
source("r/api_ingest/archive_transform.R")

DATA_ROOT <- "/Users/strophios/immigration_project/00_ML_data_expansion"

parse_cli <- function(args) {
  opts <- list(
    years = NULL, skeleton = FALSE, through = NULL,
    max_requests = 480L, sleep = 12,
    raw_dir = file.path(DATA_ROOT, "nyt_archive_raw"),
    out_dir = file.path(DATA_ROOT, "nyt_archive_by_year")
  )
  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (a == "--skeleton") {
      opts$skeleton <- TRUE
      i <- i + 1
      next
    }
    if (i == length(args)) stop("missing value for ", a)
    v <- args[i + 1]
    if (a == "--years") {
      opts$years <- parse_years(v)
    } else if (a == "--through") {
      opts$through <- v
    } else if (a == "--max-requests") {
      opts$max_requests <- as.integer(v)
    } else if (a == "--sleep") {
      opts$sleep <- as.numeric(v)
    } else if (a == "--raw-dir") {
      opts$raw_dir <- v
    } else if (a == "--out-dir") {
      opts$out_dir <- v
    } else {
      stop("unknown argument: ", a)
    }
    i <- i + 2
  }
  if (is.null(opts$years)) {
    stop("--years is required (e.g. --years 1996:2026)")
  }
  if (is.na(opts$max_requests) || opts$max_requests < 1) {
    stop("invalid --max-requests")
  }
  if (is.na(opts$sleep) || opts$sleep < 12) {
    stop("--sleep must be >= 12 (NYT rate limit is 5 requests/min)")
  }
  opts
}

fetch_month_with_retry <- function(year, month, sleep_s, tries = 4) {
  last_err <- NULL
  for (attempt in seq_len(tries)) {
    res <- tryCatch(nytimes::ny_archive(year, month), error = identity)
    Sys.sleep(sleep_s) # hold the rate limit whether or not the call worked
    if (!inherits(res, "error")) {
      return(res)
    }
    last_err <- res
    if (attempt < tries) {
      wait <- 30 * 2^(attempt - 1)
      cat(sprintf(
        "%d-%02d: attempt %d failed (%s); backing off %ds\n",
        year, month, attempt, conditionMessage(res), wait
      ))
      Sys.sleep(wait)
    }
  }
  stop(sprintf(
    paste0(
      "failed to fetch %d-%02d after %d attempts: %s -- ",
      "fetched months are checkpointed; re-run the same command to resume"
    ),
    year, month, tries, conditionMessage(last_err)
  ))
}

# Assemble every planned year whose full 12 months are checkpointed. Years
# planned with fewer than 12 months (skeleton, --through cap) are skipped by
# construction; existing {year}.rds files are never overwritten.
assemble_complete_years <- function(plan, raw_dir, out_dir) {
  for (yr in unique(plan$year)) {
    months <- plan$month[plan$year == yr]
    if (length(months) < 12) next
    paths <- raw_checkpoint_path(raw_dir, yr, months)
    if (!all(file.exists(paths))) next
    out_path <- file.path(out_dir, paste0(yr, ".rds"))
    if (file.exists(out_path)) next
    d <- dplyr::bind_rows(lapply(seq_along(months), function(i) {
      month_tibble(readRDS(paths[i]), yr, months[i])
    }))
    validate_archive_tibble(d)
    if (nrow(d) == 0) {
      stop("assembled year ", yr, " has 0 rows; refusing to write ", out_path)
    }
    saveRDS(d, out_path)
    cat(sprintf("assembled %d.rds: %d articles\n", yr, nrow(d)))
  }
}

main <- function() {
  opts <- parse_cli(commandArgs(trailingOnly = TRUE))
  key <- Sys.getenv("NYT_API_KEY")
  if (!nzchar(key)) {
    stop(
      "NYT_API_KEY is not set; export it before running ",
      "(the key lives in the old nyt_headlines.R)"
    )
  }
  nytimes::nytimes_key(key)

  dir.create(opts$raw_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(opts$out_dir, showWarnings = FALSE, recursive = TRUE)

  plan <- month_plan(
    opts$years, skeleton = opts$skeleton, through = opts$through
  )
  have <- file.exists(raw_checkpoint_path(opts$raw_dir, plan$year, plan$month))
  todo <- plan[!have, ]
  cat(sprintf(
    "plan: %d months | checkpointed: %d | to fetch: %d | budget: %d requests\n",
    nrow(plan), sum(have), nrow(todo), opts$max_requests
  ))

  start <- proc.time()[["elapsed"]]
  fetched <- 0
  for (i in seq_len(nrow(todo))) {
    if (fetched >= opts$max_requests) {
      cat(sprintf(
        paste0(
          "request budget (%d) reached; %d months remain. ",
          "Re-run the same command (e.g. tomorrow) to resume.\n"
        ),
        opts$max_requests, nrow(todo) - i + 1
      ))
      break
    }
    yr <- todo$year[i]
    mo <- todo$month[i]
    stories <- fetch_month_with_retry(yr, mo, sleep_s = opts$sleep)
    fetched <- fetched + 1
    saveRDS(stories, raw_checkpoint_path(opts$raw_dir, yr, mo))
    elapsed <- proc.time()[["elapsed"]] - start
    remaining <- min(nrow(todo), opts$max_requests) - i
    cat(sprintf(
      "%d-%02d: %d articles [%d/%d, %.0fs elapsed, ~%.0f min left]\n",
      yr, mo, length(stories), i, nrow(todo), elapsed,
      remaining * (elapsed / i) / 60
    ))
  }

  assemble_complete_years(plan, opts$raw_dir, opts$out_dir)
  cat("done. Convert new complete years with: ",
      "Rscript r/api_ingest/rds_to_parquet.R\n", sep = "")
}

if (sys.nframe() == 0) main()
