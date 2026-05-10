# Credit-card fraud capstone, single submission script for PH125.9x.
#
# What it does, step by step
#   1. install or load the R packages I need
#   2. download the train and test CSVs from Hugging Face if I do not have them
#   3. read the CSVs and build my feature tables
#   4. fit five candidate models on a 75% slice of the training data
#   5. score the held-out 25% slice and pick the best model and threshold by F1
#   6. refit the chosen model on the full training data and score the test set once
#   7. save the tables and example transactions the report uses
#   8. render Credit_card_fraud_report.pdf (skip with --skip-render)
#
# How I run it
#   Rscript credit_card_fraud_main.R
#   CC_TRAIN_N=900000 CC_TEST_N=300000 Rscript credit_card_fraud_main.R
#   Rscript credit_card_fraud_main.R --skip-render
#   Rscript credit_card_fraud_main.R --dry-run

options(stringsAsFactors = FALSE)
experiment_started_at <- Sys.time()

required_packages <- c("readr", "dplyr", "ggplot2", "knitr",
                       "randomForest", "rmarkdown")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(randomForest)
})

# I want this script to work even if you run it from a different folder, so
# I figure out where the script lives and use that as the project root.
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) == 1L) sub("^--file=", "", script_arg) else "credit_card_fraud_main.R"
root <- dirname(normalizePath(script_path, mustWork = FALSE))
if (!dir.exists(root)) root <- getwd()
setwd(root)

cli_args <- commandArgs(trailingOnly = TRUE)
dry_run <- "--dry-run" %in% cli_args
skip_render <- "--skip-render" %in% cli_args

TRAIN_N <- as.integer(Sys.getenv("CC_TRAIN_N", unset = "300000"))
TEST_N  <- as.integer(Sys.getenv("CC_TEST_N",  unset = "150000"))

message("Project root ", normalizePath(root))
message("Train rows cap ", TRAIN_N, " (CC_TRAIN_N)")
message("Test rows cap  ", TEST_N,  " (CC_TEST_N)")


# ---- 1. Download the CSVs if I don't have them ---------------------------

data_dir <- file.path(root, "data")
if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)

base_url  <- "https://huggingface.co/datasets/pointe77/credit-card-transaction/resolve/main"
train_csv <- file.path(data_dir, "credit_card_transaction_train.csv")
test_csv  <- file.path(data_dir, "credit_card_transaction_test.csv")

# I only download what is missing so re-runs are fast.
download_if_missing <- function(url, dest) {
  if (file.exists(dest)) {
    message("Already have ", basename(dest), ", skipping download")
    return(invisible(dest))
  }
  message("Downloading ", url)
  utils::download.file(url, destfile = dest, mode = "wb", quiet = FALSE)
  invisible(dest)
}

if (dry_run) {
  message("Dry run only. Packages and paths look fine.")
  message("Expected train CSV ", train_csv)
  message("Expected test CSV  ", test_csv)
  quit(save = "no", status = 0)
}

download_if_missing(file.path(base_url, "credit_card_transaction_train.csv"), train_csv)
download_if_missing(file.path(base_url, "credit_card_transaction_test.csv"),  test_csv)


# ---- 2. Read the CSVs and build feature tables ---------------------------

# I skip every column by default and only opt in to the ones I actually use.
# When I add a new feature later, this is the place to add it.
cols_keep <- cols(
  .default = col_skip(),
  trans_date_trans_time = col_character(),
  cc_num = col_character(),
  merchant = col_character(),
  category = col_character(),
  amt = col_double(),
  gender = col_character(),
  state = col_character(),
  lat = col_double(),
  long = col_double(),
  city_pop = col_double(),
  job = col_character(),
  dob = col_character(),
  unix_time = col_double(),
  merch_lat = col_double(),
  merch_long = col_double(),
  is_fraud = col_integer()
)

raw_train <- readr::read_csv(train_csv, n_max = TRAIN_N, col_types = cols_keep, progress = FALSE)
raw_test  <- readr::read_csv(test_csv,  n_max = TEST_N,  col_types = cols_keep, progress = FALSE)

# Great-circle distance in km between two lat/lon points. I use this for the
# customer-merchant distance feature.
haversine_km <- function(lat1, lon1, lat2, lon2) {
  rad <- pi / 180
  dlat <- (lat2 - lat1) * rad
  dlon <- (lon2 - lon1) * rad
  a <- sin(dlat / 2)^2 + cos(lat1 * rad) * cos(lat2 * rad) * sin(dlon / 2)^2
  6371 * 2 * atan2(sqrt(a), sqrt(pmax(0, 1 - a)))
}

prep <- function(df) {
  ts <- as.POSIXct(df$trans_date_trans_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
  dob <- as.Date(df$dob)
  age_years <- as.numeric(as.Date(ts) - dob) / 365.25
  age_years <- ifelse(is.na(age_years), median(age_years, na.rm = TRUE), age_years)

  # Every merchant name in this dataset starts with "fraud_". I strip it so
  # the same merchant lines up across rows.
  merchant_clean <- sub("^fraud_", "", df$merchant)

  data.frame(
    is_fraud = as.integer(df$is_fraud),
    amt = df$amt,
    amt_log = log1p(df$amt),
    hour = as.integer(format(ts, "%H")),
    day = as.integer(format(ts, "%d")),
    month = as.integer(format(ts, "%m")),
    weekday = factor(weekdays(ts)),
    age_years = age_years,
    distance_km = haversine_km(df$lat, df$long, df$merch_lat, df$merch_long),
    city_pop_log = log1p(df$city_pop),
    gender = factor(df$gender),
    category = factor(df$category),
    state = factor(df$state),
    merchant = merchant_clean,
    cc_num = df$cc_num,
    job = df$job,
    stringsAsFactors = FALSE
  )
}

cc_train <- prep(raw_train)
cc_test  <- prep(raw_test)

# I make sure train and test share the same factor levels. If a level shows
# up in test but not in train, predict() crashes when I score it later.
# merchant, cc_num, and job stay as character because they have too many
# values to use as factors. I rate-encode those further down.
for (nm in c("weekday", "gender", "category", "state")) {
  lv <- union(levels(cc_train[[nm]]), levels(cc_test[[nm]]))
  cc_train[[nm]] <- factor(as.character(cc_train[[nm]]), levels = lv)
  cc_test[[nm]]  <- factor(as.character(cc_test[[nm]]),  levels = lv)
}

meta <- list(
  train_n_read = nrow(raw_train),
  test_n_read = nrow(raw_test),
  train_rows = nrow(cc_train),
  test_rows = nrow(cc_test),
  train_rate = mean(cc_train$is_fraud),
  test_rate = mean(cc_test$is_fraud),
  source = "pointe77/credit-card-transaction",
  prepared_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
)

save(cc_train, cc_test, meta, file = file.path(root, "cc_fraud_data.RData"))
message("Saved cc_fraud_data.RData. Train P(fraud)=",
        signif(meta$train_rate, 5),
        "  Test P(fraud)=", signif(meta$test_rate, 5))


# ---- 3. Modelling helpers ------------------------------------------------

# ROC-AUC. I compute it from ranks. It is the chance the model scores a
# random fraud row above a random non-fraud row.
auc_roc <- function(y, score) {
  y <- as.integer(y)
  score <- as.numeric(score)
  ok <- is.finite(score) & !is.na(y)
  y <- y[ok]
  score <- score[ok]
  n1 <- sum(y == 1L)
  n0 <- sum(y == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  r <- rank(score)
  (sum(r[y == 1L]) - n1 * (n1 + 1) / 2) / (n0 * n1)
}

# Average precision, which is the area under the precision-recall curve.
avg_precision <- function(y, score) {
  y <- as.integer(y)
  score <- as.numeric(score)
  ok <- is.finite(score) & !is.na(y)
  y <- y[ok]
  score <- score[ok]
  ord <- order(-score)
  y <- y[ord]
  P <- sum(y == 1L)
  if (P == 0L) return(NA_real_)
  tp <- cumsum(y == 1L)
  fp <- cumsum(y == 0L)
  precision <- tp / (tp + fp)
  recall <- tp / P
  sum((c(recall[1], diff(recall))) * precision)
}

# Threshold helpers. Each one takes scores and returns a single score cutoff.
# I use type = 1 because I want the cutoff to be a real score from the data,
# not an interpolated value. That makes it easier to apply later.
threshold_top_rate <- function(score, rate) {
  as.numeric(stats::quantile(score, 1 - rate, na.rm = TRUE, type = 1))
}

threshold_best_f1 <- function(y, score) {
  y <- as.integer(y)
  score <- as.numeric(score)
  ok <- is.finite(score) & !is.na(y)
  y <- y[ok]
  score <- score[ok]
  ord <- order(-score)
  y <- y[ord]
  score <- score[ord]
  tp <- cumsum(y == 1L)
  fp <- cumsum(y == 0L)
  fn <- sum(y == 1L) - tp
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  f1 <- ifelse(precision + recall > 0,
               2 * precision * recall / (precision + recall), 0)
  score[which.max(f1)]
}

threshold_recall_target <- function(y, score, target = 0.8) {
  y <- as.integer(y)
  score <- as.numeric(score)
  ord <- order(-score)
  y <- y[ord]
  score <- score[ord]
  recall <- cumsum(y == 1L) / sum(y == 1L)
  idx <- which(recall >= target)[1]
  if (is.na(idx)) min(score, na.rm = TRUE) else score[idx]
}

# Confusion-matrix metrics for one threshold (precision, recall, F1, etc).
class_metrics <- function(y, score, threshold) {
  y <- as.integer(y)
  pred <- as.integer(score >= threshold)
  tp <- sum(pred == 1L & y == 1L)
  fp <- sum(pred == 1L & y == 0L)
  tn <- sum(pred == 0L & y == 0L)
  fn <- sum(pred == 0L & y == 1L)
  precision <- if (tp + fp > 0L) tp / (tp + fp) else NA_real_
  recall <- if (tp + fn > 0L) tp / (tp + fn) else NA_real_
  f1 <- if (is.finite(precision) && is.finite(recall) && precision + recall > 0)
          2 * precision * recall / (precision + recall) else NA_real_
  data.frame(
    threshold = threshold,
    flagged = tp + fp,
    alert_rate = mean(pred == 1L),
    tp = tp, fp = fp, tn = tn, fn = fn,
    accuracy = (tp + tn) / length(y),
    precision = precision,
    recall = recall,
    f1 = f1,
    stringsAsFactors = FALSE
  )
}

prob_metrics <- function(y, score) {
  p <- pmax(pmin(as.numeric(score), 1 - 1e-15), 1e-15)
  y <- as.integer(y)
  data.frame(
    roc_auc = auc_roc(y, score),
    pr_auc = avg_precision(y, score),
    brier = mean((y - p)^2),
    log_loss = -mean(y * log(p) + (1 - y) * log(1 - p)),
    stringsAsFactors = FALSE
  )
}

# Down-sampled (recall, precision, fpr) trace I use for the report's curves.
curve_points <- function(y, score, n_points = 250) {
  y <- as.integer(y)
  score <- as.numeric(score)
  ord <- order(-score)
  y <- y[ord]
  score <- score[ord]
  idx <- unique(pmax(1L, floor(seq(1, length(y),
                                    length.out = min(n_points, length(y))))))
  tp <- cumsum(y == 1L)[idx]
  fp <- cumsum(y == 0L)[idx]
  positives <- sum(y == 1L)
  negatives <- sum(y == 0L)
  data.frame(
    threshold = score[idx],
    recall = if (positives > 0L) tp / positives else NA_real_,
    precision = ifelse(tp + fp > 0L, tp / (tp + fp), NA_real_),
    fpr = if (negatives > 0L) fp / negatives else NA_real_,
    stringsAsFactors = FALSE
  )
}

# 75/25 stratified split. I keep both halves with positives in them, otherwise
# the validation slice could end up with almost no fraud rows when I am
# running on a smaller sample.
split_fit_idx <- function(y, frac = 0.75, seed = 1) {
  set.seed(seed)
  pos <- which(y == 1L)
  neg <- which(y == 0L)
  sort(c(sample(pos, floor(frac * length(pos))),
         sample(neg, floor(frac * length(neg)))))
}

# Smoothed historical fraud rate for fields with too many values to use as
# factors (merchant, cc_num, job).
#   rate = (positives + alpha * global_rate) / (count + alpha)
# alpha = 50 means a low-volume merchant gets pulled toward the global rate
# and a high-volume merchant keeps its empirical rate.
#
# When fit_df and new_df are the same data during training, the rate features
# are computed from the same labels the model is trained on. That is mild
# target leakage. I leave it that way for the encoded variants because they
# are an exploratory comparison. The selected rf_base model does not use
# these encodings, so the leakage does not affect the final test result.
add_rate_encoding <- function(fit_df, new_df, key, prefix, alpha = 50) {
  kfit <- fit_df[[key]]
  knew <- new_df[[key]]
  y <- fit_df$is_fraud
  global <- mean(y)
  n <- tapply(y, kfit, length)
  pos <- tapply(y, kfit, sum)
  rate <- (pos + alpha * global) / (n + alpha)
  m <- match(knew, names(rate))
  out <- data.frame(
    rate = ifelse(is.na(m), global, as.numeric(rate[m])),
    count = ifelse(is.na(m), 0, as.numeric(n[m])),
    stringsAsFactors = FALSE
  )
  names(out) <- paste0(prefix, c("_rate", "_log_count"))
  out[[paste0(prefix, "_log_count")]] <- log1p(out[[paste0(prefix, "_log_count")]])
  out
}

make_design <- function(fit_df, new_df, include_encodings = TRUE) {
  x <- data.frame(
    amt_log = new_df$amt_log,
    hour = new_df$hour,
    day = new_df$day,
    month = new_df$month,
    weekday = new_df$weekday,
    age_years = new_df$age_years,
    distance_km = new_df$distance_km,
    city_pop_log = new_df$city_pop_log,
    gender = new_df$gender,
    category = new_df$category,
    state = new_df$state
  )
  if (include_encodings) {
    for (key in c("merchant", "cc_num", "job", "category", "state")) {
      x <- cbind(x, add_rate_encoding(fit_df, new_df, key, key))
    }
    rate_cols <- grep("_rate$", names(x), value = TRUE)
    x$max_entity_rate <- do.call(pmax, x[rate_cols])
  }
  x
}

# Fit one of my five candidate models on fit_df.
fit_model <- function(id, fit_df) {
  if (id == "amount_rule") {
    return(list(id = id, type = "rule", feature = "amt_log"))
  }

  include_enc <- id %in% c("glm_encoded", "rf_encoded")
  x <- make_design(fit_df, fit_df, include_encodings = include_enc)

  if (grepl("^glm", id)) {
    dat <- x
    dat$is_fraud <- fit_df$is_fraud
    fit <- suppressWarnings(
      glm(is_fraud ~ ., data = dat, family = binomial(),
          control = glm.control(maxit = 100))
    )
    return(list(id = id, type = "glm", fit = fit, include_enc = include_enc))
  }

  if (grepl("^rf", id)) {
    dat <- x
    dat$y_fac <- factor(ifelse(fit_df$is_fraud == 1L, "Y", "N"))
    n_pos <- sum(dat$y_fac == "Y")
    n_neg <- sum(dat$y_fac == "N")

    # Per-model seed so a re-run gives me identical numbers.
    set.seed(1000 + match(id, c("rf_base", "rf_encoded"), nomatch = 0L))

    # I balance the rare class by down-sampling each tree. Without this each
    # tree would see roughly 600x more non-fraud than fraud and split badly.
    fit <- randomForest::randomForest(
      y_fac ~ ., data = dat, ntree = 180L,
      sampsize = c(min(15000L, n_neg), min(8000L, n_pos)),
      importance = TRUE
    )
    return(list(id = id, type = "rf", fit = fit, include_enc = include_enc))
  }

  stop("unknown model id ", id, call. = FALSE)
}

# Score new_df with a fitted model. I pass fit_df too because the rate
# encodings look up new_df values against the fit-set rates.
predict_model <- function(model, fit_df, new_df) {
  if (model$type == "rule") return(as.numeric(new_df[[model$feature]]))
  x <- make_design(fit_df, new_df, include_encodings = model$include_enc)
  if (model$type == "glm") {
    return(suppressWarnings(as.numeric(predict(model$fit, newdata = x, type = "response"))))
  }
  as.numeric(predict(model$fit, newdata = x, type = "prob")[, "Y"])
}


# ---- 4. Validation phase, fit on 75%, score the 25% ---------------------

fit_idx <- split_fit_idx(cc_train$is_fraud, frac = 0.75, seed = 1)
fit_df <- cc_train[fit_idx, , drop = FALSE]
val_df <- cc_train[-fit_idx, , drop = FALSE]
cat("Fit rows ", nrow(fit_df), " positives ", sum(fit_df$is_fraud), "\n")
cat("Val rows ", nrow(val_df), " positives ", sum(val_df$is_fraud), "\n")

model_ids <- c("amount_rule", "glm_base", "glm_encoded", "rf_base", "rf_encoded")

rules <- list(
  top_0.5pct = function(y, s) threshold_top_rate(s, 0.005),
  top_1pct   = function(y, s) threshold_top_rate(s, 0.01),
  top_2pct   = function(y, s) threshold_top_rate(s, 0.02),
  best_f1    = function(y, s) threshold_best_f1(y, s),
  recall80   = function(y, s) threshold_recall_target(y, s, 0.8)
)

rows <- list()
for (id in model_ids) {
  cat("Fitting ", id, "\n")
  model <- fit_model(id, fit_df)

  # Predict on the validation slice.
  val_score <- predict_model(model, fit_df, val_df)

  pm <- prob_metrics(val_df$is_fraud, val_score)
  for (rule in names(rules)) {
    th <- rules[[rule]](val_df$is_fraud, val_score)
    rows[[length(rows) + 1L]] <- cbind(
      data.frame(model = id, rule = rule, stringsAsFactors = FALSE),
      pm,
      class_metrics(val_df$is_fraud, val_score, th)
    )
  }
}
validation_results <- do.call(rbind, rows)
validation_results <- validation_results[
  order(-validation_results$f1, -validation_results$precision, na.last = TRUE), ]
rownames(validation_results) <- NULL

cat("\n=== Validation top rows ===\n")
print(head(validation_results, 20), row.names = FALSE, digits = 4)
cat("\n=== Validation recall >= 80% ===\n")
print(validation_results[validation_results$recall >= 0.8, ], row.names = FALSE, digits = 4)

# Pick the row with the best F1. Ties go to the row with higher precision.
best_selection <- validation_results[1, ]
cat("\nSelected ", best_selection$model, "/", best_selection$rule, "\n")


# ---- 5. Final held-out test ----------------------------------------------

# I refit the chosen model on the full training data and score the test set
# exactly once. This is the only place I touch cc_test.
final_model <- fit_model(best_selection$model, cc_train)

# Predict on the held-out test set. This single line is the test prediction.
test_score <- predict_model(final_model, cc_train, cc_test)

final_threshold <- best_selection$threshold

final_results <- cbind(
  data.frame(model = best_selection$model, rule = best_selection$rule,
             stringsAsFactors = FALSE),
  prob_metrics(cc_test$is_fraud, test_score),
  class_metrics(cc_test$is_fraud, test_score, final_threshold)
)

# I also score the same final model under each threshold rule so the report
# can show how the operational tradeoff works. Each rule's cutoff comes from
# the validation table, so I never re-select a threshold on the test set.
final_rule_rows <- list()
for (rule_name in names(rules)) {
  th <- validation_results$threshold[
    validation_results$model == best_selection$model &
      validation_results$rule == rule_name
  ][1]
  final_rule_rows[[length(final_rule_rows) + 1L]] <- cbind(
    data.frame(model = best_selection$model, rule = rule_name,
               stringsAsFactors = FALSE),
    prob_metrics(cc_test$is_fraud, test_score),
    class_metrics(cc_test$is_fraud, test_score, th)
  )
}
final_rule_results <- do.call(rbind, final_rule_rows)
final_rule_results <- final_rule_results[
  order(-final_rule_results$f1, -final_rule_results$recall, na.last = TRUE), ]

test_curve <- curve_points(cc_test$is_fraud, test_score)


# ---- 6. Example transactions and feature importance ---------------------

scored_test <- cc_test
scored_test$score <- test_score
scored_test$predicted_alert <- as.integer(test_score >= final_threshold)
scored_test$outcome <- ifelse(
  scored_test$predicted_alert == 1L & scored_test$is_fraud == 1L, "true positive",
  ifelse(scored_test$predicted_alert == 1L & scored_test$is_fraud == 0L, "false positive",
    ifelse(scored_test$predicted_alert == 0L & scored_test$is_fraud == 1L, "missed fraud", "true negative")))

example_cols <- c(
  "outcome", "score", "is_fraud", "predicted_alert", "amt", "category",
  "hour", "age_years", "distance_km", "city_pop_log", "gender", "state"
)

top_alert_examples <- scored_test[order(-scored_test$score), example_cols][1:12, ]

true_positive_examples <- scored_test[scored_test$outcome == "true positive", example_cols]
true_positive_examples <- true_positive_examples[order(-true_positive_examples$score), ][
  1:min(8, nrow(true_positive_examples)), ]

false_positive_examples <- scored_test[scored_test$outcome == "false positive", example_cols]
false_positive_examples <- false_positive_examples[order(-false_positive_examples$score), ][
  1:min(8, nrow(false_positive_examples)), ]

# I sort missed fraud by amount because the most expensive miss is the most
# informative one for the report.
missed_fraud_examples <- scored_test[scored_test$outcome == "missed fraud", example_cols]
missed_fraud_examples <- missed_fraud_examples[order(-missed_fraud_examples$amt), ][
  1:min(8, nrow(missed_fraud_examples)), ]

# Mean decrease in Gini for the random forest. It can favour continuous over
# categorical features, but it is good enough as a sanity check that the top
# variables match the exploratory analysis.
final_importance <- data.frame()
if (final_model$type == "rf") {
  imp <- randomForest::importance(final_model$fit, type = 2)
  final_importance <- data.frame(
    feature = rownames(imp),
    importance = as.numeric(imp[, 1]),
    stringsAsFactors = FALSE
  )
  final_importance <- final_importance[order(-final_importance$importance), ]
}

cat("\n=== Final test ===\n")
print(final_results, row.names = FALSE, digits = 4)
cat("\n=== Final test, all threshold rules ===\n")
print(final_rule_results, row.names = FALSE, digits = 4)
cat("\n=== Top feature importance ===\n")
print(head(final_importance, 12), row.names = FALSE, digits = 4)
cat("Test positives ", sum(cc_test$is_fraud), " of ", nrow(cc_test), "\n")
cat("Precision lift ", round(final_results$precision / mean(cc_test$is_fraud), 2), "x\n")


# ---- 7. Save the tables the report uses ---------------------------------

experiment_runtime_sec <- as.numeric(difftime(Sys.time(), experiment_started_at, units = "secs"))
experiment_info <- list(
  seed_note = "Internal split uses seed=1. randomForest fits use seeds 1001 and 1002.",
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  runtime_sec = experiment_runtime_sec,
  train_rows = nrow(cc_train),
  test_rows = nrow(cc_test)
)

save(
  validation_results, final_results, final_rule_results, best_selection,
  final_threshold, final_importance, test_curve,
  top_alert_examples, true_positive_examples, false_positive_examples,
  missed_fraud_examples, experiment_info,
  file = file.path(root, "cc_fraud_experiments.RData")
)
message("Saved cc_fraud_experiments.RData")


# ---- 8. Render the PDF report -------------------------------------------

if (skip_render) {
  message("Skipping PDF render (--skip-render).")
} else {
  if (!nzchar(Sys.which("pandoc"))) {
    stop("Pandoc is required to render the report. Install with brew install pandoc",
         call. = FALSE)
  }
  if (!requireNamespace("tinytex", quietly = TRUE) && !nzchar(Sys.which("pdflatex"))) {
    stop("A LaTeX engine is required to render the report. Install TinyTeX in R with install.packages('tinytex') and then tinytex::install_tinytex()",
         call. = FALSE)
  }
  message("Rendering Credit_card_fraud_report.pdf")
  rmarkdown::render(
    input = file.path(root, "Credit_card_fraud_report.Rmd"),
    output_format = "pdf_document",
    output_file = "Credit_card_fraud_report.pdf",
    quiet = FALSE
  )
}

message("\nDone. Outputs:")
message("  - cc_fraud_data.RData")
message("  - cc_fraud_experiments.RData")
if (!skip_render) message("  - Credit_card_fraud_report.pdf")
