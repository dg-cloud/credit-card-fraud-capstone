# Credit-card fraud — validation experiments and final held-out test.
#
# This is the main method-development script. Workflow:
#   1. Split the prepared cc_train into an internal fit set + validation set.
#   2. Fit a small ladder of candidate models on the fit set.
#   3. Score the validation set; pick the best model + threshold rule by F1.
#   4. Refit the chosen model on the full cc_train and score cc_test once.
#
# Important: cc_test never participates in model selection or threshold choice —
# that would leak the answer into the metric. Thresholds are derived from
# validation predictions (not in-sample fit predictions), so the F1 reported
# during selection isn't optimistically biased by the model fitting itself.
#
# Output: cc_fraud_experiments.RData — validation table, final test results,
# example transactions for the report, feature importance, run metadata.

if (!requireNamespace("randomForest", quietly = TRUE)) stop("install.packages('randomForest')", call. = FALSE)
suppressPackageStartupMessages(library(randomForest))

experiment_started_at <- Sys.time()

# Resolve paths from the script location, same reason as in prepare_credit_card_data.R.
this_file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])
root <- if (nzchar(this_file)) dirname(normalizePath(this_file)) else getwd()
load(file.path(root, "cc_fraud_data.RData"))

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

# ROC-AUC via the rank identity. Equivalent to the Mann-Whitney U / Wilcoxon
# formula and equals the probability that a random fraud row scores above a
# random non-fraud row. NA-safe so a constant score doesn't crash the loop.
auc_roc <- function(y, score) {
  y <- as.integer(y)
  score <- as.numeric(score)
  ok <- is.finite(score) & !is.na(y)
  y <- y[ok]
  score <- score[ok]
  n1 <- sum(y == 1L); n0 <- sum(y == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  r <- rank(score)
  (sum(r[y == 1L]) - n1 * (n1 + 1) / 2) / (n0 * n1)
}

# Average precision (≈ area under the precision-recall curve). For an imbalanced
# problem like fraud, PR-AUC is more informative than ROC-AUC because both axes
# are tied to the positive class. Random ranking gives PR-AUC ≈ base rate.
avg_precision <- function(y, score) {
  y <- as.integer(y)
  score <- as.numeric(score)
  ok <- is.finite(score) & !is.na(y)
  y <- y[ok]; score <- score[ok]
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

# ---------------------------------------------------------------------------
# Threshold rules (cutoffs that turn scores into alerts)
# ---------------------------------------------------------------------------
# Each rule answers a different operational question, so the report keeps several:
#   - top_X%      → "I have capacity to review X% of transactions, alert me on those"
#   - best_f1     → "balance precision and recall, no operational constraint"
#   - recall80    → "I want to catch ≥80% of fraud, what's the smallest queue that achieves this?"
# All rules return a numeric score cutoff; class_metrics() then converts to TP/FP/etc.

threshold_top_rate <- function(score, rate) {
  # Alert the top `rate` fraction. type=1 picks an actual data point so the
  # cutoff is reachable and stable across run-to-run sampling.
  as.numeric(stats::quantile(score, 1 - rate, na.rm = TRUE, type = 1))
}

threshold_best_f1 <- function(y, score) {
  # Walk the score-ordered list, compute F1 at every position, return the score
  # at the F1 maximum. Linear-time alternative to a grid search over thresholds.
  y <- as.integer(y); score <- as.numeric(score)
  ok <- is.finite(score) & !is.na(y)
  y <- y[ok]; score <- score[ok]
  ord <- order(-score)
  y <- y[ord]; score <- score[ord]
  tp <- cumsum(y == 1L)
  fp <- cumsum(y == 0L)
  fn <- sum(y == 1L) - tp
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
  f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
  score[which.max(f1)]
}

threshold_recall_target <- function(y, score, target = 0.8) {
  # Smallest cutoff that captures `target` of all positives. If no cutoff
  # reaches the target (very rare), fall back to alerting everything.
  y <- as.integer(y); score <- as.numeric(score)
  ord <- order(-score)
  y <- y[ord]; score <- score[ord]
  recall <- cumsum(y == 1L) / sum(y == 1L)
  idx <- which(recall >= target)[1]
  if (is.na(idx)) min(score, na.rm = TRUE) else score[idx]
}

# ---------------------------------------------------------------------------
# Per-row metrics
# ---------------------------------------------------------------------------

# Confusion-matrix-derived metrics for a given threshold. Bundled into one
# data.frame so the main loop can rbind() rows cleanly.
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

# Threshold-free metrics about the score itself (useful for picking the
# strongest model before deciding what threshold to apply). Brier and log-loss
# are kept around for completeness even though the report focuses on AUCs.
prob_metrics <- function(y, score) {
  p <- pmax(pmin(as.numeric(score), 1 - 1e-15), 1e-15)   # clip for log-loss
  y <- as.integer(y)
  data.frame(
    roc_auc = auc_roc(y, score),
    pr_auc = avg_precision(y, score),
    brier = mean((y - p)^2),
    log_loss = -mean(y * log(p) + (1 - y) * log(1 - p)),
    stringsAsFactors = FALSE
  )
}

# Down-sampled (recall, precision, fpr) trace for plotting ROC and PR curves
# in the report. n_points caps the curve resolution so the saved RData stays
# small even at full file size.
curve_points <- function(y, score, n_points = 250) {
  y <- as.integer(y)
  score <- as.numeric(score)
  ord <- order(-score)
  y <- y[ord]
  score <- score[ord]
  idx <- unique(pmax(1L, floor(seq(1, length(y), length.out = min(n_points, length(y))))))
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

# ---------------------------------------------------------------------------
# Train/validation split (stratified)
# ---------------------------------------------------------------------------

# Stratified 75/25 split on cc_train. Stratification matters here because
# fraud is rare — a plain random split could leave the validation set with
# only a handful of positive rows in a smaller sample run.
split_fit_idx <- function(y, frac = 0.75, seed = 1) {
  set.seed(seed)
  pos <- which(y == 1L); neg <- which(y == 0L)
  sort(c(sample(pos, floor(frac * length(pos))),
         sample(neg, floor(frac * length(neg)))))
}

# ---------------------------------------------------------------------------
# Smoothed target (rate) encoding for high-cardinality fields
# ---------------------------------------------------------------------------
#
# For high-cardinality keys (merchant, cc_num, job, ...) one-hot encoding is
# infeasible. Instead, replace each value with its smoothed historical fraud
# rate plus a log-count "how much do we know about this entity" feature.
#
# Smoothing formula: rate = (k + α·g) / (n + α)
#   k = positives for this key in fit data
#   n = total rows for this key in fit data
#   g = global fraud rate in fit data (prior we shrink toward)
#   α = strength of the prior (50 here — a low-volume merchant is pulled
#       strongly toward the global rate; a high-volume one is barely moved).
#
# Caveat: when called with new_df == fit_df during training, the rate features
# use the same labels that the model is fitted on. That's mild target leakage
# for the encoded variants; an out-of-fold scheme would be cleaner. The report
# notes this and the selected rf_base model doesn't use these encodings.
add_rate_encoding <- function(fit_df, new_df, key, prefix, alpha = 50) {
  kfit <- fit_df[[key]]
  knew <- new_df[[key]]
  y <- fit_df$is_fraud
  global <- mean(y)
  n <- tapply(y, kfit, length)
  pos <- tapply(y, kfit, sum)
  rate <- (pos + alpha * global) / (n + alpha)
  m <- match(knew, names(rate))                                 # NA = unseen key in new_df
  out <- data.frame(
    rate = ifelse(is.na(m), global, as.numeric(rate[m])),       # unseen → fall back to prior
    count = ifelse(is.na(m), 0, as.numeric(n[m])),
    stringsAsFactors = FALSE
  )
  names(out) <- paste0(prefix, c("_rate", "_log_count"))
  out[[paste0(prefix, "_log_count")]] <- log1p(out[[paste0(prefix, "_log_count")]])
  out
}

# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

# Build the model-ready feature frame. The base block always includes the
# structured features. include_encodings = TRUE adds the smoothed rate
# encodings plus a max_entity_rate feature (the worst rate any of the row's
# keys carries — useful for the model to spot "this row touches at least one
# very risky entity").
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

# ---------------------------------------------------------------------------
# Model fit / predict
# ---------------------------------------------------------------------------

fit_model <- function(id, fit_df) {
  if (id == "amount_rule") return(list(id = id, type = "rule", feature = "amt_log"))

  include_enc <- id %in% c("glm_encoded", "rf_encoded")
  x <- make_design(fit_df, fit_df, include_encodings = include_enc)

  if (grepl("^glm", id)) {
    dat <- x; dat$is_fraud <- fit_df$is_fraud
    # maxit raised because the encoded variant has more columns and can need
    # a few extra IRLS iterations to converge; the default 25 is too low.
    fit <- suppressWarnings(
      glm(is_fraud ~ ., data = dat, family = binomial(),
          control = glm.control(maxit = 100))
    )
    return(list(id = id, type = "glm", fit = fit, include_enc = include_enc))
  }

  if (grepl("^rf", id)) {
    dat <- x; dat$y_fac <- factor(ifelse(fit_df$is_fraud == 1L, "Y", "N"))
    n1 <- sum(dat$y_fac == "Y"); n0 <- sum(dat$y_fac == "N")

    # Deterministic per-model seed so a re-run produces identical metrics.
    # rf_base = 1001, rf_encoded = 1002.
    set.seed(1000 + match(id, c("rf_base", "rf_encoded"), nomatch = 0L))

    # Down-sampled-per-tree balancing: each tree sees up to 15k negatives and
    # 8k positives. Without this, the trees would see ~600x more negatives
    # than positives and split poorly on the rare class. 180 trees is enough
    # for the OOB estimate to settle on this sample.
    fit <- randomForest::randomForest(
      y_fac ~ ., data = dat, ntree = 180L,
      sampsize = c(min(15000L, n0), min(8000L, n1)),
      importance = TRUE
    )
    return(list(id = id, type = "rf", fit = fit, include_enc = include_enc))
  }

  stop("unknown id", call. = FALSE)
}

predict_model <- function(model, fit_df, new_df) {
  if (model$type == "rule") return(as.numeric(new_df[[model$feature]]))
  # Encodings recomputed against new_df with fit_df as the source of truth.
  # That's how leakage protection works at scoring time: encodings for the
  # held-out rows are looked up against fit-set rates only.
  x <- make_design(fit_df, new_df, include_encodings = model$include_enc)
  if (model$type == "glm") {
    return(suppressWarnings(as.numeric(predict(model$fit, newdata = x, type = "response"))))
  }
  as.numeric(predict(model$fit, newdata = x, type = "prob")[, "Y"])
}

# ---------------------------------------------------------------------------
# Validation phase: fit on 75% of cc_train, score on the other 25%
# ---------------------------------------------------------------------------

fit_idx <- split_fit_idx(cc_train$is_fraud, frac = 0.75, seed = 1)
fit_df <- cc_train[fit_idx, , drop = FALSE]
val_df <- cc_train[-fit_idx, , drop = FALSE]
cat("Fit:", nrow(fit_df), "positives:", sum(fit_df$is_fraud), "\n")
cat("Val:", nrow(val_df), "positives:", sum(val_df$is_fraud), "\n")

# Candidate ladder ordered by complexity. Reading the validation table
# top-to-bottom should show whether each step earned its complexity.
model_ids <- c("amount_rule", "glm_base", "glm_encoded", "rf_base", "rf_encoded")

# Threshold rules. Each is a function (y, score) → cutoff. y is unused for the
# top-rate rules but kept in the signature so they share the same shape.
rules <- list(
  top_0.5pct = function(y, s) threshold_top_rate(s, 0.005),
  top_1pct   = function(y, s) threshold_top_rate(s, 0.01),
  top_2pct   = function(y, s) threshold_top_rate(s, 0.02),
  best_f1    = function(y, s) threshold_best_f1(y, s),
  recall80   = function(y, s) threshold_recall_target(y, s, 0.8)
)

rows <- list()
fitted <- list()
for (id in model_ids) {
  cat("Fitting", id, "\n")
  model <- fit_model(id, fit_df)
  fitted[[id]] <- model

  # Critical: thresholds are derived from val_score, NOT fit_score. Picking
  # the threshold on in-sample fit data would be optimistic for the random
  # forests (their fit-set scores are nearly perfect), and the chosen cutoff
  # would not transfer well to held-out data.
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

# Best by validation F1 wins. Ties broken by precision (stricter queue first).
best_selection <- validation_results[1, ]
cat("\nSelected:", best_selection$model, best_selection$rule, "\n")

# ---------------------------------------------------------------------------
# Final phase: refit on full cc_train, score cc_test once
# ---------------------------------------------------------------------------

# Refit on the full cc_train so the production model uses every available row.
# The validation-derived threshold stays the same — we already chose it on
# held-out data; reapplying it to the test scores is the correct policy.
final_model <- fit_model(best_selection$model, cc_train)
test_score <- predict_model(final_model, cc_train, cc_test)
final_threshold <- best_selection$threshold

final_results <- cbind(
  data.frame(model = best_selection$model, rule = best_selection$rule, stringsAsFactors = FALSE),
  prob_metrics(cc_test$is_fraud, test_score),
  class_metrics(cc_test$is_fraud, test_score, final_threshold)
)

# Same model under each threshold rule, using the validation-derived threshold
# for that rule. Lets the report show the operational tradeoff (top-1% catches
# more fraud at lower precision, etc.) without re-selecting on test.
final_rule_rows <- list()
for (rule_name in names(rules)) {
  th <- validation_results$threshold[
    validation_results$model == best_selection$model &
      validation_results$rule == rule_name
  ][1]
  final_rule_rows[[length(final_rule_rows) + 1L]] <- cbind(
    data.frame(model = best_selection$model, rule = rule_name, stringsAsFactors = FALSE),
    prob_metrics(cc_test$is_fraud, test_score),
    class_metrics(cc_test$is_fraud, test_score, th)
  )
}
final_rule_results <- do.call(rbind, final_rule_rows)
final_rule_results <- final_rule_results[
  order(-final_rule_results$f1, -final_rule_results$recall, na.last = TRUE), ]

# Curve points for the report's ROC and PR plots.
test_curve <- curve_points(cc_test$is_fraud, test_score)

# ---------------------------------------------------------------------------
# Inspectable example transactions for the report
# ---------------------------------------------------------------------------
# The aggregate metrics can feel abstract. Saving a handful of named example
# rows lets the report show what each error type actually looks like — top
# alerts, correctly-caught fraud, false alerts, and missed fraud.

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

# Top 12 alerts (highest scores), regardless of whether they were correct.
top_alert_examples <- scored_test[order(-scored_test$score), example_cols][1:12, ]

# Top 8 of each outcome class, sorted by score (or amount for missed fraud,
# since the most-expensive miss is the most informative).
true_positive_examples <- scored_test[scored_test$outcome == "true positive", example_cols]
true_positive_examples <- true_positive_examples[order(-true_positive_examples$score), ][
  1:min(8, nrow(true_positive_examples)), ]

false_positive_examples <- scored_test[scored_test$outcome == "false positive", example_cols]
false_positive_examples <- false_positive_examples[order(-false_positive_examples$score), ][
  1:min(8, nrow(false_positive_examples)), ]

missed_fraud_examples <- scored_test[scored_test$outcome == "missed fraud", example_cols]
missed_fraud_examples <- missed_fraud_examples[order(-missed_fraud_examples$amt), ][
  1:min(8, nrow(missed_fraud_examples)), ]

# Feature importance only makes sense for the random forest. Mean decrease in
# Gini (type=2) is the standard impurity-based importance — caveat that it can
# overstate continuous variables vs categorical ones, but it's adequate for a
# sanity check that the top features match the exploratory analysis.
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
cat("\n=== Final test: selected model under all threshold rules ===\n")
print(final_rule_results, row.names = FALSE, digits = 4)
cat("\n=== Top feature importance ===\n")
print(head(final_importance, 12), row.names = FALSE, digits = 4)
cat("Test positives:", sum(cc_test$is_fraud), "of", nrow(cc_test), "\n")
cat("Precision lift:", round(final_results$precision / mean(cc_test$is_fraud), 2), "x\n")

# ---------------------------------------------------------------------------
# Persist everything the report needs
# ---------------------------------------------------------------------------

experiment_runtime_sec <- as.numeric(difftime(Sys.time(), experiment_started_at, units = "secs"))
experiment_info <- list(
  seed_note   = "Internal split uses seed=1; randomForest fits use deterministic seeds 1001/1002.",
  r_version   = paste(R.version$major, R.version$minor, sep = "."),
  runtime_sec = experiment_runtime_sec,
  train_rows  = nrow(cc_train),
  test_rows   = nrow(cc_test)
)

save(
  validation_results, final_results, final_rule_results, best_selection,
  final_threshold, final_importance, test_curve,
  top_alert_examples, true_positive_examples, false_positive_examples, missed_fraud_examples,
  experiment_info,
  file = file.path(root, "cc_fraud_experiments.RData")
)
message("Saved: ", file.path(root, "cc_fraud_experiments.RData"))
