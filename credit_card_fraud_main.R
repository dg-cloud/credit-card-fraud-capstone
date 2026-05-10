# Main submission script for the PH125.9x credit-card fraud capstone.
#
# This is the one script you can run to reproduce the workflow:
#   1. download the public Hugging Face train/test CSV files if missing
#   2. prepare the modeling feature tables
#   3. fit/compare models and evaluate the held-out test set
#   4. optionally render the PDF report
#
# The detailed implementation is kept in helper scripts for readability:
#   - prepare_credit_card_data.R
#   - experiment_credit_card_models.R
#   - render_submit_pdf.R
#
# Example:
#   CC_TRAIN_N=900000 CC_TEST_N=300000 Rscript credit_card_fraud_main.R
#   Rscript credit_card_fraud_main.R --skip-render
#   Rscript credit_card_fraud_main.R --dry-run

options(stringsAsFactors = FALSE)

required_packages <- c("readr", "dplyr", "ggplot2", "knitr", "randomForest", "rmarkdown")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

args <- commandArgs(trailingOnly = TRUE)
dry_run <- "--dry-run" %in% args
skip_render <- "--skip-render" %in% args

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_arg) == 1L) sub("^--file=", "", script_arg) else "credit_card_fraud_main.R"
root <- dirname(normalizePath(script_path, mustWork = FALSE))
if (!dir.exists(root) || !file.exists(file.path(root, "prepare_credit_card_data.R"))) root <- getwd()
setwd(root)

message("Project root: ", normalizePath(root))
message("Training row cap: ", Sys.getenv("CC_TRAIN_N", unset = "300000"), " (set CC_TRAIN_N to change)")
message("Test row cap: ", Sys.getenv("CC_TEST_N", unset = "150000"), " (set CC_TEST_N to change)")

download_if_missing <- function(url, dest) {
  if (file.exists(dest)) {
    message("Data exists, skipping: ", dest)
    return(invisible(dest))
  }
  message("Downloading: ", url)
  utils::download.file(url, destfile = dest, mode = "wb", quiet = FALSE)
  invisible(dest)
}

data_dir <- file.path(root, "data")
if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)

base_url <- "https://huggingface.co/datasets/pointe77/credit-card-transaction/resolve/main"
train_csv <- file.path(data_dir, "credit_card_transaction_train.csv")
test_csv <- file.path(data_dir, "credit_card_transaction_test.csv")

if (dry_run) {
  message("Dry run only: validated package availability and script paths.")
  message("Expected train CSV: ", train_csv)
  message("Expected test CSV:  ", test_csv)
  quit(save = "no", status = 0)
}

download_if_missing(file.path(base_url, "credit_card_transaction_train.csv"), train_csv)
download_if_missing(file.path(base_url, "credit_card_transaction_test.csv"), test_csv)

message("\nStep 1: preparing feature tables")
source(file.path(root, "prepare_credit_card_data.R"), local = new.env(parent = globalenv()))

message("\nStep 2: fitting models and evaluating held-out test set")
source(file.path(root, "experiment_credit_card_models.R"), local = new.env(parent = globalenv()))

if (!skip_render) {
  message("\nStep 3: rendering report PDF")
  source(file.path(root, "render_submit_pdf.R"), local = new.env(parent = globalenv()))
} else {
  message("\nSkipping PDF render because --skip-render was supplied.")
}

message("\nDone. Main outputs:")
message("  - ", file.path(root, "cc_fraud_data.RData"))
message("  - ", file.path(root, "cc_fraud_experiments.RData"))
if (!skip_render) message("  - ", file.path(root, "Credit_card_fraud_report.pdf"))
