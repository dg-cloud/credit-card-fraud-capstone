# Credit-card fraud — prepare train/test feature tables from Hugging Face CSVs.
#
# Source: pointe77/credit-card-transaction
# Uses the dataset's existing train/test split. CC_TRAIN_N / CC_TEST_N cap the
# number of rows read from each CSV for fast iteration; unset them or raise
# them well above file size to read the full files.
#
# Output: cc_fraud_data.RData with cc_train, cc_test, meta.

options(stringsAsFactors = FALSE)

# Fail fast with an actionable message rather than letting a missing-package
# error bubble up after the user has already read 350MB of CSV.
if (!requireNamespace("readr", quietly = TRUE)) stop("install.packages('readr')", call. = FALSE)
if (!requireNamespace("dplyr", quietly = TRUE)) stop("install.packages('dplyr')", call. = FALSE)

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

# Resolve paths from the script's own location rather than getwd(). Lets the
# script run from any working directory (RStudio, Rscript from outside this
# folder, etc.) without breaking the data/ lookup below.
this_file <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])
root <- if (nzchar(this_file)) dirname(normalizePath(this_file)) else getwd()
train_csv <- file.path(root, "data", "credit_card_transaction_train.csv")
test_csv <- file.path(root, "data", "credit_card_transaction_test.csv")

if (!file.exists(train_csv) || !file.exists(test_csv)) {
  stop("Missing data CSVs. Run: ./download_data.sh", call. = FALSE)
}

# Defaults are intentionally smaller than the full files. Running the whole
# pipeline at full scale takes too long for routine iteration; override via
# the environment variables when producing the final submission render.
TRAIN_N <- as.integer(Sys.getenv("CC_TRAIN_N", unset = "300000"))
TEST_N <- as.integer(Sys.getenv("CC_TEST_N", unset = "150000"))
message("Reading train rows: ", TRAIN_N, " (CC_TRAIN_N)")
message("Reading test rows:  ", TEST_N, " (CC_TEST_N)")

# Skip every column by default and explicitly opt in to what the model uses.
# Cuts memory on the large CSVs and makes the feature surface obvious to anyone
# reading the script — adding a column means adding it here.
cols_keep <- cols(
  .default = col_skip(),
  trans_date_trans_time = col_character(),  # parsed to POSIXct in prep()
  cc_num = col_character(),                  # identifier string, never numeric
  merchant = col_character(),
  category = col_character(),
  amt = col_double(),
  gender = col_character(),
  state = col_character(),
  lat = col_double(),
  long = col_double(),
  city_pop = col_double(),
  job = col_character(),
  dob = col_character(),                     # parsed to Date in prep() for age_years
  unix_time = col_double(),
  merch_lat = col_double(),
  merch_long = col_double(),
  is_fraud = col_integer()
)

raw_train <- readr::read_csv(train_csv, n_max = TRAIN_N, col_types = cols_keep, progress = FALSE)
raw_test  <- readr::read_csv(test_csv,  n_max = TEST_N,  col_types = cols_keep, progress = FALSE)

# Great-circle distance in km between two lat/lon points. The model uses this
# as customer↔merchant distance to pick up card-not-present-style cases where
# the merchant geocode sits far from the cardholder address.
haversine_km <- function(lat1, lon1, lat2, lon2) {
  rad <- pi / 180
  dlat <- (lat2 - lat1) * rad
  dlon <- (lon2 - lon1) * rad
  a <- sin(dlat / 2)^2 + cos(lat1 * rad) * cos(lat2 * rad) * sin(dlon / 2)^2
  6371 * 2 * atan2(sqrt(a), sqrt(pmax(0, 1 - a)))   # 6371 km ≈ Earth's radius
}

# Build the modelling-ready feature table from a raw CSV slice. Everything that
# needs the timestamp, geo, or DOB lives here so the experiment script never
# touches raw columns.
prep <- function(df) {
  ts <- as.POSIXct(df$trans_date_trans_time, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
  dob <- as.Date(df$dob)
  trans_date <- as.Date(ts)
  age_years <- as.numeric(trans_date - dob) / 365.25

  # The dataset prefixes every merchant name with "fraud_". Strip it so the
  # same merchant lines up across rows; otherwise rate encoding sees every
  # merchant as unseen.
  merchant_clean <- sub("^fraud_", "", df$merchant)

  data.frame(
    is_fraud      = as.integer(df$is_fraud),
    amt           = df$amt,
    amt_log       = log1p(df$amt),                        # heavy right tail in raw amt — log scale is friendlier to both glm and rf
    hour          = as.integer(format(ts, "%H")),
    day           = as.integer(format(ts, "%d")),
    month         = as.integer(format(ts, "%m")),
    weekday       = factor(weekdays(ts)),
    age_years     = ifelse(is.na(age_years),               # missing DOB → median age (rare; keeps the row in the model)
                           median(age_years, na.rm = TRUE),
                           age_years),
    distance_km   = haversine_km(df$lat, df$long, df$merch_lat, df$merch_long),
    city_pop_log  = log1p(df$city_pop),                   # long-tailed; same reason as amt_log
    gender        = factor(df$gender),
    category      = factor(df$category),
    state         = factor(df$state),
    merchant      = merchant_clean,                       # kept as character; only the encoded models consume it
    cc_num        = df$cc_num,                            # ditto
    job           = df$job,                               # ditto
    stringsAsFactors = FALSE
  )
}

cc_train <- prep(raw_train)
cc_test  <- prep(raw_test)

# Align low-cardinality factor levels across train/test. A factor level present
# in cc_test but missing from cc_train would crash predict() downstream. The
# high-cardinality keys (merchant, cc_num, job) stay as character — they're
# consumed by rate encoding in the experiment script, not as factors.
for (nm in c("weekday", "gender", "category", "state")) {
  lv <- union(levels(cc_train[[nm]]), levels(cc_test[[nm]]))
  cc_train[[nm]] <- factor(as.character(cc_train[[nm]]), levels = lv)
  cc_test[[nm]]  <- factor(as.character(cc_test[[nm]]),  levels = lv)
}

# Small metadata block so the report can sanity-check what it loaded — catches
# the "forgot to re-prep after changing CC_TRAIN_N" case.
meta <- list(
  train_n_read = nrow(raw_train),
  test_n_read  = nrow(raw_test),
  train_rows   = nrow(cc_train),
  test_rows    = nrow(cc_test),
  train_rate   = mean(cc_train$is_fraud),
  test_rate    = mean(cc_test$is_fraud),
  source       = "pointe77/credit-card-transaction",
  prepared_at  = format(Sys.time(), tz = "UTC", usetz = TRUE)
)

save(cc_train, cc_test, meta, file = file.path(root, "cc_fraud_data.RData"))
message("Saved: ", file.path(root, "cc_fraud_data.RData"))
message("Train P(fraud)=", signif(meta$train_rate, 5), "  Test P(fraud)=", signif(meta$test_rate, 5))
