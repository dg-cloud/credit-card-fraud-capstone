#!/usr/bin/env bash
# Optional shell helper to grab the train and test CSVs from Hugging Face
# into ./data/. credit_card_fraud_main.R does this in R already, so this
# script is only here for people who would rather download outside R.
# Idempotent. If a file already exists it is skipped.

set -euo pipefail

# Always run from the script's own folder so the data/ path is correct
# regardless of where I call it from.
cd "$(dirname "$0")"
mkdir -p data

base="https://huggingface.co/datasets/pointe77/credit-card-transaction/resolve/main"

# --fail makes curl exit with an error on HTTP errors. Without it curl would
# happily write an HTML error page into the CSV and confuse the next step.
# --retry handles the occasional flaky connection. -L follows redirects, which
# the Hugging Face CDN uses.
if [ ! -f data/credit_card_transaction_train.csv ]; then
  curl -L --fail --retry 3 -o data/credit_card_transaction_train.csv "$base/credit_card_transaction_train.csv"
fi
if [ ! -f data/credit_card_transaction_test.csv ]; then
  curl -L --fail --retry 3 -o data/credit_card_transaction_test.csv "$base/credit_card_transaction_test.csv"
fi

ls -lh data/credit_card_transaction_*.csv
