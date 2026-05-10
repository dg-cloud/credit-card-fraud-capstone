#!/usr/bin/env bash
# Download the train/test CSVs from the pointe77/credit-card-transaction
# dataset on Hugging Face into ./data/. Idempotent: skips files that already
# exist, so re-running after a partial download is safe.

set -euo pipefail

# Always run from the script's directory so the data/ path is correct
# regardless of the caller's working directory.
cd "$(dirname "$0")"
mkdir -p data

base="https://huggingface.co/datasets/pointe77/credit-card-transaction/resolve/main"

# --fail makes curl exit non-zero on HTTP errors (otherwise it would silently
# write an HTML error page into the CSV file). --retry handles transient
# network blips. -L follows redirects (HF uses CDN redirects).
if [ ! -f data/credit_card_transaction_train.csv ]; then
  curl -L --fail --retry 3 -o data/credit_card_transaction_train.csv "$base/credit_card_transaction_train.csv"
fi
if [ ! -f data/credit_card_transaction_test.csv ]; then
  curl -L --fail --retry 3 -o data/credit_card_transaction_test.csv "$base/credit_card_transaction_test.csv"
fi

ls -lh data/credit_card_transaction_*.csv
