# Credit-card fraud capstone

HarvardX PH125.9x "choose your own" capstone. Trains a fraud-screening model on
the `pointe77/credit-card-transaction` Hugging Face dataset and turns the model
score into a small, fraud-rich review queue. Full write-up is in
`Credit_card_fraud_report.pdf`.

## Run it

One command, end-to-end. It downloads the CSVs, prepares features, fits and
compares models, evaluates the held-out test set, and re-renders the PDF.

```sh
CC_TRAIN_N=900000 CC_TEST_N=300000 Rscript credit_card_fraud_main.R
```

`CC_TRAIN_N` and `CC_TEST_N` cap the rows read from each CSV. Drop them or
raise them well above file size to use the full data. Expect proportionally
longer runtime.

Prereqs: R 4.4+, `pandoc`, and a LaTeX engine (TinyTeX is the easiest path,
`install.packages("tinytex"); tinytex::install_tinytex()`). Missing R packages
are installed automatically by the script.

Other flags:

- `Rscript credit_card_fraud_main.R --skip-render` runs the modelling but
  skips the PDF render.
- `Rscript credit_card_fraud_main.R --dry-run` checks packages and paths
  without doing any work.

## Result

A seeded random forest (`rf_base`) using transaction amount, time of
transaction, customer geography and demographics, and category. The threshold
is picked on a held-out 25% validation slice of the training file and then
carried unchanged to the held-out test set. On the test sample:

- precision 0.83 at recall 0.74 (F1 0.78)
- ROC-AUC 0.99, PR-AUC 0.82
- alert queue is roughly 180x richer in fraud than the base transaction
  stream (test-set fraud rate is about 0.0046)

A wider top-1% rule on the same model gets recall above 0.86 at precision
about 0.44. The report (§8) walks through the precision/recall tradeoff in
detail and the example tables in §9 show what the model misses and why.

## Files

| File | Purpose |
| --- | --- |
| `credit_card_fraud_main.R` | Single submission script that runs the full pipeline |
| `Credit_card_fraud_report.Rmd` | Report source (R Markdown to PDF) |
| `Credit_card_fraud_report.pdf` | Rendered report, the submission deliverable |
| `download_data.sh` | Optional shell helper for downloading the CSVs |
