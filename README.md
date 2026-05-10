# Credit-card fraud capstone

Standalone PH125.9x choose-your-own project using the Hugging Face dataset
`pointe77/credit-card-transaction`.

## Workflow

```sh
CC_TRAIN_N=900000 CC_TEST_N=300000 Rscript credit_card_fraud_main.R
```

The scripts keep the modeling workflow explicit:

- `credit_card_fraud_main.R`: main submission script; downloads data, prepares features, runs experiments, and renders the PDF.
- `download_data.sh`: optional shell helper for downloading the dataset.
- `prepare_credit_card_data.R`: reads the train/test CSVs and builds features.
- `experiment_credit_card_models.R`: uses an internal train/validation split for model and threshold selection, then evaluates once on the held-out test rows.
- `Credit_card_fraud_report.Rmd`: final report source.
- `render_submit_pdf.R`: renders `Credit_card_fraud_report.pdf`.

The current sampled run selected `rf_base`, a seeded random forest, with a threshold chosen from validation predictions and then carried to the held-out test set. On the held-out test sample it achieved ROC-AUC near 0.99, PR-AUC near 0.82, precision near 0.83, recall near 0.74, and roughly 180x fraud concentration over the base rate. The top-1% alert rule captured more than 0.86 recall with lower precision, giving a clear operational tradeoff.
