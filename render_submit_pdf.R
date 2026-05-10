# Render the credit-card fraud capstone report to PDF.
#
# Wraps rmarkdown::render() with three preflight checks that point at the
# specific install command for the missing piece — a missing LaTeX engine in
# the middle of a long R session is annoying enough that fast, actionable
# error messages here pay for themselves.

if (!requireNamespace("rmarkdown", quietly = TRUE)) {
  stop("Install rmarkdown first: install.packages('rmarkdown')", call. = FALSE)
}

if (!nzchar(Sys.which("pandoc"))) {
  stop("Pandoc is required. Install with: brew install pandoc", call. = FALSE)
}

# tinytex is the easiest LaTeX path in R. If it's not installed, fall back to
# checking for a system pdflatex (e.g. an existing TeX Live install) before
# giving up.
if (!requireNamespace("tinytex", quietly = TRUE) && !nzchar(Sys.which("pdflatex"))) {
  stop(
    "A LaTeX engine is required. Install TinyTeX in R with: ",
    "install.packages('tinytex'); tinytex::install_tinytex()",
    call. = FALSE
  )
}

# Resolve the script's directory so render() finds the Rmd and the RData
# files regardless of the caller's working directory.
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE)[1])
root <- if (length(file_arg) == 1L && nzchar(file_arg)) dirname(normalizePath(file_arg)) else getwd()
setwd(root)

message("Rendering Credit_card_fraud_report.pdf ...")
rmarkdown::render(
  input = "Credit_card_fraud_report.Rmd",
  output_format = "pdf_document",
  output_file = "Credit_card_fraud_report.pdf",
  quiet = FALSE
)
message("Done: ", file.path(root, "Credit_card_fraud_report.pdf"))
