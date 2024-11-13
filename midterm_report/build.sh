#!/usr/bin/env bash

set -e

pdflatex midterm_report
bibtex midterm_report
pdflatex midterm_report
pdflatex midterm_report
