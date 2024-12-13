#!/usr/bin/env bash

set -e

pdflatex final_report
bibtex final_report
pdflatex final_report
pdflatex final_report
