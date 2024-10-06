#!/usr/bin/env bash

set -e

lualatex proposal
bibtex proposal
lualatex proposal
lualatex proposal

