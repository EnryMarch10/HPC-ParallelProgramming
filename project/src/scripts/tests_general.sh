#!/bin/bash

RESULTS=./results/cpu_general

mkdir -p "$RESULTS"
echo "### $0> Start CPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
DIR_RESULTS="$RESULTS" "$(dirname "$0")/cpu_general_tests.sh"
echo "### $0> Ended CPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
