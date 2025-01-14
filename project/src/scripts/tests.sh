#!/bin/bash

RESULTS_CPU=./results/cpu
RESULTS_GPU=./results/gpu

mkdir -p "$RESULTS_CPU"
mkdir -p "$RESULTS_GPU"
echo "### $0> Start CPU general specific tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
"$(dirname "$0")/tests_general_specific.sh"
echo "### $0> Ended CPU general specific tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
echo "### $0> Start CPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
DIR_RESULTS="$RESULTS_CPU" "$(dirname "$0")/cpu_tests.sh"
echo "### $0> Ended CPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
echo "### $0> Start GPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
DIR_RESULTS="$RESULTS_GPU" "$(dirname "$0")/gpu_tests.sh"
echo "### $0> Ended GPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
