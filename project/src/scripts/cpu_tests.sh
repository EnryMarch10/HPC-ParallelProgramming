#!/bin/bash

if [ -z "$DIR_RESULTS" ]; then
  echo "$0> The variable 'DIR_RESULTS' is unset." >&2
  exit 1
fi

if [ ! -d "$DIR_RESULTS" ]; then
  echo "$0> The variable 'DIR_RESULTS=$DIR_RESULTS' is not a valid directory." >&2
  exit 1
fi

if [ $# -gt 1 ]; then
  echo "$0> Usage: $0 [N_REPS]" >&2
  exit 1
fi

N_REPS=10

if [ $# -eq 1 ]; then
  if [[ "$1" =~ ^-?[0-9]+$ ]]; then
    N_REPS="$1"
  else
    echo "$0> N_REPS has to be an integer number" >&2
    exit 1
  fi
fi

make purge
if [ $? -ne 0 ]; then
  echo "$0> Command 'make purge' failed. Aborting script." >&2
  exit 1
fi

make omp
if [ $? -ne 0 ]; then
  echo "$0> Command 'make omp' failed. Aborting script." >&2
  exit 1
fi

make input
if [ $? -ne 0 ]; then
  echo "$0> Command 'make input' failed. Aborting script." >&2
  exit 1
fi

shopt -s nullglob

for SOURCE in omp-*.c; do
  if [[ $SOURCE != *-test.c ]]; then
    echo "# $0> Testing $SOURCE with STATIC at `date "+%Y-%m-%d %I:%M:%S %p"`"
    NAME="$( basename -s .c "$SOURCE" )"

    echo -n "TIME: " > "$DIR_RESULTS/strong-scaling-$NAME.txt"
    date "+%Y-%m-%d %I:%M:%S %p" >> "$DIR_RESULTS/strong-scaling-$NAME.txt"
    echo "SOURCE: $SOURCE static" >> "$DIR_RESULTS/strong-scaling-$NAME.txt"
    echo >> "$DIR_RESULTS/strong-scaling-$NAME.txt"
    mkdir -p "./output/strong-scaling/$NAME"
    INPUT=./input/worst-N50000-D10.in OUTPUT="./output/strong-scaling/$NAME/worst-N50000-D10" "$(dirname "$0")/demo-strong-scaling.sh" "./build/bin/$NAME" "$N_REPS" >> "$DIR_RESULTS/strong-scaling-$NAME.txt"

    echo -n "TIME: " > "$DIR_RESULTS/weak-scaling-$NAME.txt"
    date "+%Y-%m-%d %I:%M:%S %p" >> "$DIR_RESULTS/weak-scaling-$NAME.txt"
    echo "SOURCE: $SOURCE static" >> "$DIR_RESULTS/weak-scaling-$NAME.txt"
    echo >> "$DIR_RESULTS/weak-scaling-$NAME.txt"
    mkdir -p ./input/weak-scaling
    mkdir -p "./output/weak-scaling/$NAME"
    INPUT_MAX_DIM=50000 PROG_INPUT=./build/bin/inputgen DIR_INPUT=./input/weak-scaling DIR_OUTPUT="./output/weak-scaling/$NAME" "$(dirname "$0")/demo-weak-scaling.sh" "./build/bin/$NAME" "$N_REPS" >> "$DIR_RESULTS/weak-scaling-$NAME.txt"

    echo -n "TIME: " > "$DIR_RESULTS/tput-$NAME.txt"
    date "+%Y-%m-%d %I:%M:%S %p" >> "$DIR_RESULTS/tput-$NAME.txt"
    echo "SOURCE: $SOURCE static" >> "$DIR_RESULTS/tput-$NAME.txt"
    echo >> "$DIR_RESULTS/tput-$NAME.txt"
    mkdir -p "./output/tput-omp/$NAME"
    INPUT_MAX_DIM=100000 PROG_INPUT=./build/bin/inputgen DIR_INPUT=./input/weak-scaling DIR_OUTPUT="./output/tput-omp/$NAME" "$(dirname "$0")/demo-tput-openmp.sh" "./build/bin/$NAME" "$N_REPS" >> "$DIR_RESULTS/tput-$NAME.txt"

    echo -n "TIME: " > "$DIR_RESULTS/tput-1-$NAME.txt"
    date "+%Y-%m-%d %I:%M:%S %p" >> "$DIR_RESULTS/tput-1-$NAME.txt"
    echo "SOURCE: $SOURCE static" >> "$DIR_RESULTS/tput-1-$NAME.txt"
    echo >> "$DIR_RESULTS/tput-1-$NAME.txt"
    mkdir -p "./output/tput-omp-1/$NAME"
    INPUT_MAX_DIM=100000 PROG_INPUT=./build/bin/inputgen DIR_INPUT=./input/weak-scaling DIR_OUTPUT="./output/tput-omp-1/$NAME" "$(dirname "$0")/demo-tput-openmp-1.sh" "./build/bin/$NAME" "$N_REPS" >> "$DIR_RESULTS/tput-1-$NAME.txt"
  fi
done
