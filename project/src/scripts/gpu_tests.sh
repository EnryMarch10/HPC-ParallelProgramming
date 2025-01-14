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

echo "$0> make"

make purge
if [ $? -ne 0 ]; then
echo "$0> Command 'make purge' failed. Aborting script." >&2
exit 1
fi

make cuda NVCFLAGS="-DNO_CUDA_CHECK_ERROR -Wno-deprecated-gpu-targets"
if [ $? -ne 0 ]; then
  echo "$0> Command 'make cuda NVCFLAGS=\"-DNO_CUDA_CHECK_ERROR -Wno-deprecated-gpu-targets\"' failed. Aborting script." >&2
  exit 1
fi

make input
if [ $? -ne 0 ]; then
  echo "$0> Command 'make input' failed. Aborting script." >&2
  exit 1
fi

shopt -s nullglob

for SOURCE in cuda-*.cu; do
  echo "# $0> Testing $SOURCE at `date "+%Y-%m-%d %I:%M:%S %p"`"
  NAME="$( basename -s .cu "$SOURCE" )"
  echo -n "TIME: " > "$DIR_RESULTS/tput-$NAME.txt"
  date "+%Y-%m-%d %I:%M:%S %p" >> "$DIR_RESULTS/tput-$NAME.txt"
  echo "SOURCE: $SOURCE" >> "$DIR_RESULTS/tput-$NAME.txt"
  echo >> "$DIR_RESULTS/tput-$NAME.txt"
  mkdir -p ./input/weak-scaling
  mkdir -p "./output/tput/$NAME"
  INPUT_MAX_DIM=100000 PROG_INPUT=./build/bin/inputgen DIR_INPUT=./input/weak-scaling DIR_OUTPUT="./output/tput/$NAME" "$(dirname "$0")/demo-tput-cuda.sh" "./build/bin/$NAME" "$N_REPS" >> "$DIR_RESULTS/tput-$NAME.txt"
done

shopt -u nullglob
