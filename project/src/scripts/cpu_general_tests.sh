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

for IN in ./input/*.in; do
  echo "# $0> Testing on input $IN at `date "+%Y-%m-%d %I:%M:%S %p"`"
  NAME_IN="$( basename -s .in "$IN" )"

  N_SOURCES="$(ls omp-*-test.c 2>/dev/null | wc -l)"

  for SOURCE in omp-*-test.c; do
    echo "# $0> Testing $SOURCE with STATIC at `date "+%Y-%m-%d %I:%M:%S %p"` ($IN)"
    NAME="$( basename -s .c "$SOURCE" )"

    if [ "$N_SOURCES" -eq 1 ]; then
      RESULT="$DIR_RESULTS/cpu-general-sta-$NAME_IN.txt"
    else
      RESULT="$DIR_RESULTS/cpu-general-sta-$NAME_IN-$NAME.txt"
    fi
    echo -n "TIME: " > "$RESULT"
    date "+%Y-%m-%d %I:%M:%S %p" >> "$RESULT"
    echo "SOURCE: $SOURCE static" >> "$RESULT"
    echo >> "$RESULT"
    mkdir -p "./output/cpu-general-sta/$NAME"
    SCHEDULE="static" INPUT="$IN" DIR_OUTPUT="./output/cpu-general-sta/$NAME" "$(dirname "$0")/demo-openmp.sh" "./build/bin/$NAME" "$N_REPS" >> "$RESULT"

    if [ "$N_SOURCES" -eq 1 ]; then
      RESULT="$DIR_RESULTS/cpu-general-1-sta-$NAME_IN.txt"
    else
      RESULT="$DIR_RESULTS/cpu-general-1-sta-$NAME_IN-$NAME.txt"
    fi
    echo -n "TIME: " > "$RESULT"
    date "+%Y-%m-%d %I:%M:%S %p" >> "$RESULT"
    echo "SOURCE: $SOURCE static" >> "$RESULT"
    echo >> "$RESULT"
    mkdir -p "./output/cpu-general-1-sta/$NAME"
    SCHEDULE="static" INPUT="$IN" DIR_OUTPUT="./output/cpu-general-1-sta/$NAME" "$(dirname "$0")/demo-openmp-1.sh" "./build/bin/$NAME" "$N_REPS" >> "$RESULT"
  done

  for CHUNK_SIZE in 950 1000 1050; do
    for SOURCE in omp-*-test.c; do
      echo "# $0> Testing $SOURCE with CHUNK_SIZE=$CHUNK_SIZE at `date "+%Y-%m-%d %I:%M:%S %p"` ($IN)"
      NAME="$( basename -s .c "$SOURCE" )"

      if [ "$N_SOURCES" -eq 1 ]; then
        RESULT="$DIR_RESULTS/cpu-general-dyn$CHUNK_SIZE-$NAME_IN.txt"
      else
        RESULT="$DIR_RESULTS/cpu-general-dyn$CHUNK_SIZE-$NAME_IN-$NAME.txt"
      fi
      echo -n "TIME: " > "$RESULT"
      date "+%Y-%m-%d %I:%M:%S %p" >> "$RESULT"
      echo "SOURCE: $SOURCE dynamic CHUNK_SIZE=$CHUNK_SIZE" >> "$RESULT"
      echo >> "$RESULT"
      mkdir -p "./output/cpu-general-dyn$CHUNK_SIZE/$NAME"
      SCHEDULE="dynamic,$CHUNK_SIZE" INPUT="$IN" DIR_OUTPUT="./output/cpu-general-dyn$CHUNK_SIZE/$NAME" "$(dirname "$0")/demo-openmp.sh" "./build/bin/$NAME" "$N_REPS" >> "$RESULT"

      if [ "$N_SOURCES" -eq 1 ]; then
        RESULT="$DIR_RESULTS/cpu-general-1-dyn$CHUNK_SIZE-$NAME_IN.txt"
      else
        RESULT="$DIR_RESULTS/cpu-general-1-dyn$CHUNK_SIZE-$NAME_IN-$NAME.txt"
      fi
      echo -n "TIME: " > "$RESULT"
      date "+%Y-%m-%d %I:%M:%S %p" >> "$RESULT"
      echo "SOURCE: $SOURCE dynamic CHUNK_SIZE=$CHUNK_SIZE" >> "$RESULT"
      echo >> "$RESULT"
      mkdir -p "./output/cpu-general-1-dyn$CHUNK_SIZE/$NAME"
      SCHEDULE="dynamic,$CHUNK_SIZE" INPUT="$IN" DIR_OUTPUT="./output/cpu-general-1-dyn$CHUNK_SIZE/$NAME" "$(dirname "$0")/demo-openmp-1.sh" "./build/bin/$NAME" "$N_REPS" >> "$RESULT"
    done
  done
done

shopt -u nullglob
