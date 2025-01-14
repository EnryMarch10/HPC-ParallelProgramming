#!/bin/bash

if [ -z "$DIR_INPUT" ]; then
  echo "$0> The variable 'DIR_INPUT' is unset." >&2
  exit 1
fi

if [ ! -d "$DIR_INPUT" ]; then
  echo "$0> The variable 'DIR_INPUT=$DIR_INPUT' is not a valid directory." >&2
  exit 1
fi

if [ -z "$DIR_OUTPUT" ]; then
  echo "$0> The variable 'DIR_OUTPUT' is unset." >&2
  exit 1
fi

if [ ! -d "$DIR_OUTPUT" ]; then
  echo "$0> The variable 'DIR_INPUT=$DIR_OUTPUT' is not a valid directory." >&2
  exit 1
fi

if [ -z "$INPUT_MAX_DIM" ]; then
  echo "$0> The variable 'INPUT_MAX_DIM' is unset." >&2
  exit 1
fi

if [[ ! "$INPUT_MAX_DIM" =~ ^-?[0-9]+$ ]]; then
  echo "$0> The variable 'INPUT_MAX_DIM' should be a valid integer" >&2
  exit 1
fi

readonly CORES=`cat /proc/cpuinfo | grep processor | wc -l`
readonly N0="$( echo "$INPUT_MAX_DIM / sqrt($CORES) + 0.5" | bc -l -q | cut -d. -f1 )" # ($INPUT_MAX_DIM / sqrt($CORES))
readonly D=10

if [ $# -gt 2 ] || [ $# -lt 1 ]; then
  echo "$0> Usage: $0 PROG [N_REPS]" >&2
  exit 1
fi

N_REPS=10

if [ $# -eq 2 ]; then
  if [[ "$2" =~ ^-?[0-9]+$ ]]; then
    N_REPS="$2"
  else
    echo "$0> Usage: $0 PROG [N_REPS]" >&2
    echo "$0> Error: N_REPS has to be an integer number" >&2
    exit 1
  fi
fi

PROG="$1"

if [ ! -f "$PROG" ]; then
  echo >&2
  echo "$0> $PROG not found" >&2
  echo >&2
  exit 1
fi

echo -n "p"
for n in `seq $N_REPS`; do
  echo -n -e "\tt$n"
done
echo

for p in `seq $CORES`; do
  PROB_SIZE=`echo "$N0 * sqrt($p) + 0.5" | bc -l -q | cut -d. -f1`
  INPUT="${DIR_INPUT}/worst-N${PROB_SIZE}-D${D}.in"
  if [ ! -e "$INPUT" ]; then
    "$PROG_INPUT" "$PROB_SIZE" "$D" > "$INPUT"
    if [ $? -ne 0 ]; then
      echo "$0> $PROG_INPUT failed while creating '${INPUT}'. Aborting script." >&2
      exit 1
    fi
  fi
  OUTPUT="${DIR_OUTPUT}/worst-N${PROB_SIZE}-D${D}"
  echo "INPUT: $INPUT"
  echo "OUTPUT: ${OUTPUT}-R{1-${N_REPS}}.out"
  echo -n -e "$CORES\t"
  for rep in `seq $N_REPS`; do
    RESULT="$( "$PROG" < "$INPUT" 2>&1 > "${OUTPUT}-R${rep}.out" )"
    EXEC_TIME="$( echo "$RESULT" | grep "Execution time" | sed 's/Execution time (s) //' )"
    echo -n -e "${EXEC_TIME}\t"
  done
  echo
done
