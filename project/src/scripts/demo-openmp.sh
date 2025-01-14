#!/bin/bash

if [ -z "$INPUT" ]; then
  echo "$0> The variable 'INPUT' is unset. Aborting script." >&2
  exit 1
fi

if [ ! -e "$INPUT" ]; then
  echo "$0> The variable 'INPUT=$INPUT' is not a valid file. Aborting script." >&2
  exit 1
elif [ ! -r "$INPUT" ]; then
  echo "$0> The variable 'INPUT=$INPUT' is a file than cannot be read, try to change grants. Aborting script." >&2
  exit 1
fi

if [ -z "$DIR_OUTPUT" ]; then
  echo "$0> The variable 'DIR_OUTPUT' is unset. Aborting script." >&2
  exit 1
fi

if [ ! -d "$DIR_OUTPUT" ]; then
  echo "$0> The variable 'DIR_INPUT=$DIR_OUTPUT' is not a valid directory. Aborting script." >&2
  exit 1
fi

readonly CORES=`cat /proc/cpuinfo | grep processor | wc -l`

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

INPUT_NAME="$( basename -s .in "$INPUT" )"
OUTPUT="${DIR_OUTPUT}/${INPUT_NAME}"
echo "INPUT: $INPUT"
echo "OUTPUT: ${OUTPUT}-P${CORES}-R{1-${N_REPS}}.out"
echo -n -e "$CORES\t"

for rep in `seq $N_REPS`; do
  if [ -z "$SCHEDULE" ]; then
    RESULT="$( OMP_NUM_THREADS=$CORES "$PROG" < "$INPUT" 2>&1 > "${OUTPUT}-P${CORES}-R${rep}.out" )"
  else
    RESULT="$( OMP_SCHEDULE="$SCHEDULE" OMP_NUM_THREADS=$CORES "$PROG" < "$INPUT" 2>&1 > "${OUTPUT}-P${CORES}-R${rep}.out" )"
  fi
  EXEC_TIME="$( echo "$RESULT" | grep "Execution time" | sed 's/Execution time (s) //' )"
  echo -n -e "${EXEC_TIME}\t"
done
echo
