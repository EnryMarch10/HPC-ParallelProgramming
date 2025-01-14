#!/bin/bash

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

if [ ! -f "$INPUT" ]; then
  echo >&2
  echo "$0> $INPUT not found" >&2
  echo >&2
  exit 1
fi

echo "INPUT: $INPUT"
echo

echo -n "p"
for n in `seq $N_REPS`; do
  echo -n -e "\tt$n"
done
echo

for p in `seq $CORES`; do
  echo "OUTPUT: ${OUTPUT}-P${p}-R{1-${N_REPS}}.out"
  echo -n -e "$p\t"
  for rep in `seq $N_REPS`; do
    RESULT="$( OMP_NUM_THREADS=$p "$PROG" < "$INPUT" 2>&1 > "${OUTPUT}-P${p}-R${rep}.out" )"
    EXEC_TIME="$( echo "$RESULT" | grep "Execution time" | sed 's/Execution time (s) //' )"
    echo -n -e "${EXEC_TIME}\t"
  done
  echo
done
