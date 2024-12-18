#!/bin/bash

readonly PROG=./build/bin/omp-skyline
readonly INPUT=./input/strong-scaling/worst-N100000-D10.in
readonly OUTPUT=./output/strong-scaling/worst-N100000-D10
readonly CORES=`cat /proc/cpuinfo | grep processor | wc -l`
readonly N_REPS=10

make omp > /dev/null
if [ $? -ne 0 ]; then
  echo "Make failed. Aborting script." >&2
  exit 1
fi

if [ ! -f "$PROG" ]; then
    echo >&2
    echo "$PROG not found" >&2
    echo >&2
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo >&2
    echo "$INPUT not found" >&2
    echo >&2
    exit 1
fi

echo -e "p\tt1\tt2\tt3\tt4\tt5\tt6\tt7\tt8\tt9\tt10"

for p in `seq $CORES`; do
    echo -n -e "$p\t"
    for rep in `seq $N_REPS`; do
        RESULT="$( OMP_NUM_THREADS=$p "$PROG" < "$INPUT" 2>&1 > "${OUTPUT}-P${p}.out" )"
        EXEC_TIME="$( echo "$RESULT" | grep "Execution time" | sed 's/Execution time (s) //' )"
        echo -n -e "${EXEC_TIME}\t"
    done
    echo
done
