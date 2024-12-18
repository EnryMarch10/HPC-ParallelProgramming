#!/bin/bash

readonly PROG_INPUT=./build/bin/inputgen
readonly PROG=./build/bin/omp-skyline
readonly CORES=`cat /proc/cpuinfo | grep processor | wc -l`
readonly N_REPS=10
readonly N0=22361 # (100000 / sqrt(20))
readonly D=10

make omp > /dev/null
if [ $? -ne 0 ]; then
  echo "make failed. Aborting script." >&2
  exit 1
fi

if [ ! -f "$PROG" ]; then
    echo >&2
    echo "$PROG not found" >&2
    echo >&2
    exit 1
fi

echo -e "p\tt1\tt2\tt3\tt4\tt5\tt6\tt7\tt8\tt9\tt10"

for p in `seq $CORES`; do
    echo -n -e "$p\t"
    # Calculates the size rounding to the closest integer
    PROB_SIZE=`echo "$N0 * sqrt($p) + 0.5" | bc -l -q | cut -d. -f1`
    INPUT="./input/weak-scaling/worst-N${PROB_SIZE}-D${D}.in"
    OUTPUT="./output/weak-scaling/worst-N${PROB_SIZE}-D${D}"
    if [ ! -e "$INPUT" ]; then
        "$PROG_INPUT" "$PROB_SIZE" "$D" > "$INPUT"
        if [ $? -ne 0 ]; then
        echo "$PROG_INPUT failed while creating '${INPUT}'. Aborting script." >&2
        exit 1
        fi
    fi
    for rep in `seq $N_REPS`; do
        RESULT="$( OMP_NUM_THREADS=$p "$PROG" < "$INPUT" 2>&1 > "${OUTPUT}-P${p}.out" )"
        EXEC_TIME="$( echo "$RESULT" | grep "Execution time" | sed 's/Execution time (s) //' )"
        echo -n -e "${EXEC_TIME}\t"
    done
    echo
done
