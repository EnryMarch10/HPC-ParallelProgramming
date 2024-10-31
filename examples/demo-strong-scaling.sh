#!/bin/bash

# This script executes the parallel `omp-matmul` program with an
# increasing number of threads p, from 1 up to the number of logical
# cores. For each run, we use the same input size so that the
# execution times can be used to compute the speedup and the strong
# scaling efficiency. Each run is repeated `NREPS` times; the script
# prints all individual execution times on standard output.
#-----------------------------------------------------------------------
# NOTE: the problem size `PROB_SIZE` is the number of rows (or
# columns) of the input matrices. You may want to change the value
# according to the performance of the text machine. Ideally, the
# problem size should be large enough (at least a few seconds, ideally
# 10s or more) so that the OpenMP and scheduling overheads are
# minimized.

# Last updated on 2024-09-24
# Moreno Marzolla <https://www.moreno.marzolla.name/>

PROG=./omp-matmul       # name of the executable
PROB_SIZE=1500          # problem size; you may want to change this
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of (logical) cores
NREPS=5                 # number of replications.

if [ ! -f "$PROG" ]; then
    echo
    echo "$PROG not found"
    echo
    exit 1
fi

echo -e "p\tt1\tt2\tt3\tt4\tt5"

for p in `seq $CORES`; do
    echo -n -e "$p\t"
    for rep in `seq $NREPS`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $PROB_SIZE | grep "Execution time" | sed 's/Execution time //' )"
        echo -n -e "${EXEC_TIME}\t"
    done
    echo ""
done
