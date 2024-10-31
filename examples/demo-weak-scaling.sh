#!/bin/bash

# This script executes the parallel `omp-matmul` program with an
# increasing number of threads $p$, from 1 up to the number of logical
# cores. For each value of $p$, the program sets the input size so
# that the amount of work executed by each OpenMP thread is kept
# constant. For the `omp-matmul` program, this means that the input
# size (i.e., the side of the input matrix) is $N0 \times (p^(1/3))$
# were $N0$ is a user-defined constant and represents the input size
# when $p=1$.
#
#-----------------------------------------------------------------------
# NOTE: the computation of the problem size above is valid ONLY FOR
# THIS PROGRAM; therefore, the input size for computing the weak
# scaling efficiency should be defined by considering the actual
# computational cost of the algorithm under test. The `omp-matmul`
# program requires serial time $O(n^3)$, but other algorithms may have
# a different cost and therefore require a different formula for the
# input size.

# Last updated 2024-09-24
# Moreno Marzolla <https://www.moreno.marzolla.name/>

PROG=./omp-matmul       # name of the executable
N0=1024                 # base problem size; you may want to change this
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of (logical) cores
NREPS=5                 # number of replications

if [ ! -f "$PROG" ]; then
    echo
    echo "$PROG not found"
    echo
    exit 1
fi

echo -e "p\tt1\tt2\tt3\tt4\tt5"

for p in `seq $CORES`; do
    echo -n -e "$p\t"
    # the standard `bc` program does not provide a built-in function
    # for computing cube roots. Therefore, we need to use logarithm
    # and exponent and use the identity:
    #
    # x^(1/3) = e(l(x)/3)
    #
    # where $l(x)$ is the natural (base "e") logarithm of $x$, and
    # $e(x)$ is "e" raised to the $x$-th power $(e^x)$. The input size
    # with $p$ processors is therefore
    #
    # N0 * e(l(p)/3))
    PROB_SIZE=`echo "$N0 * e(l($p)/3)" | bc -l -q`
    for rep in `seq $NREPS`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $PROB_SIZE | grep "Execution time" | sed 's/Execution time //' )"
        echo -n -e "${EXEC_TIME}\t"
    done
    echo ""
done
