# Simulate the `schedule()` clause

OpenMP allows the use of the `schedule(static)` and `schedule(dynamic)` clauses to choose how to assign loop iterations to OpenMP
threads. The purpose of this exercise is to simulate these clauses using the `omp parallel` construct only.

The file [omp-schedule.c](https://www.moreno.marzolla.name/teaching/HPC/handouts/omp-schedule.c) contains a serial program that
creates two arrays `vin[]` and `vout[]` of length $n$ such that `vout[i] = Fib(vin[i])` for each $i$, where `Fib(k)` the k-th
number of the Fibonacci sequence.
`Fib(k)` is intentionally computed using the inefficient recursive algorithm, so that the computation time varies widely depending
on $k$.

There are two functions, `do_static()` and `do_dynamic()` that perform the computation above.

Follow these steps:

1. Modify `do_static()` to distribute the loop iterations as would be done by the `schedule(static, chunk_size)` clause using
   the `omp parallel` directive (not `omp for`).
2. Modify `do_dynamic()` to distribute the loop iterations according to the master-worker paradigm, as would be done by the
   `schedule(dynamic, chunk_size)` clause. Again, use the `omp parallel` directive (not `omp for`).

See the source code for suggestions.

To compile:

```shell
gcc -std=c99 -Wall -Wpedantic -Werror -fopenmp omp-schedule.c -o omp-schedule
```

To execute:

```shell
./omp-schedule [n]
```

Example:

```shell
OMP_NUM_THREADS=2 ./omp-schedule
```

## Files

- [omp-schedule.c](https://www.moreno.marzolla.name/teaching/HPC/handouts/omp-schedule.c)
