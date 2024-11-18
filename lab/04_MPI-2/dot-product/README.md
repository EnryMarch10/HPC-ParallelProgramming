# Dot product

The file [mpi-dot.c](base/mpi-dot.c) contains a MPI program that computes the dot product between two arrays `x[]` and `y[]` of
length $n$.
The dot product $s$ of two arrays `x[]` and `y[]` is defined as:

$$
s = \sum_{i = 0}^{n-1} x[i] \times y[i]
$$

In the provided program, the master performs the whole computation and is therefore not parallel.
The goal of this exercise is to write a parallel version.
Assume that, at the beginning of the program, `x[]` and `y[]` are known only to the master.
Therefore, they must be distributed across the processes.
Each process computes the scalar product of the assigned portions of the arrays; the master then uses `MPI_Reduce()` to sum the
partial results and compute $s$.

You may initially assume that $n$ is an exact multiple of the number of MPI processes $P$; then, relax this assumption and modify
the program so that it works with any array length $n$.
The simpler solution is to distribute the arrays using `MPI_Scatter()` and let the master take care of any excess data.
Another possibility is to use `MPI_Scatterv()` to distribute the input unevenly across the processes.

To compile:

```shell
mpicc -std=c99 -Wall -Wpedantic -Werror mpi-dot.c -o mpi-dot -lm
```

To execute:

```shell
mpirun -n P ./mpi-dot [n]
```

Example:

```shell
mpirun -n 4 ./mpi-dot 1000
```

## Files

- [mpi-dot.c](base/mpi-dot.c)
