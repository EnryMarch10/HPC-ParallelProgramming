# MPI - First occurrence of a value in a vector

Write a MPI program that solves the following problem.
Given a non-empty integer array `v[0..N-1]` of length $N$, and an integer $k$, find the position (index) of the first occurrence
of $k$ in `v[]`; if $k$ is not present, the result must be $N$.

For example, if `v[] = {3, 15, -1, 15, 21, 15, 7}` and `k = 15`, the result is 1, since `v[1]` is the first occurrence of `15`.
If $k$ were 37, the result is 7, since 37 is not present and the length of the array must be returned.

You may assume that:

- the array length $N$ is much larger than the number $P$ of MPI processes.

- The array length $N$ is an integer multiple of $P$.

- At the beginning, the array length $N$ and the value $k$ are known by all processes; however, the content of `v[]` is known by
  process 0 only.

- At the end, process 0 should receive the result.

To compile:

```shell
mpicc -std=c99 -Wall -Wpedantic mpi-first-pos.c -o mpi-first-pos
```

To execute:

```shell
mpirun -n 4 ./mpi-first-pos [N [k]]
```

This program initializes the input array as `v[] = {0, 1, ..., N/4, 0, 1, ..., N/4, ...}`.

Example:

```shell
mpirun -n 4 ./mpi-first-pos 1000 -73
```

should return 1000 (not fount);

```shell
mpirun -n 4 ./mpi-first-pos 1000 132
```

should return 132.

## Files

- [mpi-first-pos.c](base/mpi-first-pos.c)
