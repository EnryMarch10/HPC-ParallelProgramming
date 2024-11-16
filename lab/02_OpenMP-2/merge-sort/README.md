# Merge Sort with OpenMP tasks

The file [omp-merge-sort.c](base/omp-merge-sort.c) contains a recursive
implementation of the *Merge Sort* algorithm.
The program uses *Selection Sort* when the size of the sub-vector is less than a user-defined cutoff value;
this is a standard optimization that avoids the overhead of recursive calls on small vectors.

The program generates and sorts a random permutation of $0, 1, \ldots, n-1$;
it if therefore easy to check the correctness of the result, since it must be the sequence $0, 1, \ldots, n-1$.

The goal is to parallelize the Merge Sort algorithm using OpenMP tasks as follows:

- The recursion starts inside a parallel region; only one process starts the recursion.
- Create two tasks for the two recursive calls; pay attention to the visibility (scope) of variables.
- Wait for the two sub-tasks to complete before starting the *merge* step.

Measure the execution time of the parallel program and compare it with the serial implementation.
To get meaningful results, choose an input size that requires at least a few seconds to be sorted using all available cores.

To compile:

```shell
gcc -std=c99 -Wall -Wpedantic -Werror -fopenmp omp-merge-sort.c -o omp-merge-sort
```

To execute:

```shell
./omp-merge-sort 50000
```

## Files

- [omp-merge-sort.c](base/omp-merge-sort.c)
