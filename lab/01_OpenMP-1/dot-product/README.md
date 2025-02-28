# Dot product

The file [omp-dot.c](base/omp-dot.c) contains a serial program that computes the
dot product of two arrays $v1[]$ and $v2[]$. The program receives the array lengths $n$ as the only parameter on the command line.
The arrays are initialized deterministically, so that their scalar product is known without computing it explicitly.
The dot product of $v1[]$ and $v2[]$ is defined as:

$$ \sum_{i = 0}^{n-1} v1[i] \times v2[i] $$

The goal of this exercise is to parallelize the serial program using the `omp parallel` construct with the appropriate clauses.
It is instructive to begin without using the `omp parallel for` directive and computing the endpoints of the iterations
explicitly.
To this aim, let $P$ be the size of the OpenMP team; partition the arrays into $P$ blocks of approximately uniform size.
Thread $p(0 \leq p < P)$ computes the dot product `my_p` of the subvectors with indices `my_start,..., my_end - 1`:

$$ my \textunderscore p: = \sum_{i=my \textunderscore start}^{my \textunderscore end-1} v1[i] \times v2[i] $$

There are several ways to accumulate partial results. One possibility is to store the value computed by thread $p$ on
`partial_p[p]`, where `partial_p[]` is an array of length $P$.
In this way each thread writes on different elements of `partial_p[]` and no race conditions are possible.
After the parallel region completes, the master thread computes the final result by summing the content of `partial_p[]`.
Be sure to handle the case where is not an integer multiple of $P$ correctly.

The solution above is instructive but tedious and inefficient. Unless there are specific reasons to do so, in practice you should
use the `omp parallel for` directive with the `reduction()` clause, and let the compiler take care of everything.

To compile:

```shell
gcc -std=c99 -Wall -Wpedantic -Werror -fopenmp omp-dot.c -o omp-dot
```

To execute:

```shell
./omp-dot [n]
```

For example, if you want to use two OpenMP threads:

```shell
OMP_NUM_THREADS=2 ./omp-dot 1000000
```

## File

- [omp-dot.c](base/omp-dot.c)
