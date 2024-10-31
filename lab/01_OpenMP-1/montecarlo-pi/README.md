# Monte Carlo approximation of $\pi$

The file [omp-pi.c](https://www.moreno.marzolla.name/teaching/HPC/handouts/omp-pi.c) implements a serial *Monte Carlo* algorithm
for computing the approximate value of $\pi$. Monte Carlo algorithms use pseudo-random numbers to evaluate some function of
interest.

![Monte Carlo](img/pi_Monte_Carlo.svg)

*Monte Carlo computation of the value of $\pi$.*

The idea is simple (see the figure). We generate $N$ random points uniformly distributed over a square with corners at $(-1, -1)$
and $(1, 1)$, and count the number of points falling inside the circle with center $(0, 0)$ and unitary radius.

Then, we have:

$$ \frac{\text{N. of points inside the circle}}{\text{Total n. of points}}
   \approx \frac{\text{Area of circle}}{\text{Area of enclosing square}} $$

from which, substituting the appropriate variables:

$$ \frac{x}{N} \approx \frac{\pi}{4} $$

hence $\pi \approx 4x / N$. This estimate becomes more accurate as the number of points $N$ increases.

The goal of this exercise is to modify the serial program to make use of shared-memory parallelism using OpenMP.

## The hard (and inefficient) way

Start with a version that uses the omp parallel construct. Let $P$ be the number of OpenMP threads; then, the program operates as
follows:

1. The user specifies the number $N$ of points to generate as a command-line parameter, and the number $P$ of OpenMP threads using
   the OMP_NUM_THREADS environment variable.
2. Thread $p$ generates points using `generate_points()` and stores the result in `inside[p]`. `inside[]` is an integer array of
   length $P$ that must be declared outside the parallel region, since it must be shared across all OpenMP threads.

At the end of the parallel region, the master (thread 0) computes $x$ as the sum of the content of `inside[]`; from this the
estimate of $\pi$ can be computed as above.

You may initially assume that the number $N$ of points is a multiple of $P$; when you get a working program, relax this assumption
to make the computation correct for any value of $N$.

## The better way

A better approach is to let the compiler parallelize the “for” loop in `generate_points()` using `omp parallel` and `omp for`.
There is a problem, though: function `int rand(void)` is not thread-safe since it modifies a global state variable, so it can not
be called concurrently by multiple threads. Instead, we use int `rand_r(unsigned int *seed)` which is thread-safe but requires
that each thread keeps a local seed. We split the `omp parallel` and `omp for` directives, so that a different local seed can be
given to each thread like so:

```C
#pragma omp parallel default(none) shared(n, n_inside)
{
        const int my_id = omp_get_thread_num();
        /* Initialization of my_seed is arbitrary */
        unsigned int my_seed = 17 + 19 * my_id;
        ...
#pragma omp for reduction(+:n_inside)
        for (int i = 0; i < n; i++) {
                /* call rand_r(&my_seed) here... */
                ...
        }
        ...
}
```

Compile with:

```shell
gcc -std=c99 -Wall -Wpedantic -Werror -fopenmp omp-pi.c -o omp-pi -lm
```

Run with:

```shell
./omp-pi [N]
```

For example, to compute the approximate value of $\pi$ using $P = 4$ OpenMP threads and $N = 20000$ points:

```shell
OMP_NUM_THREADS=4 ./omp-pi 20000
```

## Files

- [omp-pi.c](https://www.moreno.marzolla.name/teaching/HPC/handouts/omp-pi.c)
