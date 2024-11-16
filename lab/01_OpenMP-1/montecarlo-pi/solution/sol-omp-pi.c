/****************************************************************************
 *
 * omp-pi.c - Monte Carlo approximation of PI
 *
 * Copyright (C) 2017--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/* The rand_r() function is available only if _XOPEN_SOURCE=600 */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs */

/* Generate `n` random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1. */
unsigned int generate_points(unsigned int n)
{
#if 0
    /* This version uses neither the "parallel for" nor the
       "reduction" directives. It is instructive to try to parallelize
       the "for" loop by hand, but in practice you should never do
       that unless there are specific reasons. */
    const int n_threads = omp_get_max_threads();
    unsigned int my_n_inside[n_threads];

#pragma omp parallel num_threads(n_threads) default(none) shared(n, my_n_inside, n_threads)
    {
        const int my_id = omp_get_thread_num();
        /* We make sure that exactly `n` points are generated. Note
           that the right-hand side of the assignment can NOT be
           simplified algebraically, since the '/' operator here is
           the truncated integer division and a/c + b/c != (a+b)/c
           (e.g., a=5, b=5, c=2, a/c + b/c == 4, (a+b)/c == 5). */
        const unsigned int local_n = (n * (my_id + 1)) / n_threads - (n * my_id) / n_threads;
        unsigned int my_seed = 17 + 19 * my_id;
        my_n_inside[my_id] = 0;
        for (int i = 0; i < local_n; i++) {
            /* Generate two random values in the range [-1, 1] */
            const double x = (2.0 * rand_r(&my_seed) / (double) RAND_MAX) - 1.0;
            const double y = (2.0 * rand_r(&my_seed) / (double) RAND_MAX) - 1.0;
            if (x * x + y * y <= 1.0) {
                my_n_inside[my_id]++;
            }
        }
    } /* end of the parallel region */
    unsigned int n_inside = 0;
    for (int i = 0; i < n_threads; i++) {
        n_inside += my_n_inside[i];
    }
    return n_inside;
#else
    unsigned int n_inside = 0;
    /* This is a case where it is necessary to split the "omp
       parallel" and "omp for" directives. Indeed, each thread uses a
       private `my_seed` variable to keep track of the seed of the
       pseudo-random number generator. The simplest way to create such
       variable is to first create a parallel region, and define a
       local (private) variable `my_seed` before using the `omp for`
       construct to execute the loop in parallel. */
#pragma omp parallel default(none) shared(n, n_inside)
    {
        const int my_id = omp_get_thread_num();
        unsigned int my_seed = 17 + 19 * my_id;
#pragma omp for reduction(+:n_inside)
        for (int i=0; i<n; i++) {
            /* Generate two random values in the range [-1, 1] */
            const double x = (2.0 * rand_r(&my_seed) / (double) RAND_MAX) - 1.0;
            const double y = (2.0 * rand_r(&my_seed) / (double) RAND_MAX) - 1.0;
            if (x * x + y * y <= 1.0) {
                n_inside++;
            }
        }
    } /* end of the parallel region */
    return n_inside;
#endif
}

int main(int argc, char *argv[])
{
    unsigned int n_points = 10000;
    unsigned int n_inside;
    const double PI_EXACT = 3.14159265358979323846;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n_points]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n_points = atol(argv[1]);
    }

    printf("Generating %u points...\n", n_points);
    const double tstart = omp_get_wtime();
    n_inside = generate_points(n_points);
    const double elapsed = omp_get_wtime() - tstart;
    const double pi_approx = 4.0 * n_inside / (double) n_points;
    printf("PI approximation %f, exact %f, error %f%%\n", pi_approx, PI_EXACT, 100.0*fabs(pi_approx - PI_EXACT) / PI_EXACT);
    printf("Elapsed time: %f\n", elapsed);

    return EXIT_SUCCESS;
}
