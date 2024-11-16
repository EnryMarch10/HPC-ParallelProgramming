/****************************************************************************
 *
 * omp-dot.c - Dot product
 *
 * Copyright (C) 2018--2023 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

void fill(int *v1, int *v2, size_t n)
{
    const int seq1[3] = { 3, 7, 18};
    const int seq2[3] = {12, 0, -2};
    for (size_t i = 0; i < n; i++) {
        v1[i] = seq1[i % 3];
        v2[i] = seq2[i % 3];
    }
}

int dot(const int *v1, const int *v2, size_t n)
{
#if 0
    /* This version uses neither "parallel for" nor "reduction"
       directives; although this solution should not be used in
       practice, it is instructive to try it. */
    const int P = omp_get_max_threads();
    int partial_p[P];
#pragma omp parallel default(none) shared(P, v1, v2, n, partial_p)
    {
        const int my_id = omp_get_thread_num();
        const size_t my_start = (n * my_id) / P;
        const size_t my_end = (n * (my_id + 1)) / P;
        int my_p = 0;
        /* printf("Thread %d P=%d my_start=%lu my_end=%lu\n", my_id, P, (unsigned long) my_start, (unsigned long) my_end); */
        for (size_t j = my_start; j < my_end; j++) {
            my_p += v1[j] * v2[j];
        }
        partial_p[my_id] = my_p;
        /* printf("partial_sum[%d]=%d\n", my_id, partial_sum[my_id]); */
    } /* implicit barrier here */

    /* we are outside a parallel region, so what follows is done by
       the master only */
    int result = 0;
    for (int i = 0; i < P; i++) {
        result += partial_p[i];
    }
#else
    /* This is the efficient solution that relies on the "parallel
       for" and "reduction" directives */
    int result = 0;
#pragma omp parallel for default(none) shared(v1, v2, n) reduction(+:result)
    for (size_t i = 0; i < n; i++) {
        result += v1[i] * v2[i];
    }
#endif
    return result;
}

int main(int argc, char *argv[])
{
    size_t n = 10 * 1024 * 1024l; /* array length */
    const size_t n_max = 512 * 1024 * 1024l; /* max length */
    int *v1, *v2;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atol(argv[1]);
    }

    if (n > n_max) {
        fprintf(stderr, "FATAL: Array too long (requested length=%lu, maximum length=%lu\n", (unsigned long) n, (unsigned long) n_max);
        return EXIT_FAILURE;
    }

    printf("Initializing array of length %lu\n", (unsigned long) n);
    v1 = (int *) malloc(n * sizeof(v1[0]));
    assert(v1 != NULL);
    v2 = (int *) malloc(n * sizeof(v2[0]));
    assert(v2 != NULL);
    fill(v1, v2, n);

    const int expect = n % 3 == 0 ? 0 : 36;

    const double tstart = omp_get_wtime();
    const int result = dot(v1, v2, n);
    const double elapsed = omp_get_wtime() - tstart;

    if (result == expect) {
        printf("Test OK\n");
    } else {
        printf("Test FAILED: expected %d, got %d\n", expect, result);
    }
    printf("Elapsed time: %f\n", elapsed);
    free(v1);
    free(v2);

    return EXIT_SUCCESS;
}
