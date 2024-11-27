/****************************************************************************
 *
 * cuda-odd-even.cu - Odd-even sort
 *
 * Copyright (C) 2017--2024 by Moreno Marzolla <https://www.moreno.marzolla.name/>
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

#include "hpc.h"

/* if *a > *b, swap them. Otherwise do nothing */
void cmp_and_swap(int *a, int *b)
{
    if (*a > *b) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

/* Odd-even transposition sort */
void odd_even_sort(int *v, int n)
{
    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            /* (even, odd) comparisons */
            for (int i = 0; i < n - 1; i += 2) {
                cmp_and_swap(&v[i], &v[i + 1]);
            }
        } else {
            /* (odd, even) comparisons */
            for (int i = 1; i < n - 1; i += 2) {
                cmp_and_swap(&v[i], &v[i + 1]);
            }
        }
    }
}

/**
 * Return a random integer in the range [a..b]
 */
int randab(int a, int b)
{
    return a + (rand() % (b - a + 1));
}

/**
 * Fill vector x with a random permutation of the integers 0..n-1
 */
void fill(int *x, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = i;
    }
    for(int i = 0; i < n - 1; i++) {
        const int j = randab(i, n - 1);
        const int tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

/**
 * Check correctness of the result
 */
int check(const int *x, int n)
{
    for (int i = 0; i < n; i++) {
        if (x[i] != i) {
            fprintf(stderr, "Check FAILED: x[%d]=%d, expected %d\n", i, x[i], i);
            return 0;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main(int argc, char *argv[])
{
    int *x;
    int n = 128 * 1024;
    const int MAX_N = 512 * 1024 * 1024;
    double tstart, elapsed;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [len]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > MAX_N) {
        fprintf(stderr, "FATAL: the maximum length is %d\n", MAX_N);
        return EXIT_FAILURE;
    }

    const size_t SIZE = n * sizeof(*x);

    /* Allocate space for x on host */
    x = (int *) malloc(SIZE);
    assert(x != NULL);
    fill(x, n);

    tstart = hpc_gettime();
    odd_even_sort(x, n);
    elapsed = hpc_gettime() - tstart;
    printf("Sorted %d elements in %f seconds\n", n, elapsed);

    /* Check result */
    check(x, n);

    /* Cleanup */
    free(x);
    return EXIT_SUCCESS;
}
