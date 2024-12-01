/****************************************************************************
 *
 * cuda-matsum.cu - Matrix-matrix addition
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
#include <math.h>
#include <assert.h>

#include "hpc.h"

void matsum(float *p, float *q, float *r, int n)
{
    /* [TODO] Modify the body of this function to
       - allocate memory on the device
       - copy p and q to the device
       - call an appropriate kernel
       - copy the result from the device to the host
       - free memory on the device
    */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            r[i * n + j] = p[i * n + j] + q[i * n + j];
        }
    }
}

/* Initialize square matrix p of size nxn */
void fill(float *p, int n)
{
    int k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p[i * n + j] = k;
            k = (k + 1) % 1000;
        }
    }
}

/* Check result */
int check(float *r, int n)
{
    int k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabsf(r[i * n + j] - 2.0 * k) > 1e-5) {
                fprintf(stderr, "Check FAILED: r[%d][%d] = %f, expeted %f\n", i, j, r[i * n + j], 2.0 * k);
                return 0;
            }
            k = (k + 1) % 1000;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main(int argc, char *argv[])
{
    float *p, *q, *r;
    int n = 1024;
    const int max_n = 5000;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n > max_n) {
        fprintf(stderr, "FATAL: the maximum allowed matrix size is %d\n", max_n);
        return EXIT_FAILURE;
    }

    printf("Matrix size: %d x %d\n", n, n);

    const size_t size = n * n * sizeof(*p);

    /* Allocate space for p, q, r */
    p = (float *) malloc(size);
    assert(p != NULL);
    fill(p, n);
    q = (float *) malloc(size);
    assert(q != NULL);
    fill(q, n);
    r = (float *) malloc(size);
    assert(r != NULL);

    const double tstart = hpc_gettime();
    matsum(p, q, r, n);
    const double elapsed = hpc_gettime() - tstart;

    printf("Elapsed time (including data movement): %f\n", elapsed);
    printf("Throughput (Melements/s): %f\n", n * n / (1e6 * elapsed));

    /* Check result */
    check(r, n);

    /* Cleanup */
    free(p);
    free(q);
    free(r);

    return EXIT_SUCCESS;
}
